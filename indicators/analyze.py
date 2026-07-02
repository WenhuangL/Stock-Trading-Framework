"""
analyze.py
----------
Quantitative analysis toolkit for the stock trading framework.

Provides:
  - Technical indicators  : RSI, MACD, Bollinger Bands, SMA, EMA, ATR, VWAP
  - Signal generation     : composite buy/sell signals from multiple indicators
  - DB utilities          : load historical price data from the local SQLite DB
  - Visualization         : price charts with overlaid indicators
  - Backtesting           : simple signal-based strategy simulator

Designed to work alongside webscraper.py (data collection) and
alpaca_test.py (order execution).  All heavy computation stays here;
other modules import what they need.

Requirements
------------
    pip install pandas numpy matplotlib scipy
"""

import sqlite3
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Database path (mirrors webscraper.py) ─────────────────────────────────────
_DB_PATH = Path("") / "analysis.db"


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def load_table(table_name: str, db_path: Path = _DB_PATH) -> pd.DataFrame:
    """
    Load an entire table from the local SQLite database into a DataFrame.

    Parameters
    ----------
    table_name : str
        Name of the table to load (e.g. 'yahoo_table', 'msn_table').
    db_path : Path
        Path to the SQLite .db file. Defaults to analysis/analysis.db.

    Returns
    -------
    pd.DataFrame
        All rows from the table, or an empty DataFrame if the table
        doesn't exist or the DB file is missing.
    """
    if not db_path.exists():
        print(f"[load_table] Database not found at '{db_path}'.")
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception as exc:
        print(f"[load_table] Could not read '{table_name}': {exc}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_ticker_history(
    ticker: str,
    table_name: str = "yahoo_table",
    db_path: Path = _DB_PATH,
) -> pd.DataFrame:
    """
    Load all stored rows for a single ticker from the database.

    Parameters
    ----------
    ticker : str
        Ticker symbol to filter on (e.g. 'AAPL').
    table_name : str
        Table to query.
    db_path : Path
        Path to the SQLite .db file.

    Returns
    -------
    pd.DataFrame
        Rows for the requested ticker sorted by date, or an empty
        DataFrame if no data is found.
    """
    if not db_path.exists():
        print(f"[load_ticker_history] Database not found at '{db_path}'.")
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {table_name} WHERE Ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker.upper(),),
        )
    except Exception as exc:
        print(f"[load_ticker_history] Query failed: {exc}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def list_tables(db_path: Path = _DB_PATH) -> list[str]:
    """Return the names of all tables in the database."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


# =============================================================================
# TECHNICAL INDICATORS
# All functions accept a pd.DataFrame with at minimum a 'close' column
# (lowercase, as returned by Alpaca's bars.df) and return a pd.Series
# unless noted otherwise.
# =============================================================================

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Measures momentum by comparing average gains to average losses over
    *period* trading days.  Values above 70 are traditionally considered
    overbought; values below 30 oversold.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100].
    """
    close = data["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence / Divergence (MACD).

    Returns a DataFrame with three columns:
      - 'macd'       : fast EMA − slow EMA
      - 'signal'     : EMA of the MACD line
      - 'histogram'  : macd − signal  (positive = bullish momentum)

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    fast, slow, signal : int
        EMA periods (classic defaults: 12, 26, 9).
    """
    close = data["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=data.index,
    )


def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns a DataFrame with columns:
      - 'middle'  : simple moving average
      - 'upper'   : middle + num_std × rolling std
      - 'lower'   : middle − num_std × rolling std
      - 'width'   : (upper − lower) / middle  (band-width as % of price)
      - 'pct_b'   : position of price within the bands [0, 1]

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    period : int
        Rolling window for the SMA (default 20).
    num_std : float
        Number of standard deviations for band width (default 2).
    """
    close = data["close"]
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower,
         "width": width, "pct_b": pct_b},
        index=data.index,
    )


def calculate_sma(data: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    period : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        SMA values.
    """
    return data["close"].rolling(window=period).mean()


def calculate_dollar_volume(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Rolling average dollar volume (close x volume).

    The single most important liquidity metric for thin / micro-cap names:
    share-volume alone is misleading because a $2 stock trading 500k shares
    ($1M/day) is far more liquid than a $200 stock trading the same share count
    would suggest.  Micro-cap strategies gate on a *dollar*-volume floor, not a
    share-volume floor.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'close' and 'volume' columns.
    period : int
        Rolling window in trading days (default 20).

    Returns
    -------
    pd.Series
        Rolling mean of (close * volume), in dollars.
    """
    return (data["close"] * data["volume"]).rolling(window=period).mean()


def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    period : int
        Span for the EMA.

    Returns
    -------
    pd.Series
        EMA values.
    """
    return data["close"].ewm(span=period, adjust=False).mean()


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    A measure of volatility.  Useful for position sizing and setting
    dynamic stop-loss levels.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', and 'close' columns.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        ATR values in price units.
    """
    high = data["high"]
    low = data["low"]
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price (VWAP).

    Calculated cumulatively over the supplied data window.  Most meaningful
    for intraday data; on daily bars it acts as a long-run fair-value line.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', 'close', and 'volume' columns.

    Returns
    -------
    pd.Series
        Cumulative VWAP values.
    """
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    cum_tp_vol = (typical_price * data["volume"]).cumsum()
    cum_vol = data["volume"].cumsum()
    return cum_tp_vol / cum_vol


def calculate_historical_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Annualised Historical Volatility (HV).

    Computed as the rolling standard deviation of log returns, scaled to
    an annual figure by multiplying by √252 (trading days per year).

    This is the primary volatility metric used by the universe selector to
    identify stocks in the 'sweet-spot' range (~20–60% annualised HV) that
    generate enough daily movement for the EOD reversion strategy without
    being so erratic that moves are purely fundamental rather than noise.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    period : int
        Rolling window in trading days (default 20 ≈ one calendar month).

    Returns
    -------
    pd.Series
        Annualised HV values. Values are decimals (e.g. 0.35 = 35%).
        First (period - 1) rows will be NaN.
    """
    log_returns = np.log(data["close"] / data["close"].shift(1))
    return log_returns.rolling(window=period).std() * np.sqrt(252)


def calculate_donchian_channel(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel (N-day price channel).

    Returns the highest high and lowest low over a rolling window.
    Used by the swing strategy to identify price breakouts above/below
    the recent range without look-ahead bias.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', and 'close' columns.
    period : int
        Look-back window in bars (default 20).

    Returns
    -------
    pd.DataFrame
        Columns: 'upper' (period high), 'lower' (period low),
                 'mid' (average of upper and lower).
        Note: the current bar is excluded from the channel so that a
        breakout on today's close does not compare against today's bar
        (avoids trivial same-bar membership).
    """
    upper = data["high"].shift(1).rolling(window=period).max()
    lower = data["low"].shift(1).rolling(window=period).min()
    mid   = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid}, index=data.index)


def calculate_atr_percentile(
    data: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 60,
) -> pd.Series:
    """
    ATR Percentile — current ATR ranked within its recent distribution.

    Returns a value in [0, 1] where 0.0 means the current ATR is at
    the lowest point of the lookback window and 1.0 means the highest.
    Values below 0.35 indicate a volatility contraction (squeeze).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'high', 'low', and 'close' columns.
    atr_period : int
        Period for the ATR calculation (default 14).
    lookback : int
        Rolling window over which to rank the current ATR (default 60).

    Returns
    -------
    pd.Series
        Percentile rank of current ATR, values in [0, 1].
    """
    atr = calculate_atr(data, period=atr_period)

    def _rank(window):
        if len(window) < 2:
            return np.nan
        current = window.iloc[-1]
        mn, mx = window.min(), window.max()
        if mx == mn:
            return 0.5
        return float((current - mn) / (mx - mn))

    return atr.rolling(window=lookback).apply(_rank, raw=False)


def calculate_relative_volume(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Relative Volume (RVOL).

    The ratio of today's volume to the N-day average volume.
    A value of 2.0 means the stock is trading at twice its normal volume,
    which is a strong indicator of heightened retail participation or a
    significant news event — both of which increase the likelihood of
    noise-driven price moves that mean-revert.

    Used by the universe selector as part of the retail sentiment proxy:
    stocks with consistently elevated relative volume attract more retail
    attention and therefore more sentiment-driven selling on bad days.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'volume' column.
    lookback : int
        Rolling window for the baseline average (default 20 days).

    Returns
    -------
    pd.Series
        RVOL values. 1.0 = average volume. First (lookback - 1) rows NaN.
    """
    avg_vol = data["volume"].rolling(window=lookback).mean()
    return data["volume"] / avg_vol


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a composite buy / sell signal DataFrame from multiple indicators.

    Each indicator casts a vote of +1 (bullish) or -1 (bearish).
    The 'composite_score' column sums all votes; a score ≥ +2 triggers a
    'buy' signal and ≤ -2 triggers a 'sell' signal.

    indicators used
    ---------------
    - RSI          : buy < 35, sell > 65
    - MACD         : buy when histogram turns positive, sell when negative
    - Bollinger    : buy near lower band (pct_b < 0.2), sell near upper (> 0.8)
    - SMA crossover: buy when 20-day EMA crosses above 50-day SMA

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at minimum a 'close' column.  'high', 'low', and
        'volume' unlock ATR / VWAP but are not required for signals.

    Returns
    -------
    pd.DataFrame
        Original data enriched with indicator columns and:
          'composite_score'  – integer vote tally
          'signal'           – 'buy', 'sell', or 'hold'
    """
    df = data.copy()

    # ── indicators ────────────────────────────────────────────────────────────
    df["rsi"]     = calculate_rsi(df)
    macd_df       = calculate_macd(df)
    df["macd"]    = macd_df["macd"]
    df["macd_sig"]= macd_df["signal"]
    df["macd_hist"]= macd_df["histogram"]
    bb            = calculate_bollinger_bands(df)
    df["bb_upper"]= bb["upper"]
    df["bb_lower"]= bb["lower"]
    df["bb_pct_b"]= bb["pct_b"]
    df["ema_20"]  = calculate_ema(df, 20)
    df["sma_50"]  = calculate_sma(df, 50)

    if {"high", "low", "volume"}.issubset(df.columns):
        df["atr"]  = calculate_atr(df)
        df["vwap"] = calculate_vwap(df)

    # ── Votes ─────────────────────────────────────────────────────────────────
    df["vote_rsi"]  = np.where(df["rsi"] < 35, 1, np.where(df["rsi"] > 65, -1, 0))
    df["vote_macd"] = np.where(df["macd_hist"] > 0, 1, np.where(df["macd_hist"] < 0, -1, 0))
    df["vote_bb"]   = np.where(df["bb_pct_b"] < 0.2, 1, np.where(df["bb_pct_b"] > 0.8, -1, 0))
    # EMA-20 crossing above SMA-50 → buy; crossing below → sell
    cross = df["ema_20"] - df["sma_50"]
    df["vote_cross"] = np.where(
        (cross > 0) & (cross.shift(1) <= 0), 1,
        np.where((cross < 0) & (cross.shift(1) >= 0), -1, 0),
    )

    df["composite_score"] = (
        df["vote_rsi"] + df["vote_macd"] + df["vote_bb"] + df["vote_cross"]
    )
    df["signal"] = np.where(
        df["composite_score"] >= 2, "buy",
        np.where(df["composite_score"] <= -2, "sell", "hold"),
    )

    return df


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest(
    data: pd.DataFrame,
    initial_cash: float = 10_000.0,
    take_profit_pct: float = 0.02,
    stop_loss_pct: float = 0.02,
) -> dict:
    """
    Simple signal-driven backtest using the composite signals from
    generate_signals().

    Rules
    -----
    - On a 'buy' signal:  spend all available cash to buy shares at that
      day's close price, with a TP at +take_profit_pct and SL at
      -stop_loss_pct.
    - On a 'sell' signal or TP/SL breach: close the position at close price.
    - One open position at a time.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column (and ideally all columns needed by
        generate_signals).
    initial_cash : float
        Starting portfolio value in dollars.
    take_profit_pct : float
        Profit target as a decimal (default 0.02 = 2 %).
    stop_loss_pct : float
        Stop-loss threshold as a decimal (default 0.02 = 2 %).

    Returns
    -------
    dict with keys:
        'trades'         – list of completed trade dicts
        'equity_curve'   – pd.Series of portfolio value over time
        'final_value'    – float
        'total_return'   – float (e.g. 0.15 = +15 %)
        'num_trades'     – int
        'win_rate'       – float  (winning trades / total trades)
        'max_drawdown'   – float  (maximum peak-to-trough decline)
    """
    df = generate_signals(data).reset_index(drop=True)

    cash: float = initial_cash
    shares: float = 0.0
    entry_price: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    in_position: bool = False

    equity_values: list[float] = []
    trades: list[dict] = []

    for i, row in df.iterrows():
        price: float = float(row["close"])

        # ── Check TP / SL if in a position ───────────────────────────────────
        if in_position:
            exit_reason: Optional[str] = None
            exit_price: float = price

            if price >= tp_price:
                exit_reason = "take_profit"
            elif price <= sl_price:
                exit_reason = "stop_loss"
            elif row["signal"] == "sell":
                exit_reason = "signal"

            if exit_reason:
                proceeds = shares * exit_price
                pnl = proceeds - (shares * entry_price)
                trades.append(
                    {
                        "entry_index": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
                        "pnl": round(pnl, 2),
                        "exit_reason": exit_reason,
                    }
                )
                cash += proceeds
                shares = 0.0
                in_position = False

        # ── Open a new position on buy signal ─────────────────────────────────
        if not in_position and row["signal"] == "buy" and price > 0:
            shares = cash / price
            entry_price = price
            tp_price = round(entry_price * (1 + take_profit_pct), 4)
            sl_price = round(entry_price * (1 - stop_loss_pct), 4)
            cash = 0.0
            in_position = True

        # Portfolio value at end of this bar
        equity_values.append(cash + shares * price)

    # Close any remaining open position at last price
    if in_position:
        final_price = float(df["close"].iloc[-1])
        proceeds = shares * final_price
        pnl = proceeds - (shares * entry_price)
        trades.append(
            {
                "entry_index": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": final_price,
                "shares": shares,
                "pnl": round(pnl, 2),
                "exit_reason": "end_of_data",
            }
        )
        equity_values[-1] = proceeds

    equity_curve = pd.Series(equity_values, index=df.index)
    final_value = equity_curve.iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash

    winning_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_value": round(final_value, 2),
        "total_return": round(total_return, 4),
        "num_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_price_and_indicators(
    data: pd.DataFrame,
    symbol: str = "",
    show_volume: bool = True,
    show_macd: bool = True,
    show_rsi: bool = True,
) -> None:
    """
    Plot a full technical analysis chart with price, Bollinger Bands,
    moving averages, volume, MACD, and RSI.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.  If 'high', 'low', 'volume' are
        present they will be plotted too.
    symbol : str
        Ticker label shown in the chart title.
    show_volume : bool
        Include a volume sub-chart (requires 'volume' column).
    show_macd : bool
        Include a MACD sub-chart.
    show_rsi : bool
        Include an RSI sub-chart.
    """
    df = generate_signals(data.copy())

    has_volume = "volume" in df.columns and show_volume

    # Determine subplot layout dynamically
    n_subplots = 1 + int(has_volume) + int(show_macd) + int(show_rsi)
    height_ratios = [4] + [1] * (n_subplots - 1)

    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(14, 3 * n_subplots + 2),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    if n_subplots == 1:
        axes = [axes]

    ax_idx = 0

    # ── Panel 1: Price + Bollinger Bands + Moving Averages + Signals ─────────
    ax_price = axes[ax_idx]
    ax_idx += 1

    ax_price.plot(df.index, df["close"], color="#1f77b4", linewidth=1.5, label="Close")
    ax_price.plot(df.index, df["ema_20"], color="#ff7f0e", linewidth=1.0, linestyle="--", label="EMA 20")
    ax_price.plot(df.index, df["sma_50"], color="#2ca02c", linewidth=1.0, linestyle="--", label="SMA 50")
    ax_price.plot(df.index, df["bb_upper"], color="#9467bd", linewidth=0.8, linestyle=":", label="BB Upper")
    ax_price.plot(df.index, df["bb_lower"], color="#9467bd", linewidth=0.8, linestyle=":", label="BB Lower")
    ax_price.fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.07, color="#9467bd")

    # Buy / sell markers
    buys  = df[df["signal"] == "buy"]
    sells = df[df["signal"] == "sell"]
    ax_price.scatter(buys.index,  buys["close"],  marker="^", color="#2ca02c", s=80, zorder=5, label="Buy signal")
    ax_price.scatter(sells.index, sells["close"], marker="v", color="#d62728", s=80, zorder=5, label="Sell signal")

    ax_price.set_title(f"{symbol} — Technical Analysis" if symbol else "Technical Analysis", fontsize=13)
    ax_price.set_ylabel("Price ($)")
    ax_price.legend(loc="upper left", fontsize=7, ncol=3)
    ax_price.grid(alpha=0.3)

    # ── Panel 2 (optional): Volume ────────────────────────────────────────────
    if has_volume:
        ax_vol = axes[ax_idx]
        ax_idx += 1
        colors = ["#2ca02c" if c >= o else "#d62728"
                  for c, o in zip(df["close"], df["close"].shift(1).fillna(df["close"]))]
        ax_vol.bar(df.index, df["volume"], color=colors, width=0.8, alpha=0.7)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(alpha=0.3)

    # ── Panel 3 (optional): MACD ──────────────────────────────────────────────
    if show_macd:
        ax_macd = axes[ax_idx]
        ax_idx += 1
        ax_macd.plot(df.index, df["macd"],     color="#1f77b4", linewidth=1.2, label="MACD")
        ax_macd.plot(df.index, df["macd_sig"], color="#ff7f0e", linewidth=1.0, linestyle="--", label="Signal")
        pos_hist = df["macd_hist"].clip(lower=0)
        neg_hist = df["macd_hist"].clip(upper=0)
        ax_macd.bar(df.index, pos_hist, color="#2ca02c", alpha=0.5, width=0.8)
        ax_macd.bar(df.index, neg_hist, color="#d62728", alpha=0.5, width=0.8)
        ax_macd.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left", fontsize=7)
        ax_macd.grid(alpha=0.3)

    # ── Panel 4 (optional): RSI ───────────────────────────────────────────────
    if show_rsi:
        ax_rsi = axes[ax_idx]
        ax_rsi.plot(df.index, df["rsi"], color="#8c564b", linewidth=1.2, label="RSI 14")
        ax_rsi.axhline(70, color="#d62728", linewidth=0.8, linestyle="--", label="Overbought (70)")
        ax_rsi.axhline(30, color="#2ca02c", linewidth=0.8, linestyle="--", label="Oversold (30)")
        ax_rsi.fill_between(df.index, df["rsi"], 70, where=(df["rsi"] >= 70), alpha=0.15, color="#d62728")
        ax_rsi.fill_between(df.index, df["rsi"], 30, where=(df["rsi"] <= 30), alpha=0.15, color="#2ca02c")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.legend(loc="upper left", fontsize=7)
        ax_rsi.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_equity_curve(
    backtest_result: dict,
    symbol: str = "",
    initial_cash: float = 10_000.0,
) -> None:
    """
    Plot the equity curve produced by backtest(), annotating each
    trade entry/exit and the final performance summary.

    Parameters
    ----------
    backtest_result : dict
        The dictionary returned by backtest().
    symbol : str
        Ticker label shown in the chart title.
    initial_cash : float
        Starting cash (used to draw the breakeven line).
    """
    curve   = backtest_result["equity_curve"]
    trades  = backtest_result["trades"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(curve.index, curve.values, color="#1f77b4", linewidth=1.5, label="Portfolio Value")
    ax.axhline(initial_cash, color="gray", linewidth=0.8, linestyle="--", label="Starting Cash")

    for trade in trades:
        color = "#2ca02c" if trade["pnl"] >= 0 else "#d62728"
        ax.axvline(trade["entry_index"], color=color, linewidth=0.6, alpha=0.4)

    tr_pct = backtest_result["total_return"] * 100
    md_pct = backtest_result["max_drawdown"] * 100
    wr_pct = backtest_result["win_rate"] * 100
    summary = (
        f"Final: ${backtest_result['final_value']:,.2f}  |  "
        f"Return: {tr_pct:+.1f}%  |  "
        f"Max DD: {md_pct:.1f}%  |  "
        f"Trades: {backtest_result['num_trades']}  |  "
        f"Win rate: {wr_pct:.0f}%"
    )
    ax.set_title(
        f"{symbol} — Backtest Equity Curve\n{summary}" if symbol
        else f"Backtest Equity Curve\n{summary}",
        fontsize=11,
    )
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_db_snapshot(
    table_name: str = "yahoo_table",
    top_n: int = 20,
    db_path: Path = _DB_PATH,
) -> None:
    """
    Visualize the most recent snapshot from a database table as a
    horizontal bar chart of 1-day price change, coloured by direction.

    Parameters
    ----------
    table_name : str
        Table to read (default 'yahoo_table').
    top_n : int
        Number of tickers to display (sorted by absolute 1D change).
    db_path : Path
        Path to the SQLite .db file.
    """
    df = load_table(table_name, db_path)
    if df.empty:
        print("[plot_db_snapshot] No data found.")
        return

    required = {"Ticker", "1D Change", "date"}
    if not required.issubset(df.columns):
        print(f"[plot_db_snapshot] Table must contain columns: {required}")
        return

    # Keep most recent date only
    df["date"] = pd.to_datetime(df["date"])
    latest = df[df["date"] == df["date"].max()].copy()

    # Parse "4.23%" → 4.23
    latest["change_num"] = (
        latest["1D Change"]
        .str.replace("%", "", regex=False)
        .astype(float, errors="ignore")
    )
    latest = latest.dropna(subset=["change_num"])
    latest = latest.nlargest(top_n, "change_num")
    latest = latest.sort_values("change_num")

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in latest["change_num"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(latest) * 0.35)))
    ax.barh(latest["Ticker"], latest["change_num"], color=colors, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("1-Day Change (%)")
    ax.set_title(
        f"Top {top_n} Movers — {latest['date'].iloc[-1].strftime('%Y-%m-%d')} "
        f"(from '{table_name}')"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# QUICK SUMMARY
# =============================================================================

def summarize(data: pd.DataFrame, symbol: str = "") -> None:
    """
    Print a concise summary of the latest indicator readings and signal
    for a given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    symbol : str
        Optional ticker label.
    """
    df = generate_signals(data)
    last = df.iloc[-1]

    label = f" [{symbol}]" if symbol else ""
    print(f"\n{'='*46}")
    print(f"  Analysis Summary{label}  —  latest bar")
    print(f"{'='*46}")
    print(f"  Close price      : ${float(last['close']):.2f}")
    print(f"  RSI (14)         : {float(last['rsi']):.1f}")
    print(f"  MACD histogram   : {float(last['macd_hist']):.4f}")
    print(f"  Bollinger %B     : {float(last['bb_pct_b']):.2f}")
    print(f"  EMA 20 / SMA 50  : {float(last['ema_20']):.2f} / {float(last['sma_50']):.2f}")
    if "atr" in df.columns:
        print(f"  ATR (14)         : {float(last['atr']):.2f}")
    print(f"  Composite score  : {int(last['composite_score'])}")
    print(f"  Signal           : {str(last['signal']).upper()}")
    print(f"{'='*46}\n")


# =============================================================================
# EXAMPLE USAGE (run directly for a quick demo)
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf

    DEMO_SYMBOL = "NVDA"
    print(f"Downloading demo data for {DEMO_SYMBOL}...")
    raw = yf.download(DEMO_SYMBOL, period="6mo", progress=False)

    # yfinance uses Title-case columns; normalise to lowercase for consistency
    raw.columns = [c.lower() for c in raw.columns]

    summarize(raw, symbol=DEMO_SYMBOL)

    result = backtest(raw, initial_cash=10_000)
    print(f"Backtest result: {result['total_return']*100:+.1f}% total return "
          f"over {result['num_trades']} trades  |  win rate {result['win_rate']*100:.0f}%")

    plot_price_and_indicators(raw, symbol=DEMO_SYMBOL)
    plot_equity_curve(result, symbol=DEMO_SYMBOL)
