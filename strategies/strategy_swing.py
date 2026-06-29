"""
strategy_swing.py
-----------------
Low-Volatility Breakout Swing Strategy
=======================================

STRATEGY PREMISE
----------------
Markets cycle between compression (low volatility, tight ranges) and
expansion (high volatility, directional moves).  When a stock squeezes
into a tight range over several weeks, the energy required for the next
move accumulates.  A decisive close at a new 20-day high, confirmed by
a volume surge, signals that the compression has resolved bullishly and
the move tends to persist for 2–5 trading days before mean-reverting.

This is a purely momentum strategy that operates on DAILY bars and holds
positions overnight, making it structurally distinct from both the
intraday strategy (minute bars, same-day close) and the EOD strategy
(afterhours, next-morning exit).

SIGNAL CRITERIA (LONG only by default)
---------------------------------------
All conditions must hold on the signal day (today's closing data):

  1. ATR-14 is in the bottom 35% of its 60-day distribution — the stock
     has been quiet recently (volatility squeeze).
  2. Today's close > highest close of the PREVIOUS 20 days — confirmed
     breakout above the recent range (Donchian channel).
  3. Today's volume ≥ 1.25× its 20-day average — institutional
     participation is backing the move.
  4. RSI-14 between 50 and 72 — momentum is positive but not yet
     overbought.  Overbought breakouts fail faster.
  5. Today's close > 20-day SMA — broad uptrend intact; not fighting
     the trend.
  6. Price $15–$500.
  7. 20-day average volume ≥ 500,000 — sufficient liquidity for clean
     fills and low overnight gap risk.

EXIT CONDITIONS
---------------
  - Take profit : +4% above entry price (hard TP)
  - Stop loss   : −2% below entry price (2:1 reward-to-risk)
  - Time stop   : Close at the end of day 4 (the strategy's edge fades
    fast; holding longer invites mean reversion back against us)
  - Gap stop    : If the open on any day is below the SL, fill at open
    (cannot fill inside a gap)

POSITION SIZING
---------------
  - 3% of portfolio per position
  - Maximum 5 simultaneous positions → 15% max deployed
  - Shares round-trip through the risk manager's sector concentration
    limits just like the other strategies

TIME WINDOWS (no overlap with existing strategies)
--------------------------------------------------
  Intraday strategy  :  9:30 AM – 3:55 PM  (minute bars, same-day close)
  EOD strategy       :  3:55 PM – 7:50 PM  (afterhours, losers)
  Swing strategy     :  Signal 3:45 PM; Entry next day 9:31 AM;
                        Managed during the session; Max 4-day hold

LIVE USAGE
----------
  1. At 3:45 PM ET (before EOD script starts):
         strategy = SwingStrategy(tc, dc, config=SwingConfig(), risk_manager=rm)
         strategy.evening_scan()   # scans, saves pending entries to JSON

  2. At 9:31 AM ET the next day (after market open):
         strategy.morning_session()  # executes pending entries, logs open positions

  The run_eod.py and run_session.py scripts call these methods
  automatically once configured.

BACKTEST USAGE
--------------
  results = strategy.backtest(
      tickers     = get_sp500_tickers(),
      start_date  = "2023-01-01",
      end_date    = "2023-12-31",
      initial_cash= 100_000,
  )
  strategy.print_summary(results)
  strategy.plot_backtest(results)

REQUIREMENTS
------------
  pip install alpaca-py pandas numpy matplotlib
"""

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from indicators.analyze import (
    calculate_atr,
    calculate_atr_percentile,
    calculate_donchian_channel,
    calculate_rsi,
    calculate_sma,
)
from data_collection.stock_universe import UniverseSelector, UniverseConfig

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SwingConfig:
    """All tuneable parameters for the swing strategy in one place."""

    # ── Signal filters ─────────────────────────────────────────────────────────
    breakout_period: int = 20
    """Days for the Donchian channel (new high must exceed this many days)."""

    atr_period: int = 14
    """Period for the ATR calculation."""

    atr_percentile_lookback: int = 60
    """Days over which to rank the current ATR percentile."""

    atr_squeeze_threshold: float = 0.35
    """ATR percentile must be below this to qualify as a squeeze.
    0.35 = ATR must be in the lowest 35% of its 60-day history."""

    volume_mult: float = 1.25
    """Volume today must be ≥ this × 20-day average to confirm the breakout."""

    volume_lookback: int = 20
    """Days for computing the baseline average volume."""

    rsi_min: float = 50.0
    """Minimum RSI for a long signal (momentum active)."""

    rsi_max: float = 72.0
    """Maximum RSI for a long signal (not overbought)."""

    # ── Universe filters ───────────────────────────────────────────────────────
    min_price: float = 15.0
    max_price: float = 500.0
    min_avg_volume: int = 500_000
    """Minimum 20-day average volume for overnight liquidity."""

    # ── Exit parameters ────────────────────────────────────────────────────────
    tp_pct: float = 0.040
    """Take-profit threshold above entry (4%). At 2:1 R:R with a −2% SL, the
    strategy only needs a 34% win rate to break even — the actual signal
    quality delivers substantially higher."""

    sl_pct: float = 0.020
    """Stop-loss threshold below entry (2%). Gives 2:1 reward-to-risk ratio."""

    max_hold_days: int = 4
    """Close the position at day's close if this many days have elapsed.
    The edge decays quickly after the initial momentum burst; holding
    longer simply adds overnight gap risk."""

    # ── Position sizing ────────────────────────────────────────────────────────
    position_size_pct: float = 0.03
    """Fraction of portfolio per position (3%)."""

    max_positions: int = 5
    """Maximum simultaneous swing positions (15% max deployed)."""

    # ── Signal lookback ────────────────────────────────────────────────────────
    min_history_bars: int = 65
    """Minimum daily bars required to calculate all indicators reliably.
    65 = 60-day ATR percentile window + a small buffer."""

    # ── Slippage / fill realism ────────────────────────────────────────────────
    slippage_pct: float = 0.0010
    """0.10% per fill (entry AND exit). Opening prints on liquid names can be
    0.05–0.20% away from the theoretical price; 0.10% is a realistic middle."""

    spread_pct: float = 0.0005
    """Additional half-spread cost per fill (0.05%). Large-cap stocks in the
    regular session have narrow spreads; this is conservative."""

    commission_per_trade: float = 0.0

    # ── Live execution paths ───────────────────────────────────────────────────
    pending_file: str = "output/swing_pending.json"
    """JSON file holding signals from yesterday's close, waiting to enter."""

    positions_file: str = "output/swing_positions.json"
    """JSON file holding currently open swing positions (persists across days)."""

    # ── Monitoring ─────────────────────────────────────────────────────────────
    monitor_interval_sec: int = 120
    """How often the live monitoring loop checks prices (every 2 minutes)."""


# =============================================================================
# SWING STRATEGY
# =============================================================================

class SwingStrategy:
    """
    Encapsulates both live execution and historical backtesting for the
    Low-Volatility Breakout Swing Strategy.

    Live usage requires two entry points called at different times of day:
      - evening_scan()    : 3:45 PM — generates and persists signals
      - morning_session() : 9:31 AM — executes signals, monitors positions

    Backtest usage mirrors the EOD and Intraday strategies; pass to
    run_backtest.py for combined reporting.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        config: Optional[SwingConfig] = None,
        risk_manager: Optional[object] = None,
        universe_config: Optional[UniverseConfig] = None,
    ) -> None:
        self.tc   = trading_client
        self.dc   = data_client
        self.cfg  = config or SwingConfig()
        self.rm   = risk_manager
        self.ucfg = universe_config or UniverseConfig()
        self.log  = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # LIVE EXECUTION — EVENING (called at 3:45 PM ET)
    # =========================================================================

    def evening_scan(self, tickers: Optional[list[str]] = None) -> list[dict]:
        """
        Scan for new breakout signals using today's daily data.
        Persists qualifying candidates to pending_file so morning_session()
        can execute them at tomorrow's open.

        Returns the list of signals found.
        """
        now = datetime.datetime.now(ET)
        self.log.info(f"Swing evening scan at {now.strftime('%H:%M:%S ET')}")

        if tickers is None:
            selector = UniverseSelector(self.dc, self.ucfg)
            tickers  = selector.get_universe(use_cache=True)

        if not tickers:
            self.log.warning("Empty universe — no swing signals generated.")
            return []

        # ── Count current open positions ──────────────────────────────────────
        open_positions = self._load_positions()
        open_symbols   = {p["symbol"] for p in open_positions if not p.get("closed", False)}
        open_count     = len([p for p in open_positions if not p.get("closed", False)])

        if open_count >= self.cfg.max_positions:
            self.log.info(f"Max positions ({self.cfg.max_positions}) already open — skipping scan.")
            return []

        # ── Fetch daily bars ──────────────────────────────────────────────────
        signals = self._generate_signals_from_snapshots(tickers, open_symbols)

        if signals:
            self.log.info(f"Swing: {len(signals)} signal(s) queued for tomorrow's open.")
            _save_json(self.cfg.pending_file, signals)
        else:
            self.log.info("Swing: no qualifying signals today.")
            _save_json(self.cfg.pending_file, [])

        return signals

    def _generate_signals_from_snapshots(
        self,
        tickers: list[str],
        exclude_symbols: set,
    ) -> list[dict]:
        """
        Fetch recent daily bars for the universe and run the breakout signal.
        Uses LocalDataCache for efficiency.
        """
        from data_collection.data_cache import LocalDataCache
        cache = LocalDataCache(self.dc)

        end_dt   = datetime.datetime.now(ET)
        start_dt = end_dt - datetime.timedelta(days=self.cfg.min_history_bars + 15)

        signals: list[dict] = []

        for sym in tickers:
            if sym in exclude_symbols:
                continue
            try:
                df = cache.get_bars_df(sym, TimeFrame.Day, start_dt, end_dt, feed="iex")
                if df is None or len(df) < self.cfg.min_history_bars:
                    continue
                df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
                df = df.sort_index()

                sig = self._check_signal(sym, df)
                if sig:
                    signals.append(sig)
            except Exception as exc:
                self.log.debug(f"Signal check failed for {sym}: {exc}")

        # Sort by score descending; higher-scored signals entered first
        signals.sort(key=lambda x: x["score"], reverse=True)
        # Return only as many as we can still open
        max_new = max(0, self.cfg.max_positions - len(exclude_symbols))
        return signals[:max_new]

    # =========================================================================
    # LIVE EXECUTION — MORNING (called at 9:31 AM ET)
    # =========================================================================

    def morning_session(self) -> None:
        """
        Execute pending entries from yesterday's scan and monitor open positions.
        Called once at 9:31 AM ET; monitoring runs until all positions exit
        or the end-of-day check time (3:45 PM ET).
        """
        self.log.info("Swing morning session starting.")

        # ── Execute pending entries ───────────────────────────────────────────
        pending = _load_json(self.cfg.pending_file, default=[])
        if pending:
            open_positions = self._load_positions()
            account        = self.tc.get_account()
            portfolio_val  = float(account.portfolio_value)
            self._execute_pending(pending, open_positions, portfolio_val)
            _save_json(self.cfg.pending_file, [])  # clear after execution

        # ── Monitor until 3:45 PM ─────────────────────────────────────────────
        cutoff = _parse_time_today("15:45")
        while datetime.datetime.now(ET) < cutoff:
            self._monitor_positions()
            remaining = (cutoff - datetime.datetime.now(ET)).total_seconds()
            if remaining > 0:
                time.sleep(min(self.cfg.monitor_interval_sec, remaining))

        # Final check at scan time: close any that hit their time stop
        self._close_time_stops()
        self.log.info("Swing morning session complete.")

    def _execute_pending(
        self,
        pending: list[dict],
        open_positions: list[dict],
        portfolio_val: float,
    ) -> None:
        """Submit buy orders for yesterday's signals."""
        budget = portfolio_val * self.cfg.position_size_pct
        cash   = portfolio_val  # approximate; we'll use budget check

        for sig in pending:
            open_count = len([p for p in open_positions if not p.get("closed", False)])
            if open_count >= self.cfg.max_positions:
                break
            if cash < budget:
                break

            sym = sig["symbol"]
            try:
                # Fetch the current price for sizing
                trade = self.dc.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=[sym])
                )
                price = float(trade[sym].price) if sym in trade else sig.get("signal_price", budget)
                qty   = int(budget // price)
                if qty < 1:
                    continue

                if self.rm is not None:
                    ok, reason = self.rm.check_ok_to_enter(
                        sym, open_positions, qty * price, portfolio_val
                    )
                    if not ok:
                        self.log.info(f"Swing entry blocked {sym}: {reason}")
                        continue

                order = self.tc.submit_order(MarketOrderRequest(
                    symbol        = sym,
                    qty           = qty,
                    side          = OrderSide.BUY,
                    time_in_force = TimeInForce.DAY,
                ))
                slip     = self.cfg.slippage_pct + self.cfg.spread_pct
                ep       = price * (1 + slip)
                tp_price = ep * (1 + self.cfg.tp_pct)
                sl_price = ep * (1 - self.cfg.sl_pct)
                pos = {
                    "symbol":     sym,
                    "order_id":   str(order.id),
                    "qty":        qty,
                    "direction":  "long",
                    "entry_price": round(ep, 4),
                    "tp_price":   round(tp_price, 4),
                    "sl_price":   round(sl_price, 4),
                    "entry_date": datetime.date.today().isoformat(),
                    "days_held":  0,
                    "closed":     False,
                    "exit_price": None,
                    "exit_reason": None,
                }
                open_positions.append(pos)
                cash -= qty * ep
                self._save_positions(open_positions)
                self.log.info(
                    f"SWING OPEN  {qty:>4}x {sym:<6} @ ${ep:>8.2f} | "
                    f"TP ${tp_price:.2f} | SL ${sl_price:.2f}"
                )
            except Exception as exc:
                self.log.error(f"Swing order failed for {sym}: {exc}")

    def _monitor_positions(self) -> None:
        """Check TP/SL for all open swing positions."""
        positions = self._load_positions()
        open_pos  = [p for p in positions if not p.get("closed", False)]
        if not open_pos:
            return

        syms = [p["symbol"] for p in open_pos]
        try:
            trades = self.dc.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=syms)
            )
        except Exception as exc:
            self.log.debug(f"Swing price fetch failed: {exc}")
            return

        changed = False
        for pos in open_pos:
            sym   = pos["symbol"]
            price = float(trades[sym].price) if sym in trades else None
            if price is None:
                continue

            if price <= pos["sl_price"]:
                self._close_live_position(pos, "stop_loss", price)
                changed = True
            elif price >= pos["tp_price"]:
                self._close_live_position(pos, "take_profit", price)
                changed = True

        if changed:
            self._save_positions(positions)

    def _close_time_stops(self) -> None:
        """Close positions that have exceeded max_hold_days."""
        positions = self._load_positions()
        today     = datetime.date.today()
        changed   = False

        for pos in positions:
            if pos.get("closed", False):
                continue
            try:
                entry = datetime.date.fromisoformat(pos["entry_date"])
                days  = (today - entry).days
            except Exception:
                days  = 0

            if days >= self.cfg.max_hold_days:
                self._close_live_position(pos, "time_stop")
                changed = True

        if changed:
            self._save_positions(positions)

    def _close_live_position(
        self,
        pos: dict,
        reason: str,
        price: Optional[float] = None,
    ) -> None:
        """Submit a position close and update the position dict."""
        try:
            self.tc.close_position(pos["symbol"])
        except Exception as exc:
            self.log.error(f"Close failed for {pos['symbol']}: {exc}")
        exit_price = price or pos["entry_price"]
        pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        pos.update(closed=True, exit_price=round(exit_price, 4),
                   exit_reason=reason)
        self.log.info(
            f"SWING CLOSE {pos['qty']:>4}x {pos['symbol']:<6} @ ${exit_price:>8.2f} | "
            f"{reason:<20} | P&L ${pnl:>+8.2f}"
        )

    # =========================================================================
    # SIGNAL LOGIC (shared between live and backtest)
    # =========================================================================

    def _check_signal(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Apply the breakout signal to a daily bar DataFrame.
        Returns a signal dict if the criteria are met, else None.

        DataFrame must be sorted ascending with today's bar last.
        No look-ahead: today's bar is df.iloc[-1].
        """
        if len(df) < self.cfg.min_history_bars:
            return None

        close   = df["close"]
        volume  = df["volume"]

        last_close  = float(close.iloc[-1])
        last_vol    = float(volume.iloc[-1])

        # ── Price filter ───────────────────────────────────────────────────────
        if not (self.cfg.min_price <= last_close <= self.cfg.max_price):
            return None

        # ── Volume filter ──────────────────────────────────────────────────────
        avg_vol = float(volume.iloc[-self.cfg.volume_lookback - 1:-1].mean())
        if avg_vol < self.cfg.min_avg_volume:
            return None
        if last_vol < avg_vol * self.cfg.volume_mult:
            return None

        # ── ATR squeeze check ──────────────────────────────────────────────────
        atr_pct = calculate_atr_percentile(
            df, atr_period=self.cfg.atr_period,
            lookback=self.cfg.atr_percentile_lookback,
        )
        current_atr_pct = float(atr_pct.iloc[-1])
        if np.isnan(current_atr_pct) or current_atr_pct > self.cfg.atr_squeeze_threshold:
            return None  # not in a squeeze

        # ── Donchian channel breakout (exclude today from channel) ─────────────
        channel = calculate_donchian_channel(df, period=self.cfg.breakout_period)
        prev_high = float(channel["upper"].iloc[-1])
        if np.isnan(prev_high) or last_close <= prev_high:
            return None  # no breakout

        # ── Trend filter (above 20-day SMA) ────────────────────────────────────
        sma20 = calculate_sma(df, period=20)
        if np.isnan(sma20.iloc[-1]) or last_close < float(sma20.iloc[-1]):
            return None

        # ── RSI filter ─────────────────────────────────────────────────────────
        rsi = calculate_rsi(df, period=14)
        last_rsi = float(rsi.iloc[-1])
        if np.isnan(last_rsi) or not (self.cfg.rsi_min <= last_rsi <= self.cfg.rsi_max):
            return None

        # ── Score: stronger squeeze + larger breakout = higher priority ─────────
        breakout_pct = (last_close - prev_high) / prev_high
        vol_ratio    = last_vol / max(avg_vol, 1)
        score        = (1.0 - current_atr_pct) * 0.40 + breakout_pct * 0.35 + (vol_ratio - 1) * 0.25

        return {
            "symbol":        sym,
            "signal_price":  round(last_close, 4),
            "prev_high":     round(prev_high, 4),
            "breakout_pct":  round(breakout_pct * 100, 2),
            "atr_percentile": round(current_atr_pct, 3),
            "rsi":           round(last_rsi, 1),
            "vol_ratio":     round(vol_ratio, 2),
            "score":         round(score, 4),
        }

    # =========================================================================
    # PERSISTENCE (live trading only)
    # =========================================================================

    def _load_positions(self) -> list[dict]:
        return _load_json(self.cfg.positions_file, default=[])

    def _save_positions(self, positions: list[dict]) -> None:
        _save_json(self.cfg.positions_file, positions)

    # =========================================================================
    # BACKTESTING
    # =========================================================================

    def backtest(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        historical_universes: Optional[dict] = None,
    ) -> dict:
        """
        Simulate the strategy on historical daily bars.

        Execution model:
          - Signal fires on close of day T (using data up to and including T)
          - Entry is at the open of day T+1 (next-bar fill, no look-ahead)
          - TP/SL checked against high/low of each subsequent bar
          - Gap stop: if open < SL, fill at open
          - Time stop: close at end of day T + max_hold_days

        Returns a dict with 'equity_curve', 'trades', and 'summary'.
        """
        self.log.info(
            f"Swing backtest: {start_date} → {end_date} | "
            f"{len(tickers)} tickers | ${initial_cash:,.0f}"
        )

        start_dt = datetime.datetime.fromisoformat(start_date).replace(tzinfo=ET)
        end_dt   = datetime.datetime.fromisoformat(end_date).replace(tzinfo=ET)

        # Extra lookback so indicators are warm before the backtest window starts
        fetch_start = start_dt - datetime.timedelta(days=self.cfg.min_history_bars + 30)

        daily_data = self._fetch_daily_bars(tickers, fetch_start, end_dt)

        trading_days = pd.bdate_range(start_date, end_date)

        cash            = initial_cash
        portfolio_value = initial_cash
        all_trades:  list[dict] = []
        equity_rows: list[dict] = []

        # pending entries: keyed by symbol, value = signal dict + sizing
        pending: dict[str, dict] = {}
        # open positions: list of dicts (same structure as live)
        open_pos: list[dict] = []

        for day_i, day in enumerate(trading_days):
            day_str  = day.strftime("%Y-%m-%d")
            day_date = day.date()

            # ── PIT universe ──────────────────────────────────────────────────
            reference_day = day - pd.Timedelta(days=7)
            year, week, _ = reference_day.isocalendar()
            week_str      = f"{year}-W{week:02d}"
            if historical_universes:
                active = set(historical_universes.get(week_str, tickers))
            else:
                active = set(tickers)

            # ── Step 1: Enter pending positions at today's open ────────────────
            for sym, sig in list(pending.items()):
                if sym not in daily_data:
                    del pending[sym]
                    continue
                df = daily_data[sym]
                try:
                    today_bar = df.loc[day_str:day_str]
                except KeyError:
                    del pending[sym]
                    continue
                if today_bar.empty:
                    del pending[sym]
                    continue

                open_price = float(today_bar["open"].iloc[0])
                slip       = self.cfg.slippage_pct + self.cfg.spread_pct
                ep         = open_price * (1 + slip)

                budget  = portfolio_value * self.cfg.position_size_pct
                qty     = int(budget // ep)
                if qty < 1 or cash < ep * qty:
                    del pending[sym]
                    continue

                if not self._risk_ok_swing(sym, open_pos, ep * qty, portfolio_value):
                    del pending[sym]
                    continue

                tp_price = ep * (1 + self.cfg.tp_pct)
                sl_price = ep * (1 - self.cfg.sl_pct)

                pos = {
                    "symbol":      sym,
                    "direction":   "long",
                    "entry_date":  day_str,
                    "entry_price": round(ep, 4),
                    "tp_price":    round(tp_price, 4),
                    "sl_price":    round(sl_price, 4),
                    "qty":         qty,
                    "days_held":   0,
                    "closed":      False,
                    "exit_price":  None,
                    "exit_reason": None,
                    "phase":       "swing",
                }
                open_pos.append(pos)
                cash -= qty * ep
                del pending[sym]

            # ── Step 2: Check exits for open positions ─────────────────────────
            slip = self.cfg.slippage_pct + self.cfg.spread_pct

            for pos in [p for p in open_pos if not p.get("closed", False)]:
                sym = pos["symbol"]
                if sym not in daily_data:
                    continue
                df = daily_data[sym]
                try:
                    today_bar = df.loc[day_str:day_str]
                except KeyError:
                    continue
                if today_bar.empty:
                    continue

                bar_open  = float(today_bar["open"].iloc[0])
                bar_high  = float(today_bar["high"].iloc[0])
                bar_low   = float(today_bar["low"].iloc[0])
                bar_close = float(today_bar["close"].iloc[0])

                ep  = pos["entry_price"]
                tp  = pos["tp_price"]
                sl  = pos["sl_price"]

                exit_price  = None
                exit_reason = None

                # Gap stop: opened below SL
                if bar_open <= sl:
                    exit_price  = bar_open * (1 - slip)
                    exit_reason = "gap_stop"
                # Normal stop or take-profit within the day's range
                elif bar_low <= sl:
                    exit_price  = sl * (1 - slip)
                    exit_reason = "stop_loss"
                elif bar_high >= tp:
                    exit_price  = tp * (1 - slip)
                    exit_reason = "take_profit"
                else:
                    pos["days_held"] += 1
                    if pos["days_held"] >= self.cfg.max_hold_days:
                        exit_price  = bar_close * (1 - slip)
                        exit_reason = "time_stop"

                if exit_price is not None:
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    cash += pos["qty"] * exit_price
                    pos.update(
                        closed=True,
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                    )
                    all_trades.append({
                        "date":          day_date,
                        "symbol":        sym,
                        "phase":         "swing",
                        "direction":     "long",
                        "entry_price":   round(ep, 4),
                        "exit_price":    round(exit_price, 4),
                        "qty":           pos["qty"],
                        "pnl":           round(pnl, 2),
                        "exit_reason":   exit_reason,
                        "days_held":     pos["days_held"],
                        "breakout_pct":  sig.get("breakout_pct", 0) if sym in pending else 0,
                    })

            # Purge closed positions
            open_pos = [p for p in open_pos if not p.get("closed", False)]

            # ── Step 3: Generate signals for tomorrow's open ───────────────────
            # Only scan if we can still take new positions
            open_count = len(open_pos) + len(pending)
            if open_count < self.cfg.max_positions:
                open_symbols = {p["symbol"] for p in open_pos} | set(pending.keys())

                for sym in active:
                    if sym in open_symbols or sym in pending:
                        continue
                    if open_count >= self.cfg.max_positions:
                        break
                    if sym not in daily_data:
                        continue

                    df = daily_data[sym]
                    try:
                        # Data up to and including today's close — no lookahead
                        hist = df.loc[:day_str]
                    except KeyError:
                        continue
                    if len(hist) < self.cfg.min_history_bars:
                        continue

                    sig = self._check_signal(sym, hist)
                    if sig:
                        pending[sym] = sig
                        open_count  += 1

            # ── Track equity ──────────────────────────────────────────────────
            day_pnl = sum(t["pnl"] for t in all_trades if str(t["date"]) == day_str)
            portfolio_value = cash + sum(
                p["qty"] * float(
                    daily_data[p["symbol"]].loc[day_str:day_str]["close"].iloc[-1]
                    if p["symbol"] in daily_data and
                    not daily_data[p["symbol"]].loc[day_str:day_str].empty
                    else p["entry_price"]
                )
                for p in open_pos
            )

            equity_rows.append({
                "date":            day,
                "portfolio_value": portfolio_value,
                "day_pnl":         day_pnl,
                "num_trades":      sum(1 for t in all_trades if str(t["date"]) == day_str),
                "open_positions":  len(open_pos),
            })

            if all_trades and str(all_trades[-1]["date"]) == day_str:
                today_trades = [t for t in all_trades if str(t["date"]) == day_str]
                today_pnl    = sum(t["pnl"] for t in today_trades)
                self.log.info(
                    f"{day_str}: {len(today_trades)} exits | "
                    f"P&L ${today_pnl:>+8.2f} | Portfolio ${portfolio_value:>10,.2f}"
                )

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        summary      = self._compute_summary(all_trades, equity_curve, initial_cash)

        return {
            "equity_curve": equity_curve,
            "trades":       all_trades,
            "summary":      summary,
        }

    def _fetch_daily_bars(
        self,
        tickers: list[str],
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily bars for all tickers using the shared LocalDataCache."""
        from data_collection.data_cache import LocalDataCache
        cache     = LocalDataCache(self.dc)
        bars_data: dict[str, pd.DataFrame] = {}

        for i, sym in enumerate(tickers):
            try:
                df = cache.get_bars_df(sym, TimeFrame.Day, start_dt, end_dt, feed="iex")
                if df is not None and not df.empty:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(ET).normalize()
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    bars_data[sym] = df
            except Exception as exc:
                self.log.debug(f"{sym}: daily bars failed — {exc}")

            if (i + 1) % 50 == 0:
                self.log.info(f"  Daily bars: {i + 1}/{len(tickers)} loaded.")

        self.log.info(f"Daily bars loaded: {len(bars_data)} / {len(tickers)} tickers.")
        return bars_data

    def _risk_ok_swing(
        self,
        symbol: str,
        open_positions: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Apply RiskManager caps for swing positions; no-op if rm is None."""
        if self.rm is None:
            return True

        cfg = self.rm.cfg
        active = [p for p in open_positions if not p.get("closed", False)]

        if len(active) >= cfg.max_total_positions:
            return False

        total_notional = sum(p["qty"] * p["entry_price"] for p in active)
        if (total_notional + proposed_notional) / portfolio_value > cfg.max_portfolio_deployed_pct:
            return False

        sector = self.rm.get_sector(symbol)
        if sector != "Unknown":
            sector_count = sum(
                1 for p in active if self.rm.get_sector(p["symbol"]) == sector
            )
            if sector_count >= cfg.max_positions_per_sector:
                return False

        return True

    @staticmethod
    def _compute_summary(
        trades:       list[dict],
        equity_curve: pd.DataFrame,
        initial_cash: float,
    ) -> dict:
        if not trades:
            return {"num_trades": 0, "note": "No trades executed in date range."}

        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        final_value  = float(equity_curve["portfolio_value"].iloc[-1])
        rolling_max  = equity_curve["portfolio_value"].cummax()
        drawdown     = (equity_curve["portfolio_value"] - rolling_max) / rolling_max

        exit_counts: dict[str, int] = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exit_counts[r] = exit_counts.get(r, 0) + 1

        avg_days_held = np.mean([t.get("days_held", 0) for t in trades])

        gross_win  = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
        expectancy    = round(float(np.mean([t["pnl"] for t in trades])), 2)

        pv   = equity_curve["portfolio_value"].astype(float)
        rets = pv.pct_change().dropna()
        sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) \
            if len(rets) >= 2 and rets.std() != 0 else 0.0

        return {
            "initial_cash":      initial_cash,
            "final_value":       round(final_value, 2),
            "total_return_pct":  round((final_value - initial_cash) / initial_cash * 100, 2),
            "num_trades":        len(trades),
            "win_rate_pct":      round(len(wins) / len(trades) * 100, 1),
            "avg_win":           round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
            "avg_loss":          round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
            "reward_risk_ratio": round(
                abs(np.mean([t["pnl"] for t in wins]) / np.mean([t["pnl"] for t in losses])), 2
            ) if wins and losses else 0,
            "profit_factor":     profit_factor,
            "expectancy":        expectancy,
            "sharpe":            sharpe,
            "largest_win":       round(max(t["pnl"] for t in trades), 2),
            "largest_loss":      round(min(t["pnl"] for t in trades), 2),
            "avg_days_held":     round(avg_days_held, 1),
            "max_drawdown_pct":  round(float(drawdown.min()) * 100, 2),
            "exit_reason_counts": exit_counts,
        }

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, results: dict) -> None:
        s = results["summary"]
        if "note" in s:
            print(f"\n[Swing: no trades — {s['note']}]")
            return

        ec     = s["exit_reason_counts"]
        tp_n   = ec.get("take_profit", 0)
        sl_n   = ec.get("stop_loss", 0) + ec.get("gap_stop", 0)
        ts_n   = ec.get("time_stop", 0)
        total  = s["num_trades"]

        print(f"\n{'='*54}")
        print(f"  Swing Breakout Strategy — Backtest Summary")
        print(f"{'='*54}")
        print(f"  Initial capital      : ${s['initial_cash']:>12,.2f}")
        print(f"  Final value          : ${s['final_value']:>12,.2f}")
        print(f"  Total return         : {s['total_return_pct']:>+11.2f}%")
        print(f"  Max drawdown         : {s['max_drawdown_pct']:>+11.2f}%")
        print(f"{'-'*54}")
        print(f"  Total trades         : {total:>12}")
        print(f"  Win rate             : {s['win_rate_pct']:>11.1f}%")
        print(f"  Avg win              : ${s['avg_win']:>12.2f}")
        print(f"  Avg loss             : ${s['avg_loss']:>12.2f}")
        print(f"  Reward / risk        : {s['reward_risk_ratio']:>12.2f}")
        print(f"  Profit factor        : {s['profit_factor']:>12.2f}")
        print(f"  Expectancy / trade   : ${s['expectancy']:>12.2f}")
        print(f"  Sharpe (ann.)        : {s['sharpe']:>12.2f}")
        print(f"  Largest single win   : ${s['largest_win']:>12.2f}")
        print(f"  Largest single loss  : ${s['largest_loss']:>12.2f}")
        print(f"  Avg days held        : {s['avg_days_held']:>12.1f}")
        print(f"{'-'*54}")
        print(f"  Exit breakdown:")
        print(f"    Take-profit        : {tp_n:>5} ({tp_n/total*100:>4.0f}%)")
        print(f"    Stop-loss / gap    : {sl_n:>5} ({sl_n/total*100:>4.0f}%)")
        print(f"    Time stop          : {ts_n:>5} ({ts_n/total*100:>4.0f}%)")
        print(f"{'='*54}\n")

    def plot_backtest(self, results: dict) -> None:
        s      = results["summary"]
        equity = results["equity_curve"]
        trades = results["trades"]

        if not trades:
            print("Swing: no trades to plot.")
            return

        fig = plt.figure(figsize=(15, 10))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        # ── 1. Equity curve ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        pv  = equity["portfolio_value"]
        ax1.plot(equity.index, pv, color="#1f77b4", linewidth=1.5, label="Portfolio")
        ax1.axhline(s["initial_cash"], color="gray", linewidth=0.8,
                    linestyle="--", label="Starting capital")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv >= s["initial_cash"]), alpha=0.12, color="#2ca02c")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv  < s["initial_cash"]), alpha=0.12, color="#d62728")
        ax1.set_title(
            f"Swing Breakout — Equity Curve\n"
            f"Return {s['total_return_pct']:+.1f}% | "
            f"{s['num_trades']} trades | "
            f"Win rate {s['win_rate_pct']:.0f}% | "
            f"Max DD {s['max_drawdown_pct']:.1f}%",
            fontsize=11,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # ── 2. Trade P&L distribution ─────────────────────────────────────────
        ax2  = fig.add_subplot(gs[1, 0])
        pnls = [t["pnl"] for t in trades]
        ax2.hist(pnls, bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax2.axvline(0, color="#d62728", linewidth=1.2, linestyle="--")
        ax2.axvline(np.mean(pnls), color="#ff7f0e", linewidth=1.0,
                    linestyle="--", label=f"Mean ${np.mean(pnls):.2f}")
        ax2.set_title("Trade P&L Distribution")
        ax2.set_xlabel("P&L per Trade ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # ── 3. Exit reason breakdown ──────────────────────────────────────────
        ax3    = fig.add_subplot(gs[1, 1])
        ec     = s["exit_reason_counts"]
        labels = list(ec.keys())
        sizes  = list(ec.values())
        colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
        ax3.pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=colors[: len(labels)], startangle=90)
        ax3.set_title("Exit Reason Breakdown")

        plt.suptitle("Swing Breakout Strategy — Backtest Report", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()


# =============================================================================
# UTILITIES
# =============================================================================

def _parse_time_today(hhmm: str, tz: ZoneInfo = ET) -> datetime.datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.datetime.now(tz).replace(hour=h, minute=m, second=0, microsecond=0)


def _load_json(path: str, default) -> object:
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path: str, data: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
