"""
strategy_rsi2.py
----------------
RSI-2 Mean Reversion Strategy (Connors RSI-2 System)
=====================================================

STRATEGY PREMISE
----------------
Short-period RSI measures extreme short-term momentum exhaustion.  On S&P 500
stocks in established uptrends (price > SMA(200)), an RSI(2) reading below 10
signals a temporary oversold condition that tends to revert within a few trading
days as normal buying pressure returns.

This is the Connors RSI-2 system, one of the most robustly back-tested
short-term mean-reversion edges documented in academic and practitioner
literature.  Confining the universe to large-cap S&P 500 names in uptrends
minimises the risk of catching "falling knives" where a fundamental break is
mistaken for a mean-revertible dip.

SIGNAL CRITERIA (LONG only)
----------------------------
All conditions must hold on the signal day (today's close):

  1. Price > SMA(200) — confirmed uptrend; not fighting a structural decline.
  2. RSI(2) < BUY_THRESH (default 10) — deeply oversold on a 2-day time horizon.
  3. Price $10–$500, 20-day avg volume ≥ 500,000 — liquidity floor.

EXIT CONDITIONS (in priority order)
------------------------------------
  1. Pending exit (from yesterday's RSI/SMA signal): fills at today's OPEN.
  2. Gap stop: if today's open ≤ stop price, fill at open.
  3. Hard stop: if today's low ≤ stop price, fill at stop (-8% from entry).
  4. RSI(2) close above EXIT_RSI (default 60): queue exit for next open.
  5. Price closes above SMA(5): queue exit for next open.
  6. Time stop: close at end of day MAX_HOLD_DAYS (fills at close).

EXECUTION MODEL
---------------
  - Signal fires on CLOSE of day T → entry at OPEN of day T+1
  - RSI/SMA(5) exits fire on CLOSE of T+N → fill at OPEN of T+N+1
  - Hard stops fill same-day (using bar's open/low)
  - Time stop fills at bar's close (immediate)

POSITION SIZING
---------------
  - 5% of portfolio per position
  - Maximum 10 simultaneous positions → 50% max deployed
  - RiskManager sector and total concentration caps still apply

BACKTEST USAGE
--------------
    results = strategy.backtest(
        tickers    = rsi2_universe_tickers,
        start_date = "2023-01-01",
        end_date   = "2023-12-31",
        initial_cash = 100_000,
        historical_universes = rsi2_pit_map,   # from build_historical_universes.py --sma200-filter
    )
    strategy.print_summary(results)
    strategy.plot_backtest(results)

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy matplotlib
"""

import datetime
import logging
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from indicators.analyze import calculate_rsi, calculate_sma
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
class Rsi2Config:
    """All tuneable parameters for the RSI-2 mean-reversion strategy."""

    # ── Signal ─────────────────────────────────────────────────────────────────
    rsi_period: int = 2
    """RSI look-back period.  RSI(2) is hyper-sensitive and ideal for catching
    1–5 day oversold extremes on large-cap names."""

    buy_thresh: float = 10.0
    """Enter long when RSI(2) closes BELOW this.  10 = deeply oversold;
    can be relaxed up to ~35 for more frequent signals at the cost of edge."""

    exit_rsi: float = 60.0
    """Queue an exit when RSI(2) closes ABOVE this.  60 signals sufficient
    mean-reversion; 70+ is overbought but waiting that long reduces returns."""

    # ── Trend / exit filters ───────────────────────────────────────────────────
    sma_trend_period: int = 200
    """Price must close ABOVE SMA(N) on the signal day.  SMA(200) is the
    standard long-term trend filter; below it we skip — falling knives only."""

    sma_exit_period: int = 5
    """Close above SMA(N) also queues an exit (fills next open).  SMA(5) is
    the Connors standard; it captures a quick normalisation back to the
    short-term mean."""

    # ── Risk / exits ───────────────────────────────────────────────────────────
    stop_loss_pct: float = 0.08
    """-8% hard stop from entry.  Wider than usual because RSI-2 positions
    are intended to absorb continued sell-pressure before reverting; a tight
    stop would be stopped out on normal noise before the edge plays out."""

    max_hold_days: int = 10
    """Close at bar's close on day N if no other exit triggers.  Prevents
    capital from being tied up in a position that never reverts."""

    # ── Position sizing ────────────────────────────────────────────────────────
    position_size_pct: float = 0.05
    """Fraction of portfolio per position (5%)."""

    max_positions: int = 10
    """Maximum simultaneous RSI-2 positions (50% max deployed at 5% each)."""

    # ── Universe filters ───────────────────────────────────────────────────────
    min_price: float = 10.0
    max_price: float = 500.0
    min_avg_volume: int = 500_000
    """20-day avg volume floor; ensures fills are realistic."""

    # ── Slippage / fill realism ────────────────────────────────────────────────
    slippage_pct: float = 0.001
    """0.10% per fill — conservative for liquid large caps at market open."""

    spread_pct: float = 0.0005
    """Additional half-spread (0.05%) per fill."""

    commission_per_trade: float = 0.0

    # ── Lookback ───────────────────────────────────────────────────────────────
    min_history_bars: int = 210
    """Minimum daily bars needed before signals are evaluated.
    210 = SMA(200) + 10-bar buffer."""


# =============================================================================
# RSI-2 STRATEGY
# =============================================================================

class Rsi2Strategy:
    """
    Connors RSI-2 Mean Reversion Strategy.

    Buys S&P 500 stocks in established uptrends when RSI(2) becomes deeply
    oversold.  Exits when RSI(2) recovers or price crosses back above SMA(5),
    with a hard stop and a time-based safety valve.

    Live trading is not yet implemented — backtest is the primary interface.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        config: Optional[Rsi2Config] = None,
        risk_manager: Optional[object] = None,
        universe_config: Optional[UniverseConfig] = None,
    ) -> None:
        self.tc   = trading_client
        self.dc   = data_client
        self.cfg  = config or Rsi2Config()
        self.rm   = risk_manager
        self.ucfg = universe_config or UniverseConfig()
        self.log  = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # SIGNAL LOGIC (shared between live and backtest)
    # =========================================================================

    def _check_signal(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Apply entry criteria to a daily bar DataFrame sliced through today's close.
        No lookahead: today's bar is df.iloc[-1].
        Returns a signal dict if all conditions are met, else None.
        """
        if len(df) < self.cfg.min_history_bars:
            return None

        close  = df["close"]
        volume = df["volume"]

        last_close = float(close.iloc[-1])
        if not (self.cfg.min_price <= last_close <= self.cfg.max_price):
            return None

        # 20-bar average volume (exclude today to avoid partial-day inflation)
        avg_vol = float(volume.iloc[-21:-1].mean())
        if avg_vol < self.cfg.min_avg_volume:
            return None

        # SMA(200) trend filter — long side only
        sma200_ser = calculate_sma(df, period=self.cfg.sma_trend_period)
        sma200_val = float(sma200_ser.iloc[-1])
        if np.isnan(sma200_val) or last_close <= sma200_val:
            return None

        # RSI(2) oversold entry signal
        rsi_ser  = calculate_rsi(df, period=self.cfg.rsi_period)
        last_rsi = float(rsi_ser.iloc[-1])
        if np.isnan(last_rsi) or last_rsi >= self.cfg.buy_thresh:
            return None

        return {
            "symbol":       sym,
            "signal_price": round(last_close, 4),
            "rsi2":         round(last_rsi, 2),
            "sma200":       round(sma200_val, 4),
            "avg_volume":   round(avg_vol, 0),
        }

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_daily_bars(
        self,
        tickers: list[str],
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily bars for all tickers via LocalDataCache."""
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
                self.log.debug("%s: daily bars failed — %s", sym, exc)

            if (i + 1) % 50 == 0:
                self.log.info("  Daily bars: %d/%d loaded.", i + 1, len(tickers))

        self.log.info(
            "Daily bars loaded: %d / %d tickers.", len(bars_data), len(tickers)
        )
        return bars_data

    # =========================================================================
    # RISK CHECK
    # =========================================================================

    def _risk_ok_rsi2(
        self,
        symbol: str,
        open_positions: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Apply RiskManager caps; always True if rm is None."""
        if self.rm is None:
            return True

        cfg    = self.rm.cfg
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
        Simulate the RSI-2 strategy on historical daily bars.

        Execution model (no lookahead):
          - Signal fires on CLOSE of day T → entry at OPEN of day T+1
          - RSI exit / SMA(5) cross: observed on close of T+N → fill at open T+N+1
          - Gap stop: T+N+1 open ≤ stop price → fill at open
          - Hard stop: bar low ≤ stop price → fill at stop (intraday)
          - Time stop: close at end of day T + max_hold_days

        Returns dict with 'equity_curve', 'trades', and 'summary'.
        """
        self.log.info(
            "RSI-2 backtest: %s to %s | %d tickers | $%s",
            start_date, end_date, len(tickers), f"{initial_cash:,.0f}",
        )

        start_dt    = datetime.datetime.fromisoformat(start_date).replace(tzinfo=ET)
        end_dt      = datetime.datetime.fromisoformat(end_date).replace(tzinfo=ET)
        fetch_start = start_dt - datetime.timedelta(days=self.cfg.min_history_bars + 30)

        daily_data   = self._fetch_daily_bars(tickers, fetch_start, end_dt)
        trading_days = pd.bdate_range(start_date, end_date)

        cash            = initial_cash
        portfolio_value = initial_cash
        all_trades:  list[dict] = []
        equity_rows: list[dict] = []

        pending_entries: dict[str, dict] = {}  # sym → signal dict from yesterday's close
        open_pos:        list[dict]      = []  # currently open positions

        slip = self.cfg.slippage_pct + self.cfg.spread_pct

        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")

            # ── PIT universe ──────────────────────────────────────────────────
            reference_day  = day - pd.Timedelta(days=7)
            year, week, _  = reference_day.isocalendar()
            week_str       = f"{year}-W{week:02d}"
            active         = set(
                historical_universes.get(week_str, tickers)
                if historical_universes else tickers
            )

            # ── Step 1: Enter pending entries at today's open ─────────────────
            for sym, sig in list(pending_entries.items()):
                if sym not in daily_data:
                    del pending_entries[sym]
                    continue
                try:
                    today_bar = daily_data[sym].loc[day_str:day_str]
                except KeyError:
                    del pending_entries[sym]
                    continue
                if today_bar.empty:
                    del pending_entries[sym]
                    continue

                bar_open = float(today_bar["open"].iloc[0])
                ep       = bar_open * (1 + slip)

                budget = portfolio_value * self.cfg.position_size_pct
                qty    = int(budget // ep)
                if qty < 1 or cash < ep * qty:
                    del pending_entries[sym]
                    continue

                active_count = len([p for p in open_pos if not p["closed"]])
                if active_count >= self.cfg.max_positions:
                    del pending_entries[sym]
                    continue

                if not self._risk_ok_rsi2(sym, open_pos, ep * qty, portfolio_value):
                    del pending_entries[sym]
                    continue

                sl_price = ep * (1 - self.cfg.stop_loss_pct)
                pos = {
                    "symbol":              sym,
                    "direction":           "long",
                    "entry_date":          day_str,
                    "entry_price":         round(ep, 4),
                    "sl_price":            round(sl_price, 4),
                    "qty":                 qty,
                    "days_held":           0,
                    "closed":              False,
                    "pending_exit":        False,
                    "pending_exit_reason": None,
                    "exit_price":          None,
                    "exit_reason":         None,
                    "phase":               "rsi2",
                    "rsi2_at_entry":       sig.get("rsi2", 0.0),
                }
                open_pos.append(pos)
                cash -= qty * ep
                del pending_entries[sym]

            # ── Step 2: Check exits for open positions ────────────────────────
            for pos in [p for p in open_pos if not p["closed"]]:
                sym = pos["symbol"]
                if sym not in daily_data:
                    continue
                try:
                    today_bar = daily_data[sym].loc[day_str:day_str]
                except KeyError:
                    continue
                if today_bar.empty:
                    continue

                bar_open  = float(today_bar["open"].iloc[0])
                bar_low   = float(today_bar["low"].iloc[0])
                bar_close = float(today_bar["close"].iloc[0])
                ep        = pos["entry_price"]
                sl        = pos["sl_price"]

                # ── 2A: Execute pending exit from yesterday's RSI/SMA signal ──
                if pos["pending_exit"]:
                    exit_price  = bar_open * (1 - slip)
                    exit_reason = pos["pending_exit_reason"]
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    cash += pos["qty"] * exit_price
                    pos.update(
                        closed=True,
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                    )
                    all_trades.append(self._trade_record(pos, day_str, pnl))
                    continue

                # ── 2B: Gap stop (opened below stop price) ────────────────────
                if bar_open <= sl:
                    exit_price  = bar_open * (1 - slip)
                    exit_reason = "gap_stop"
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    cash += pos["qty"] * exit_price
                    pos.update(
                        closed=True,
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                    )
                    all_trades.append(self._trade_record(pos, day_str, pnl))
                    continue

                # ── 2C: Intraday hard stop ────────────────────────────────────
                if bar_low <= sl:
                    exit_price  = sl * (1 - slip)
                    exit_reason = "stop_loss"
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    cash += pos["qty"] * exit_price
                    pos.update(
                        closed=True,
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                    )
                    all_trades.append(self._trade_record(pos, day_str, pnl))
                    continue

                # ── 2D/2E: EOD RSI/SMA exit signals (queue for next open) ─────
                try:
                    hist = daily_data[sym].loc[:day_str]
                except KeyError:
                    hist = pd.DataFrame()

                if not hist.empty:
                    rsi_ser  = calculate_rsi(hist, period=self.cfg.rsi_period)
                    last_rsi = float(rsi_ser.iloc[-1])

                    if not np.isnan(last_rsi) and last_rsi >= self.cfg.exit_rsi:
                        pos["pending_exit"]        = True
                        pos["pending_exit_reason"] = "rsi_exit"
                        continue  # skip SMA and days_held checks this bar

                    sma5_ser  = calculate_sma(hist, period=self.cfg.sma_exit_period)
                    last_sma5 = float(sma5_ser.iloc[-1])

                    if not np.isnan(last_sma5) and bar_close > last_sma5:
                        pos["pending_exit"]        = True
                        pos["pending_exit_reason"] = "sma_exit"
                        continue

                # ── 2F: Increment days held; check time stop ──────────────────
                pos["days_held"] += 1
                if pos["days_held"] >= self.cfg.max_hold_days:
                    exit_price  = bar_close * (1 - slip)
                    exit_reason = "time_stop"
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    cash += pos["qty"] * exit_price
                    pos.update(
                        closed=True,
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                    )
                    all_trades.append(self._trade_record(pos, day_str, pnl))

            # Purge fully closed positions
            open_pos = [p for p in open_pos if not p["closed"]]

            # ── Step 3: Generate new entry signals from today's close ──────────
            open_symbols = {p["symbol"] for p in open_pos} | set(pending_entries.keys())
            open_count   = len(open_pos) + len(pending_entries)

            if open_count < self.cfg.max_positions:
                new_signals: list[dict] = []

                for sym in active:
                    if sym in open_symbols or sym not in daily_data:
                        continue
                    try:
                        hist = daily_data[sym].loc[:day_str]
                    except KeyError:
                        continue
                    if len(hist) < self.cfg.min_history_bars:
                        continue

                    sig = self._check_signal(sym, hist)
                    if sig is not None:
                        new_signals.append(sig)

                # Most oversold (lowest RSI) gets highest priority
                new_signals.sort(key=lambda x: x["rsi2"])

                for sig in new_signals:
                    if open_count >= self.cfg.max_positions:
                        break
                    sym = sig["symbol"]
                    if sym in open_symbols:
                        continue
                    pending_entries[sym] = sig
                    open_symbols.add(sym)
                    open_count += 1

            # ── Step 4: Track equity (mark open positions to today's close) ───
            portfolio_value = cash + sum(
                p["qty"] * float(
                    daily_data[p["symbol"]].loc[day_str:day_str]["close"].iloc[-1]
                    if p["symbol"] in daily_data
                    and not daily_data[p["symbol"]].loc[day_str:day_str].empty
                    else p["entry_price"]
                )
                for p in open_pos
            )

            day_pnl = sum(t["pnl"] for t in all_trades if t["date"] == day_str)
            day_n   = sum(1 for t in all_trades if t["date"] == day_str)
            equity_rows.append({
                "date":            day,
                "portfolio_value": portfolio_value,
                "day_pnl":         day_pnl,
                "num_trades":      day_n,
                "open_positions":  len(open_pos),
            })

            if day_n > 0:
                self.log.info(
                    "%s: %d exit(s) | P&L $%+.2f | Portfolio $%s",
                    day_str, day_n, day_pnl, f"{portfolio_value:,.2f}",
                )

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        summary      = self._compute_summary(all_trades, equity_curve, initial_cash)

        return {
            "equity_curve": equity_curve,
            "trades":       all_trades,
            "summary":      summary,
        }

    @staticmethod
    def _trade_record(pos: dict, day_str: str, pnl: float) -> dict:
        return {
            "date":          day_str,
            "symbol":        pos["symbol"],
            "phase":         "rsi2",
            "direction":     "long",
            "entry_date":    pos["entry_date"],
            "entry_price":   round(pos["entry_price"], 4),
            "exit_price":    round(pos["exit_price"], 4),
            "qty":           pos["qty"],
            "pnl":           round(pnl, 2),
            "exit_reason":   pos["exit_reason"],
            "days_held":     pos["days_held"],
            "rsi2_at_entry": pos.get("rsi2_at_entry", 0.0),
        }

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

        avg_days   = float(np.mean([t.get("days_held", 0) for t in trades]))
        gross_win  = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
        expectancy    = round(float(np.mean([t["pnl"] for t in trades])), 2)

        pv     = equity_curve["portfolio_value"].astype(float)
        rets   = pv.pct_change().dropna()
        sharpe = (
            round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
            if len(rets) >= 2 and rets.std() != 0
            else 0.0
        )

        rsi2_vals = [
            t["rsi2_at_entry"] for t in trades if t.get("rsi2_at_entry", 0) > 0
        ]

        return {
            "initial_cash":       initial_cash,
            "final_value":        round(final_value, 2),
            "total_return_pct":   round((final_value - initial_cash) / initial_cash * 100, 2),
            "num_trades":         len(trades),
            "win_rate_pct":       round(len(wins) / len(trades) * 100, 1),
            "avg_win":            round(float(np.mean([t["pnl"] for t in wins])), 2) if wins else 0.0,
            "avg_loss":           round(float(np.mean([t["pnl"] for t in losses])), 2) if losses else 0.0,
            "reward_risk_ratio":  round(
                abs(float(np.mean([t["pnl"] for t in wins])) /
                    float(np.mean([t["pnl"] for t in losses]))), 2
            ) if wins and losses else 0.0,
            "profit_factor":      profit_factor,
            "expectancy":         expectancy,
            "sharpe":             sharpe,
            "largest_win":        round(max(t["pnl"] for t in trades), 2),
            "largest_loss":       round(min(t["pnl"] for t in trades), 2),
            "avg_days_held":      round(avg_days, 1),
            "max_drawdown_pct":   round(float(drawdown.min()) * 100, 2),
            "avg_rsi2_at_entry":  round(float(np.mean(rsi2_vals)), 2) if rsi2_vals else 0.0,
            "exit_reason_counts": exit_counts,
        }

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, results: dict) -> None:
        s = results["summary"]
        if "note" in s:
            print(f"\n[RSI-2: no trades — {s['note']}]")
            return

        ec    = s["exit_reason_counts"]
        rsi_n = ec.get("rsi_exit", 0)
        sma_n = ec.get("sma_exit", 0)
        sl_n  = ec.get("stop_loss", 0) + ec.get("gap_stop", 0)
        ts_n  = ec.get("time_stop", 0)
        total = s["num_trades"]

        print(f"\n{'='*54}")
        print(f"  RSI-2 Mean Reversion Strategy — Backtest Summary")
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
        print(f"  Avg RSI(2) at entry  : {s['avg_rsi2_at_entry']:>12.2f}")
        print(f"{'-'*54}")
        print(f"  Exit breakdown:")
        print(f"    RSI exit           : {rsi_n:>5} ({rsi_n/total*100:>4.0f}%)")
        print(f"    SMA(5) exit        : {sma_n:>5} ({sma_n/total*100:>4.0f}%)")
        print(f"    Stop-loss / gap    : {sl_n:>5} ({sl_n/total*100:>4.0f}%)")
        print(f"    Time stop          : {ts_n:>5} ({ts_n/total*100:>4.0f}%)")
        print(f"{'='*54}\n")

    def plot_backtest(self, results: dict) -> None:
        s      = results["summary"]
        equity = results["equity_curve"]
        trades = results["trades"]

        if not trades:
            print("RSI-2: no trades to plot.")
            return

        fig = plt.figure(figsize=(15, 10))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        # ── 1. Equity curve ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        pv  = equity["portfolio_value"]
        ax1.plot(equity.index, pv, color="#1f77b4", linewidth=1.5, label="Portfolio")
        ax1.axhline(
            s["initial_cash"], color="gray", linewidth=0.8,
            linestyle="--", label="Starting capital",
        )
        ax1.fill_between(
            equity.index, s["initial_cash"], pv,
            where=(pv >= s["initial_cash"]), alpha=0.12, color="#2ca02c",
        )
        ax1.fill_between(
            equity.index, s["initial_cash"], pv,
            where=(pv  < s["initial_cash"]), alpha=0.12, color="#d62728",
        )
        ax1.set_title(
            f"RSI-2 Mean Reversion — Equity Curve\n"
            f"Return {s['total_return_pct']:+.1f}%  |  "
            f"{s['num_trades']} trades  |  "
            f"Win rate {s['win_rate_pct']:.0f}%  |  "
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
        ax2.axvline(
            float(np.mean(pnls)), color="#ff7f0e", linewidth=1.0,
            linestyle="--", label=f"Mean ${float(np.mean(pnls)):.2f}",
        )
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
        colors = ["#2ca02c", "#17becf", "#d62728", "#ff7f0e", "#9467bd"]
        ax3.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=colors[: len(labels)], startangle=90,
        )
        ax3.set_title("Exit Reason Breakdown")

        plt.suptitle(
            "RSI-2 Mean Reversion Strategy — Backtest Report", fontsize=13, y=1.01
        )
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # LIVE EXECUTION STUBS (backtest only for now)
    # =========================================================================

    def evening_scan(self, tickers: Optional[list[str]] = None) -> list[dict]:
        """Placeholder — live trading not yet implemented."""
        self.log.info("Rsi2Strategy.evening_scan() — live trading not yet implemented.")
        return []

    def morning_session(self) -> None:
        """Placeholder — live trading not yet implemented."""
        self.log.info("Rsi2Strategy.morning_session() — live trading not yet implemented.")
