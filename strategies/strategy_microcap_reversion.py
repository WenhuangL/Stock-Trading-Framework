"""
strategy_microcap_reversion.py
------------------------------
Micro-Cap Mean-Reversion Strategy — long-only liquidity provision.

STRATEGY PREMISE
----------------
Micro-cap mean reversion works for one structural reason: in a stock that trades
a few hundred thousand dollars a day, a single motivated or panicked participant
can shove the price far past any reasonable value, and there is almost no
arbitrage capital standing by to stop them.  The overshoot is bigger and the
snap-back more reliable than anywhere in large caps.  These names are
under-covered, retail-dominated, and structurally off-limits to the institutions
who would otherwise compete the inefficiency away.  When someone dumps shares
into a thin book on no news, we are the patient counterparty to a
price-insensitive seller, and the wide spread is partly the *fee* we collect for
providing that liquidity.

THE EDGE IS NOT THE SIGNAL — IT IS DATA INTEGRITY AND FILLS
----------------------------------------------------------
A micro-cap backtest is the single most likely of any strategy to be a mirage,
for three specific reasons, each of which this module confronts explicitly:

  1. SURVIVORSHIP BIAS is extreme — delisted "corpses" silently vanish from most
     databases, so the price history is missing exactly the disasters.  We cannot
     manufacture the missing corpses, so instead we *inject* them: a Monte-Carlo
     survivorship stress-test (`_survivorship_stress`) overrides a hazard-rate
     fraction of trades to a catastrophic loss and reports the de-biased profit
     factor alongside the raw (optimistic) one.  The raw number is labelled a
     mirage on purpose.

  2. HEROIC FILL ASSUMPTIONS — we credit ourselves prices at which no liquidity
     existed.  This module uses LIMIT ORDERS ONLY with probabilistic non-fill:
     a buy limit fills only if the next bar actually trades through it, a wide
     half-spread is charged on every leg, and gap-through stops fill at the
     gapped-open (worse), not at the stop price.  Reversion must clear two
     half-spreads before a cent of profit.

  3. DIRTY DATA — bad ticks and reverse-split distortion.  Alpaca daily bars are
     split-UNadjusted, and micro-caps reverse-split constantly.  A recent-split /
     bad-tick guard (`_split_or_badtick`) rejects names showing an implausible
     overnight ratio.

SIGNAL CRITERIA (long-only; the short side is unborrowable + squeeze-prone)
---------------------------------------------------------------------------
Niche 1 — NO-NEWS OVERSOLD LIQUIDATION (the bread and butter):
    close down >= decline_threshold over lookback_days
    AND today's volume >= vol_mult x 20-day avg volume
    AND RSI(rsi_period) < rsi_buy_thresh
Niche 2 — PANIC GAP-DOWN SYMPATHY:
    single-day drop >= gap_threshold
    AND the peer small-cap ETF (IWM) fell >= peer_down_threshold the same day
    AND the name's drop is not wildly more extreme than the peer's (idiosyncrasy
        guard — a far bigger drop suggests name-specific news, not sympathy)

Both niches additionally require every SAFETY filter to pass (price band,
dollar-volume floor, no-news proxy, split/bad-tick guard).

EXECUTION MODEL (all on fully-closed daily bars — never intraday, no lookahead)
-------------------------------------------------------------------------------
  - Signal fires on CLOSE of day T -> a buy LIMIT is queued for day T+1.
  - The limit fills on T+1 only if bar_low <= limit (price improvement to the
    open if it gaps below).  Otherwise it is retried for fill_retry_days then
    cancelled — you often cannot get in at the price you saw.
  - Exit priority (pessimistic: adverse events assumed first within a bar):
      1. pending SMA-reversion exit queued yesterday -> fill at open
      2. gap stop (open <= stop)  -> fill at open (+ halt haircut)
      3. intraday hard stop (low <= stop) -> fill at stop
      4. reversion target (high >= target) -> fill at target (the winner path)
      5. SMA(exit_sma) reversion -> queue exit for next open
      6. time stop after max_hold_days -> fill at close

POSITION SIZING IS THE REAL RISK CONTROL, NOT STOPS
---------------------------------------------------
Stops barely work when a thin name gaps straight through them, so risk is
controlled by tiny per-name size (position_size_pct ~1%) spread across many
names (max_positions ~40).  The edge is purely statistical; breadth is what
survives the fat per-name tail.

BACKTEST USAGE
--------------
    strat   = MicrocapReversionStrategy(trading_client, data_client)
    results = strat.backtest(
        tickers    = microcap_tickers,
        start_date = "2022-06-01",
        end_date   = "2023-12-31",
        historical_universes = microcap_pit_map,  # from build_microcap_universe.py
    )
    strat.print_summary(results)   # prints RAW and SURVIVORSHIP-STRESSED metrics

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

from indicators.analyze import calculate_rsi, calculate_sma, calculate_dollar_volume
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
class MicrocapReversionConfig:
    """All tuneable parameters for the micro-cap mean-reversion strategy."""

    # ── Niche 1: no-news oversold liquidation ──────────────────────────────────
    lookback_days: int = 3
    """Window over which the cumulative decline is measured (1-3 days)."""

    decline_threshold: float = 0.20
    """Enter when close is down >= this over lookback_days.  0.20 = -20%."""

    vol_mult: float = 1.5
    """Today's volume must be >= vol_mult x 20-day average (elevated liquidation)."""

    rsi_period: int = 2
    rsi_buy_thresh: float = 10.0
    """RSI(2) < 10 confirms a short-term oversold extreme."""

    # ── Niche 2: panic gap-down sympathy ───────────────────────────────────────
    enable_gap_sympathy: bool = True
    gap_threshold: float = 0.10
    """Single-day drop >= this qualifies as a panic day.  0.10 = -10%."""

    peer_etf: str = "IWM"
    """Small-cap benchmark used to detect an indiscriminate sector sell-off.
    IWM (Russell 2000); IWC (micro-cap) is an alternative."""

    peer_down_threshold: float = 0.02
    """The peer ETF must be down >= this the same day for a sympathy entry."""

    idio_multiple: float = 3.0
    """Reject a sympathy entry when the name's drop exceeds idio_multiple x the
    peer's drop — a far bigger fall suggests name-specific news, not sympathy."""

    # ── Safety filters (the survivability core) ────────────────────────────────
    min_price: float = 1.0
    """Exclude sub-$1 names outright (penny-stock / imminent-delist territory)."""

    max_price: float = 15.0
    """Micro-cap price ceiling."""

    min_dollar_volume: float = 250_000.0
    """20-day average dollar volume floor (close x volume).  Below this the wide
    spread and non-fills eat the entire edge."""

    dollar_vol_period: int = 20

    max_single_day_drop: float = 0.55
    """No-news proxy: reject when the worst single-day drop in the lookback is
    beyond this — a drop that violent is almost certainly information / a halt /
    a corporate action, not a liquidity overshoot.  The genuine no-news / solvency
    filter (Alpaca news keyword scan) runs live in evening_scan(); Alpaca lacks the
    history to reproduce it in-backtest, so this price-shape proxy stands in and
    its weakness is acknowledged."""

    split_up_ratio: float = 2.5
    """Reverse-split / bad-tick guard: reject when any bar in the recent window
    shows close/prev_close > this (a raw, split-UNadjusted reverse-split jump)."""

    exclude_recent_split_days: int = 60
    """Window (bars) scanned by the split/bad-tick guard."""

    symbol_blacklist: frozenset = frozenset()

    # ── Exits ──────────────────────────────────────────────────────────────────
    target_revert_pct: float = 0.12
    """Primary reversion target: exit when price recovers this far above entry."""

    exit_sma: int = 5
    """Secondary reversion exit: close >= SMA(exit_sma) queues a next-open exit."""

    stop_loss_pct: float = 0.20
    """Hard stop from entry.  Wide and gap-aware — for thin names the stop barely
    works, so sizing (not the stop) is the real control."""

    halt_slippage_pct: float = 0.05
    """Extra deterministic haircut applied to a gap-through stop fill, modelling a
    thin name that gaps / halts straight through the stop.  Set 0 to disable."""

    model_halts: bool = True

    max_hold_days: int = 4
    """Reversion is fast or the thesis was wrong — close after N days."""

    # ── Position sizing / breadth ──────────────────────────────────────────────
    position_size_pct: float = 0.01
    """Fraction of portfolio per position (1%).  Tiny by design — breadth, not
    per-name conviction, is the risk control."""

    max_positions: int = 40
    """Large by design: the edge is statistical and any single name can zero."""

    # ── Honest fill model ──────────────────────────────────────────────────────
    entry_limit_offset_pct: float = 0.0
    """Buy limit is placed at signal_close x (1 - this).  0 = at the signal close;
    raise it to demand price improvement (fewer fills, better entries)."""

    fill_retry_days: int = 1
    """A queued buy limit is retried for this many bars before being cancelled."""

    spread_pct: float = 0.015
    """Half-spread per fill (1.5%).  Micro-cap books are wide; charged on both
    legs, so reversion must clear ~2 x this before profit."""

    slippage_pct: float = 0.005
    """Additional slippage per fill (0.5%)."""

    commission_per_trade: float = 0.0

    # ── Survivorship stress-test ───────────────────────────────────────────────
    survivorship_stress: bool = True
    """Inject the delisted 'corpses' the survivorship-biased universe omits."""

    delisting_hazard_annual: float = 0.08
    """Annual probability that a distressed name we bought was actually on its way
    to zero.  8% is a conservative base rate for the deeply-oversold cohort."""

    delisting_loss_pct: float = 0.85
    """Loss taken when a stressed trade is flipped to a corpse (-85%)."""

    stress_mc_runs: int = 300
    stress_seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────────
    daily_bar_feed: str = "iex"
    """Feed for daily OHLCV.  IEX matches the free live feed and the universe
    scoring, keeping the backtest internally consistent."""

    min_history_bars: int = 40
    """Minimum daily bars before signals evaluate (dollar-vol + SMA warm-up)."""


# =============================================================================
# MICRO-CAP MEAN-REVERSION STRATEGY
# =============================================================================

class MicrocapReversionStrategy:
    """
    Long-only micro-cap mean reversion.

    Two entry niches (no-news oversold liquidation + panic gap-down sympathy)
    feed one unified exit engine.  The class deliberately mirrors the public
    contract of the other strategies (Rsi2Strategy in particular) so the runner,
    the combined summary, and trade_db treat it identically.  Live trading is not
    yet wired; the backtest is the primary interface, with evening_scan() a
    live-candidate stub that applies the real Alpaca-news solvency filter.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        config: Optional[MicrocapReversionConfig] = None,
        risk_manager: Optional[object] = None,
        universe_config: Optional[UniverseConfig] = None,
    ) -> None:
        self.tc   = trading_client
        self.dc   = data_client
        self.cfg  = config or MicrocapReversionConfig()
        self.rm   = risk_manager
        self.ucfg = universe_config or UniverseConfig()
        self.log  = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # SAFETY FILTERS
    # =========================================================================

    def _split_or_badtick(self, df: pd.DataFrame) -> bool:
        """True if a recent bar shows an implausible overnight up-jump — the raw,
        split-unadjusted signature of a reverse split or a bad tick."""
        window = df["close"].iloc[-self.cfg.exclude_recent_split_days:]
        if len(window) < 2:
            return False
        ratio = window / window.shift(1)
        return bool((ratio > self.cfg.split_up_ratio).any())

    def _passes_safety(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """Shared safety gate for both niches.  Returns a context dict (last_close,
        avg_dollar_vol) when the name is tradeable, else None."""
        if sym in self.cfg.symbol_blacklist:
            return None
        if len(df) < self.cfg.min_history_bars:
            return None

        last_close = float(df["close"].iloc[-1])
        if not (self.cfg.min_price <= last_close <= self.cfg.max_price):
            return None

        dvol = calculate_dollar_volume(df, period=self.cfg.dollar_vol_period)
        avg_dollar_vol = float(dvol.iloc[-1])
        if np.isnan(avg_dollar_vol) or avg_dollar_vol < self.cfg.min_dollar_volume:
            return None

        # No-news proxy: reject drops too violent to be liquidity-driven.
        daily_ret = df["close"].pct_change().iloc[-self.cfg.lookback_days:]
        if float(daily_ret.min()) <= -self.cfg.max_single_day_drop:
            return None

        # Reverse-split / bad-tick guard.
        if self._split_or_badtick(df):
            return None

        return {"last_close": last_close, "avg_dollar_vol": round(avg_dollar_vol, 0)}

    # =========================================================================
    # SIGNAL LOGIC  (df sliced through today's close — no lookahead)
    # =========================================================================

    def _check_oversold_signal(self, sym: str, df: pd.DataFrame) -> Optional[dict]:
        """Niche 1: no-news oversold liquidation."""
        ctx = self._passes_safety(sym, df)
        if ctx is None:
            return None

        close  = df["close"]
        volume = df["volume"]
        last_close = ctx["last_close"]

        if len(close) <= self.cfg.lookback_days:
            return None
        ref_close = float(close.iloc[-1 - self.cfg.lookback_days])
        if ref_close <= 0:
            return None
        decline = (last_close - ref_close) / ref_close
        if decline > -self.cfg.decline_threshold:
            return None

        avg_vol = float(volume.iloc[-21:-1].mean())
        if avg_vol <= 0 or float(volume.iloc[-1]) < self.cfg.vol_mult * avg_vol:
            return None

        rsi_ser  = calculate_rsi(df, period=self.cfg.rsi_period)
        last_rsi = float(rsi_ser.iloc[-1])
        if np.isnan(last_rsi) or last_rsi >= self.cfg.rsi_buy_thresh:
            return None

        return {
            "symbol":         sym,
            "direction":      "long",
            "signal_type":    "oversold",
            "signal_price":   round(last_close, 4),
            "decline_pct":    round(decline * 100, 2),
            "rsi2":           round(last_rsi, 2),
            "dollar_vol":     ctx["avg_dollar_vol"],
            # priority: deeper decline first
            "rank":           abs(decline),
        }

    def _check_gap_signal(self, sym: str, df: pd.DataFrame, peer_ret: Optional[float]) -> Optional[dict]:
        """Niche 2: panic gap-down sympathy.  peer_ret is today's peer-ETF return."""
        if not self.cfg.enable_gap_sympathy:
            return None
        if peer_ret is None or peer_ret > -self.cfg.peer_down_threshold:
            return None  # no indiscriminate sell-off today

        ctx = self._passes_safety(sym, df)
        if ctx is None:
            return None

        close = df["close"]
        if len(close) < 2:
            return None
        prev_close = float(close.iloc[-2])
        last_close = ctx["last_close"]
        if prev_close <= 0:
            return None
        day_drop = (last_close - prev_close) / prev_close
        if day_drop > -self.cfg.gap_threshold:
            return None

        # Idiosyncrasy guard: a drop far larger than the peer's suggests own news.
        if abs(day_drop) > self.cfg.idio_multiple * abs(peer_ret):
            return None

        return {
            "symbol":         sym,
            "direction":      "long",
            "signal_type":    "gap_sympathy",
            "signal_price":   round(last_close, 4),
            "decline_pct":    round(day_drop * 100, 2),
            "peer_ret_pct":   round(peer_ret * 100, 2),
            "dollar_vol":     ctx["avg_dollar_vol"],
            "rank":           abs(day_drop),
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
        """Fetch daily bars for all tickers via LocalDataCache (mirrors RSI-2)."""
        from data_collection.data_cache import LocalDataCache
        cache     = LocalDataCache(self.dc)
        bars_data: dict[str, pd.DataFrame] = {}

        for i, sym in enumerate(tickers):
            try:
                df = cache.get_bars_df(sym, TimeFrame.Day, start_dt, end_dt, feed=self.cfg.daily_bar_feed)
                if df is not None and not df.empty:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(ET).normalize()
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    bars_data[sym] = df
            except Exception as exc:
                self.log.debug("%s: daily bars failed — %s", sym, exc)

            if (i + 1) % 100 == 0:
                self.log.info("  Daily bars: %d/%d loaded.", i + 1, len(tickers))

        self.log.info("Daily bars loaded: %d / %d tickers.", len(bars_data), len(tickers))
        return bars_data

    # =========================================================================
    # RISK CHECK
    # =========================================================================

    def _risk_ok(
        self,
        symbol: str,
        open_positions: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Apply RiskManager caps; always True if rm is None.  (Sector caps won't
        resolve for off-index micro-caps — breadth substitutes.)"""
        if self.rm is None:
            return True
        cfg    = self.rm.cfg
        active = [p for p in open_positions if not p.get("closed", False)]
        if len(active) >= cfg.max_total_positions:
            return False
        total_notional = sum(p["qty"] * p["entry_price"] for p in active)
        if (total_notional + proposed_notional) / portfolio_value > cfg.max_portfolio_deployed_pct:
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
        Simulate the long-only micro-cap reversion strategy on daily bars.
        Returns dict with 'equity_curve', 'trades', and 'summary'.
        """
        self.log.info(
            "Micro-cap reversion backtest: %s to %s | %d tickers | $%s | gap_sympathy=%s",
            start_date, end_date, len(tickers), f"{initial_cash:,.0f}",
            self.cfg.enable_gap_sympathy,
        )

        start_dt    = datetime.datetime.fromisoformat(start_date).replace(tzinfo=ET)
        end_dt      = datetime.datetime.fromisoformat(end_date).replace(tzinfo=ET)
        fetch_start = start_dt - datetime.timedelta(days=self.cfg.min_history_bars + 120)

        fetch_tickers = list(tickers)
        if self.cfg.enable_gap_sympathy and self.cfg.peer_etf not in fetch_tickers:
            fetch_tickers.append(self.cfg.peer_etf)
        daily_data   = self._fetch_daily_bars(fetch_tickers, fetch_start, end_dt)
        trading_days = pd.bdate_range(start_date, end_date)

        cash            = initial_cash
        portfolio_value = initial_cash
        all_trades:  list[dict] = []
        equity_rows: list[dict] = []
        num_signals = 0
        num_filled  = 0

        # pending_entries[sym] = {signal dict, "limit": float, "days_pending": int}
        pending_entries: dict[str, dict] = {}
        open_pos:        list[dict]      = []

        slip = self.cfg.slippage_pct + self.cfg.spread_pct

        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")

            # ── Point-in-time universe (weekly, mirrors RSI-2) ────────────────
            reference_day = day - pd.Timedelta(days=7)
            year, week, _ = reference_day.isocalendar()
            week_str      = f"{year}-W{week:02d}"
            if historical_universes:
                _pit = historical_universes.get(week_str)
                active_universe = set(_pit) if _pit is not None else set()
            else:
                active_universe = set(tickers)

            # ── Peer ETF return for the day (gap-sympathy gate) ───────────────
            peer_ret: Optional[float] = None
            if self.cfg.enable_gap_sympathy and self.cfg.peer_etf in daily_data:
                try:
                    phist = daily_data[self.cfg.peer_etf].loc[:day_str]
                    if len(phist) >= 2 and phist.index[-1].strftime("%Y-%m-%d") == day_str:
                        peer_ret = float(phist["close"].iloc[-1] / phist["close"].iloc[-2] - 1.0)
                except (KeyError, IndexError):
                    peer_ret = None

            # ── Step 1: attempt to fill queued buy limits at/through the bar ──
            for sym, pend in list(pending_entries.items()):
                if sym not in daily_data:
                    del pending_entries[sym]
                    continue
                try:
                    today_bar = daily_data[sym].loc[day_str:day_str]
                except KeyError:
                    del pending_entries[sym]
                    continue
                if today_bar.empty:
                    # No bar today (halt / no trade) — count a pending day, maybe expire.
                    pend["days_pending"] += 1
                    if pend["days_pending"] > self.cfg.fill_retry_days:
                        del pending_entries[sym]
                    continue

                bar_open = float(today_bar["open"].iloc[0])
                bar_low  = float(today_bar["low"].iloc[0])
                limit    = pend["limit"]

                if bar_low <= limit:
                    # Limit is marketable: price-improve to the open if it gapped
                    # below the limit, otherwise fill at the limit.
                    raw_fill = bar_open if bar_open <= limit else limit
                    ep = raw_fill * (1 + slip)          # buy at ask
                    budget = portfolio_value * self.cfg.position_size_pct
                    qty    = int(budget // ep)
                    active_count = len([p for p in open_pos if not p["closed"]])
                    if (qty < 1 or cash < ep * qty
                            or active_count >= self.cfg.max_positions
                            or not self._risk_ok(sym, open_pos, ep * qty, portfolio_value)):
                        del pending_entries[sym]
                        continue

                    sig = pend["sig"]
                    pos = {
                        "symbol":              sym,
                        "direction":           "long",
                        "signal_type":         sig["signal_type"],
                        "entry_date":          day_str,
                        "entry_price":         round(ep, 4),
                        "sl_price":            round(ep * (1 - self.cfg.stop_loss_pct), 4),
                        "target_price":        round(ep * (1 + self.cfg.target_revert_pct), 4),
                        "qty":                 qty,
                        "days_held":           0,
                        "closed":              False,
                        "pending_exit":        False,
                        "pending_exit_reason": None,
                        "exit_price":          None,
                        "exit_reason":         None,
                        "phase":               "microcap",
                        "decline_pct":         sig.get("decline_pct", 0.0),
                        "dollar_vol":          sig.get("dollar_vol", 0.0),
                        "rsi2_at_entry":       sig.get("rsi2", 0.0),
                    }
                    open_pos.append(pos)
                    cash -= qty * ep
                    num_filled += 1
                    del pending_entries[sym]
                else:
                    # Limit never traded — you did not get in at your price.
                    pend["days_pending"] += 1
                    if pend["days_pending"] > self.cfg.fill_retry_days:
                        del pending_entries[sym]

            # ── Step 2: exits for open positions ──────────────────────────────
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
                bar_high  = float(today_bar["high"].iloc[0])
                bar_low   = float(today_bar["low"].iloc[0])
                bar_close = float(today_bar["close"].iloc[0])
                ep        = pos["entry_price"]
                sl        = pos["sl_price"]
                tgt       = pos["target_price"]

                def _exit(exit_price: float, reason: str) -> None:
                    pnl = (exit_price - ep) * pos["qty"] - self.cfg.commission_per_trade
                    nonlocal cash
                    cash += pos["qty"] * exit_price
                    pos.update(closed=True, exit_price=round(exit_price, 4), exit_reason=reason)
                    all_trades.append(self._trade_record(pos, day_str, pnl))

                # 2A: pending SMA-reversion exit queued yesterday — sell at open (bid)
                if pos["pending_exit"]:
                    _exit(bar_open * (1 - slip), pos["pending_exit_reason"])
                    continue

                # 2B: gap stop — open gaps below stop (+ deterministic halt haircut)
                if bar_open <= sl:
                    halt = self.cfg.halt_slippage_pct if self.cfg.model_halts else 0.0
                    _exit(bar_open * (1 - slip - halt), "gap_stop")
                    continue

                # 2C: intraday hard stop — assumed BEFORE the target (pessimistic)
                if bar_low <= sl:
                    _exit(sl * (1 - slip), "stop_loss")
                    continue

                # 2D: reversion target — the winner path (sell at target/bid)
                if bar_high >= tgt:
                    raw = bar_open if bar_open >= tgt else tgt
                    _exit(raw * (1 - slip), "target")
                    continue

                # 2E: SMA(exit_sma) reversion — queue a next-open exit
                try:
                    hist = daily_data[sym].loc[:day_str]
                except KeyError:
                    hist = pd.DataFrame()
                if not hist.empty:
                    sma_ser  = calculate_sma(hist, period=self.cfg.exit_sma)
                    last_sma = float(sma_ser.iloc[-1])
                    if not np.isnan(last_sma) and bar_close >= last_sma:
                        pos["pending_exit"]        = True
                        pos["pending_exit_reason"] = "sma_exit"
                        continue

                # 2F: time stop — sell at close (bid)
                pos["days_held"] += 1
                if pos["days_held"] >= self.cfg.max_hold_days:
                    _exit(bar_close * (1 - slip), "time_stop")

            open_pos = [p for p in open_pos if not p["closed"]]

            # ── Step 3: generate new signals from today's close ───────────────
            open_symbols = {p["symbol"] for p in open_pos} | set(pending_entries.keys())
            open_count   = len(open_pos) + len(pending_entries)

            if open_count < self.cfg.max_positions:
                signals: list[dict] = []
                for sym in active_universe:
                    if sym in open_symbols or sym not in daily_data:
                        continue
                    try:
                        hist = daily_data[sym].loc[:day_str]
                    except KeyError:
                        continue
                    if len(hist) < self.cfg.min_history_bars:
                        continue
                    # Only evaluate names whose latest bar is actually today.
                    if hist.index[-1].strftime("%Y-%m-%d") != day_str:
                        continue

                    sig = self._check_oversold_signal(sym, hist)
                    if sig is None:
                        sig = self._check_gap_signal(sym, hist, peer_ret)
                    if sig is not None:
                        signals.append(sig)

                # Priority: deepest overshoot first.
                signals.sort(key=lambda x: x["rank"], reverse=True)
                num_signals += len(signals)

                for sig in signals:
                    if open_count >= self.cfg.max_positions:
                        break
                    sym = sig["symbol"]
                    if sym in open_symbols:
                        continue
                    limit = sig["signal_price"] * (1 - self.cfg.entry_limit_offset_pct)
                    pending_entries[sym] = {"sig": sig, "limit": round(limit, 4), "days_pending": 0}
                    open_symbols.add(sym)
                    open_count += 1

            # ── Step 4: mark-to-market (long only) ────────────────────────────
            mtm = 0.0
            for p in open_pos:
                try:
                    cur = float(daily_data[p["symbol"]].loc[day_str:day_str]["close"].iloc[-1])
                except (KeyError, IndexError):
                    cur = p["entry_price"]
                mtm += p["qty"] * cur

            portfolio_value = cash + mtm

            day_pnl = sum(t["pnl"] for t in all_trades if t["date"] == day_str)
            day_n   = sum(1 for t in all_trades if t["date"] == day_str)
            equity_rows.append({
                "date":            day,
                "portfolio_value": portfolio_value,
                "day_pnl":         day_pnl,
                "num_trades":      day_n,
                "open_positions":  len(open_pos),
            })

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        summary      = self._compute_summary(
            all_trades, equity_curve, initial_cash, self.cfg,
            num_signals=num_signals, num_filled=num_filled,
        )

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
            "phase":         "microcap",
            "direction":     "long",
            "signal_type":   pos.get("signal_type"),
            "entry_date":    pos["entry_date"],
            "entry_price":   round(pos["entry_price"], 4),
            "exit_price":    round(pos["exit_price"], 4),
            "qty":           pos["qty"],
            "pnl":           round(pnl, 2),
            "exit_reason":   pos["exit_reason"],
            "days_held":     pos["days_held"],
            "decline_pct":   pos.get("decline_pct", 0.0),
            "dollar_vol":    pos.get("dollar_vol", 0.0),
            "rsi2_at_entry": pos.get("rsi2_at_entry", 0.0),
        }

    # =========================================================================
    # SURVIVORSHIP STRESS-TEST
    # =========================================================================

    @staticmethod
    def _survivorship_stress(trades: list[dict], initial_cash: float, cfg) -> dict:
        """Inject the delisted corpses the (survivorship-biased) universe omits.

        For each Monte-Carlo pass, every trade is independently flipped — with a
        hazard-rate probability scaled by its holding period — to a catastrophic
        `delisting_loss_pct` loss on its committed capital.  Returns the median /
        5th-percentile stressed profit factor and total return, so the honest
        (de-biased) numbers sit right next to the raw ones."""
        if not trades:
            return {}
        rng = np.random.default_rng(cfg.stress_seed)
        base_pnl = np.array([t["pnl"] for t in trades], dtype=float)
        notional = np.array(
            [t["entry_price"] * t["qty"] for t in trades], dtype=float
        )
        days = np.array([max(t.get("days_held", 1), 1) for t in trades], dtype=float)
        # Per-trade probability the name was actually a corpse over its hold.
        p_delist = 1.0 - np.exp(-cfg.delisting_hazard_annual * days / 252.0)
        corpse_pnl = -notional * cfg.delisting_loss_pct

        pfs, rets = [], []
        for _ in range(cfg.stress_mc_runs):
            hit = rng.random(len(trades)) < p_delist
            pnl = np.where(hit, corpse_pnl, base_pnl)
            gw = pnl[pnl > 0].sum()
            gl = -pnl[pnl <= 0].sum()
            pfs.append(gw / gl if gl > 0 else np.inf)
            rets.append(pnl.sum() / initial_cash * 100.0)

        pfs_f = np.array([p for p in pfs if np.isfinite(p)])
        return {
            "profit_factor_stress":   round(float(np.median(pfs_f)), 2) if len(pfs_f) else float("inf"),
            "stress_return_pct_med":  round(float(np.median(rets)), 2),
            "stress_return_pct_p05":  round(float(np.percentile(rets, 5)), 2),
            "stress_return_pct_p95":  round(float(np.percentile(rets, 95)), 2),
            "stress_corpse_rate_pct": round(float(np.mean(p_delist)) * 100, 2),
        }

    # =========================================================================
    # SUMMARY
    # =========================================================================

    @staticmethod
    def _compute_summary(
        trades:       list[dict],
        equity_curve: pd.DataFrame,
        initial_cash: float,
        cfg,
        num_signals:  int = 0,
        num_filled:   int = 0,
    ) -> dict:
        if not trades:
            return {
                "num_trades": 0,
                "initial_cash": initial_cash,
                "final_value": round(float(equity_curve["portfolio_value"].iloc[-1]), 2)
                    if not equity_curve.empty else initial_cash,
                "total_return_pct": 0.0,
                "win_rate_pct": 0.0,
                "num_signals": num_signals,
                "num_filled": num_filled,
                "fill_rate_pct": round(num_filled / num_signals * 100, 1) if num_signals else 0.0,
                "note": "No trades executed in date range.",
            }

        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        oversold = [t for t in trades if t.get("signal_type") == "oversold"]
        gap      = [t for t in trades if t.get("signal_type") == "gap_sympathy"]

        final_value = float(equity_curve["portfolio_value"].iloc[-1])
        rolling_max = equity_curve["portfolio_value"].cummax()
        drawdown    = (equity_curve["portfolio_value"] - rolling_max) / rolling_max

        exit_counts: dict[str, int] = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exit_counts[r] = exit_counts.get(r, 0) + 1

        gross_win  = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")

        pv     = equity_curve["portfolio_value"].astype(float)
        rets   = pv.pct_change().dropna()
        sharpe = (
            round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
            if len(rets) >= 2 and rets.std() != 0 else 0.0
        )

        summary = {
            "initial_cash":       initial_cash,
            "final_value":        round(final_value, 2),
            "total_return_pct":   round((final_value - initial_cash) / initial_cash * 100, 2),
            "num_trades":         len(trades),
            "oversold_trades":    len(oversold),
            "gap_trades":         len(gap),
            "win_rate_pct":       round(len(wins) / len(trades) * 100, 1),
            "avg_win":            round(float(np.mean([t["pnl"] for t in wins])), 2) if wins else 0.0,
            "avg_loss":           round(float(np.mean([t["pnl"] for t in losses])), 2) if losses else 0.0,
            "profit_factor":      profit_factor,
            "expectancy":         round(float(np.mean([t["pnl"] for t in trades])), 2),
            "sharpe":             sharpe,
            "largest_win":        round(max(t["pnl"] for t in trades), 2),
            "largest_loss":       round(min(t["pnl"] for t in trades), 2),
            "avg_days_held":      round(float(np.mean([t.get("days_held", 0) for t in trades])), 1),
            "max_drawdown_pct":   round(float(drawdown.min()) * 100, 2),
            "num_signals":        num_signals,
            "num_filled":         num_filled,
            "fill_rate_pct":      round(num_filled / num_signals * 100, 1) if num_signals else 0.0,
            "exit_reason_counts": exit_counts,
        }

        if cfg.survivorship_stress:
            summary.update(
                MicrocapReversionStrategy._survivorship_stress(trades, initial_cash, cfg)
            )

        return summary

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, results: dict) -> None:
        s = results["summary"]
        if s.get("num_trades", 0) == 0:
            print(f"\n[Micro-cap: no trades - signals={s.get('num_signals',0)}, "
                  f"fill_rate={s.get('fill_rate_pct',0)}%]")
            return

        ec = s["exit_reason_counts"]
        tgt_n = ec.get("target", 0)
        sma_n = ec.get("sma_exit", 0)
        sl_n  = ec.get("stop_loss", 0) + ec.get("gap_stop", 0)
        ts_n  = ec.get("time_stop", 0)
        total = s["num_trades"]

        print(f"\n{'='*62}")
        print(f"  Micro-Cap Mean Reversion - Backtest Summary")
        print(f"{'='*62}")
        print(f"  Initial capital      : ${s['initial_cash']:>12,.2f}")
        print(f"  Final value          : ${s['final_value']:>12,.2f}")
        print(f"  Total return (RAW)   : {s['total_return_pct']:>+11.2f}%")
        print(f"  Max drawdown         : {s['max_drawdown_pct']:>+11.2f}%")
        print(f"{'-'*62}")
        print(f"  Signals generated    : {s['num_signals']:>12}")
        print(f"  Limit orders filled  : {s['num_filled']:>12}  "
              f"(fill rate {s['fill_rate_pct']:.1f}%)")
        print(f"  Total trades         : {total:>12}  "
              f"(oversold:{s['oversold_trades']} / gap:{s['gap_trades']})")
        print(f"  Win rate             : {s['win_rate_pct']:>11.1f}%")
        print(f"  Avg win              : ${s['avg_win']:>12.2f}")
        print(f"  Avg loss             : ${s['avg_loss']:>12.2f}")
        print(f"  Profit factor (RAW)  : {s['profit_factor']:>12.2f}")
        print(f"  Expectancy / trade   : ${s['expectancy']:>12.2f}")
        print(f"  Sharpe (ann.)        : {s['sharpe']:>12.2f}")
        print(f"  Avg days held        : {s['avg_days_held']:>12.1f}")
        print(f"{'-'*62}")
        print(f"  Exit breakdown:")
        print(f"    Reversion target   : {tgt_n:>5} ({tgt_n/total*100:>4.0f}%)")
        print(f"    SMA reversion      : {sma_n:>5} ({sma_n/total*100:>4.0f}%)")
        print(f"    Stop-loss / gap    : {sl_n:>5} ({sl_n/total*100:>4.0f}%)")
        print(f"    Time stop          : {ts_n:>5} ({ts_n/total*100:>4.0f}%)")
        if "profit_factor_stress" in s:
            print(f"{'-'*62}")
            print(f"  SURVIVORSHIP STRESS-TEST (de-biased - trust these, not RAW):")
            print(f"    Implied corpse rate: {s['stress_corpse_rate_pct']:>10.2f}% of trades")
            print(f"    Profit factor      : {s['profit_factor_stress']:>10.2f}  "
                  f"(RAW was {s['profit_factor']:.2f})")
            print(f"    Return  median     : {s['stress_return_pct_med']:>+9.2f}%")
            print(f"    Return  5th pctile : {s['stress_return_pct_p05']:>+9.2f}%  (bad-luck tail)")
            print(f"    Return 95th pctile : {s['stress_return_pct_p95']:>+9.2f}%")
        print(f"{'-'*62}")
        print(f"  NOTE: RAW metrics are survivorship-biased and assume every limit")
        print(f"  fill was real. Treat the stress-test PF as the honest estimate.")
        print(f"{'='*62}\n")

    def plot_backtest(self, results: dict) -> None:
        s      = results["summary"]
        equity = results["equity_curve"]
        trades = results["trades"]

        if not trades:
            print("Micro-cap: no trades to plot.")
            return

        fig = plt.figure(figsize=(15, 12))
        gs  = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        pv  = equity["portfolio_value"]
        ax1.plot(equity.index, pv, color="#1f77b4", linewidth=1.5, label="Portfolio (RAW)")
        ax1.axhline(s["initial_cash"], color="gray", linewidth=0.8, linestyle="--",
                    label="Starting capital")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv >= s["initial_cash"]), alpha=0.12, color="#2ca02c")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv < s["initial_cash"]), alpha=0.12, color="#d62728")
        pf_stress = s.get("profit_factor_stress", float("nan"))
        ax1.set_title(
            f"Micro-Cap Mean Reversion — Equity Curve (RAW, survivorship-biased)\n"
            f"RAW Return {s['total_return_pct']:+.1f}%  |  {s['num_trades']} trades  |  "
            f"Win {s['win_rate_pct']:.0f}%  |  Fill {s['fill_rate_pct']:.0f}%  |  "
            f"Stressed PF {pf_stress:.2f}",
            fontsize=11,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 2. Trade P&L distribution
        ax2  = fig.add_subplot(gs[1, 0])
        pnls = [t["pnl"] for t in trades]
        ax2.hist(pnls, bins=25, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax2.axvline(0, color="#d62728", linewidth=1.2, linestyle="--")
        ax2.axvline(float(np.mean(pnls)), color="black", linewidth=0.8, linestyle=":",
                    label=f"Mean ${float(np.mean(pnls)):.2f}")
        ax2.set_title("Trade P&L Distribution")
        ax2.set_xlabel("P&L per Trade ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. Exit reason breakdown
        ax3    = fig.add_subplot(gs[1, 1])
        ec     = s["exit_reason_counts"]
        labels = list(ec.keys())
        sizes  = list(ec.values())
        colors = ["#2ca02c", "#17becf", "#d62728", "#ff7f0e", "#9467bd"]
        ax3.pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=colors[: len(labels)], startangle=90)
        ax3.set_title("Exit Reason Breakdown")

        # 4. RAW vs stressed return comparison
        ax4 = fig.add_subplot(gs[2, :])
        cats = ["RAW\n(biased)", "Stress\nmedian", "Stress\n5th pctile"]
        vals = [
            s["total_return_pct"],
            s.get("stress_return_pct_med", 0.0),
            s.get("stress_return_pct_p05", 0.0),
        ]
        bar_colors = ["#7f7f7f", "#2ca02c" if vals[1] >= 0 else "#d62728",
                      "#2ca02c" if vals[2] >= 0 else "#d62728"]
        bars = ax4.bar(cats, vals, color=bar_colors, width=0.5, alpha=0.85)
        ax4.axhline(0, color="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (1 if v >= 0 else -1),
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
        ax4.set_ylabel("Total Return (%)")
        ax4.set_title("RAW vs Survivorship-Stressed Return (why the RAW number is a mirage)")
        ax4.grid(alpha=0.3, axis="y")

        plt.suptitle("Micro-Cap Mean Reversion — Backtest Report", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # LIVE EXECUTION STUBS
    # =========================================================================

    def evening_scan(self, tickers: Optional[list[str]] = None) -> list[dict]:
        """Live candidate scan.

        Unlike the backtest (which must use a price-shape no-news proxy because
        Alpaca lacks the news history), the live path applies the REAL solvency
        filter: RiskManager.scan_news drops any name carrying a
        bankruptcy / fraud / SEC / FDA / delisting headline.  Order placement is
        intentionally NOT implemented — live micro-cap execution needs a real
        limit-order / non-fill layer this backtest-first module does not yet ship.
        """
        self.log.info("MicrocapReversionStrategy.evening_scan() — candidate scan only "
                      "(live order placement not yet implemented).")
        candidates: list[dict] = []
        if not tickers:
            return candidates

        end_dt   = datetime.datetime.now(ET)
        start_dt = end_dt - datetime.timedelta(days=self.cfg.min_history_bars + 120)
        daily    = self._fetch_daily_bars(list(tickers), start_dt, end_dt)

        for sym, df in daily.items():
            sig = self._check_oversold_signal(sym, df)
            if sig is None and self.cfg.enable_gap_sympathy:
                # Live sympathy check would need today's peer return; skipped here.
                continue
            if sig is None:
                continue
            # Real no-news / solvency filter (live only).
            if self.rm is not None:
                try:
                    all_clear, alerts = self.rm.scan_news([sym], lookback_min=60 * 24 * 3)
                    if not all_clear or alerts:
                        self.log.info("  %s dropped by news filter: %s", sym,
                                      alerts[0] if alerts else "critical")
                        continue
                except Exception as exc:
                    self.log.debug("news scan failed for %s: %s", sym, exc)
            candidates.append(sig)

        self.log.info("evening_scan: %d candidate(s) passed filters.", len(candidates))
        return candidates

    def morning_session(self) -> None:
        """Placeholder — live order placement not yet implemented."""
        self.log.info("MicrocapReversionStrategy.morning_session() — not yet implemented.")
