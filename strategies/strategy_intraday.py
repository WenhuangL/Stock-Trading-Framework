"""
strategy_intraday.py
--------------------
Three-phase intraday trading strategy for the full regular session.

PHASE 1 — Opening Range Breakout (ORB)  9:30 – 10:30 AM ET
    Waits for the first 15 minutes to establish a price range, then enters
    on a confirmed breakout in either direction.  The opening imbalance of
    institutional and retail orders creates directional momentum that
    typically persists for 30–60 minutes after the initial breakout.

    Entry  : 1-min bar closes outside the range with volume > 1.2× average
    TP     : 2.0× range size from entry
    SL     : 0.6× range size back inside range
    Window : entries 9:45–10:00 AM, exits by 10:30 AM

PHASE 2 — VWAP Reversion  11:00 AM – 2:30 PM ET
    Identifies stocks that have moved more than 1.5× their ATR away from
    VWAP and shows early signs of exhaustion (declining volume, momentum
    shift). Bets that price reverts toward VWAP — the institutional
    fair-value anchor — during the low-volume midday window.

    Entry  : price > 1.5 ATR from VWAP, last 5-bar avg volume < 80% of day avg
    TP     : within 0.2% of VWAP
    SL     : 2.2× ATR from VWAP (further extension = thesis wrong)
    Window : 11:00 AM–2:00 PM entries, max 90-minute hold, close by 2:30 PM

PHASE 3 — Power Hour  3:05 – 3:50 PM ET
    At the start of the final hour, institutional end-of-day positioning
    amplifies the day's dominant trend.  Enters in the trend direction
    on a VWAP pullback/hold for a short, high-probability push.

    Entry  : stock trending consistently above/below VWAP all day,
             pullback to within 0.3 ATR of VWAP at 3:05–3:30 PM
    TP     : +1.0% (longs), -1.0% (shorts)
    SL     : -0.5% (longs), +0.5% (shorts)   → 2:1 reward-to-risk
    Window : entries 3:05–3:30 PM, hard close 3:50 PM

PREMARKET PREPARATION  8:00 – 9:25 AM ET
    Builds a ranked watchlist from the universe for ORB and power-hour
    consideration.  Scores each stock on: gap size, premarket RVOL,
    news catalyst presence, and futures alignment.  No orders are placed
    in premarket — preparation only.

SHARED POSITION MANAGEMENT
    All three phases share a single open-position list and a monitoring
    loop. A position opened in Phase 1 is correctly managed through
    Phases 2 and 3 if it has not yet hit TP or SL.  The risk_manager
    is checked before every new entry and every 5 minutes during the
    session.

BACKTEST
    Fetches 1-minute regular-session bars per stock for the full date
    range (one API call per stock).  Simulates all three phases per day
    with a shared capital pool, using high/low within each bar for
    conservative SL/TP fill assumptions.  Returns a results dict that
    drives plot_backtest() and print_summary().

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy matplotlib yfinance
"""

import datetime
import logging
import time
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from indicators.analyze import (calculate_rsi, calculate_macd, calculate_bollinger_bands,
                                 calculate_atr, calculate_vwap, calculate_sma, calculate_ema,
                                 calculate_relative_volume, generate_signals
)
from risk.risk_manager import RiskManager
from data_collection.stock_universe import UniverseSelector, UniverseConfig

ET = ZoneInfo("America/New_York")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IntradayConfig:
    """All tuneable parameters for the three intraday phases."""

    # ── Phase 1: ORB ───────────────────────────────────────────────────────────
    orb_range_bars:        int   = 15      # first N bars establish the range (9:30-9:44)
    orb_entry_vol_mult:    float = 1.50    # breakout bar volume must exceed range avg × this
    orb_tp_mult:           float = 0.75   # TP = range_size × this above/below entry
    orb_sl_mult:           float = 0.50   # SL = range_size × this inside the range
    orb_entry_cutoff:      str   = "10:00" # no new ORB entries after this time
    orb_phase_end:         str   = "10:30" # close all ORB positions by this time
    orb_min_rvol:          float = 1.50   # minimum relative volume at open
    orb_min_range_pct:     float = 0.004  # range must be ≥ 0.4% of price (not too tight)
    orb_max_range_pct:     float = 0.025  # range must be ≤ 2.5% of price (not too wide)
    orb_position_size_pct: float = 0.025  # 2.5% of portfolio per position
    orb_max_positions:     int   = 2
    orb_min_entry_time:    str   = "09:52"  # skip ORB signal bars before this time

    # ── ORB catalyst gate (Step 1) ─────────────────────────────────────────────
    orb_require_catalyst:  bool  = False
    """Catalyst gating for ORB. In LIVE trading, when True a stock only qualifies
    for ORB if Alpaca's news API found at least one headline for that symbol in
    the 2 hours before open.

    IMPORTANT — backtest honesty: the backtest does NOT have point-in-time news,
    so `_rank_orb_candidates` is called with `catalyst_symbols=None` and the gate
    is a no-op there. Previously this defaulted to True, which gave the false
    impression that backtested ORB results were catalyst-filtered when they were
    not. It now defaults to False so the config matches what the backtest actually
    measures (unfiltered ORB). Wire historical news into the backtest before
    flipping this back on, or treat live and backtest as different systems."""

    orb_news_lookback_min: int   = 120
    """How many minutes before market open to look back for catalyst news.
    120 = 7:30–9:30 AM ET window. Extend to 240 to catch overnight headlines."""

    # ── Phase 2: VWAP Reversion ────────────────────────────────────────────────
    # EXPERIMENT (reversion-windows branch): widened to the full session to test
    # whether the reversion edge extends into the 9:45-11:00 AM and 3:00-3:55 PM
    # windows. Baseline (11:00-14:30 window) = +9.43%. Segment results by entry
    # time to see which windows actually carry edge before locking these in.
    vwap_entry_start:      str   = "09:45" # was 11:00 — test morning reversion
    vwap_entry_cutoff:     str   = "15:45" # was 14:30 — test afternoon/close reversion
    vwap_phase_end:        str   = "15:55"  # was 15:00 — hold into the close
    vwap_extension_atr:    float = 2.50   # price must be > N×ATR from VWAP
    vwap_sl_atr:           float = 1.00   # SL distance in ATR (was 1.40 — segmented opt found a tighter stop wins: full-year 13.46%->14.32%, validated out-of-sample H1 6.85->7.25 / H2 6.16->6.57, PF 3.48->4.16)
    vwap_tp_pct:           float = 0.002  # TP within 0.2% of VWAP
    vwap_tp_extension_atr: float = 1.00  # extend TP this many ATR past VWAP (0 = exit at VWAP)
    vwap_breakeven_atr:    float = 1.00   # slide SL to entry after this many ATR of recovery
    vwap_vol_decay:        float = 0.80   # recent vol must be < 80% of day average
    vwap_signal_vol_max:   float = 1.3   # signal bar vol must be < this × day avg (was hard-coded 1.5)
    vwap_min_atr:          float = 0.05  # skip stocks with tiny ATR (low-vol / near-penny)
    vwap_max_hold_bars:    int   = 90     # 90-minute max hold on 1-min bars
    vwap_sl_grace_bars:    int   = 25    # skip SL evaluation for first N bars after entry
    vwap_position_size_pct:float = 0.015  # 1.5% of portfolio per position
    vwap_max_positions:    int   = 6
    vwap_atr_period:       int   = 14

    # ── VWAP dead zone (Step 3) ────────────────────────────────────────────────
    # EXPERIMENT: disabled (start==end==0 → empty range) so every time-of-day
    # window is measurable. Re-introduce a targeted dead zone afterward if the
    # midday segment proves unprofitable.
    vwap_dead_zone_start:  int   = 0
    """Bar index where the midday dead zone begins (~11:15 AM on 1-min bars,
    counting from bar 0 = 9:30 AM open). No new VWAP reversion entries are
    opened between dead_zone_start and dead_zone_end. Liquidity is lowest
    midday; extensions drift rather than snap back. Shifted from 120 (11:30 AM)
    to 105 (11:15 AM) to block fill-bar leakage from signals at the boundary."""

    vwap_dead_zone_end:    int   = 0
    """EXPERIMENT: 0 disables the dead zone (the `start <= i < end` test is never
    true when start==end==0). Original value 210 blocked midday entries. Restore
    a targeted window here if the segmented results show midday drift losses."""

    # ── VWAP SPY alignment (Step 3) ────────────────────────────────────────────
    vwap_spy_alignment:    bool  = True
    """When True, only take LONG VWAP reversion entries when SPY is trading
    above its own session VWAP. Prevents fading extensions in a structurally
    down market where mean reversion is unlikely.
    Short entries are always allowed regardless of SPY direction."""

    # ── Phase 3: Power Hour ────────────────────────────────────────────────────
    power_entry_start:     str   = "15:00"
    power_entry_cutoff:    str   = "15:35"
    power_hard_close:      str   = "15:55"
    power_tp_pct:          float = 0.010  # +1.0% TP
    power_sl_pct:          float = 0.005  # -0.5% SL  →  2:1 R:R
    power_vwap_entry_atr:  float = 0.50   # enter only when within 0.5 ATR of VWAP
    power_trend_bars:      int   = 60     # look back this many bars for trend detection
    power_trend_threshold: float = 0.55   # fraction of bars on VWAP side to call a trend
    power_position_size_pct:float= 0.020  # 2.0% of portfolio per position
    power_max_positions:   int   = 4

    # ── Premarket preparation ──────────────────────────────────────────────────
    pm_gap_min_pct:        float = 0.005  # minimum gap to qualify (0.5%)
    pm_gap_max_pct:        float = 0.080  # maximum gap (>8% often fills)
    pm_rvol_min:           float = 1.30   # premarket RVOL threshold
    pm_watchlist_size:     int   = 25     # number of candidates to track at open

    # ── Common ────────────────────────────────────────────────────────────────
    min_price:             float = 15.0
    monitor_interval_sec:  int   = 30     # how often the live monitor loop fires
    risk_check_interval_sec: int = 300    # how often intraday risk check fires (5 min)
    daily_dd_halt_pct: float = 0.004      # halt new positions if day P&L < -0.4% of portfolio

    # ── Slippage / fill realism ────────────────────────────────────────────────
    slippage_pct: float = 0.0004  # 0.04% per fill — market impact + reversion entry-timing friction on liquid large-caps
    use_next_bar_fill: bool = True  # fill at next bar's open, not signal bar's close

    # ── Transaction costs ──────────────────────────────────────────────────────
    spread_pct: float = 0.0002
    """Half bid-ask spread applied to entry AND exit, on top of slippage.
    0.0002 (~$0.02 on a $100 stock) reflects the effective IEX spread on
    liquid large-caps. Combined with slippage_pct (0.0004) this gives ~0.12%
    total round-trip cost — the realistic center for top-100 large-caps via
    Alpaca IEX. Validated against the 2023 backtest: VWAP-only nets +7.9%.
    Calibrate against real live fills once enough paper trades accumulate."""

    commission_per_trade: float = 0.0
    """Flat commission charged once per round-trip trade. Alpaca equities are
    commission-free, so 0.0 is realistic; set it to model a different broker."""

    # ── Backtest data feed ─────────────────────────────────────────────────────
    backtest_feed: str = "iex"
    """Data feed used when fetching historical bars for the backtest.
    'iex' matches live trading (free Alpaca tier) and naturally excludes
    any ticker with no IEX coverage, just like live. Use 'sip' for the
    broadest historical data (consolidated tape, paid tier)."""

    # ── Fix 7: ORB minimum breakout margin ────────────────────────────────────
    orb_breakout_margin_pct: float = 0.002
    """A breakout bar's close must exceed the range boundary by at least this
    fraction before the signal fires. 0.002 = close must be > r_high * 1.002
    (longs) or < r_low * 0.998 (shorts). Eliminates marginal one-tick
    breakouts that immediately reverse."""

    # ── Fix 8: SPY regime filter for ORB ──────────────────────────────────────
    orb_regime_filter: bool = True
    """Skip ORB longs when SPY is in a sharp opening downtrend and skip ORB
    shorts when SPY is in a sharp opening uptrend. Breakouts against the broad
    tape are low-probability."""

    orb_regime_long_skip_threshold: float = -0.010
    """Skip ORB longs if SPY's 30-bar opening return (bars 0-30, 9:30-10:00 AM)
    is below this value. -0.01 = SPY down more than 1% from its open."""

    orb_regime_short_skip_threshold: float = 0.000
    """Skip ORB shorts if SPY's 30-bar opening return is above this value.
    0.0 = skip shorts any time SPY is up at all from its open (bull-bias filter)."""

    # ── Fix 9: ATR-based position sizing ──────────────────────────────────────
    orb_risk_per_trade_pct: float = 0.005
    """Dollar risk per ORB trade as a fraction of portfolio. 0.005 = 0.5%.
    qty = risk_dollars / sl_per_share, capped by orb_max_notional_pct."""

    orb_max_notional_pct: float = 0.025
    """Hard notional cap per ORB position (2.5% of portfolio). Matches the old
    fixed-notional size so the regime filter and breakout margin changes are
    isolated from position sizing changes. Raise once the strategy is
    profitable to let ATR sizing fully express itself."""

    vwap_risk_per_trade_pct: float = 0.016
    """Dollar risk per VWAP trade as a fraction of portfolio (1.6%)."""

    vwap_max_notional_pct: float = 0.05
    """Hard notional cap per VWAP position (5.0% of portfolio). Matches the
    old fixed-notional vwap_position_size_pct."""

    power_risk_per_trade_pct: float = 0.005
    """Dollar risk per Power Hour trade as a fraction of portfolio (0.5%)."""

    power_max_notional_pct: float = 0.020
    """Hard notional cap per Power Hour position (2.0% of portfolio). Matches
    the old fixed-notional power_position_size_pct."""


# =============================================================================
# INTRADAY STRATEGY
# =============================================================================

class IntradayStrategy:
    """
    Orchestrates all three intraday phases for live trading and backtesting.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        risk_manager: RiskManager,
        config: Optional[IntradayConfig] = None,
        universe_config: Optional[UniverseConfig] = None,
    ) -> None:
        self.tc   = trading_client
        self.dc   = data_client
        self.rm   = risk_manager
        self.cfg  = config or IntradayConfig()
        self.ucfg = universe_config or UniverseConfig()
        self.log  = logging.getLogger(self.__class__.__name__)

        self._positions:  list[dict]  = []   # shared across all phases
        self._watchlist:  list[dict]  = []   # prepared in premarket
        self._universe:   list[str]   = []   # loaded once per session
        self._last_risk_check:        float = 0.0

    # =========================================================================
    # PREMARKET PREPARATION  (call at 8:00 AM ET)
    # =========================================================================

    def prepare_watchlist(self) -> list[dict]:
        """
        Score the universe against premarket conditions and return a ranked
        watchlist for use at the open.
        """
        self.log.info("Preparing premarket watchlist...")

        selector       = UniverseSelector(self.dc, self.ucfg)
        self._universe = selector.get_universe(use_cache=True)

        if not self._universe:
            self.log.warning("Universe empty — cannot prepare watchlist.")
            return []

        # Futures direction (SPY premarket change)
        futures_bias = self._get_spy_premarket_direction()

        # Fetch snapshots for scoring
        candidates: list[dict] = []
        BATCH = 500
        for i in range(0, len(self._universe), BATCH):
            batch = self._universe[i : i + BATCH]
            try:
                snaps = self.dc.get_stock_snapshots(
                    StockSnapshotRequest(symbol_or_symbols=batch, feed=DataFeed.IEX)
                )
                for sym, snap in snaps.items():
                    if not snap.daily_bar or not snap.prev_daily_bar:
                        continue
                    prev_close = snap.prev_daily_bar.close
                    current    = snap.daily_bar.close
                    if prev_close < self.cfg.min_price:
                        continue
                    gap = (current - prev_close) / prev_close
                    if abs(gap) < self.cfg.pm_gap_min_pct or abs(gap) > self.cfg.pm_gap_max_pct:
                        continue

                    candidates.append({
                        "symbol":        sym,
                        "prev_close":    prev_close,
                        "premarket_price": current,
                        "gap_pct":       gap,
                        "gap_direction": 1 if gap > 0 else -1,
                        "futures_aligned": (gap > 0) == (futures_bias > 0),
                    })
            except Exception as exc:
                self.log.warning(f"Snapshot batch {i} failed: {exc}")

        if not candidates:
            self.log.info("No gap candidates found in premarket.")
            return []

        # Fetch premarket RVOL using latest quote volume estimate
        self._score_watchlist_candidates(candidates, futures_bias)

        # Sort by composite score, take top N
        candidates.sort(key=lambda x: x.get("pm_score", 0), reverse=True)
        self._watchlist = candidates[: self.cfg.pm_watchlist_size]

        self.log.info(
            f"Watchlist ready: {len(self._watchlist)} candidates | "
            f"Top: {[c['symbol'] for c in self._watchlist[:5]]}"
        )
        return self._watchlist

    def _score_watchlist_candidates(
        self, candidates: list[dict], futures_bias: float
    ) -> None:
        """Add a pm_score to each candidate dict in-place."""
        # News check (batch: scan for symbols that have news)
        news_symbols: set[str] = set()
        _, alerts = self.rm.scan_news(
            symbols=[c["symbol"] for c in candidates[:50]],
            lookback_min=120,
        )
        # Crude: any alert mentions → mark symbol as having news
        for c in candidates:
            if any(c["symbol"].lower() in a for a in alerts):
                news_symbols.add(c["symbol"])

        for c in candidates:
            gap_abs = abs(c["gap_pct"])
            has_cat = c["symbol"] in news_symbols

            # Gap score: peaks at 2-3%, decays on both sides
            gap_score = np.exp(-0.5 * ((gap_abs - 0.025) / 0.015) ** 2)

            # Futures alignment
            futures_score = 0.8 if c["futures_aligned"] else 0.2

            # Catalyst
            catalyst_score = 1.0 if has_cat else 0.3

            c["has_catalyst"] = has_cat  # used by _scan_orb_entries gate
            c["pm_score"] = (
                0.35 * gap_score
                + 0.15 * futures_score
                + 0.20 * catalyst_score
                + 0.30 * 0.5   # RVOL placeholder (live premarket vol not easily accessible)
            )

    def _get_spy_premarket_direction(self) -> float:
        """Returns SPY gap fraction (positive = up, negative = down)."""
        try:
            snaps = self.dc.get_stock_snapshots(
                StockSnapshotRequest(symbol_or_symbols=["SPY", "QQQ"], feed=DataFeed.IEX)
            )
            changes = []
            for sym in ["SPY", "QQQ"]:
                s = snaps.get(sym)
                if s and s.daily_bar and s.prev_daily_bar:
                    changes.append(
                        (s.daily_bar.close - s.prev_daily_bar.close) / s.prev_daily_bar.close
                    )
            return float(np.mean(changes)) if changes else 0.0
        except Exception:
            return 0.0

    # =========================================================================
    # LIVE EXECUTION ENTRY POINT  (blocking, 9:30 AM → 3:55 PM)
    # =========================================================================

    def run(self) -> dict:
        """
        Execute all three phases sequentially. Blocks until 3:55 PM ET then
        returns a session summary dict.  Call prepare_watchlist() first.
        """
        self.log.info("Intraday session starting.")
        account = self.tc.get_account()
        portfolio_value = float(account.portfolio_value)

        # ── VWAP Reversion owns the full session (09:45 – 15:55) ──────────────
        # ORB and Power Hour removed from the live session. Both were confirmed
        # money-losers in the 2023 backtest (large-caps mean-revert intraday, so
        # continuation/breakout bets lose). ORB's blocking 9:30-10:30 loop also
        # prevented VWAP from capturing the morning reversion window, which the
        # backtest showed is the single most profitable window of the day
        # (avg +$13.30/trade). _phase_vwap now runs the whole session: it enters
        # 09:45-15:45, then holds and closes all positions through
        # vwap_phase_end (15:55).
        _wait_until(self.cfg.vwap_entry_start)   # 09:45 (also serves as the market-open wait)
        self.log.info("Phase: VWAP Reversion (full session 09:45–15:55)")
        self._phase_vwap(portfolio_value)

        # Final close — safety net for anything _phase_vwap left open.
        remaining = [p for p in self._positions if not p.get("closed", False)]
        if remaining:
            self.log.info(f"Closing {len(remaining)} remaining position(s) at session end.")
            for pos in remaining:
                self._close_live_position(pos, reason="session_end")

        session_trades = [p for p in self._positions if p.get("closed", False)]
        total_pnl = sum(p.get("pnl", 0.0) for p in session_trades)
        self.log.info(
            f"Session complete. {len(session_trades)} trades | P&L ${total_pnl:+.2f}"
        )
        return {"trades": session_trades, "total_pnl": total_pnl}

    # =========================================================================
    # PHASE 1: ORB
    # =========================================================================

    def _phase_orb(self, portfolio_value: float) -> None:
        """Opening Range Breakout: 9:30 – 10:30 AM."""
        entry_cutoff = _parse_time_today(self.cfg.orb_entry_cutoff)
        phase_end    = _parse_time_today(self.cfg.orb_phase_end)
        range_lock   = _parse_time_today("09:45")  # range established after 15 bars
        range_data: dict[str, dict] = {}  # symbol → {high, low, avg_vol, range_size}
        last_risk = time.time()

        while datetime.datetime.now(ET) < phase_end:
            now = datetime.datetime.now(ET)

            # Risk check every 5 minutes
            if time.time() - last_risk > self.cfg.risk_check_interval_sec:
                prices = self._fetch_current_prices()
                pv     = self._estimate_portfolio_value(portfolio_value, prices)
                status = self.rm.intraday_monitor(self._positions, prices, pv)
                if status["action"] == "emergency_close":
                    self.rm.emergency_close_all(self._positions, status["reason"])
                    return
                last_risk = time.time()

            # Monitor all existing positions
            self._shared_monitor()

            if now >= range_lock and now < entry_cutoff:
                # Fetch current intraday bars and look for new ORB setups
                self._scan_orb_entries(range_data, portfolio_value)

            time.sleep(self.cfg.monitor_interval_sec)

        # Hard close all ORB positions at phase end
        for pos in [p for p in self._positions
                    if not p.get("closed", False) and p.get("phase") == "orb"]:
            self._close_live_position(pos, reason="orb_time_stop")

    def _scan_orb_entries(
        self, range_data: dict, portfolio_value: float
    ) -> None:
        """Identify and enter ORB breakouts from the current watchlist."""
        open_symbols = {p["symbol"] for p in self._positions if not p.get("closed", False)}
        orb_count    = sum(
            1 for p in self._positions
            if not p.get("closed", False) and p.get("phase") == "orb"
        )
        if orb_count >= self.cfg.orb_max_positions:
            return

        # Catalyst gate
        if self.cfg.orb_require_catalyst:
            candidates = [
                c["symbol"] for c in self._watchlist
                if c["symbol"] not in open_symbols
                and c.get("has_catalyst", False)
            ][:20]
        else:
            candidates = [c["symbol"] for c in self._watchlist
                          if c["symbol"] not in open_symbols][:20]
        if not candidates:
            return

        try:
            today     = datetime.date.today()
            start_dt  = datetime.datetime.combine(today, datetime.time(9, 30), tzinfo=ET)
            resp      = self.dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols = candidates,
                timeframe         = TimeFrame.Minute,
                start             = start_dt,
                end               = datetime.datetime.now(ET),
                feed              = DataFeed.IEX,
            ))
        except Exception as exc:
            self.log.warning(f"ORB bar fetch failed: {exc}")
            return

        for sym in candidates:
            if orb_count >= self.cfg.orb_max_positions:
                break
            try:
                df = _extract_symbol_df(resp, sym)
                if df is None or len(df) < self.cfg.orb_range_bars + 1:
                    continue

                range_bars  = df.iloc[: self.cfg.orb_range_bars]
                r_high      = float(range_bars["high"].max())
                r_low       = float(range_bars["low"].min())
                r_size      = r_high - r_low
                r_avg_vol   = float(range_bars["volume"].mean())
                price       = float(df["close"].iloc[-1])

                if r_size / price < self.cfg.orb_min_range_pct:
                    continue
                if r_size / price > self.cfg.orb_max_range_pct:
                    continue

                first_5_vol  = float(df["volume"].iloc[:5].mean())
                hist_avg_vol = float(df["volume"].mean())
                rvol         = first_5_vol / hist_avg_vol if hist_avg_vol > 0 else 1.0
                if rvol < self.cfg.orb_min_rvol:
                    continue

                last = df.iloc[-1]
                if (float(last["close"]) > r_high and
                        float(last["volume"]) > r_avg_vol * self.cfg.orb_entry_vol_mult):
                    direction = "long"
                    entry     = r_high
                    tp        = entry + r_size * self.cfg.orb_tp_mult
                    sl        = entry - r_size * self.cfg.orb_sl_mult
                elif (float(last["close"]) < r_low and
                      float(last["volume"]) > r_avg_vol * self.cfg.orb_entry_vol_mult):
                    direction = "short"
                    entry     = r_low
                    tp        = entry - r_size * self.cfg.orb_tp_mult
                    sl        = entry + r_size * self.cfg.orb_sl_mult
                else:
                    continue

                sl_per_share = abs(entry - sl)
                if sl_per_share > 0:
                    risk_dollars = portfolio_value * self.cfg.orb_risk_per_trade_pct
                    max_qty      = int((portfolio_value * self.cfg.orb_max_notional_pct) / price)
                    qty          = min(int(risk_dollars / sl_per_share), max_qty)
                else:
                    qty = int((portfolio_value * self.cfg.orb_max_notional_pct) / price)
                if qty < 1:
                    continue
                notional = qty * price

                ok, reason = self.rm.check_ok_to_enter(
                    sym, self._positions, notional, portfolio_value
                )
                if not ok:
                    self.log.debug(f"ORB entry blocked for {sym}: {reason}")
                    continue

                self._submit_and_track(
                    sym, qty, direction, entry, tp, sl, "orb", price
                )
                orb_count += 1

            except Exception as exc:
                self.log.debug(f"ORB scan error for {sym}: {exc}")

    # =========================================================================
    # PHASE 2: VWAP REVERSION
    # =========================================================================

    def _phase_vwap(self, portfolio_value: float) -> None:
        """VWAP Reversion: 11:00 AM – 2:30 PM."""
        entry_cutoff = _parse_time_today(self.cfg.vwap_entry_cutoff)
        phase_end    = _parse_time_today(self.cfg.vwap_phase_end)
        last_scan    = 0.0
        last_risk    = time.time()

        while datetime.datetime.now(ET) < phase_end:
            now = datetime.datetime.now(ET)

            if time.time() - last_risk > self.cfg.risk_check_interval_sec:
                prices = self._fetch_current_prices()
                pv     = self._estimate_portfolio_value(portfolio_value, prices)
                status = self.rm.intraday_monitor(self._positions, prices, pv)
                if status["action"] == "emergency_close":
                    self.rm.emergency_close_all(self._positions, status["reason"])
                    return
                last_risk = time.time()

            self._shared_monitor()

            if (now < entry_cutoff and
                    time.time() - last_scan > 300):  # scan every 5 min
                self._scan_vwap_entries(portfolio_value)
                last_scan = time.time()

            time.sleep(self.cfg.monitor_interval_sec)

        for pos in [p for p in self._positions
                    if not p.get("closed", False) and p.get("phase") == "vwap"]:
            self._close_live_position(pos, reason="vwap_phase_end")

    def _scan_vwap_entries(self, portfolio_value: float) -> None:
        """Scan for VWAP extension setups across the universe."""
        open_symbols  = {p["symbol"] for p in self._positions if not p.get("closed", False)}
        vwap_count    = sum(
            1 for p in self._positions
            if not p.get("closed", False) and p.get("phase") == "vwap"
        )
        if vwap_count >= self.cfg.vwap_max_positions:
            return

        candidates = [s for s in self._universe if s not in open_symbols][:40]
        if not candidates:
            return

        try:
            today    = datetime.date.today()
            start_dt = datetime.datetime.combine(today, datetime.time(9, 30), tzinfo=ET)
            resp     = self.dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols = candidates,
                timeframe         = TimeFrame.Minute,
                start             = start_dt,
                end               = datetime.datetime.now(ET),
                feed              = DataFeed.IEX,
            ))
        except Exception as exc:
            self.log.warning(f"VWAP bar fetch failed: {exc}")
            return

        scored: list[tuple[float, str, dict]] = []

        # ── SPY VWAP state for alignment filter ───────────────────────────────
        spy_above_vwap: Optional[bool] = None
        if self.cfg.vwap_spy_alignment:
            try:
                today_str = datetime.date.today().isoformat()
                spy_resp  = self.dc.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols = ["SPY"],
                    timeframe         = TimeFrame.Minute,
                    start             = datetime.datetime.combine(
                        datetime.date.today(), datetime.time(9, 30), tzinfo=ET),
                    end               = datetime.datetime.now(ET),
                    feed              = DataFeed.IEX,
                ))
                spy_df = _extract_symbol_df(spy_resp, "SPY")
                if spy_df is not None and len(spy_df) >= 30:
                    spy_vwap     = calculate_vwap(spy_df)
                    spy_price    = float(spy_df["close"].iloc[-1])
                    spy_vwap_val = float(spy_vwap.iloc[-1])
                    spy_above_vwap = spy_price > spy_vwap_val
            except Exception as exc:
                self.log.debug(f"SPY VWAP fetch failed: {exc}")

        # ── Dead zone check ───────────────────────────────────────────────────
        now_bar_est = int(
            (datetime.datetime.now(ET) -
             datetime.datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)
            ).total_seconds() / 60
        )
        dead_zone_active = (
            self.cfg.vwap_dead_zone_start
            <= now_bar_est
            < self.cfg.vwap_dead_zone_end
        )
        if dead_zone_active:
            self.log.debug("VWAP dead zone active — skipping entry scan.")
            return

        for sym in candidates:
            if vwap_count >= self.cfg.vwap_max_positions:
                break
            try:
                df = _extract_symbol_df(resp, sym)
                if df is None or len(df) < 30:
                    continue

                vwap_series = calculate_vwap(df)
                atr_series  = calculate_atr(df, period=self.cfg.vwap_atr_period)
                current_vwap= float(vwap_series.iloc[-1])
                current_atr = float(atr_series.iloc[-1])
                price       = float(df["close"].iloc[-1])

                if current_atr <= 0:
                    continue

                distance    = price - current_vwap
                dist_in_atr = abs(distance) / current_atr

                if dist_in_atr < self.cfg.vwap_extension_atr:
                    continue

                day_avg_vol = float(df["volume"].mean())
                recent_vol  = float(df["volume"].iloc[-5:].mean())
                if day_avg_vol > 0 and recent_vol / day_avg_vol >= self.cfg.vwap_vol_decay:
                    continue

                direction = "short" if distance > 0 else "long"

                if (self.cfg.vwap_spy_alignment
                        and spy_above_vwap is not None
                        and direction == "long"
                        and not spy_above_vwap):
                    continue

                entry     = price
                tp        = current_vwap  # target VWAP
                sl_dist   = current_atr * self.cfg.vwap_sl_atr
                sl        = (price + sl_dist) if direction == "short" else (price - sl_dist)

                vol_decay_score = max(0, 1 - recent_vol / day_avg_vol) if day_avg_vol > 0 else 0
                score = dist_in_atr * 0.6 + vol_decay_score * 0.4
                scored.append((score, sym, {
                    "direction": direction, "entry": entry, "tp": tp, "sl": sl,
                }))

            except Exception as exc:
                self.log.debug(f"VWAP scan error for {sym}: {exc}")

        scored.sort(reverse=True)
        for score, sym, params in scored:
            if vwap_count >= self.cfg.vwap_max_positions:
                break
            price   = params["entry"]
            sl_dist = abs(price - params["sl"])
            if sl_dist > 0:
                risk_dollars = portfolio_value * self.cfg.vwap_risk_per_trade_pct
                max_qty      = int((portfolio_value * self.cfg.vwap_max_notional_pct) / price)
                qty          = min(int(risk_dollars / sl_dist), max_qty)
            else:
                qty = int((portfolio_value * self.cfg.vwap_max_notional_pct) / price)
            if qty < 1:
                continue
            notional = qty * price
            ok, reason = self.rm.check_ok_to_enter(
                sym, self._positions, notional, portfolio_value
            )
            if not ok:
                continue
            self._submit_and_track(
                sym, qty, params["direction"],
                params["entry"], params["tp"], params["sl"], "vwap", price
            )
            vwap_count += 1

    # =========================================================================
    # PHASE 3: POWER HOUR
    # =========================================================================

    def _phase_power(self, portfolio_value: float) -> None:
        """Power Hour: 3:05 – 3:50 PM."""
        entry_cutoff = _parse_time_today(self.cfg.power_entry_cutoff)
        hard_close   = _parse_time_today(self.cfg.power_hard_close)
        last_scan    = 0.0
        last_risk    = time.time()

        while datetime.datetime.now(ET) < hard_close:
            now = datetime.datetime.now(ET)

            if time.time() - last_risk > 120:  # check risk every 2 min in power hour
                prices = self._fetch_current_prices()
                pv     = self._estimate_portfolio_value(portfolio_value, prices)
                status = self.rm.intraday_monitor(self._positions, prices, pv)
                if status["action"] == "emergency_close":
                    self.rm.emergency_close_all(self._positions, status["reason"])
                    return
                last_risk = time.time()

            self._shared_monitor()

            if (now < entry_cutoff and
                    time.time() - last_scan > 180):  # scan every 3 min
                self._scan_power_entries(portfolio_value)
                last_scan = time.time()

            time.sleep(self.cfg.monitor_interval_sec)

        # Hard close all power-hour and any other remaining positions
        for pos in [p for p in self._positions if not p.get("closed", False)]:
            self._close_live_position(pos, reason="power_hard_close")

    def _scan_power_entries(self, portfolio_value: float) -> None:
        """Find trend-confirmed, VWAP-pullback setups for power hour."""
        open_symbols = {p["symbol"] for p in self._positions if not p.get("closed", False)}
        power_count  = sum(
            1 for p in self._positions
            if not p.get("closed", False) and p.get("phase") == "power"
        )
        if power_count >= self.cfg.power_max_positions:
            return

        candidates = [s for s in self._universe if s not in open_symbols][:40]
        if not candidates:
            return

        try:
            today    = datetime.date.today()
            start_dt = datetime.datetime.combine(today, datetime.time(9, 30), tzinfo=ET)
            resp     = self.dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols = candidates,
                timeframe         = TimeFrame.Minute,
                start             = start_dt,
                end               = datetime.datetime.now(ET),
                feed              = DataFeed.IEX,
            ))
        except Exception as exc:
            self.log.warning(f"Power bar fetch failed: {exc}")
            return

        scored: list[tuple[float, str, dict]] = []

        for sym in candidates:
            try:
                df = _extract_symbol_df(resp, sym)
                if df is None or len(df) < self.cfg.power_trend_bars + 10:
                    continue

                vwap_series = calculate_vwap(df)
                atr_series  = calculate_atr(df, period=self.cfg.vwap_atr_period)
                ema20       = calculate_ema(df, period=20)
                sma50       = calculate_sma(df, period=50)

                price       = float(df["close"].iloc[-1])
                vwap_now    = float(vwap_series.iloc[-1])
                atr_now     = float(atr_series.iloc[-1])
                ema_now     = float(ema20.iloc[-1])
                sma_now     = float(sma50.iloc[-1])

                if atr_now <= 0:
                    continue

                # Trend check: historical side of VWAP AND active structural trend
                lookback   = min(self.cfg.power_trend_bars, len(df))
                closes     = df["close"].iloc[-lookback:]
                vwaps      = vwap_series.iloc[-lookback:]
                above_frac = float((closes > vwaps).mean())

                if above_frac >= self.cfg.power_trend_threshold and ema_now > sma_now:
                    direction    = "long"
                    trend_score  = above_frac
                elif (1 - above_frac) >= self.cfg.power_trend_threshold and ema_now < sma_now:
                    direction    = "short"
                    trend_score  = 1 - above_frac
                else:
                    continue  # no consistent trend

                # VWAP pullback entry: price near VWAP
                dist_from_vwap = abs(price - vwap_now) / atr_now
                if dist_from_vwap > self.cfg.power_vwap_entry_atr:
                    continue

                entry = price
                if direction == "long":
                    tp = entry * (1 + self.cfg.power_tp_pct)
                    sl = entry * (1 - self.cfg.power_sl_pct)
                else:
                    tp = entry * (1 - self.cfg.power_tp_pct)
                    sl = entry * (1 + self.cfg.power_sl_pct)

                scored.append((trend_score, sym, {
                    "direction": direction, "entry": entry, "tp": tp, "sl": sl
                }))

            except Exception as exc:
                self.log.debug(f"Power scan error for {sym}: {exc}")

        scored.sort(reverse=True)
        for _, sym, params in scored:
            if power_count >= self.cfg.power_max_positions:
                break
            budget = portfolio_value * self.cfg.power_position_size_pct
            qty    = int(budget // params["entry"])
            if qty < 1:
                continue
            ok, reason = self.rm.check_ok_to_enter(
                sym, self._positions, budget, portfolio_value
            )
            if not ok:
                continue
            self._submit_and_track(
                sym, qty, params["direction"],
                params["entry"], params["tp"], params["sl"], "power", params["entry"]
            )
            power_count += 1

    # =========================================================================
    # SHARED POSITION MANAGEMENT
    # =========================================================================

    def _shared_monitor(self) -> None:
        """Check TP/SL for every open position regardless of phase."""
        open_positions = [p for p in self._positions if not p.get("closed", False)]
        if not open_positions:
            return

        symbols = [p["symbol"] for p in open_positions]
        try:
            trades  = self.dc.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbols)
            )
        except Exception as exc:
            self.log.debug(f"Price fetch in monitor failed: {exc}")
            return

        for pos in open_positions:
            sym   = pos["symbol"]
            price = trades[sym].price if sym in trades else pos["entry_price"]

            if pos["direction"] == "long":
                if price <= pos["sl_price"]:
                    self._close_live_position(pos, reason="stop_loss", price=price)
                elif price >= pos["tp_price"]:
                    self._close_live_position(pos, reason="take_profit", price=price)
            else:  # short
                if price >= pos["sl_price"]:
                    self._close_live_position(pos, reason="stop_loss", price=price)
                elif price <= pos["tp_price"]:
                    self._close_live_position(pos, reason="take_profit", price=price)

            # VWAP-specific: max hold time
            if pos.get("phase") == "vwap" and "entry_time" in pos:
                held = (datetime.datetime.now(ET) - pos["entry_time"]).seconds / 60
                if held >= self.cfg.vwap_max_hold_bars:
                    self._close_live_position(pos, reason="vwap_max_hold", price=price)

    def _submit_and_track(
        self,
        symbol: str,
        qty: int,
        direction: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        phase: str,
        current_price: float,
    ) -> None:
        """Submit a market order and add to the shared position tracker."""
        side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        try:
            order = self.tc.submit_order(MarketOrderRequest(
                symbol        = symbol,
                qty           = qty,
                side          = side,
                time_in_force = TimeInForce.DAY,
            ))
            self._positions.append({
                "symbol":      symbol,
                "order_id":    order.id,
                "qty":         qty,
                "direction":   direction,
                "entry_price": current_price,
                "tp_price":    tp_price,
                "sl_price":    sl_price,
                "phase":       phase,
                "entry_time":  datetime.datetime.now(ET),
                "closed":      False,
                "exit_price":  None,
                "exit_reason": None,
                "pnl":         None,
            })
            self.log.info(
                f"OPEN [{phase.upper():5}] {direction:5} {qty:>4}× {symbol:<6} "
                f"@ ${current_price:.2f} | TP ${tp_price:.2f} | SL ${sl_price:.2f}"
            )
        except Exception as exc:
            self.log.error(f"Order failed for {symbol}: {exc}")

    def _close_live_position(
        self, pos: dict, reason: str, price: Optional[float] = None
    ) -> None:
        try:
            self.tc.close_position(pos["symbol"])
        except Exception as exc:
            self.log.error(f"Close failed for {pos['symbol']}: {exc}")
        exit_price = price or pos["entry_price"]
        if pos["direction"] == "long":
            pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["qty"]
        pos.update(closed=True, exit_price=exit_price, exit_reason=reason, pnl=pnl)
        self.log.info(
            f"CLOSE [{pos['phase'].upper():5}] {pos['symbol']:<6} "
            f"@ ${exit_price:.2f} | {reason:<22} | P&L ${pnl:>+8.2f}"
        )

    def _fetch_current_prices(self) -> dict[str, float]:
        syms = [p["symbol"] for p in self._positions if not p.get("closed", False)]
        if not syms:
            return {}
        try:
            trades = self.dc.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=syms)
            )
            return {s: t.price for s, t in trades.items()}
        except Exception:
            return {}

    def _estimate_portfolio_value(
        self, base_value: float, prices: dict[str, float]
    ) -> float:
        pnl = 0.0
        for p in self._positions:
            if not p.get("closed", False) and p["symbol"] in prices:
                px = prices[p["symbol"]]
                if p["direction"] == "long":
                    pnl += (px - p["entry_price"]) * p["qty"]
                else:
                    pnl += (p["entry_price"] - px) * p["qty"]
        return base_value + pnl

    # =========================================================================
    # BACKTESTING
    # =========================================================================

    def _load_minute_store(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """Load 1-minute bars for all tickers into a dict. Separated so the
        optimizer can call this once and reuse the result across grid runs."""
        start_dt = datetime.datetime.fromisoformat(start_date).replace(tzinfo=ET)
        end_dt   = datetime.datetime.fromisoformat(end_date).replace(tzinfo=ET)
        minute_store: dict[str, pd.DataFrame] = {}
        fetch_tickers = list(dict.fromkeys(["SPY"] + list(tickers)))
        from data_collection.data_cache import LocalDataCache
        cache = LocalDataCache(self.dc)
        for sym in fetch_tickers:
            try:
                df = cache.get_bars_df(sym, TimeFrame.Minute, start_dt, end_dt,
                                       feed=self.cfg.backtest_feed)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index).tz_convert(ET)
                    minute_store[sym] = df
                    self.log.info(f"  {sym}: {len(df)} bars loaded.")
                else:
                    self.log.warning(f"  {sym}: no data returned.")
            except Exception as exc:
                self.log.warning(f"  {sym}: fetch/cache failed — {exc}")
        return minute_store

    def backtest(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        historical_universes: dict = None,
        _minute_store: dict = None,
    ) -> dict:
        """
        Simulate all three phases on historical 1-minute data.
        Pass _minute_store to skip data loading (used by the grid optimizer).
        """
        self.log.info(
            f"Backtest: {start_date} -> {end_date} | "
            f"{len(tickers)} tickers | ${initial_cash:,.0f}"
        )

        # ── Fetch all minute data per ticker ──────────────────────────────────
        if _minute_store is not None:
            minute_store = _minute_store
            self.log.debug(f"Using {len(minute_store)} pre-loaded tickers.")
        else:
            self.log.info("Fetching 1-minute bars (one call per ticker)...")
            minute_store = self._load_minute_store(tickers, start_date, end_date)

        if not minute_store:
            self.log.error("No minute data fetched. Aborting backtest.")
            return {}

        # ── Walk each trading day ─────────────────────────────────────────────
        cash            = initial_cash
        portfolio_value = initial_cash
        all_trades:      list[dict] = []
        all_candidates:  list[dict] = []
        equity_rows: list[dict] = []

        trading_days = pd.bdate_range(start_date, end_date)

        for day in trading_days:
            day_result = self._backtest_day(day, minute_store, portfolio_value, cash, historical_universes=historical_universes)
            cash             = day_result["ending_cash"]
            portfolio_value  = cash  # intraday-only, no overnight positions
            all_trades.extend(day_result["trades"])
            all_candidates.extend(day_result.get("candidates", []))
            equity_rows.append({
                "date":            day,
                "portfolio_value": portfolio_value,
                "day_pnl":         day_result["day_pnl"],
                "orb_pnl":         day_result.get("orb_pnl", 0),
                "vwap_pnl":        day_result.get("vwap_pnl", 0),
                "power_pnl":       day_result.get("power_pnl", 0),
                "num_trades":      len(day_result["trades"]),
            })
            if day_result["trades"]:
                self.log.info(
                    f"{day.strftime('%Y-%m-%d')}: {len(day_result['trades'])} trades | "
                    f"P&L ${day_result['day_pnl']:>+8.2f} | "
                    f"Portfolio ${portfolio_value:>10,.2f}"
                )

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        summary      = _compute_backtest_summary(all_trades, equity_curve, initial_cash)
        return {"equity_curve": equity_curve, "trades": all_trades,
                "candidates": all_candidates, "summary": summary}

    def _backtest_day(
            self,
            day: pd.Timestamp,
            minute_store: dict[str, pd.DataFrame],
            portfolio_value: float,
            cash: float,
            historical_universes: dict = None,
    ) -> dict:
        day_date = day.date()
        all_trades = []
        day_pnl = 0.0
        phase_pnl = {"orb": 0.0, "vwap": 0.0, "power": 0.0}

        # ── PIT universe selection ────────────────────────────────────────────
        reference_day = day - pd.Timedelta(days=7)
        year, week, _ = reference_day.isocalendar()
        week_str = f"{year}-W{week:02d}"

        if historical_universes:
            active_tickers = historical_universes.get(week_str, list(minute_store.keys()))
        else:
            active_tickers = list(minute_store.keys())

        # ── Build today's per-symbol DataFrames (MUST be at this indent level) ──
        day_data: dict[str, pd.DataFrame] = {}
        day_str = day.strftime('%Y-%m-%d')

        for sym, df in minute_store.items():
            if sym not in active_tickers and sym != "SPY":
                continue
            try:
                today_df = df.loc[day_str:day_str]
                if len(today_df) >= 30:
                    day_data[sym] = today_df.copy()
            except KeyError:
                continue

        if not day_data:
            return {
                "trades": [], "ending_cash": cash,
                "day_pnl": 0.0, "orb_pnl": 0.0, "vwap_pnl": 0.0, "power_pnl": 0.0,
            }

        # Notional budgets used only for the cash-sufficiency guard before each trade
        orb_max_notional   = portfolio_value * self.cfg.orb_max_notional_pct
        vwap_max_notional  = portfolio_value * self.cfg.vwap_max_notional_pct
        power_max_notional = portfolio_value * self.cfg.power_max_notional_pct

        # Tracks every position opened today so the risk gate can enforce the
        # same total / sector / deployment caps the live strategy uses.
        opened_today: list[dict] = []

        # ── Compute SPY state for this day ────────────────────────────────────
        spy_above_vwap: Optional[bool] = None   # static snapshot used by ranking
        spy_above_vwap_by_ts = None              # per-bar series used by _sim_vwap
        spy_orb_return: Optional[float] = None  # first-30-bar return for regime filter
        spy_day_return: Optional[float] = None  # open-to-close return stored on each trade
        if "SPY" in day_data:
            try:
                spy_df   = day_data["SPY"]
                spy_vwap = calculate_vwap(spy_df)
                spy_above_vwap_by_ts = spy_df["close"] > spy_vwap
                idx = min(150, len(spy_df) - 1)
                spy_above_vwap = bool(spy_above_vwap_by_ts.iloc[idx])
                # SPY return from open to bar 30 (~10:00 AM) for ORB regime gate
                if len(spy_df) >= 31:
                    spy_open_px = float(spy_df["open"].iloc[0])
                    if spy_open_px > 0:
                        spy_orb_return = (float(spy_df["close"].iloc[30]) - spy_open_px) / spy_open_px
                # Full-day SPY return for trade-level context in the database
                if len(spy_df) >= 2:
                    spy_open_full = float(spy_df["open"].iloc[0])
                    if spy_open_full > 0:
                        spy_day_return = (float(spy_df["close"].iloc[-1]) - spy_open_full) / spy_open_full
            except Exception:
                spy_above_vwap = None
                spy_above_vwap_by_ts = None
                spy_orb_return = None
                spy_day_return = None

        # ── SPY regime gate ───────────────────────────────────────────────────
        spy_regime = "flat"
        if spy_orb_return is not None:
            if spy_orb_return > 0.010:
                spy_regime = "bull"
            elif spy_orb_return < -0.005:
                spy_regime = "bear"

        # ── Phase 1: ORB (bull days only) ─────────────────────────────────────
        orb_candidates = self._rank_orb_candidates(day_data, catalyst_symbols=None)
        orb_filled = 0

        if spy_regime == "bull":
            for sym in orb_candidates:
                if orb_filled >= self.cfg.orb_max_positions:
                    break
                if cash < orb_max_notional:
                    break
                if day_pnl / portfolio_value <= -self.cfg.daily_dd_halt_pct:
                    break
                if not self._risk_ok(sym, opened_today, orb_max_notional, portfolio_value):
                    continue
                trade = self._sim_orb(sym, day_data[sym], portfolio_value,
                                       spy_orb_return=spy_orb_return)
                if trade:
                    trade["date"] = day_date
                    trade["spy_day_return"] = spy_day_return
                    cash   += trade["pnl"]
                    day_pnl += trade["pnl"]
                    phase_pnl["orb"] += trade["pnl"]
                    all_trades.append(trade)
                    opened_today.append({"symbol": sym, "qty": trade["qty"],
                                         "entry_price": trade["entry_price"], "closed": False})
                    orb_filled += 1

        # ── Phase 2: VWAP Reversion ───────────────────────────────────────────
        vwap_candidates = self._rank_vwap_candidates(day_data, spy_above_vwap=spy_above_vwap)
        vwap_filled = 0
        day_candidates: list[dict] = []

        for rank_pos, (score, sym, sig) in enumerate(vwap_candidates, start=1):
            cand_base = {"symbol": sym, "date": day_date, "rank_score": score,
                         "rank_position": rank_pos, "spy_regime": spy_regime, **sig}
            halt = (vwap_filled >= self.cfg.vwap_max_positions
                    or cash < vwap_max_notional
                    or day_pnl / portfolio_value <= -self.cfg.daily_dd_halt_pct)
            if halt:
                day_candidates.append({**cand_base, "was_traded": 0})
                continue
            if not self._risk_ok(sym, opened_today, vwap_max_notional, portfolio_value):
                day_candidates.append({**cand_base, "was_traded": 0})
                continue
            trade = self._sim_vwap(sym, day_data[sym], portfolio_value,
                                    spy_above_vwap_by_ts=spy_above_vwap_by_ts,
                                    spy_open_return=spy_orb_return,
                                    spy_regime=spy_regime)
            if trade:
                trade["date"] = day_date
                trade["spy_day_return"] = spy_day_return
                cash    += trade["pnl"]
                day_pnl += trade["pnl"]
                phase_pnl["vwap"] += trade["pnl"]
                all_trades.append(trade)
                opened_today.append({"symbol": sym, "qty": trade["qty"],
                                     "entry_price": trade["entry_price"], "closed": False})
                vwap_filled += 1
            day_candidates.append({**cand_base, "was_traded": 1 if trade else 0})

        # ── Phase 3: Power Hour ───────────────────────────────────────────────
        power_candidates = self._rank_power_candidates(day_data)
        power_filled = 0

        # Power Hour temporarily disabled for comparison — re-enable by removing this guard
        if False:
            for sym in power_candidates:
                if power_filled >= self.cfg.power_max_positions:
                    break
                if cash < power_max_notional:
                    break
                if day_pnl / portfolio_value <= -self.cfg.daily_dd_halt_pct:
                    break
                if not self._risk_ok(sym, opened_today, power_max_notional, portfolio_value):
                    continue
                trade = self._sim_power(sym, day_data[sym], portfolio_value)
                if trade:
                    trade["date"] = day_date
                    trade["spy_day_return"] = spy_day_return
                    cash    += trade["pnl"]
                    day_pnl += trade["pnl"]
                    phase_pnl["power"] += trade["pnl"]
                    all_trades.append(trade)
                    opened_today.append({"symbol": sym, "qty": trade["qty"],
                                         "entry_price": trade["entry_price"], "closed": False})
                    power_filled += 1

        return {
            "trades":      all_trades,
            "candidates":  day_candidates,
            "ending_cash": cash,
            "day_pnl":     round(day_pnl, 2),
            "orb_pnl":     round(phase_pnl["orb"], 2),
            "vwap_pnl":    round(phase_pnl["vwap"], 2),
            "power_pnl":   round(phase_pnl["power"], 2),
        }

    # ── Backtest risk gate ────────────────────────────────────────────────────

    def _risk_ok(
        self,
        symbol: str,
        opened_today: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Apply the live risk-manager concentration limits inside the backtest.

        Without this, the backtest ignored the per-sector / total-position /
        deployment caps that gate every live entry, so the two systems traded
        differently. We reuse the same RiskManager thresholds here. The sector
        cap is skipped when the sector is 'Unknown' so an empty/partial sector
        map cannot silently throttle the whole book down to one bucket.
        """
        if self.rm is None:
            return True

        cfg = self.rm.cfg
        open_positions = [p for p in opened_today if not p.get("closed", False)]

        if len(open_positions) >= cfg.max_total_positions:
            return False

        total_notional = sum(p["qty"] * p["entry_price"] for p in open_positions)
        if (total_notional + proposed_notional) / portfolio_value > cfg.max_portfolio_deployed_pct:
            return False

        sector = self.rm.get_sector(symbol)
        if sector != "Unknown":
            sector_count = sum(
                1 for p in open_positions
                if self.rm.get_sector(p["symbol"]) == sector
            )
            if sector_count >= cfg.max_positions_per_sector:
                return False

        return True

    # ── Backtest candidate ranking ────────────────────────────────────────────

    def _rank_orb_candidates(
        self,
        day_data: dict[str, pd.DataFrame],
        catalyst_symbols: Optional[set] = None,
    ) -> list[str]:
        scores: list[tuple[float, str]] = []
        for sym, df in day_data.items():
            if (self.cfg.orb_require_catalyst
                    and catalyst_symbols is not None
                    and sym not in catalyst_symbols):
                continue

            if len(df) < self.cfg.orb_range_bars + 1:
                continue
            range_bars = df.iloc[: self.cfg.orb_range_bars]
            r_high     = float(range_bars["high"].max())
            r_low      = float(range_bars["low"].min())
            price      = float(df["close"].iloc[self.cfg.orb_range_bars])
            r_pct      = (r_high - r_low) / price if price > 0 else 0
            if not (self.cfg.orb_min_range_pct <= r_pct <= self.cfg.orb_max_range_pct):
                continue

            # Use bars AFTER the opening range to get a clean non-spike baseline.
            # Bars 15-90 (~9:45-10:30 AM) represent settled volume, not the open spike.
            post_range_bars = df.iloc[self.cfg.orb_range_bars:90]
            if len(post_range_bars) >= 10:
                hist_avg_vol = float(post_range_bars["volume"].mean())
            else:
                hist_avg_vol = float(df["volume"].mean())
            rvol = float(range_bars["volume"].mean()) / max(hist_avg_vol, 1)
            if rvol < self.cfg.orb_min_rvol:
                continue
            scores.append((rvol * r_pct, sym))
        scores.sort(reverse=True)
        return [s for _, s in scores]

    def _rank_vwap_candidates(
        self,
        day_data: dict[str, pd.DataFrame],
        spy_above_vwap: Optional[bool] = None,
    ) -> list[tuple[float, str, dict]]:
        scores: list[tuple[float, str, dict]] = []
        dead_zone_start  = self.cfg.vwap_dead_zone_start
        dead_zone_end    = self.cfg.vwap_dead_zone_end

        for sym, df in day_data.items():
            entry_start_idx = self._bar_idx_at(df, self.cfg.vwap_entry_start)
            entry_end_idx   = self._bar_idx_at(df, self.cfg.vwap_entry_cutoff)
            if len(df) < entry_start_idx + 15:
                continue
            scan_df     = df.iloc[: entry_end_idx]
            vwap_series = calculate_vwap(scan_df)
            atr_series  = calculate_atr(scan_df, period=self.cfg.vwap_atr_period)
            for i in range(entry_start_idx, min(entry_end_idx, len(scan_df))):
                if dead_zone_start <= i < dead_zone_end:
                    continue

                vwap_val = float(vwap_series.iloc[i])
                atr_val  = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0
                price    = float(scan_df["close"].iloc[i])
                if atr_val <= 0 or atr_val < self.cfg.vwap_min_atr:
                    continue
                dist = abs(price - vwap_val) / atr_val
                if dist >= self.cfg.vwap_extension_atr:
                    recent_vol = float(scan_df["volume"].iloc[max(0, i - 5):i].mean())
                    day_avg_vol = float(scan_df["volume"].iloc[:i].mean())
                    if day_avg_vol > 0 and recent_vol / day_avg_vol < self.cfg.vwap_vol_decay:

                        # NEW: require price to have stalled — no new extreme in last 5 bars
                        if price > vwap_val:  # extended above, looking to short
                            recent_high = float(scan_df["high"].iloc[max(0, i - 5):i].max())
                            if float(scan_df["high"].iloc[i]) > recent_high:
                                continue  # still making new highs, not exhausted yet
                        else:  # extended below, looking to long
                            recent_low = float(scan_df["low"].iloc[max(0, i - 5):i].min())
                            if float(scan_df["low"].iloc[i]) < recent_low:
                                continue  # still making new lows, not exhausted yet

                        direction_would_be = "short" if price > vwap_val else "long"

                        if (self.cfg.vwap_spy_alignment
                                and spy_above_vwap is not None
                                and direction_would_be == "long"
                                and not spy_above_vwap):
                            continue
                        sig_info = {
                            "signal_time":      str(scan_df.index[i]),
                            "direction":        direction_would_be,
                            "bar_close":        price,
                            "vwap_val":         vwap_val,
                            "atr_val":          atr_val,
                            "dist_atr":         dist,
                            "vol_decay_ratio":  recent_vol / day_avg_vol if day_avg_vol > 0 else None,
                            "signal_vol_ratio": float(scan_df["volume"].iloc[i]) / day_avg_vol if day_avg_vol > 0 else None,
                        }
                        scores.append((dist, sym, sig_info))
                        break
        scores.sort()  # ascending: lowest extension first (2.5-4.0 ATR sweet spot gets priority)
        return scores

    def _rank_power_candidates(self, day_data: dict[str, pd.DataFrame]) -> list[str]:
        scores: list[tuple[float, str]] = []
        for sym, df in day_data.items():
            power_start_idx = self._bar_idx_at(df, self.cfg.power_entry_start)
            if power_start_idx >= len(df) - 5:
                continue
            vwap_series = calculate_vwap(df)
            atr_series  = calculate_atr(df, period=self.cfg.vwap_atr_period)
            ema20       = calculate_ema(df, period=20)
            sma50       = calculate_sma(df, period=50)

            lookback    = min(self.cfg.power_trend_bars, power_start_idx)
            closes      = df["close"].iloc[power_start_idx - lookback: power_start_idx]
            vwaps       = vwap_series.iloc[power_start_idx - lookback: power_start_idx]
            above_frac  = float((closes > vwaps).mean())

            ema_now     = float(ema20.iloc[power_start_idx])
            sma_now     = float(sma50.iloc[power_start_idx])

            # Require both historical percentage and active EMA/SMA trend
            if above_frac >= self.cfg.power_trend_threshold and ema_now > sma_now:
                trend_score = above_frac
            elif (1 - above_frac) >= self.cfg.power_trend_threshold and ema_now < sma_now:
                trend_score = 1 - above_frac
            else:
                continue

            atr_now  = float(atr_series.iloc[power_start_idx])
            vwap_now = float(vwap_series.iloc[power_start_idx])
            price    = float(df["close"].iloc[power_start_idx])
            if atr_now > 0 and abs(price - vwap_now) / atr_now <= self.cfg.power_vwap_entry_atr:
                scores.append((trend_score, sym))
        scores.sort(reverse=True)
        return [s for _, s in scores]

    # ── Single-position simulators ────────────────────────────────────────────

    def _sim_orb(
            self, sym: str, df: pd.DataFrame, portfolio_value: float,
            spy_orb_return: Optional[float] = None,
    ) -> Optional[dict]:
        """Simulate one ORB trade with breakout margin, regime filter, and ATR-based sizing."""
        N = self.cfg.orb_range_bars
        if len(df) < N + 5:
            return None

        range_bars  = df.iloc[:N]
        r_high      = float(range_bars["high"].max())
        r_low       = float(range_bars["low"].min())
        r_size      = r_high - r_low
        r_avg_vol   = float(range_bars["volume"].mean())
        slip        = self.cfg.slippage_pct + self.cfg.spread_pct

        entry_cutoff_bar = 30   # ~10:00 AM
        phase_end_bar    = 60   # ~10:30 AM

        # ── Pass 1: find signal bar and compute fill price ────────────────────
        entry_price: Optional[float] = None
        direction:   Optional[str]   = None
        tp_price = sl_price = 0.0
        fill_start_bar = 0

        post_range_bars = df.iloc[N:min(90, len(df))]
        neutral_avg_vol = float(post_range_bars["volume"].mean()) if len(post_range_bars) >= 5 \
            else float(r_avg_vol)

        # Fix 7: breakout close must clear the range boundary by a minimum margin
        long_threshold  = r_high * (1 + self.cfg.orb_breakout_margin_pct)
        short_threshold = r_low  * (1 - self.cfg.orb_breakout_margin_pct)
        min_entry_bar   = self._bar_idx_at(df, self.cfg.orb_min_entry_time)

        for i in range(N, min(len(df), entry_cutoff_bar + 1)):
            if i < min_entry_bar:
                continue
            row = df.iloc[i]
            bar_close = float(row["close"])
            bar_vol   = float(row["volume"])
            vol_ok    = bar_vol > neutral_avg_vol * self.cfg.orb_entry_vol_mult

            long_signal  = bar_close > long_threshold  and vol_ok
            short_signal = bar_close < short_threshold and vol_ok

            if not long_signal and not short_signal:
                continue

            direction = "long" if long_signal else "short"

            # Fix 8: skip when broad market trend is strongly against the trade
            if self.cfg.orb_regime_filter and spy_orb_return is not None:
                if direction == "long" and spy_orb_return < self.cfg.orb_regime_long_skip_threshold:
                    return None
                if direction == "short" and spy_orb_return > self.cfg.orb_regime_short_skip_threshold:
                    return None

            if self.cfg.use_next_bar_fill:
                fill_bar_idx = i + 1
                if fill_bar_idx >= len(df):
                    return None
                raw_fill = float(df.iloc[fill_bar_idx]["open"])
            else:
                fill_bar_idx = i
                raw_fill = r_high if direction == "long" else r_low

            entry_price = raw_fill * (1 + slip) if direction == "long" \
                          else raw_fill * (1 - slip)
            tp_price = entry_price + r_size * self.cfg.orb_tp_mult if direction == "long" \
                       else entry_price - r_size * self.cfg.orb_tp_mult
            sl_price = entry_price - r_size * self.cfg.orb_sl_mult if direction == "long" \
                       else entry_price + r_size * self.cfg.orb_sl_mult

            fill_start_bar = fill_bar_idx
            break

        if entry_price is None:
            return None

        # Fix 9: ATR-based qty — risk a fixed % of portfolio, cap by max notional
        sl_per_share = abs(entry_price - sl_price)
        if sl_per_share > 0:
            risk_dollars = portfolio_value * self.cfg.orb_risk_per_trade_pct
            max_qty      = int((portfolio_value * self.cfg.orb_max_notional_pct) / entry_price)
            qty          = min(int(risk_dollars / sl_per_share), max_qty)
        else:
            qty = int((portfolio_value * self.cfg.orb_max_notional_pct) / entry_price)
        if qty < 1:
            return None

        # ── Pass 2: manage the open position ─────────────────────────────────
        elapsed = 0
        for i in range(fill_start_bar, len(df)):
            row       = df.iloc[i]
            bar_high  = float(row["high"])
            bar_low   = float(row["low"])
            bar_close = float(row["close"])
            elapsed  += 1

            unrealized = (bar_close - entry_price) if direction == "long" \
                else (entry_price - bar_close)
            breakeven_threshold = r_size * 0.5

            if unrealized >= breakeven_threshold:
                if direction == "long":
                    sl_price = max(sl_price, entry_price)
                else:
                    sl_price = min(sl_price, entry_price)

            if i >= phase_end_bar:
                exit_price  = bar_close * (1 - slip) if direction == "long" \
                              else bar_close * (1 + slip)
                exit_reason = "time_stop"
            elif direction == "long":
                if bar_low <= sl_price:
                    exit_price  = sl_price * (1 - slip)
                    exit_reason = "stop_loss"
                elif bar_high >= tp_price:
                    exit_price  = tp_price * (1 - slip)
                    exit_reason = "take_profit"
                else:
                    continue
            else:
                if bar_high >= sl_price:
                    exit_price  = sl_price * (1 + slip)
                    exit_reason = "stop_loss"
                elif bar_low <= tp_price:
                    exit_price  = tp_price * (1 + slip)
                    exit_reason = "take_profit"
                else:
                    continue

            pnl = (exit_price - entry_price) * qty if direction == "long" \
                  else (entry_price - exit_price) * qty
            pnl -= self.cfg.commission_per_trade
            return {
                "symbol": sym, "phase": "orb", "direction": direction,
                "entry_price": round(entry_price, 4), "exit_price": round(exit_price, 4),
                "tp_price": round(tp_price, 4), "sl_price": round(sl_price, 4),
                "qty": qty, "pnl": round(pnl, 2),
                "exit_reason": exit_reason, "bars_held": elapsed,
                "entry_time": str(df.index[fill_bar_idx]),
                "exit_time":  str(df.index[i]),
            }
        return None

    def _sim_vwap(
        self, sym: str, df: pd.DataFrame, portfolio_value: float,
        spy_above_vwap_by_ts=None,
        spy_open_return: Optional[float] = None,
        spy_regime: str = "flat",
    ) -> Optional[dict]:
        """Simulate one VWAP reversion trade with corrected fill tracking."""
        entry_start = self._bar_idx_at(df, self.cfg.vwap_entry_start)
        entry_end   = self._bar_idx_at(df, self.cfg.vwap_entry_cutoff)
        phase_close = self._bar_idx_at(df, self.cfg.vwap_phase_end)
        slip        = self.cfg.slippage_pct + self.cfg.spread_pct

        vwap_s = calculate_vwap(df)
        atr_s  = calculate_atr(df, period=self.cfg.vwap_atr_period)

        entry_price: Optional[float] = None
        direction:   Optional[str]   = None
        tp_price = sl_price = 0.0
        entry_bar = 0
        entry_atr = 0.0
        entry_vwap       = 0.0
        entry_dist_atr   = 0.0
        entry_vol_decay: Optional[float]  = None
        entry_signal_vol: Optional[float] = None

        for i in range(entry_start, min(len(df), phase_close)):
            row       = df.iloc[i]
            bar_high  = float(row["high"])
            bar_low   = float(row["low"])
            bar_close = float(row["close"])
            vwap_val  = float(vwap_s.iloc[i])
            atr_val   = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 0

            if atr_val <= 0 or atr_val < self.cfg.vwap_min_atr:
                continue

            if entry_price is None:
                if i >= entry_end:
                    return None
                if self.cfg.vwap_dead_zone_start <= i < self.cfg.vwap_dead_zone_end:
                    continue

                dist = abs(bar_close - vwap_val) / atr_val
                if dist < self.cfg.vwap_extension_atr:
                    continue
                recent_vol  = float(df["volume"].iloc[max(0, i - 5):i].mean())
                day_avg_vol = float(df["volume"].iloc[:i].mean())
                if day_avg_vol > 0 and recent_vol / day_avg_vol >= self.cfg.vwap_vol_decay:
                    continue

                # Skip if the signal bar itself has an abnormally high volume spike —
                # a volume explosion at the extreme often signals trend continuation,
                # not exhaustion. A true mean-reversion entry is quiet/decelerating.
                signal_bar_vol = float(df["volume"].iloc[i])
                if day_avg_vol > 0 and signal_bar_vol > day_avg_vol * self.cfg.vwap_signal_vol_max:
                    continue

                # Stall check: price must not be making a new extreme in last 5 bars.
                # Matches the exhaustion gate in _rank_vwap_candidates.
                direction_candidate = "short" if bar_close > vwap_val else "long"
                if i >= 5:
                    if direction_candidate == "short":
                        if float(df["high"].iloc[i]) > float(df["high"].iloc[max(0, i - 5):i].max()):
                            continue
                    else:
                        if float(df["low"].iloc[i]) < float(df["low"].iloc[max(0, i - 5):i].min()):
                            continue

                # Per-bar SPY alignment: only take longs when SPY is above its own VWAP
                # at this exact bar, not a stale noon snapshot.
                if (self.cfg.vwap_spy_alignment
                        and spy_above_vwap_by_ts is not None
                        and direction_candidate == "long"):
                    bar_ts = df.index[i]
                    spy_before = spy_above_vwap_by_ts[spy_above_vwap_by_ts.index <= bar_ts]
                    if not spy_before.empty and not bool(spy_before.iloc[-1]):
                        continue

                # Block VWAP shorts except in bear regimes (down-day win rate 38-50%
                # vs 9-17% on bull/flat days). Flat also skips shorts — only longs.
                if direction_candidate == "short" and spy_regime in ("bull", "flat"):
                    continue

                direction = direction_candidate

                if self.cfg.use_next_bar_fill and i + 1 < len(df):
                    raw_fill  = float(df.iloc[i + 1]["open"])
                    entry_bar = i + 1
                else:
                    raw_fill  = bar_close
                    entry_bar = i

                entry_price = raw_fill * (1 + slip) if direction == "long" \
                              else raw_fill * (1 - slip)
                sl_dist  = atr_val * self.cfg.vwap_sl_atr
                sl_price = (entry_price + sl_dist) if direction == "short" \
                           else (entry_price - sl_dist)
                tp_ext    = self.cfg.vwap_tp_extension_atr * atr_val
                tp_price  = vwap_val + tp_ext if direction == "long" \
                            else vwap_val - tp_ext
                entry_atr        = atr_val
                entry_vwap       = vwap_val
                entry_dist_atr   = dist
                entry_vol_decay  = recent_vol / day_avg_vol if day_avg_vol > 0 else None
                entry_signal_vol = signal_bar_vol / day_avg_vol if day_avg_vol > 0 else None

                # Fix 9: ATR-based qty — risk a fixed % of portfolio, cap by max notional
                if sl_dist > 0:
                    risk_dollars = portfolio_value * self.cfg.vwap_risk_per_trade_pct
                    max_qty      = int((portfolio_value * self.cfg.vwap_max_notional_pct) / entry_price)
                    qty          = min(int(risk_dollars / sl_dist), max_qty)
                else:
                    qty = int((portfolio_value * self.cfg.vwap_max_notional_pct) / entry_price)
                if qty < 1:
                    return None

            else:
                # FIXED: evaluates the bar we actually got filled on
                if i < entry_bar:
                    continue

                held = i - entry_bar

                if held >= self.cfg.vwap_max_hold_bars or i >= phase_close:
                    exit_p = bar_close * (1 - slip) if direction == "long" \
                             else bar_close * (1 + slip)
                    pnl = (exit_p - entry_price) * qty if direction == "long" \
                          else (entry_price - exit_p) * qty
                    pnl -= self.cfg.commission_per_trade
                    return {
                        "symbol": sym, "phase": "vwap", "direction": direction,
                        "entry_price": round(entry_price, 4), "exit_price": round(exit_p, 4),
                        "tp_price": round(tp_price, 4), "sl_price": round(sl_price, 4),
                        "qty": qty, "pnl": round(pnl, 2),
                        "exit_reason": "max_hold" if held >= self.cfg.vwap_max_hold_bars
                                       else "phase_end",
                        "bars_held": held,
                        "entry_time": str(df.index[entry_bar]),
                        "exit_time":  str(df.index[i]),
                        "vwap_at_entry":    round(entry_vwap, 4),
                        "atr_at_entry":     round(entry_atr, 6),
                        "dist_atr":         round(entry_dist_atr, 3),
                        "vol_decay_ratio":  round(entry_vol_decay, 4) if entry_vol_decay is not None else None,
                        "signal_vol_ratio": round(entry_signal_vol, 4) if entry_signal_vol is not None else None,
                        "spy_regime":       spy_regime,
                    }

                # Once price recovers vwap_breakeven_atr ATR toward VWAP, slide SL to entry
                if self.cfg.vwap_breakeven_atr > 0 and entry_atr > 0:
                    recovered = (bar_close - entry_price) if direction == "long" \
                                else (entry_price - bar_close)
                    if recovered >= self.cfg.vwap_breakeven_atr * entry_atr:
                        if direction == "long":
                            sl_price = max(sl_price, entry_price)
                        else:
                            sl_price = min(sl_price, entry_price)

                exit_p: Optional[float] = None
                reason: str = ""
                in_grace = held < self.cfg.vwap_sl_grace_bars
                if direction == "long":
                    if not in_grace and bar_low <= sl_price:
                        exit_p = sl_price * (1 - slip)
                        reason = "stop_loss"
                    elif bar_close >= tp_price * (1 - self.cfg.vwap_tp_pct):
                        exit_p = tp_price * (1 - slip)
                        reason = "take_profit"
                else:
                    if not in_grace and bar_high >= sl_price:
                        exit_p = sl_price * (1 + slip)
                        reason = "stop_loss"
                    elif bar_close <= tp_price * (1 + self.cfg.vwap_tp_pct):
                        exit_p = tp_price * (1 + slip)
                        reason = "take_profit"

                if exit_p is None:
                    continue

                pnl = (exit_p - entry_price) * qty if direction == "long" \
                      else (entry_price - exit_p) * qty
                pnl -= self.cfg.commission_per_trade
                return {
                    "symbol": sym, "phase": "vwap", "direction": direction,
                    "entry_price": round(entry_price, 4), "exit_price": round(exit_p, 4),
                    "tp_price": round(tp_price, 4), "sl_price": round(sl_price, 4),
                    "qty": qty, "pnl": round(pnl, 2),
                    "exit_reason": reason, "bars_held": i - entry_bar,
                    "entry_time": str(df.index[entry_bar]),
                    "exit_time":  str(df.index[i]),
                    "vwap_at_entry":    round(entry_vwap, 4),
                    "atr_at_entry":     round(entry_atr, 6),
                    "dist_atr":         round(entry_dist_atr, 3),
                    "vol_decay_ratio":  round(entry_vol_decay, 4) if entry_vol_decay is not None else None,
                    "signal_vol_ratio": round(entry_signal_vol, 4) if entry_signal_vol is not None else None,
                    "spy_regime":       spy_regime,
                }
        return None

    # Convert bar index constants to time-based index lookups
    def _bar_idx_at(self, df: pd.DataFrame, hhmm: str) -> int:
        """Return the integer position of the first bar at or after HH:MM ET."""
        h, m = map(int, hhmm.split(":"))
        target = datetime.time(h, m)
        matches = np.where(df.index.time >= target)[0]
        return int(matches[0]) if len(matches) > 0 else len(df)

    def _sim_power(
        self, sym: str, df: pd.DataFrame, portfolio_value: float,
    ) -> Optional[dict]:
        """Simulate one Power Hour trade with corrected fill tracking and structural trend logic."""
        entry_start = self._bar_idx_at(df, self.cfg.power_entry_start)  # "15:05"
        entry_end = self._bar_idx_at(df, self.cfg.power_entry_cutoff)  # "15:30"
        hard_close = self._bar_idx_at(df, self.cfg.power_hard_close)  # "15:50"
        slip = self.cfg.slippage_pct + self.cfg.spread_pct

        if entry_start >= len(df) -5:
            return None

        vwap_s = calculate_vwap(df)
        atr_s  = calculate_atr(df, period=self.cfg.vwap_atr_period)
        ema20  = calculate_ema(df, period=20)
        sma50  = calculate_sma(df, period=50)

        lookback   = min(self.cfg.power_trend_bars, entry_start)
        closes     = df["close"].iloc[entry_start - lookback: entry_start]
        vwaps      = vwap_s.iloc[entry_start - lookback: entry_start]
        above_frac = float((closes > vwaps).mean())

        ema_now    = float(ema20.iloc[entry_start])
        sma_now    = float(sma50.iloc[entry_start])

        if above_frac >= self.cfg.power_trend_threshold and ema_now > sma_now:
            direction = "long"
        elif (1 - above_frac) >= self.cfg.power_trend_threshold and ema_now < sma_now:
            direction = "short"
        else:
            return None

        entry_price: Optional[float] = None
        tp_price = sl_price = 0.0
        entry_bar = 0

        for i in range(entry_start, min(len(df), hard_close)):
            row      = df.iloc[i]
            price    = float(row["close"])
            vwap_val = float(vwap_s.iloc[i])
            atr_val  = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 0

            if entry_price is None:
                if i > entry_end:
                    return None
                if atr_val <= 0:
                    continue
                if abs(price - vwap_val) / atr_val > self.cfg.power_vwap_entry_atr:
                    continue

                if self.cfg.use_next_bar_fill and i + 1 < min(len(df), hard_close):
                    raw_fill  = float(df.iloc[i + 1]["open"])
                    entry_bar = i + 1
                else:
                    raw_fill  = price
                    entry_bar = i

                entry_price = raw_fill * (1 + slip) if direction == "long" \
                              else raw_fill * (1 - slip)
                tp_price = entry_price * (1 + self.cfg.power_tp_pct) if direction == "long" \
                           else entry_price * (1 - self.cfg.power_tp_pct)
                sl_price = entry_price * (1 - self.cfg.power_sl_pct) if direction == "long" \
                           else entry_price * (1 + self.cfg.power_sl_pct)

                # Fix 9: ATR-based qty — risk a fixed % of portfolio, cap by max notional
                sl_per_share = abs(entry_price - sl_price)
                if sl_per_share > 0:
                    risk_dollars = portfolio_value * self.cfg.power_risk_per_trade_pct
                    max_qty      = int((portfolio_value * self.cfg.power_max_notional_pct) / entry_price)
                    qty          = min(int(risk_dollars / sl_per_share), max_qty)
                else:
                    qty = int((portfolio_value * self.cfg.power_max_notional_pct) / entry_price)
                if qty < 1:
                    return None

            else:
                # FIXED: Evaluates SL/TP on the actual bar we were filled on
                if i < entry_bar:
                    continue
                bar_high = float(row["high"])
                bar_low  = float(row["low"])

                if i >= hard_close - 1:
                    exit_p = price * (1 - slip) if direction == "long" \
                             else price * (1 + slip)
                    reason = "hard_close"
                elif direction == "long":
                    if bar_low <= sl_price:
                        exit_p = sl_price * (1 - slip)
                        reason = "stop_loss"
                    elif bar_high >= tp_price:
                        exit_p = tp_price * (1 - slip)
                        reason = "take_profit"
                    else:
                        continue
                else:
                    if bar_high >= sl_price:
                        exit_p = sl_price * (1 + slip)
                        reason = "stop_loss"
                    elif bar_low <= tp_price:
                        exit_p = tp_price * (1 + slip)
                        reason = "take_profit"
                    else:
                        continue

                pnl = (exit_p - entry_price) * qty if direction == "long" \
                      else (entry_price - exit_p) * qty
                pnl -= self.cfg.commission_per_trade
                return {
                    "symbol": sym, "phase": "power", "direction": direction,
                    "entry_price": round(entry_price, 4), "exit_price": round(exit_p, 4),
                    "tp_price": round(tp_price, 4), "sl_price": round(sl_price, 4),
                    "qty": qty, "pnl": round(pnl, 2),
                    "exit_reason": reason, "bars_held": i - entry_bar,
                    "entry_time": str(df.index[entry_bar]),
                    "exit_time":  str(df.index[i]),
                }
        return None

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, results: dict) -> None:
        s = results.get("summary", {})
        if not s:
            print("[No summary data]")
            return
        print(f"\n{'='*58}")
        print(f"  Intraday Strategy Backtest Summary")
        print(f"{'='*58}")
        print(f"  Initial capital    : ${s['initial_cash']:>14,.2f}")
        print(f"  Final value        : ${s['final_value']:>14,.2f}")
        print(f"  Total return       : {s['total_return_pct']:>+13.2f}%")
        print(f"  Max drawdown       : {s['max_drawdown_pct']:>+13.2f}%")
        print(f"  Total trades       : {s['num_trades']:>14}")
        print(f"  Win rate           : {s['win_rate_pct']:>13.1f}%")
        print(f"  Reward / risk      : {s['reward_risk']:>14.2f}")
        print(f"  Profit factor      : {s.get('profit_factor', 0):>14.2f}")
        print(f"  Expectancy / trade : ${s.get('expectancy', 0):>13.2f}")
        print(f"  Sharpe (ann.)      : {s.get('sharpe', 0):>14.2f}")
        print(f"{'-'*58}")
        print(f"  Phase breakdown:")
        for phase, stats in s.get("by_phase", {}).items():
            print(f"    {phase.upper():<6}: {stats['trades']:>4} trades | "
                  f"P&L ${stats['pnl']:>+10.2f} | "
                  f"Win {stats['win_rate']*100:>5.1f}%")
        print(f"{'='*58}\n")

    def plot_backtest(self, results: dict) -> None:
        """Four-panel backtest report for the intraday strategy."""
        equity = results.get("equity_curve")
        trades = results.get("trades", [])
        s      = results.get("summary", {})

        if equity is None or not trades:
            print("No data to plot.")
            return

        fig = plt.figure(figsize=(15, 11))
        gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

        # ── 1. Equity curve ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        pv  = equity["portfolio_value"]
        ic  = s.get("initial_cash", float(pv.iloc[0]))
        ax1.plot(equity.index, pv, color="#1f77b4", lw=1.5, label="Portfolio")
        ax1.axhline(ic, color="gray", lw=0.8, ls="--", label="Starting capital")
        ax1.fill_between(equity.index, ic, pv,
                         where=(pv >= ic), alpha=0.12, color="#2ca02c")
        ax1.fill_between(equity.index, ic, pv,
                         where=(pv  < ic), alpha=0.12, color="#d62728")
        ax1.set_title(
            f"Intraday Strategy — Equity Curve\n"
            f"Return {s.get('total_return_pct',0):+.1f}% | "
            f"{s.get('num_trades',0)} trades | "
            f"Win {s.get('win_rate_pct',0):.0f}% | "
            f"Max DD {s.get('max_drawdown_pct',0):.1f}%",
            fontsize=11,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # ── 2. P&L by phase ────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        for phase, col in [("orb", "#1f77b4"), ("vwap", "#ff7f0e"), ("power", "#2ca02c")]:
            col_name = f"{phase}_pnl"
            if col_name in equity.columns:
                ax2.bar(equity.index, equity[col_name], label=phase.upper(),
                        color=col, alpha=0.7, width=0.8)
        ax2.axhline(0, color="gray", lw=0.7)
        ax2.set_title("Daily P&L by Phase")
        ax2.set_ylabel("P&L ($)")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        # ── 3. Trade P&L distribution ──────────────────────────────────────────
        ax3  = fig.add_subplot(gs[1, 1])
        pnls = [t["pnl"] for t in trades]
        ax3.hist(pnls, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax3.axvline(0, color="#d62728", lw=1.2, ls="--")
        ax3.axvline(np.mean(pnls), color="#ff7f0e", lw=1.0, ls="--",
                    label=f"Mean ${np.mean(pnls):.2f}")
        ax3.set_title("Trade P&L Distribution")
        ax3.set_xlabel("P&L per Trade ($)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # ── 4. Win rate by phase ───────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, 0])
        phases    = ["orb", "vwap", "power"]
        win_rates = [
            s.get("by_phase", {}).get(p, {}).get("win_rate", 0) * 100
            for p in phases
        ]
        avg_pnls  = [
            s.get("by_phase", {}).get(p, {}).get("avg_pnl", 0)
            for p in phases
        ]
        x = np.arange(len(phases))
        ax4.bar(x - 0.2, win_rates, width=0.35, label="Win Rate %",
                color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8)
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + 0.2, avg_pnls, width=0.35, label="Avg P&L $",
                     color=["#aec7e8", "#ffbb78", "#98df8a"], alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([p.upper() for p in phases])
        ax4.set_ylabel("Win Rate (%)")
        ax4_twin.set_ylabel("Avg P&L ($)")
        ax4.set_title("Win Rate & Avg P&L by Phase")
        ax4.axhline(50, color="gray", lw=0.7, ls="--")
        ax4.grid(axis="y", alpha=0.3)

        # ── 5. Exit reason breakdown ───────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, 1])
        ec  = s.get("exit_reasons", {})
        if ec:
            ax5.pie(list(ec.values()), labels=list(ec.keys()),
                    autopct="%1.0f%%", startangle=90,
                    colors=plt.cm.Set3.colors[:len(ec)])
            ax5.set_title("Exit Reason Breakdown")

        plt.suptitle("Intraday Three-Phase Strategy — Backtest Report",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()


# =============================================================================
# BACKTEST SUMMARY HELPER
# =============================================================================

def _compute_backtest_summary(
    trades: list[dict],
    equity_curve: pd.DataFrame,
    initial_cash: float,
) -> dict:
    if not trades:
        return {"num_trades": 0}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    final  = float(equity_curve["portfolio_value"].iloc[-1])
    dd     = ((equity_curve["portfolio_value"] -
               equity_curve["portfolio_value"].cummax()) /
              equity_curve["portfolio_value"].cummax()).min()

    by_phase: dict = {}
    for phase in ["orb", "vwap", "power"]:
        pt = [t for t in trades if t.get("phase") == phase]
        pw = [t for t in pt if t["pnl"] > 0]
        by_phase[phase] = {
            "trades":   len(pt),
            "pnl":      round(sum(t["pnl"] for t in pt), 2),
            "win_rate": len(pw) / len(pt) if pt else 0,
            "avg_pnl":  round(np.mean([t["pnl"] for t in pt]), 2) if pt else 0,
        }

    exit_reasons: dict = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    avg_win  = np.mean([t["pnl"] for t in wins])  if wins   else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
    expectancy    = round(float(np.mean([t["pnl"] for t in trades])), 2)
    sharpe        = _daily_sharpe(equity_curve)

    return {
        "initial_cash":     initial_cash,
        "final_value":      round(final, 2),
        "total_return_pct": round((final - initial_cash) / initial_cash * 100, 2),
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "num_trades":       len(trades),
        "win_rate_pct":     round(len(wins) / len(trades) * 100, 1),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "reward_risk":      round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
        "profit_factor":    profit_factor,
        "expectancy":       expectancy,
        "sharpe":           sharpe,
        "by_phase":         by_phase,
        "exit_reasons":     exit_reasons,
    }


def _daily_sharpe(equity_curve: pd.DataFrame) -> float:
    """Annualised Sharpe ratio from the daily portfolio-value series (rf = 0)."""
    try:
        pv = equity_curve["portfolio_value"].astype(float)
        rets = pv.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0:
            return 0.0
        return round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
    except Exception:
        return 0.0


# =============================================================================
# MODULE-LEVEL UTILITIES
# =============================================================================

def _extract_symbol_df(bars_resp, symbol: str) -> Optional[pd.DataFrame]:
    """Safely extract a single-symbol DataFrame from an Alpaca BarSet response."""
    try:
        full_df = bars_resp.df
        if hasattr(full_df.index, "levels"):  # MultiIndex
            syms = full_df.index.get_level_values(0)
            if symbol in syms:
                df = full_df.loc[symbol].copy()
                df.columns = [c.lower() for c in df.columns]
                return df
    except Exception:
        pass
    try:
        raw = bars_resp[symbol]
        rows = [{"open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": b.volume}
                for b in raw]
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def _parse_time_today(hhmm: str) -> datetime.datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.datetime.now(ET).replace(
        hour=h, minute=m, second=0, microsecond=0
    )


def _wait_until(hhmm: str) -> None:
    target = _parse_time_today(hhmm)
    now    = datetime.datetime.now(ET)
    if now < target:
        secs = (target - now).total_seconds()
        logging.getLogger("_wait_until").info(
            f"Waiting {secs/60:.1f} min until {hhmm} ET..."
        )
        time.sleep(max(secs, 0))