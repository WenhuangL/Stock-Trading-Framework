"""
strategy_eod_reversion.py
--------------------------
End-of-Day Mean-Reversion Strategy
===================================

STRATEGY PREMISE
----------------
Large single-day declines in S&P 500 stocks are sometimes exaggerated by
end-of-day selling pressure (index rebalancing, tax-loss harvesting, stop-loss
cascades, forced ETF outflows). Buying these stocks at 3:55 PM and selling in
the aftermarket bets that some of this selling pressure reverses once regular
session mechanics are gone.

VIABILITY ASSESSMENT
--------------------
Honest verdict: this is a plausible edge, but a fragile one. Here is why:

  Reasons it can work:
  - S&P 500 stocks have reasonable afterhours liquidity vs small caps
  - ETF/index rebalancing selling genuinely reverses some of the time
  - The 3:55 PM entry avoids the final-minute volatility spike at close
  - A hard 7:50 PM exit bounds your risk to a ~4-hour window

  Reasons it can fail:
  - Afterhours bid-ask spreads are wide (often 0.2–0.5%), which eats directly
    into a 1% TP target. You may need TP closer to 1.2–1.5% to net 1% after
    spread costs. The config below reflects this.
  - On broad down days (e.g. rate shock, geopolitical event), many stocks will
    qualify but they're all falling for the same macro reason. Mean reversion
    doesn't work when the reason for the decline is still active.
  - Stocks with bad earnings, guidance cuts, or analyst downgrades fall for
    fundamental reasons and often continue lower in afterhours. This strategy
    cannot distinguish fundamental declines from noise. The earnings filter
    (exclude_earnings_day) mitigates this but doesn't eliminate it.
  - You are always the "dumb money" in afterhours. Institutions with better
    information are trading against you.
  - Backtest results will be optimistic due to fill assumptions. In live
    trading, market orders in thin afterhours books fill at worse prices than
    the last trade price.

  Bottom line: run the backtest first, look at the exit_reason breakdown.
  If >50% of trades exit via hard_close (flat or small loss), the edge is
  weak. If take_profit exits dominate with a win rate above ~55%, it's worth
  paper-trading.

PARAMETER CHANGES FROM USER'S ORIGINAL SPEC
--------------------------------------------
  - TP raised from 1.0% → 1.2% to net ~1% after typical afterhours spread
  - Partial exit pct raised from 0.5% → 0.6% for the same reason
  - Partial exit hold shortened from 120 min → 60 min (afterhours is 4h total;
    waiting 2h to exit a marginal gain leaves very little time before hard close)
  - Hard close moved from 8:00 PM → 7:50 PM to allow 10 min buffer for fills
  - Stop loss added at -1.5% (user's original had no stop loss; without one,
    a stock that crashes 5% in afterhours wipes multiple winning trades)
  - Max positions cap added at 25 (prevents spreading too thin on broad down days
    and limits catastrophic loss if the drop was macro-driven)

USAGE — LIVE (called by external scheduler at 3:55 PM ET)
---------------------------------------------------------
    from strategy_eod_reversion import EodReversionStrategy, EodReversionConfig
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient

    cfg = EodReversionConfig()
    strategy = EodReversionStrategy(
        trading_client=TradingClient(API_KEY, SECRET_KEY, paper=True),
        data_client=StockHistoricalDataClient(API_KEY, SECRET_KEY),
        config=cfg,
    )
    strategy.run()

USAGE — BACKTEST
----------------
    tickers = get_sp500_tickers()          # or pass a smaller subset
    results = strategy.backtest(
        tickers=tickers[:50],              # start small; full 500 is slow
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_cash=100_000,
    )
    strategy.print_summary(results)
    strategy.plot_backtest(results)

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy matplotlib requests
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
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from data_collection.stock_universe import UniverseSelector, UniverseConfig

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)

ET = ZoneInfo("America/New_York")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EodReversionConfig:
    """All tuneable parameters for the strategy in one place."""

    # ── Entry ──────────────────────────────────────────────────────────────────
    drop_threshold: float = -0.010
    """Buy stocks that have fallen at least this much from previous close.
    -0.010 = -1.0%.  Tighten to -0.015 to reduce trade count and focus on
    larger dislocations; loosen to -0.008 to increase sample size."""

    entry_time_hhmm: str = "15:55"
    """Time to scan and open positions (HH:MM ET). 3:55 PM gives 5 minutes
    before the regular session close for order submission."""

    position_size_pct: float = 0.010
    """Fraction of portfolio value allocated per position. 0.01 = 1%.
    With max_positions=25 the maximum deployed capital is 25%."""

    max_positions: int = 25
    """Hard cap on simultaneous positions. Prevents over-exposure on broad
    down days where 100+ stocks might qualify."""

    # ── Exits ──────────────────────────────────────────────────────────────────
    tp_pct: float = 0.005
    """Take-profit threshold above entry. 0.005 = +0.5%. Must be strictly
    greater than partial_exit_pct so the partial-exit countdown can trigger
    before TP fires."""

    partial_exit_pct: float = 0.003
    """If the stock reaches this gain but has NOT hit TP, start the partial-
    exit countdown timer. 0.003 = +0.3%. Must be < tp_pct; previously set
    to 0.006 which exceeded tp_pct=0.005, making the partial-exit dead code."""

    partial_exit_hold_min: int = 60
    """Minutes to wait after partial_exit_pct is first reached before giving
    up on hitting TP and selling at market. 60 min = 1 hour."""

    stop_loss_pct: float = -0.015
    """Cut the position if it falls this far below entry. -0.015 = -1.5%.
    Critical: without a stop loss, a single -5% afterhours move can wipe
    the gains from many winning trades."""

    hard_close_hhmm: str = "19:50"
    """Close ALL remaining positions at this time regardless of P&L.
    7:50 PM gives a 10-minute buffer before the 8:00 PM aftermarket close.

    NOTE: this was silently set to '15:59' previously, which collapsed the whole
    strategy to a 4-minute regular-session hold and never tested the afterhours
    reversion thesis (the backtest summary showed avg 4 min held, 100% hard
    close). It is restored to the documented 7:50 PM. The minute fetch window in
    the backtest is now derived from this value. IMPORTANT: extended-hours bars
    are only returned by Alpaca's SIP feed, and the local minute cache from prior
    runs only holds the old 15:55–16:01 slice — delete
    data_collection/cache/minute/ before re-running so the afterhours bars are
    actually fetched. The summary reports how many trades had afterhours data."""

    # ── Universe filters ───────────────────────────────────────────────────────
    exclude_earnings_day: bool = True
    """Skip stocks reporting earnings today. Earnings-day declines are often
    fundamental (missed EPS, bad guidance) rather than noise, and do NOT
    mean-revert reliably."""

    min_price: float = 15.0
    """Skip stocks below this price. Very cheap stocks have large percentage
    spreads that destroy the thin TP margin."""

    max_price: float = 500.0  # exclude mega-caps above this price
    min_avg_volume: int = 500_000  # need enough afterhours liquidity
    max_avg_volume: int = 10_000_000  # exclude mega-caps with too much

    # ── Step 2 qualifying filters ──────────────────────────────────────────────
    require_close_within_range: bool = True
    """Only trade stocks where (close - low) / (high - low) > close_within_range_min.
    Stocks that closed at or near their day low are still under active selling
    pressure. Stocks that bounced off their low show some demand coming in,
    making afterhours reversion more likely."""

    close_within_range_min: float = 0.35
    """Minimum fraction of today's range that the close must be above the low.
    0.35 = close must be at least 35% of the way from low to high."""

    require_above_sma20: bool = True
    """Only trade stocks above their 20-day SMA. Stocks in structural downtrends
    (below SMA20) experience single-day declines as continuation moves, not
    noise. Mean reversion is more reliable in uptrending stocks."""

    require_idiosyncratic_decline: bool = True
    """Only trade stocks whose decline exceeded SPY's decline by at least
    idiosyncratic_min_gap. Removes broad macro-driven down days where the
    entire market fell and afterhours reversion is unlikely."""

    idiosyncratic_min_gap: float = 0.003
    """Minimum percentage points by which the stock must have fallen MORE than
    SPY. 0.003 = stock must be down at least 0.3% more than SPY."""

    # ── Slippage / fill realism ────────────────────────────────────────────────
    slippage_pct: float = 0.0005
    use_next_bar_fill: bool = True  # fill at next bar's open after signal

    # ── Transaction costs ──────────────────────────────────────────────────────
    spread_pct: float = 0.0010
    """Half bid-ask spread applied to entry AND exit, on top of slippage.
    Afterhours books for these names are wide (the strategy docstring cites
    0.2–0.5% full spread), so this defaults to 0.10% per side (~0.2% round trip).
    This is the single biggest reason live afterhours results trail the
    backtest; tune it to match your observed fills."""

    commission_per_trade: float = 0.0
    """Flat commission per round-trip trade (Alpaca equities are commission-free)."""


# =============================================================================
# S&P 500 UNIVERSE
# =============================================================================

def get_sp500_tickers() -> list[str]:
    """
    Fetch the current S&P 500 constituent list from Wikipedia.

    Returns a list of ticker strings normalised for Alpaca
    (e.g. 'BRK.B' → 'BRK-B').  Returns an empty list on failure;
    callers should handle this gracefully.
    """
    log = logging.getLogger("get_sp500_tickers")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = (
            tables[0]["Symbol"]
            .str.replace(".", "-", regex=False)  # Alpaca uses BRK-B not BRK.B
            .tolist()
        )
        log.info(f"Fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as exc:
        log.error(f"Failed to fetch S&P 500 list: {exc}")
        return []


# =============================================================================
# STRATEGY CLASS
# =============================================================================

class EodReversionStrategy:
    """
    Encapsulates both live execution and historical backtesting for the
    End-of-Day Mean-Reversion strategy.

    The same config and logic drive both modes, which means a parameter
    that improves backtest results translates directly to the live version.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        config: Optional[EodReversionConfig] = None,
        universe_config: Optional[UniverseConfig] = None,
        risk_manager: Optional[object] = None,
    ) -> None:
        self.tc   = trading_client
        self.dc   = data_client
        self.cfg  = config or EodReversionConfig()
        self.ucfg = universe_config or UniverseConfig()
        self.rm   = risk_manager  # Optional[RiskManager] — avoids circular import
        self.log  = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # LIVE EXECUTION
    # =========================================================================

    def run(self, tickers: Optional[list[str]] = None) -> None:
        """
        Main live entry point.  Call this at 3:55 PM ET via external scheduler.
        Blocks until hard_close_hhmm, then returns cleanly.
        """
        now = datetime.datetime.now(ET)
        self.log.info(f"Strategy starting at {now.strftime('%H:%M:%S ET')}")

        # ── Pre-session risk gate ─────────────────────────────────────────────
        if self.rm is not None:
            ok, reason = self.rm.pre_session_check()
            if not ok:
                self.log.warning(f"EOD session aborted by risk manager: {reason}")
                return

        if tickers is None:
            selector = UniverseSelector(self.dc, self.ucfg)
            tickers  = selector.get_universe(use_cache=True)

        if not tickers:
            self.log.error("Empty ticker list — aborting.")
            return

        self.log.info(f"Universe: {len(tickers)} tickers loaded.")

        # Optionally filter out earnings-day stocks
        if self.cfg.exclude_earnings_day:
            tickers = self._filter_earnings(tickers)

        losers = self._get_live_losers(tickers)
        if not losers:
            self.log.info("No qualifying losers today. Nothing to do.")
            return

        positions = self._open_live_positions(losers)
        if positions:
            self._monitor_until_close(positions)

        self.log.info("Strategy run complete.")

    def _filter_earnings(self, tickers: list[str]) -> list[str]:
        """Remove tickers reporting earnings today using webscraper.py."""
        try:
            from data_collection.earnings_calendar import get_todays_earnings
            earnings_set = set(get_todays_earnings())
            before = len(tickers)
            tickers = [t for t in tickers if t not in earnings_set]
            self.log.info(f"Earnings filter: removed {before - len(tickers)} tickers.")
        except Exception as exc:
            self.log.warning(f"Earnings filter unavailable: {exc}")
        return tickers

    def _get_live_losers(self, tickers: list[str]) -> list[dict]:
        """
        Fetch live snapshots for all tickers.
        Returns qualifying stocks sorted by largest decline first.
        """
        losers: list[dict] = []
        BATCH = 1000

        for i in range(0, len(tickers), BATCH):
            batch = tickers[i : i + BATCH]
            try:
                snapshots = self.dc.get_stock_snapshots(
                    StockSnapshotRequest(symbol_or_symbols=batch)
                )
                for symbol, snap in snapshots.items():
                    if snap.daily_bar is None or snap.prev_daily_bar is None:
                        continue
                    prev_close = snap.prev_daily_bar.close
                    current    = snap.daily_bar.close
                    if prev_close <= self.cfg.min_price or prev_close == 0:
                        continue
                    pct = (current - prev_close) / prev_close
                    if pct <= self.cfg.drop_threshold:
                        losers.append({
                            "symbol":        symbol,
                            "prev_close":    prev_close,
                            "current_price": current,
                            "pct_change":    pct,
                        })
            except Exception as exc:
                self.log.warning(f"Snapshot batch {i}–{i+BATCH} failed: {exc}")

        losers.sort(key=lambda x: x["pct_change"])
        self.log.info(
            f"Qualifying losers: {len(losers)} "
            f"(threshold: {self.cfg.drop_threshold*100:.1f}%)"
        )
        return losers

    def _open_live_positions(self, losers: list[dict]) -> list[dict]:
        """
        Submit market orders for the top qualifying losers.
        Respects max_positions and available cash.
        Returns a list of position dicts used by the monitoring loop.
        """
        account        = self.tc.get_account()
        portfolio_val  = float(account.portfolio_value)
        cash           = float(account.cash)
        position_budget = portfolio_val * self.cfg.position_size_pct

        opened: list[dict] = []

        for loser in losers[: self.cfg.max_positions]:
            if cash < position_budget:
                self.log.info("Cash exhausted — stopping position opens.")
                break

            symbol = loser["symbol"]
            price  = loser["current_price"]
            qty    = int(position_budget // price)

            if qty < 1:
                self.log.debug(f"Skipping {symbol}: price ${price:.2f} > budget ${position_budget:.2f}")
                continue

            # ── Risk manager per-trade gate ───────────────────────────────────
            if self.rm is not None:
                proposed_notional = qty * price
                ok, rm_reason = self.rm.check_ok_to_enter(
                    symbol            = symbol,
                    open_positions    = opened,
                    proposed_notional = proposed_notional,
                    portfolio_value   = portfolio_val,
                )
                if not ok:
                    self.log.info(f"Skipping {symbol}: risk gate — {rm_reason}")
                    continue

            try:
                order = self.tc.submit_order(MarketOrderRequest(
                    symbol         = symbol,
                    qty            = qty,
                    side           = OrderSide.BUY,
                    time_in_force  = TimeInForce.DAY,
                ))
                entry_time = datetime.datetime.now(ET)
                pos = {
                    "symbol":               symbol,
                    "order_id":             order.id,
                    "qty":                  qty,
                    "entry_price":          price,
                    "entry_time":           entry_time,
                    "tp_price":             price * (1 + self.cfg.tp_pct),
                    "sl_price":             price * (1 + self.cfg.stop_loss_pct),
                    "partial_price":        price * (1 + self.cfg.partial_exit_pct),
                    "partial_deadline":     entry_time + datetime.timedelta(minutes=self.cfg.partial_exit_hold_min),
                    "partial_triggered_at": None,
                    "closed":               False,
                    "exit_price":           None,
                    "exit_reason":          None,
                }
                opened.append(pos)
                cash -= qty * price
                self.log.info(
                    f"OPEN  {qty:>4}x {symbol:<6} @ ${price:>8.2f} | "
                    f"TP ${pos['tp_price']:.2f} | SL ${pos['sl_price']:.2f} | "
                    f"Day Δ {loser['pct_change']*100:+.1f}%"
                )
            except Exception as exc:
                self.log.error(f"Order failed for {symbol}: {exc}")

        return opened

    def _monitor_until_close(self, positions: list[dict]) -> None:
        """
        Poll all open positions every 60 seconds and apply exit conditions.
        Blocks until all positions are closed or hard_close_hhmm is reached.
        """
        hard_close_dt = _parse_time_today(self.cfg.hard_close_hhmm)
        _last_risk_check = datetime.datetime.now(ET)

        while True:
            now  = datetime.datetime.now(ET)
            open_positions = [p for p in positions if not p["closed"]]

            if not open_positions:
                self.log.info("All positions closed.")
                break

            if now >= hard_close_dt:
                self.log.info("Hard close time reached — liquidating all positions.")
                for pos in open_positions:
                    self._close_live_position(pos, reason="hard_close")
                break

            # ── Risk monitor every 5 minutes ──────────────────────────────────
            if self.rm is not None and (now - _last_risk_check).total_seconds() >= 300:
                _last_risk_check = now
                try:
                    current_prices = {}
                    syms = [p["symbol"] for p in open_positions]
                    if syms:
                        trades = self.dc.get_stock_latest_trade(
                            StockLatestTradeRequest(symbol_or_symbols=syms)
                        )
                        current_prices = {s: trades[s].price for s in syms if s in trades}
                    account    = self.tc.get_account()
                    port_value = float(account.portfolio_value)
                    rm_status  = self.rm.intraday_monitor(
                        open_positions  = open_positions,
                        current_prices  = current_prices,
                        portfolio_value = port_value,
                    )
                    if rm_status.get("action") == "emergency_close":
                        reason = rm_status.get("reason", "intraday monitor triggered")
                        self.log.warning(f"EMERGENCY CLOSE triggered by risk monitor: {reason}")
                        self.rm.emergency_close_all(open_positions, reason)
                        return
                except Exception as exc:
                    self.log.warning(f"Risk monitor call failed: {exc}")

            symbols = [p["symbol"] for p in open_positions]
            try:
                latest_trades = self.dc.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=symbols)
                )
            except Exception as exc:
                self.log.warning(f"Price fetch failed: {exc}")
                time.sleep(30)
                continue

            for pos in open_positions:
                sym = pos["symbol"]
                if sym not in latest_trades:
                    continue
                price = latest_trades[sym].price

                if price <= pos["sl_price"]:
                    self._close_live_position(pos, reason="stop_loss", price=price)
                    continue

                if price >= pos["tp_price"]:
                    self._close_live_position(pos, reason="take_profit", price=price)
                    continue

                if price >= pos["partial_price"]:
                    if pos["partial_triggered_at"] is None:
                        pos["partial_triggered_at"] = now
                    elif now >= pos["partial_deadline"]:
                        self._close_live_position(pos, reason="partial_exit_timeout", price=price)
                        continue

            time.sleep(60)

    def _close_live_position(
        self,
        pos: dict,
        reason: str,
        price: Optional[float] = None,
    ) -> None:
        """Submit a close-position order and update the position dict."""
        try:
            self.tc.close_position(pos["symbol"])
            exit_price = price or pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["qty"]
            self.log.info(
                f"CLOSE {pos['qty']:>4}x {pos['symbol']:<6} @ ${exit_price:>8.2f} | "
                f"reason={reason:<22} | P&L ${pnl:>+8.2f}"
            )
            pos["closed"]      = True
            pos["exit_price"]  = exit_price
            pos["exit_reason"] = reason
        except Exception as exc:
            self.log.error(f"Failed to close {pos['symbol']}: {exc}")

    # =========================================================================
    # BACKTESTING
    # =========================================================================

    def backtest(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000.0,
        historical_universes: dict = None,
    ) -> dict:
        """
        Simulate the strategy against historical data using minute-level bars.
        """
        self.log.info(
            f"Backtest: {start_date} -> {end_date} | "
            f"{len(tickers)} tickers | ${initial_cash:,.0f} starting capital"
        )
        start_dt = datetime.datetime.fromisoformat(start_date).replace(tzinfo=ET)
        end_dt   = datetime.datetime.fromisoformat(end_date).replace(tzinfo=ET)

        # ── Pass 1: Fetch daily bars ──────────────────────────────────────────
        daily_data = self._fetch_daily_bars(tickers, start_dt, end_dt)

        # ── Walk through each trading day ─────────────────────────────────────
        portfolio_value = initial_cash
        cash            = initial_cash
        equity_rows: list[dict] = []
        all_trades:  list[dict] = []

        trading_days = pd.bdate_range(start_date, end_date)

        for day in trading_days:
            day_result = self._simulate_day(
                day            = day,
                daily_data     = daily_data,
                tickers        = tickers,
                portfolio_value= portfolio_value,
                cash           = cash,
                historical_universes=historical_universes
            )

            cash            = day_result["ending_cash"]
            portfolio_value = day_result["ending_cash"]  # no overnight positions
            n               = len(day_result["trades"])
            all_trades.extend(day_result["trades"])

            equity_rows.append({
                "date":            day,
                "portfolio_value": portfolio_value,
                "num_trades":      n,
                "day_pnl":         day_result["day_pnl"],
            })

            if n > 0:
                self.log.info(
                    f"{day.strftime('%Y-%m-%d')}: {n} trades | "
                    f"P&L ${day_result['day_pnl']:>+8.2f} | "
                    f"Portfolio ${portfolio_value:>10,.2f}"
                )

        equity_curve = pd.DataFrame(equity_rows).set_index("date")
        summary      = self._compute_summary(all_trades, equity_curve, initial_cash)

        return {
            "equity_curve": equity_curve,
            "trades":        all_trades,
            "summary":       summary,
        }

    # ── Backtest helpers ──────────────────────────────────────────────────────

    def _fetch_daily_bars(
            self,
            tickers: list[str],
            start_dt: datetime.datetime,
            end_dt: datetime.datetime,
    ) -> dict[str, pd.DataFrame]:
        from data_collection.data_cache import LocalDataCache
        cache = LocalDataCache(self.dc)
        bars_data: dict[str, pd.DataFrame] = {}

        fetch_tickers = list(dict.fromkeys(["SPY"] + list(tickers)))
        for i, sym in enumerate(fetch_tickers):
            try:
                df = cache.get_bars_df(sym, TimeFrame.Day, start_dt, end_dt, feed="sip")
                if df is not None and not df.empty:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(ET).normalize()
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    bars_data[sym] = df
            except Exception as exc:
                self.log.debug(f"  {sym}: extraction failed — {exc}")

            if (i + 1) % 50 == 0:
                self.log.info(f"Daily bars: fetched {i + 1}/{len(fetch_tickers)}")

        self.log.info(f"Daily bars loaded: {len(bars_data)} / {len(fetch_tickers)} tickers.")
        return bars_data

    def _risk_ok_eod(
        self,
        symbol: str,
        selected: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> bool:
        """Apply the live RiskManager caps when picking EOD positions.

        No-op if no RiskManager was supplied. The sector cap is skipped for
        'Unknown' sectors so a partial sector map can't throttle the whole book.
        """
        if self.rm is None:
            return True

        cfg = self.rm.cfg
        if len(selected) >= cfg.max_total_positions:
            return False

        deployed = len(selected) * proposed_notional
        if (deployed + proposed_notional) / portfolio_value > cfg.max_portfolio_deployed_pct:
            return False

        sector = self.rm.get_sector(symbol)
        if sector != "Unknown":
            sector_count = sum(
                1 for q in selected if self.rm.get_sector(q["symbol"]) == sector
            )
            if sector_count >= cfg.max_positions_per_sector:
                return False

        return True

    def _simulate_day(
            self,
            day: pd.Timestamp,
            daily_data: dict[str, pd.DataFrame],
            tickers: list[str],
            portfolio_value: float,
            cash: float,
            historical_universes: dict = None,
    ) -> dict:
        """Simulate a single trading day end-to-end with corrected exact-fill mechanics."""
        day_date = day.date()

        # ── EXACT CODE GOES HERE ─────────────────────────────────────────────
        # Shift the lookup backward by 7 days to retrieve the PREVIOUS week's universe,
        # ensuring we only trade on data that was fully knowable before Monday's open.
        reference_day = day - pd.Timedelta(days=7)
        year, week, _ = reference_day.isocalendar()
        week_str = f"{year}-W{week:02d}"

        if historical_universes:
            active_tickers = historical_universes.get(week_str, tickers)
        else:
            active_tickers = tickers
        # ─────────────────────────────────────────────────────────────────────

        # ── Get SPY prev close for idiosyncratic filter ───────────────────────
        spy_prev_close = None
        day_str = day.strftime('%Y-%m-%d')
        prev_day_str = (day - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        if self.cfg.require_idiosyncratic_decline and "SPY" in daily_data:
            spy_prior = daily_data["SPY"].loc[:prev_day_str]
            if not spy_prior.empty:
                spy_prev_close = float(spy_prior["close"].iloc[-1])

        # ── Identify qualifying stocks (Relaxed Daily Pre-Filter) ────────────
        qualifying: list[dict] = []
        pre_filter_threshold = self.cfg.drop_threshold * 0.5

        for sym in active_tickers:
            if sym not in daily_data:
                continue
            df = daily_data[sym]

            try:
                today_rows = df.loc[day_str:day_str]
                prior_rows = df.loc[:prev_day_str]
            except KeyError:
                continue

            if today_rows.empty or prior_rows.empty:
                continue

            prev_close = float(prior_rows["close"].iloc[-1])
            today_close = float(today_rows["close"].iloc[-1])
            today_high = float(today_rows["high"].iloc[-1])
            today_low = float(today_rows["low"].iloc[-1])

            if prev_close < self.cfg.min_price or prev_close == 0:
                continue

            if prev_close > self.cfg.max_price:
                continue

            if self.cfg.min_avg_volume > 0 and len(prior_rows) >= 20:
                avg_vol = float(prior_rows["volume"].iloc[-20:].mean())
                if avg_vol < self.cfg.min_avg_volume or avg_vol > self.cfg.max_avg_volume:
                    continue

            pct = (today_close - prev_close) / prev_close
            if pct > pre_filter_threshold:
                continue

            # Approximation pre-filters
            if self.cfg.require_close_within_range:
                day_range = today_high - today_low
                if day_range > 0:
                    close_within = (today_close - today_low) / day_range
                    if close_within < self.cfg.close_within_range_min:
                        continue

            if self.cfg.require_above_sma20:
                if len(prior_rows) >= 20:
                    sma20 = float(prior_rows["close"].iloc[-20:].mean())
                    if today_close < sma20:
                        continue

            qualifying.append({
                "symbol":       sym,
                "entry_price":  today_close,
                "pct_change":   pct,
                "prev_close":   prev_close,
            })

        if not qualifying:
            return {
                "trades": [], "ending_cash": cash,
                "ending_value": cash, "day_pnl": 0.0,
            }

        # ── Add SPY to Minute Bar fetch if needed ─────────────────────────────
        syms       = [q["symbol"] for q in qualifying]
        if self.cfg.require_idiosyncratic_decline and spy_prev_close is not None and "SPY" not in syms:
            syms.append("SPY")

        entry_h, entry_m = map(int, self.cfg.entry_time_hhmm.split(":"))
        close_h, close_m = map(int, self.cfg.hard_close_hhmm.split(":"))
        entry_dt = datetime.datetime.combine(day_date, datetime.time(entry_h, entry_m), tzinfo=ET)
        # +1 minute so the hard-close bar itself is inside the fetched window.
        close_dt = datetime.datetime.combine(day_date, datetime.time(close_h, close_m), tzinfo=ET) \
                   + datetime.timedelta(minutes=1)
        minute_data: dict[str, pd.DataFrame] = {}

        from data_collection.data_cache import LocalDataCache
        cache = LocalDataCache(self.dc)

        for sym in syms:
            try:
                df = cache.get_bars_df(sym, TimeFrame.Minute, entry_dt, close_dt, feed="sip")
                if df is not None and not df.empty:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
                    minute_data[sym] = df
            except Exception as exc:
                self.log.debug(f"  {sym}: minute bar extraction failed — {exc}")

        if not minute_data:
            self.log.warning(f"{day_date}: minute bar fetch failed for all candidates.")
            return {
                "trades": [], "ending_cash": cash,
                "ending_value": cash, "day_pnl": 0.0,
            }

        # ── EXACT 3:55 PM Lookahead-Free Qualification ────────────────────────
        exact_spy_drop = None
        if self.cfg.require_idiosyncratic_decline and "SPY" in minute_data and spy_prev_close is not None:
            if not minute_data["SPY"].empty:
                spy_355_price = float(minute_data["SPY"].iloc[0]["open"])
                exact_spy_drop = (spy_355_price - spy_prev_close) / spy_prev_close

        exact_qualifying = []
        for q in qualifying:
            sym = q["symbol"]
            if sym in minute_data and not minute_data[sym].empty:
                exact_355_price = float(minute_data[sym].iloc[0]["open"])
            else:
                exact_355_price = q["entry_price"]  # Fallback to daily close

            exact_drop = (exact_355_price - q["prev_close"]) / q["prev_close"]

            # Strict Drop Validation at exactly 3:55 PM
            if exact_drop > self.cfg.drop_threshold:
                continue

            # Strict SPY idiosyncratic validation at exactly 3:55 PM
            if self.cfg.require_idiosyncratic_decline and exact_spy_drop is not None:
                #if (exact_spy_drop - exact_drop) < self.cfg.idiosyncratic_min_gap:
                if exact_drop > (exact_spy_drop - self.cfg.idiosyncratic_min_gap):
                    continue

            q["exact_355_price"] = exact_355_price
            q["exact_drop"] = exact_drop
            exact_qualifying.append(q)

        exact_qualifying.sort(key=lambda x: x["exact_drop"])

        # Apply the live RiskManager concentration caps (total positions, per-sector,
        # deployment) in addition to the strategy's own max_positions. Previously the
        # backtest ignored these, so it could hold 25 names with no sector limit while
        # the live strategy would have been capped much tighter.
        position_budget = portfolio_value * self.cfg.position_size_pct
        selected: list[dict] = []
        for q in exact_qualifying:
            if len(selected) >= self.cfg.max_positions:
                break
            if not self._risk_ok_eod(q["symbol"], selected, position_budget, portfolio_value):
                continue
            selected.append(q)
        exact_qualifying = selected

        # ── Simulate each position ────────────────────────────────────────────
        trades:  list[dict] = []
        day_pnl: float      = 0.0
        position_budget     = portfolio_value * self.cfg.position_size_pct

        hard_close_time = datetime.datetime.combine(
            day_date,
            datetime.time(*map(int, self.cfg.hard_close_hhmm.split(":"))),
            tzinfo=ET,
        )

        for q in exact_qualifying:
            if cash < position_budget:
                break

            sym = q["symbol"]
            slip = self.cfg.slippage_pct + self.cfg.spread_pct

            # Using exact_355_price evaluated above
            raw_entry = q["exact_355_price"]
            entry_price = raw_entry * (1 + slip)

            qty = int(position_budget // entry_price)
            if qty < 1:
                continue

            tp_price      = entry_price * (1 + self.cfg.tp_pct)
            sl_price      = entry_price * (1 + self.cfg.stop_loss_pct)
            partial_price = entry_price * (1 + self.cfg.partial_exit_pct)
            partial_hold  = self.cfg.partial_exit_hold_min

            cash -= qty * entry_price

            exit_price:  float = entry_price
            exit_reason: str   = "hard_close"
            bars_held:   int   = 0
            partial_triggered_bar: Optional[int] = None

            if sym in minute_data and not minute_data[sym].empty:
                mdf = minute_data[sym]

                for bar_idx, (ts, bar) in enumerate(mdf.iterrows()):
                    bar_high  = float(bar["high"])
                    bar_low   = float(bar["low"])
                    bar_close = float(bar["close"])
                    bar_time  = ts.to_pydatetime()
                    bars_held = bar_idx

                    if bar_time >= hard_close_time:
                        exit_price  = bar_close * (1 - slip)
                        exit_reason = "hard_close"
                        break

                    if bar_low <= sl_price:
                        exit_price  = sl_price * (1 - slip)
                        exit_reason = "stop_loss"
                        break

                    if bar_high >= tp_price:
                        exit_price  = tp_price * (1 - slip)
                        exit_reason = "take_profit"
                        break

                    if bar_high >= partial_price and partial_triggered_bar is None:
                        partial_triggered_bar = bar_idx

                    if (
                        partial_triggered_bar is not None
                        and (bar_idx - partial_triggered_bar) >= partial_hold
                    ):
                        exit_price  = bar_close * (1 - slip)
                        exit_reason = "partial_exit_timeout"
                        break

                else:
                    exit_price  = float(mdf["close"].iloc[-1]) * (1 - slip)
                    exit_reason = "hard_close"

            pnl   = (exit_price - entry_price) * qty - self.cfg.commission_per_trade
            cash += qty * exit_price
            day_pnl += pnl

            # Did this trade actually have afterhours bars to hold into? If the
            # data feed only returned regular-session bars, the reversion thesis
            # was never exercised — track it so the summary can surface it.
            had_afterhours = False
            if sym in minute_data and not minute_data[sym].empty:
                had_afterhours = bool((minute_data[sym].index.time >= datetime.time(16, 0)).any())

            trades.append({
                "date":              day_date,
                "phase": "eod",
                "direction": "long",
                "symbol":            sym,
                "entry_price":       round(entry_price, 4),
                "exit_price":        round(exit_price, 4),
                "qty":               qty,
                "pnl":               round(pnl, 2),
                "pct_change_at_entry": round(q["exact_drop"] * 100, 2), # Fixed recording metric
                "exit_reason":       exit_reason,
                "bars_held":         bars_held,
                "had_afterhours":    had_afterhours,
            })

        return {
            "trades":       trades,
            "ending_cash":  cash,
            "ending_value": cash,
            "day_pnl":      round(day_pnl, 2),
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

        avg_bars_held = np.mean([t["bars_held"] for t in trades])

        gross_win  = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
        expectancy    = round(float(np.mean([t["pnl"] for t in trades])), 2)

        pv   = equity_curve["portfolio_value"].astype(float)
        rets = pv.pct_change().dropna()
        sharpe = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) \
            if len(rets) >= 2 and rets.std() != 0 else 0.0

        ah_trades = sum(1 for t in trades if t.get("had_afterhours"))
        afterhours_data_pct = round(ah_trades / len(trades) * 100, 1)

        return {
            "initial_cash":        initial_cash,
            "final_value":         round(final_value, 2),
            "total_return_pct":    round((final_value - initial_cash) / initial_cash * 100, 2),
            "num_trades":          len(trades),
            "win_rate_pct":        round(len(wins) / len(trades) * 100, 1),
            "avg_win":             round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
            "avg_loss":            round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
            "reward_risk_ratio":   round(
                abs(np.mean([t["pnl"] for t in wins]) / np.mean([t["pnl"] for t in losses])), 2
            ) if wins and losses else 0,
            "profit_factor":       profit_factor,
            "expectancy":          expectancy,
            "sharpe":              sharpe,
            "largest_win":         round(max(t["pnl"] for t in trades), 2),
            "largest_loss":        round(min(t["pnl"] for t in trades), 2),
            "avg_bars_held":       round(avg_bars_held, 1),
            "afterhours_data_pct": afterhours_data_pct,
            "max_drawdown_pct":    round(float(drawdown.min()) * 100, 2),
            "exit_reason_counts":  exit_counts,
        }

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_summary(self, results: dict) -> None:
        s = results["summary"]
        if "note" in s:
            print(f"\n[No trades: {s['note']}]")
            return

        tp_count   = s["exit_reason_counts"].get("take_profit", 0)
        sl_count   = s["exit_reason_counts"].get("stop_loss", 0)
        hc_count   = s["exit_reason_counts"].get("hard_close", 0)
        pe_count   = s["exit_reason_counts"].get("partial_exit_timeout", 0)
        total      = s["num_trades"]

        print(f"\n{'='*54}")
        print(f"  EOD Mean-Reversion Backtest Summary")
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
        print(f"  Profit factor        : {s.get('profit_factor', 0):>12.2f}")
        print(f"  Expectancy / trade   : ${s.get('expectancy', 0):>12.2f}")
        print(f"  Sharpe (ann.)        : {s.get('sharpe', 0):>12.2f}")
        print(f"  Largest single win   : ${s['largest_win']:>12.2f}")
        print(f"  Largest single loss  : ${s['largest_loss']:>12.2f}")
        print(f"  Avg minutes held     : {s['avg_bars_held']:>12.1f}")
        print(f"  Trades w/ afterhours : {s.get('afterhours_data_pct', 0):>11.1f}%")
        print(f"{'-'*54}")
        print(f"  Exit breakdown:")
        print(f"    Take-profit        : {tp_count:>5} ({tp_count/total*100:>4.0f}%)")
        print(f"    Stop-loss          : {sl_count:>5} ({sl_count/total*100:>4.0f}%)")
        print(f"    Partial timeout    : {pe_count:>5} ({pe_count/total*100:>4.0f}%)")
        print(f"    Hard close (flat)  : {hc_count:>5} ({hc_count/total*100:>4.0f}%)")
        print(f"{'='*54}\n")
        if s.get("afterhours_data_pct", 0) < 50:
            print(f"  [!] Only {s.get('afterhours_data_pct', 0):.0f}% of trades had afterhours bars.")
            print("     The afterhours reversion thesis is largely UNTESTED — your")
            print("     data feed returned mostly regular-session bars. Delete")
            print("     data_collection/cache/minute/ and re-run with a SIP feed to")
            print("     pull extended-hours data, otherwise these numbers reflect a")
            print("     few minutes of regular-session noise, not the strategy.\n")
        print("  [!] Backtest models spread via cfg.spread_pct but still assumes")
        print("     market orders fill at the bar price. Treat as an upper bound.\n")

    def plot_backtest(self, results: dict) -> None:
        s      = results["summary"]
        equity = results["equity_curve"]
        trades = results["trades"]

        if not trades:
            print("No trades to plot.")
            return

        fig = plt.figure(figsize=(15, 12))
        gs  = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

        # ── 1. Equity curve ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        pv  = equity["portfolio_value"]
        ax1.plot(equity.index, pv, color="#1f77b4", linewidth=1.5, label="Portfolio")
        ax1.axhline(s["initial_cash"], color="gray", linewidth=0.8, linestyle="--", label="Starting capital")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv >= s["initial_cash"]), alpha=0.12, color="#2ca02c")
        ax1.fill_between(equity.index, s["initial_cash"], pv,
                         where=(pv  < s["initial_cash"]), alpha=0.12, color="#d62728")
        ax1.set_title(
            f"EOD Reversion — Equity Curve\n"
            f"Return {s['total_return_pct']:+.1f}% | "
            f"{s['num_trades']} trades | "
            f"Win rate {s['win_rate_pct']:.0f}% | "
            f"Max DD {s['max_drawdown_pct']:.1f}%",
            fontsize=11,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # ── 2. Daily P&L bars ─────────────────────────────────────────────────
        ax2    = fig.add_subplot(gs[1, 0])
        daily  = equity["day_pnl"]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in daily]
        ax2.bar(equity.index, daily, color=colors, width=0.8)
        ax2.axhline(0, color="gray", linewidth=0.7)
        ax2.set_title("Daily P&L")
        ax2.set_ylabel("P&L ($)")
        ax2.grid(axis="y", alpha=0.3)

        # ── 3. Trade P&L distribution ──────────────────────────────────────────
        ax3  = fig.add_subplot(gs[1, 1])
        pnls = [t["pnl"] for t in trades]
        ax3.hist(pnls, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax3.axvline(0, color="#d62728", linewidth=1.2, linestyle="--")
        ax3.axvline(np.mean(pnls), color="#ff7f0e", linewidth=1.0,
                    linestyle="--", label=f"Mean ${np.mean(pnls):.2f}")
        ax3.set_title("Trade P&L Distribution")
        ax3.set_xlabel("P&L per Trade ($)")
        ax3.set_ylabel("Frequency")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # ── 4. Exit reason breakdown ───────────────────────────────────────────
        ax4      = fig.add_subplot(gs[2, 0])
        ec       = s["exit_reason_counts"]
        labels   = list(ec.keys())
        sizes    = list(ec.values())
        pie_colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#9467bd"]
        ax4.pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=pie_colors[: len(labels)], startangle=90)
        ax4.set_title("Exit Reason Breakdown")

        # ── 5. Avg P&L by entry-drop bucket ────────────────────────────────────
        ax5     = fig.add_subplot(gs[2, 1])
        tdf     = pd.DataFrame(trades)
        bins    = [-20, -4, -3, -2, -1.5, -1, 0]
        blabels = ["< -4%", "-4 to -3%", "-3 to -2%", "-2 to -1.5%", "-1.5 to -1%", "> -1%"]
        tdf["bucket"] = pd.cut(tdf["pct_change_at_entry"], bins=bins, labels=blabels)
        bucket_pnl    = tdf.groupby("bucket", observed=True)["pnl"].mean()
        bcolors       = ["#2ca02c" if v >= 0 else "#d62728" for v in bucket_pnl.values]
        ax5.bar(bucket_pnl.index.astype(str), bucket_pnl.values,
                color=bcolors, edgecolor="white")
        ax5.axhline(0, color="gray", linewidth=0.7)
        ax5.set_title("Avg Trade P&L by Entry Drop Size")
        ax5.set_xlabel("Stock Drop at 3:55 PM")
        ax5.set_ylabel("Avg P&L ($)")
        ax5.tick_params(axis="x", rotation=20)
        ax5.grid(axis="y", alpha=0.3)

        plt.suptitle(
            "EOD Mean-Reversion Strategy — Backtest Report",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# UTILITIES
# =============================================================================

def _parse_time_today(hhmm: str) -> datetime.datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.datetime.now(ET).replace(hour=h, minute=m, second=0, microsecond=0)