"""
risk_manager.py
---------------
Centralised risk management for the full trading framework.

Provides three layers of protection:

  Layer 1 — Pre-session gate (9:25 AM ET)
      Checks macro conditions before any position is opened.
      If any condition fails, the entire trading session is aborted.
      Conditions: VIX level, broad market direction, critical news.

  Layer 2 — Intraday monitoring (called every 5 minutes during session)
      Monitors ongoing conditions while positions are open.
      Can halt new entries without closing existing positions,
      or trigger emergency close of all positions on extreme events.
      Conditions: VIX spike, portfolio drawdown, correlation collapse.

  Layer 3 — Real-time news scanning (called every 60 seconds)
      Scans Alpaca's news feed for critical keywords.
      Two tiers: WARNING (halt new entries) and CRITICAL (close all).

SECTOR DATA
-----------
Sector classification is loaded from the universe cache built by
stock_universe.py.  If the cache is missing, it falls back to Wikipedia.
Sector data is used to enforce concentration limits (max N positions
per sector simultaneously).

USAGE
-----
    from risk_manager import RiskManager, RiskConfig
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient

    rm = RiskManager(
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True),
        data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY),
        api_key        = API_KEY,
        secret_key     = SECRET_KEY,
    )

    ok, reason = rm.pre_session_check()
    if not ok:
        print(f"Session aborted: {reason}")
        exit()

    # During trading loop:
    status = rm.intraday_monitor(open_positions, current_prices, portfolio_value)
    if status["action"] == "emergency_close":
        rm.emergency_close_all(open_positions, status["reason"])

REQUIREMENTS
------------
    pip install alpaca-py yfinance pandas requests
"""

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient

# Alpaca news — import path may vary by alpaca-py version; verify if ImportError
try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    _NEWS_CLIENT_AVAILABLE = True
except ImportError:
    try:
        from alpaca.data import NewsClient
        from alpaca.data.requests import NewsRequest
        _NEWS_CLIENT_AVAILABLE = True
    except ImportError:
        _NEWS_CLIENT_AVAILABLE = False

ET  = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

# ── News keyword tiers ────────────────────────────────────────────────────────
# CRITICAL → emergency close all positions immediately
_CRITICAL_KEYWORDS = frozenset([
    "market halt", "trading halt", "exchange closed", "exchange halt",
    "circuit breaker", "trading suspended", "market suspended",
    "emergency rate cut", "emergency fed", "systemic risk",
    "financial crisis", "market closure", "nyse halt", "nasdaq halt",
])

# WARNING → stop opening new positions, keep monitoring existing ones
_WARNING_KEYWORDS = frozenset([
    "bankruptcy", "chapter 11", "fraud", "accounting fraud",
    "sec charges", "sec investigation", "doj investigation", "doj charges",
    "fda rejection", "fda hold", "clinical hold",
    "geopolitical", "military action", "war", "terrorist attack",
    "pandemic", "outbreak", "public health emergency",
    "natural disaster", "earthquake", "hurricane",
    "credit default", "sovereign default", "debt ceiling",
    "margin call", "liquidity crisis", "bank run",
])


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """All risk thresholds in one place. Tune these first when live results differ."""

    # ── Pre-session gate ───────────────────────────────────────────────────────
    vix_max: float = 25.0
    """Don't trade if VIX opens above this. Above 25, intraday mean-reversion
    strategies historically show significantly negative expected value."""

    spy_daily_drop_max: float = -0.015
    """Don't trade if SPY is down more than this from yesterday's close.
    -0.015 = -1.5%. Broad market selling contaminates individual setups."""

    premarket_news_lookback_min: int = 60
    """Minutes of news history to scan at session start."""

    # ── Intraday circuit breakers ──────────────────────────────────────────────
    vix_intraday_spike: float = 0.15
    """If VIX rises more than this % from session-open VIX, halt new entries.
    0.15 = 15% intraday spike (e.g. VIX goes from 18 → 20.7)."""

    max_daily_drawdown_pct: float = 0.020
    """Close all positions and stop trading if portfolio drops this much from
    session-open value. 0.02 = 2.0%."""

    correlation_close_threshold: float = 0.75
    """If more than this fraction of open positions are simultaneously adverse
    (moving against us), trigger emergency close. Classic sign of macro shock."""

    news_scan_interval_sec: int = 60
    """How often to scan the news feed during active trading."""

    # ── Position concentration limits ─────────────────────────────────────────
    max_positions_per_sector: int = 3
    """Maximum simultaneous positions in any single GICS sector."""

    max_total_positions: int = 12
    """Hard cap across all strategies simultaneously."""

    max_portfolio_deployed_pct: float = 0.30
    """Maximum fraction of portfolio in open positions at any one time. 0.30 = 30%."""

    # ── Risk event log ─────────────────────────────────────────────────────────
    event_log_path: str = "output/risk_events.log"

    # ── Sector cache ──────────────────────────────────────────────────────────
    universe_cache_path: str = "output/universe_cache.json"


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Central risk authority for all trading strategies.

    Every strategy must call pre_session_check() before opening positions,
    and intraday_monitor() on a regular loop while positions are open.
    The emergency_close_all() method is the last line of defence.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        api_key: str,
        secret_key: str,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self.tc         = trading_client
        self.dc         = data_client
        self.api_key    = api_key
        self.secret_key = secret_key
        self.cfg        = config or RiskConfig()
        self.log        = logging.getLogger(self.__class__.__name__)

        # Session state — populated by pre_session_check()
        self.session_start_value:  Optional[float] = None
        self.session_start_vix:    Optional[float] = None
        self.session_halted:       bool            = False
        self.new_entries_halted:   bool            = False
        self.halt_reason:          str             = ""

        # News client
        self._news_client: Optional[object] = None
        if _NEWS_CLIENT_AVAILABLE:
            try:
                self._news_client = NewsClient(api_key, secret_key)
            except Exception as exc:
                self.log.warning(f"News client init failed: {exc}")

        # Sector map
        self._sector_map: dict[str, str] = self._load_sector_map()

    # =========================================================================
    # LAYER 1 — PRE-SESSION GATE
    # =========================================================================

    def pre_session_check(self) -> tuple[bool, str]:
        """
        Run all pre-session checks. Call this at 9:25 AM ET.

        Returns
        -------
        (ok_to_trade: bool, reason: str)
            If ok_to_trade is False, do not open any positions today.
        """
        self.log.info("Running pre-session risk checks...")

        # Record starting portfolio value for daily drawdown tracking
        try:
            account = self.tc.get_account()
            self.session_start_value = float(account.portfolio_value)
            self.log.info(f"Session start value: ${self.session_start_value:,.2f}")
        except Exception as exc:
            return False, f"Could not fetch account value: {exc}"

        # ── VIX check ─────────────────────────────────────────────────────────
        vix_ok, vix_level, _ = self.check_vix()
        self.session_start_vix = vix_level
        if not vix_ok:
            reason = f"VIX={vix_level:.1f} exceeds threshold {self.cfg.vix_max}"
            self._log_event("PRE_SESSION_ABORT", reason)
            return False, reason

        # ── Market direction check ─────────────────────────────────────────────
        mkt_ok, spy_change = self.check_market_direction()
        if not mkt_ok:
            reason = f"SPY day change {spy_change*100:.2f}% below threshold {self.cfg.spy_daily_drop_max*100:.1f}%"
            self._log_event("PRE_SESSION_ABORT", reason)
            return False, reason

        # ── News check ────────────────────────────────────────────────────────
        news_ok, alerts = self.scan_news(lookback_min=self.cfg.premarket_news_lookback_min)
        if not news_ok:
            reason = f"Critical news detected: {'; '.join(alerts[:3])}"
            self._log_event("PRE_SESSION_ABORT", reason)
            return False, reason

        self.log.info(
            f"Pre-session checks PASSED | VIX={vix_level:.1f} | "
            f"SPY={spy_change*100:+.2f}% | News=clear"
        )
        return True, "ok"

    # =========================================================================
    # LAYER 2 — INTRADAY MONITORING
    # =========================================================================

    def intraday_monitor(
        self,
        open_positions: list[dict],
        current_prices: dict[str, float],
        portfolio_value: float,
    ) -> dict:
        """
        Called every 5 minutes during the trading session.

        Parameters
        ----------
        open_positions : list[dict]
            Each dict must have: symbol, entry_price, qty, direction ('long'/'short').
        current_prices : dict[str, float]
            Latest price for each symbol in open_positions.
        portfolio_value : float
            Current total portfolio value.

        Returns
        -------
        dict with keys:
            'action'  : 'continue' | 'halt_entries' | 'emergency_close'
            'reason'  : str explanation
            'details' : dict of individual check results
        """
        if self.session_halted:
            return {"action": "emergency_close", "reason": self.halt_reason, "details": {}}

        details: dict = {}

        # ── Portfolio drawdown ────────────────────────────────────────────────
        if self.session_start_value:
            dd = (portfolio_value - self.session_start_value) / self.session_start_value
            details["drawdown_pct"] = dd
            if dd <= -self.cfg.max_daily_drawdown_pct:
                reason = f"Daily drawdown {dd*100:.2f}% hit limit {self.cfg.max_daily_drawdown_pct*100:.1f}%"
                self._log_event("EMERGENCY_CLOSE", reason)
                self.session_halted = True
                self.halt_reason    = reason
                return {"action": "emergency_close", "reason": reason, "details": details}

        # ── VIX intraday spike ────────────────────────────────────────────────
        vix_ok, vix_now, vix_change = self.check_vix()
        details["vix"]        = vix_now
        details["vix_change"] = vix_change

        if not vix_ok or (
            self.session_start_vix and
            vix_change >= self.cfg.vix_intraday_spike
        ):
            reason = f"VIX spike: {vix_now:.1f} (+{vix_change*100:.1f}% from session open)"
            self._log_event("HALT_ENTRIES", reason)
            self.new_entries_halted = True
            self.halt_reason        = reason
            details["vix_spike"]    = True
            # Spike alone doesn't force close — existing positions keep their stops
            # But if VIX exceeds absolute max, close everything
            if vix_now > self.cfg.vix_max * 1.5:
                self.session_halted = True
                return {"action": "emergency_close", "reason": reason, "details": details}
            return {"action": "halt_entries", "reason": reason, "details": details}

        # ── Correlation collapse ───────────────────────────────────────────────
        if open_positions:
            collapse, pct_adverse = self._check_correlation_collapse(
                open_positions, current_prices
            )
            details["pct_adverse"]        = pct_adverse
            details["correlation_collapse"] = collapse
            if collapse:
                reason = (
                    f"Correlation collapse: {pct_adverse*100:.0f}% of positions adverse "
                    f"(threshold {self.cfg.correlation_close_threshold*100:.0f}%)"
                )
                self._log_event("EMERGENCY_CLOSE", reason)
                self.session_halted = True
                self.halt_reason    = reason
                return {"action": "emergency_close", "reason": reason, "details": details}

        # ── News scan ─────────────────────────────────────────────────────────
        news_ok, alerts = self.scan_news(lookback_min=5)
        details["news_alerts"] = alerts
        if not news_ok:
            reason = f"Critical news: {'; '.join(alerts[:2])}"
            self._log_event("EMERGENCY_CLOSE", reason)
            self.session_halted = True
            self.halt_reason    = reason
            return {"action": "emergency_close", "reason": reason, "details": details}
        if alerts:  # WARNING tier only
            reason = f"News warning: {alerts[0]}"
            self._log_event("HALT_ENTRIES", reason)
            self.new_entries_halted = True
            return {"action": "halt_entries", "reason": reason, "details": details}

        # Restore entry permission if all checks pass
        if self.new_entries_halted:
            self.log.info("Risk conditions normalised — resuming new entries.")
            self.new_entries_halted = False
            self.halt_reason        = ""

        return {"action": "continue", "reason": "all checks passed", "details": details}

    # =========================================================================
    # INDIVIDUAL CHECKS
    # =========================================================================

    def check_vix(self) -> tuple[bool, float, float]:
        """
        Fetch current VIX via yfinance.

        Returns
        -------
        (below_threshold: bool, current_vix: float, pct_change_from_session_open: float)
        """
        try:
            vix_df = yf.download("^VIX", period="2d", interval="1m", progress=False)
            if vix_df.empty:
                self.log.warning("VIX data unavailable — assuming safe.")
                return True, 0.0, 0.0
            current_vix = float(vix_df["Close"].iloc[-1])
            pct_change  = (
                (current_vix - self.session_start_vix) / self.session_start_vix
                if self.session_start_vix else 0.0
            )
            ok = current_vix < self.cfg.vix_max
            return ok, current_vix, pct_change
        except Exception as exc:
            self.log.warning(f"VIX check failed: {exc}")
            return True, 0.0, 0.0  # fail-open to avoid blocking on data outage

    def check_market_direction(self) -> tuple[bool, float]:
        """
        Compare SPY's current price to its previous session close.

        Returns
        -------
        (above_threshold: bool, pct_change: float)
        """
        try:
            snaps = self.dc.get_stock_snapshots(
                StockSnapshotRequest(symbol_or_symbols=["SPY"])
            )
            spy = snaps.get("SPY")
            if not spy or not spy.daily_bar or not spy.prev_daily_bar:
                return True, 0.0
            change = (spy.daily_bar.close - spy.prev_daily_bar.close) / spy.prev_daily_bar.close
            return change > self.cfg.spy_daily_drop_max, change
        except Exception as exc:
            self.log.warning(f"Market direction check failed: {exc}")
            return True, 0.0

    def scan_news(
        self,
        symbols: Optional[list[str]] = None,
        lookback_min: int = 5,
    ) -> tuple[bool, list[str]]:
        """
        Scan recent news for critical and warning keywords.

        Returns
        -------
        (all_clear: bool, alerts: list[str])
            all_clear is False only for CRITICAL keywords.
            alerts contains both critical and warning headlines.
        """
        since = datetime.datetime.now(ET) - datetime.timedelta(minutes=lookback_min)
        headlines: list[str] = []

        # ── Alpaca news ───────────────────────────────────────────────────────
        if self._news_client:
            try:
                req = NewsRequest(
                    symbols  = symbols,
                    start    = since,
                    limit    = 50,
                )
                articles = self._news_client.get_news(req)
                for article in (articles.news if hasattr(articles, "news") else articles):
                    headlines.append(article.headline.lower())
            except Exception as exc:
                self.log.debug(f"Alpaca news scan error: {exc}")

        # ── Fallback: Yahoo Finance RSS for broad market news ─────────────────
        if not headlines:
            try:
                rss = requests.get(
                    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
                    timeout=5
                ).text
                import re
                titles = re.findall(r"<title>(.*?)</title>", rss, re.DOTALL)
                headlines = [t.lower().strip() for t in titles[2:22]]  # skip feed title
            except Exception:
                pass

        # ── Classify headlines ────────────────────────────────────────────────
        critical_hits: list[str] = []
        warning_hits:  list[str] = []

        for hl in headlines:
            if any(kw in hl for kw in _CRITICAL_KEYWORDS):
                critical_hits.append(hl[:80])
            elif any(kw in hl for kw in _WARNING_KEYWORDS):
                warning_hits.append(hl[:80])

        all_alerts = critical_hits + warning_hits
        all_clear  = len(critical_hits) == 0

        if critical_hits:
            self.log.critical(f"CRITICAL NEWS: {critical_hits[0]}")
        elif warning_hits:
            self.log.warning(f"News warning: {warning_hits[0]}")

        return all_clear, all_alerts

    def check_ok_to_enter(
        self,
        symbol: str,
        open_positions: list[dict],
        proposed_notional: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        Quick pre-trade check before opening any individual position.
        Call this immediately before submitting an order.

        Returns
        -------
        (ok: bool, reason: str)
        """
        if self.session_halted:
            return False, "Session halted"
        if self.new_entries_halted:
            return False, f"New entries halted: {self.halt_reason}"

        # Total position count
        open_count = sum(1 for p in open_positions if not p.get("closed", False))
        if open_count >= self.cfg.max_total_positions:
            return False, f"Max positions ({self.cfg.max_total_positions}) reached"

        # Portfolio deployment cap
        total_notional = sum(
            p.get("qty", 0) * p.get("entry_price", 0)
            for p in open_positions if not p.get("closed", False)
        )
        if (total_notional + proposed_notional) / portfolio_value > self.cfg.max_portfolio_deployed_pct:
            return False, f"Portfolio deployment cap ({self.cfg.max_portfolio_deployed_pct*100:.0f}%) reached"

        # Sector concentration
        sector = self.get_sector(symbol)
        sector_count = sum(
            1 for p in open_positions
            if not p.get("closed", False) and self.get_sector(p["symbol"]) == sector
        )
        if sector_count >= self.cfg.max_positions_per_sector:
            return False, f"Sector '{sector}' at max ({self.cfg.max_positions_per_sector} positions)"

        return True, "ok"

    # =========================================================================
    # EMERGENCY CLOSE
    # =========================================================================

    def emergency_close_all(self, open_positions: list[dict], reason: str) -> None:
        """
        Close every open position immediately via market order.
        Uses Alpaca's close_all_positions() for speed, then marks each
        position dict as closed.
        """
        self.log.critical(f"EMERGENCY CLOSE ALL — reason: {reason}")
        self._log_event("EMERGENCY_CLOSE_ALL", reason)
        try:
            self.tc.close_all_positions(cancel_orders=True)
            self.log.info("close_all_positions() submitted successfully.")
        except Exception as exc:
            self.log.error(f"close_all_positions() failed: {exc}")
            # Fallback: close each individually
            for pos in open_positions:
                if not pos.get("closed", False):
                    try:
                        self.tc.close_position(pos["symbol"])
                    except Exception as e2:
                        self.log.error(f"Individual close failed for {pos['symbol']}: {e2}")

        for pos in open_positions:
            if not pos.get("closed", False):
                pos["closed"]      = True
                pos["exit_reason"] = f"emergency_close: {reason[:40]}"

        self.session_halted = True

    # =========================================================================
    # SECTOR / CONCENTRATION UTILITIES
    # =========================================================================

    def get_sector(self, symbol: str) -> str:
        """Return GICS sector for symbol, or 'Unknown' if not in map."""
        return self._sector_map.get(symbol.upper(), "Unknown")

    def _load_sector_map(self) -> dict[str, str]:
        """
        Load sector data from universe cache first, then Wikipedia fallback.
        Returns dict mapping ticker → GICS sector name.
        """
        # Try universe cache
        cache_path = Path(self.cfg.universe_cache_path)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    payload = json.load(f)
                # Only trust the cache when it actually contains sectors. An empty
                # map would make every symbol 'Unknown' and silently break the
                # sector-concentration cap, so fall through to Wikipedia instead.
                if payload.get("sector_map"):
                    self.log.info(
                        f"Sector map loaded from cache ({len(payload['sector_map'])} stocks)."
                    )
                    return payload["sector_map"]
            except Exception:
                pass

        # Fallback: scrape Wikipedia S&P 500 table
        self.log.info("Building sector map from Wikipedia...")
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            df = tables[0][["Symbol", "GICS Sector"]].dropna()
            sector_map = {
                row["Symbol"].replace(".", "-"): row["GICS Sector"]
                for _, row in df.iterrows()
            }
            self.log.info(f"Sector map built: {len(sector_map)} stocks.")
            return sector_map
        except Exception as exc:
            self.log.warning(f"Sector map unavailable: {exc}")
            return {}

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _check_correlation_collapse(
        self,
        open_positions: list[dict],
        current_prices: dict[str, float],
    ) -> tuple[bool, float]:
        """
        Check if an unusual fraction of positions are simultaneously losing.
        Returns (is_collapsing: bool, fraction_adverse: float).
        """
        active = [p for p in open_positions if not p.get("closed", False)]
        if len(active) < 3:  # too few positions to detect a pattern
            return False, 0.0

        adverse = 0
        for pos in active:
            sym   = pos["symbol"]
            price = current_prices.get(sym, pos["entry_price"])
            if pos.get("direction", "long") == "long":
                if price < pos["entry_price"]:
                    adverse += 1
            else:  # short
                if price > pos["entry_price"]:
                    adverse += 1

        frac = adverse / len(active)
        return frac >= self.cfg.correlation_close_threshold, frac

    def _log_event(self, event_type: str, detail: str) -> None:
        """Append a timestamped risk event to the event log file."""
        path = Path(self.cfg.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts  = datetime.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
        msg = f"[{ts}] [{event_type}] {detail}\n"
        try:
            with open(path, "a") as f:
                f.write(msg)
        except Exception:
            pass  # logging failure must never crash the strategy
        self.log.warning(f"RISK EVENT [{event_type}]: {detail}")

    @property
    def can_enter(self) -> bool:
        """True if new position entries are currently permitted."""
        return not self.session_halted and not self.new_entries_halted


# =============================================================================
# PUBLIC HELPER: add sector map to universe cache
# =============================================================================

def build_and_cache_sector_map(cache_path: str = "output/universe_cache.json") -> dict[str, str]:
    """
    Fetch S&P 500 sector data from Wikipedia and write it into the
    universe cache JSON.  Called by stock_universe.py after building
    a fresh universe.  Call manually if the cache exists but sector
    data is missing.
    """
    path = Path(cache_path)
    try:
        tables     = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df         = tables[0][["Symbol", "GICS Sector"]].dropna()
        sector_map = {row["Symbol"].replace(".", "-"): row["GICS Sector"] for _, row in df.iterrows()}
    except Exception as exc:
        log.warning(f"Could not build sector map: {exc}")
        return {}

    if path.exists():
        try:
            with open(path) as f:
                payload = json.load(f)
            payload["sector_map"] = sector_map
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            log.info(f"Sector map ({len(sector_map)} stocks) written to universe cache.")
        except Exception as exc:
            log.warning(f"Could not update cache with sector map: {exc}")

    return sector_map
