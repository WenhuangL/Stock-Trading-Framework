"""
stock_universe.py
-----------------
Builds and caches a dynamic stock universe optimised for the EOD mean-
reversion strategy (and any future strategy that benefits from liquid,
moderately volatile, retail-sentiment-driven stocks).

THREE SELECTION CRITERIA
------------------------
1. HIGH VOLUME
   Stocks must trade enough daily volume that afterhours bid-ask spreads
   remain manageable. Low-volume stocks have wide spreads that consume the
   thin profit margins the strategy targets.
   Metric: average daily volume over the lookback window.
   Weight: 35% of composite score.

2. SWEET-SPOT VOLATILITY
   Too little volatility → rarely generates qualifying 1%+ daily drops.
   Too much → moves are driven by genuine news, not noise (no reversion).
   A 20–60% annualised historical volatility range is the target zone.
   Metric: annualised 20-day historical volatility (log returns).
   Weight: 30% of composite score. Score peaks at the target HV and
   decays symmetrically on either side using a Gaussian curve, so
   stocks right in the sweet spot are strongly preferred.

3. RETAIL SENTIMENT PROXY
   Retail traders create noise-driven selling that reverts. Two measurable
   proxies for retail participation without an external sentiment feed:
   a) Relative volume: if recent volume is elevated vs the 30-day average,
      retail interest is active right now. (Weight: 15%)
   b) Price momentum: retail money chases recent winners. Stocks up 10-30%
      over the past 20 days attract the most retail eyeballs and therefore
      the most sentiment-driven selling on bad days. (Weight: 20%)

CANDIDATE POOL
--------------
S&P 500 + Nasdaq 100 (~600 unique tickers after deduplication). This gives
a large enough pool to be selective while keeping API calls manageable.
All candidates pass a basic price filter before scoring.

TICKER SOURCING
---------------
Tickers are fetched from three sources in priority order, with automatic
fallback so the universe build never silently produces zero candidates:

  1. Yahoo Finance index constituent API  (^GSPC for S&P 500)
  2. Nasdaq public API                    (nasdaq100 list endpoint)
  3. Wikipedia with spoofed User-Agent    (last resort)

The same three-source pattern is used for both the S&P 500 and Nasdaq 100
pools, so if any single source is down the others cover it.

CACHING
-------
The universe is expensive to build (requires 30-day bars for ~600 tickers).
Results are cached to a JSON file with a configurable TTL (default 7 days).
The strategy reads from cache on weekdays and rebuilds on the first run
after the TTL expires (typically Monday morning before market open).

USAGE
-----
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    from alpaca.data.historical import StockHistoricalDataClient

    dc       = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    selector = UniverseSelector(dc)

    # Returns cached list if fresh, otherwise rebuilds
    tickers = selector.get_universe()

    # Force a rebuild ignoring the cache
    tickers = selector.get_universe(use_cache=False)

    # Inspect scores and visualize
    scored_df = selector.build_universe(return_scores=True)
    selector.plot_scores(scored_df)

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy matplotlib requests
"""

import datetime
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from indicators.analyze import calculate_historical_volatility

ET  = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

# Shared browser-like headers reused across all HTTP requests in this module.
# Same pattern as scrapers.py and earnings_calendar.py — these headers are
# what prevent 403s from Yahoo, Nasdaq, and Wikipedia.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UniverseConfig:
    """All tuneable parameters for universe construction and scoring."""

    # ── Candidate filters (applied before scoring) ─────────────────────────────
    min_price: float = 10.0
    """Exclude stocks below this price. Cheap stocks have large % spreads."""

    max_price: float = 500.0
    """Exclude very expensive stocks where 1% of portfolio buys <1 share."""

    min_avg_daily_volume: int = 1_000_000
    """Minimum average daily volume. Below this, afterhours spreads are
    typically too wide for the strategy's thin TP targets."""

    # ── Volatility sweet spot ──────────────────────────────────────────────────
    target_annualized_hv: float = 0.40
    """Ideal annualised historical volatility (40%). Stocks here move enough
    to generate 1%+ daily drops regularly without being genuinely erratic."""

    hv_score_sigma: float = 0.15
    """Controls how quickly the volatility score decays away from the target.
    Gaussian width: score = exp(-0.5 * ((hv - target) / sigma)^2).
    0.15 means a stock at 25% or 55% HV still scores ~60% of maximum."""

    # ── Lookback windows ───────────────────────────────────────────────────────
    volume_lookback_days: int  = 30
    """Days of history used to compute average daily volume."""

    hv_period: int = 20
    """Days used for the rolling historical volatility calculation."""

    momentum_period: int = 20
    """Days used for the price momentum calculation (simple return)."""

    rel_volume_recent: int = 5
    """Recent window for relative volume numerator (5-day avg / 30-day avg)."""

    # ── Scoring weights (must sum to 1.0) ──────────────────────────────────────
    weight_volume:     float = 0.35
    weight_volatility: float = 0.30
    weight_momentum:   float = 0.20
    weight_rel_volume: float = 0.15

    # ── Output ────────────────────────────────────────────────────────────────
    top_n: int = 150
    """Number of top-scoring tickers to include in the final universe."""

    # ── Cache ──────────────────────────────────────────────────────────────────
    cache_path: str = "output/universe_cache.json"
    """File path for the cached universe list."""

    cache_ttl_days: int = 7
    """How many days before the cache is considered stale and rebuilt."""


# =============================================================================
# TICKER FETCHING  (replaces the old Wikipedia-only helpers)
# =============================================================================

def _normalize_tickers(raw: list[str]) -> list[str]:
    """
    Normalize a raw ticker list to Alpaca format:
      - Strip whitespace
      - Uppercase
      - Deduplicate while preserving order

    Note: Alpaca uses dots in tickers (BRK.B, BF.B), NOT dashes.
    Do NOT convert dots to dashes here — that breaks Alpaca bar requests.
    Wikipedia historically used dots; the Nasdaq/Yahoo APIs already return
    the correct Alpaca-compatible format.
    """
    seen: set[str] = set()
    result: list[str] = []
    for t in raw:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ── S&P 500 sources ───────────────────────────────────────────────────────────

def _sp500_from_yahoo() -> list[str]:
    """
    Fetch S&P 500 constituents from Yahoo Finance's index constituent endpoint.
    This is the same API pattern already used in scrapers.py.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/index/constituents"
    params = {"symbol": "^GSPC"}
    headers = {
        **_HEADERS,
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com",
    }
    try:
        resp = _SESSION.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        constituents = (
            data.get("indexConstituentStatistics", {})
                .get("constituents", [])
        )
        tickers = [c["symbol"] for c in constituents if c.get("symbol")]
        if tickers:
            log.info("S&P 500 via Yahoo Finance API: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("S&P 500 Yahoo fetch failed: %s", exc)
        return []


def _sp500_from_nasdaq_api() -> list[str]:
    """
    Fetch S&P 500 constituents from Nasdaq's public screener API.
    Same pattern as _scrape_nasdaq() in earnings_calendar.py.

    The sp500 endpoint response shape differs slightly from the nasdaq100
    endpoint: the inner 'data' key may be absent, with 'rows' sitting
    directly under the top-level 'data' key. Both shapes are handled.
    """
    url = "https://api.nasdaq.com/api/quote/list-type/sp500"
    headers = {
        **_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
        "Origin": "https://www.nasdaq.com",
    }
    try:
        resp = _SESSION.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        outer = data.get("data") or {}
        # Shape A: data -> data -> rows  (same as nasdaq100)
        # Shape B: data -> rows          (sp500 variant)
        inner = outer.get("data") or outer
        rows  = inner.get("rows") if isinstance(inner, dict) else None
        if not rows:
            rows = outer.get("rows") or []
        tickers = [r["symbol"] for r in rows if isinstance(r, dict) and r.get("symbol")]
        if tickers:
            log.info("S&P 500 via Nasdaq API: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("S&P 500 Nasdaq API fetch failed: %s", exc)
        return []


def _sp500_from_wikipedia() -> list[str]:
    """
    Last-resort S&P 500 fetch from Wikipedia using a spoofed User-Agent
    to avoid the 403 that pandas' default UA receives.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _HEADERS["User-Agent"]})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read()
        tables = pd.read_html(html)
        tickers = tables[0]["Symbol"].dropna().str.strip().tolist()
        if tickers:
            log.info("S&P 500 via Wikipedia: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("S&P 500 Wikipedia fetch failed: %s", exc)
        return []


def get_sp500_tickers() -> list[str]:
    """
    Return current S&P 500 constituents in Alpaca format.
    Tries Yahoo Finance -> Nasdaq API -> Wikipedia, returning the first
    successful non-empty result.
    """
    for fn in (_sp500_from_yahoo, _sp500_from_nasdaq_api, _sp500_from_wikipedia):
        tickers = _normalize_tickers(fn())
        if tickers:
            return tickers
    log.error("All S&P 500 sources failed — returning empty list.")
    return []


# ── Nasdaq 100 sources ────────────────────────────────────────────────────────

def _ndx100_from_nasdaq_api() -> list[str]:
    """
    Fetch Nasdaq-100 constituents from Nasdaq's public list-type endpoint.
    Most reliable source for NDX100; same auth pattern as earnings_calendar.py.
    """
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    headers = {
        **_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
        "Origin": "https://www.nasdaq.com",
    }
    try:
        resp = _SESSION.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = (
            data.get("data", {})
                .get("data", {})
                .get("rows", [])
        )
        tickers = [r["symbol"] for r in rows if r.get("symbol")]
        if tickers:
            log.info("Nasdaq-100 via Nasdaq API: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("Nasdaq-100 Nasdaq API fetch failed: %s", exc)
        return []


def _ndx100_from_yahoo() -> list[str]:
    """
    Fetch Nasdaq-100 constituents from Yahoo Finance's index constituent endpoint.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/index/constituents"
    params = {"symbol": "^NDX"}
    headers = {
        **_HEADERS,
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com",
    }
    try:
        resp = _SESSION.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        constituents = (
            data.get("indexConstituentStatistics", {})
                .get("constituents", [])
        )
        tickers = [c["symbol"] for c in constituents if c.get("symbol")]
        if tickers:
            log.info("Nasdaq-100 via Yahoo Finance API: %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("Nasdaq-100 Yahoo fetch failed: %s", exc)
        return []


def _ndx100_from_wikipedia() -> list[str]:
    """
    Last-resort Nasdaq-100 fetch from Wikipedia with spoofed User-Agent.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _HEADERS["User-Agent"]})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read()
        tables = pd.read_html(html)
        # The constituents table is the one with a "Ticker" column
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].dropna().str.strip().tolist()
                if tickers:
                    log.info("Nasdaq-100 via Wikipedia: %d tickers", len(tickers))
                    return tickers
        return []
    except Exception as exc:
        log.warning("Nasdaq-100 Wikipedia fetch failed: %s", exc)
        return []


def get_nasdaq100_tickers() -> list[str]:
    """
    Return current Nasdaq-100 constituents in Alpaca format.
    Tries Nasdaq API -> Yahoo Finance -> Wikipedia.
    """
    for fn in (_ndx100_from_nasdaq_api, _ndx100_from_yahoo, _ndx100_from_wikipedia):
        tickers = _normalize_tickers(fn())
        if tickers:
            return tickers
    log.error("All Nasdaq-100 sources failed — returning empty list.")
    return []


# ── Combined candidate pool ───────────────────────────────────────────────────

def get_candidate_pool() -> list[str]:
    """
    Return the union of S&P 500 and Nasdaq-100, deduplicated and normalized.
    This ~600-ticker pool is large enough to be selective while keeping
    API calls to roughly 6-7 batch calls for daily bars.

    Each index is fetched independently (with its own fallback chain) so a
    failure on one does not affect the other.
    """
    sp500  = get_sp500_tickers()
    ndx100 = get_nasdaq100_tickers()
    combined = list(dict.fromkeys(sp500 + ndx100))  # preserves order, deduplicates
    log.info(
        "Candidate pool: %d S&P500 + %d NDX100 = %d unique tickers",
        len(sp500), len(ndx100), len(combined),
    )
    return combined


# =============================================================================
# UNIVERSE SELECTOR
# =============================================================================

class UniverseSelector:
    """
    Scores and ranks a candidate pool of stocks against the three selection
    criteria, caching the result to avoid rebuilding daily.
    """

    def __init__(
        self,
        data_client: StockHistoricalDataClient,
        config: Optional[UniverseConfig] = None,
    ) -> None:
        self.dc  = data_client
        self.cfg = config or UniverseConfig()
        self.log = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_universe(self, use_cache: bool = True) -> list[str]:
        """
        Main entry point. Returns the scored universe as a list of tickers,
        highest-scoring first.

        If use_cache=True and a valid (non-expired) cache exists, the cached
        list is returned immediately without any API calls. Otherwise, a full
        rebuild is triggered (takes 1-3 minutes depending on pool size).

        Parameters
        ----------
        use_cache : bool
            Set False to force a rebuild even if the cache is fresh.

        Returns
        -------
        list[str]
            Top-N ticker symbols ranked by composite score.
        """
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self.log.info("Using cached universe: %d tickers.", len(cached))
                return cached

        self.log.info("Building fresh universe (this may take a minute)...")
        scored_df = self.build_universe(return_scores=True)
        tickers   = scored_df["symbol"].tolist()
        self._save_cache(tickers)
        # Refresh sector data in cache so RiskManager can load it without
        # a separate Wikipedia scrape.
        try:
            from risk.risk_manager import build_and_cache_sector_map
            build_and_cache_sector_map(cache_path=self.cfg.cache_path)
        except Exception as exc:
            self.log.warning("Sector map update failed: %s", exc)
        return tickers

    def build_universe(self, return_scores: bool = False) -> "list[str] | pd.DataFrame":
        """
        Full rebuild: fetch bars, calculate metrics, score, rank, return.

        Parameters
        ----------
        return_scores : bool
            If True, returns the full scored DataFrame (useful for inspection
            and plotting). If False, returns a plain list of tickers.

        Returns
        -------
        list[str] or pd.DataFrame
        """
        candidates = get_candidate_pool()
        bars_data  = self._fetch_bars(candidates)

        if not bars_data:
            self.log.error("No bar data returned — cannot build universe.")
            return [] if not return_scores else pd.DataFrame()

        scored_df = self._score_tickers(bars_data)
        top       = scored_df.head(self.cfg.top_n)

        self.log.info(
            "Universe built: %d candidates scored, top %d selected.",
            len(scored_df), len(top),
        )
        return top if return_scores else top["symbol"].tolist()

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_bars(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Batch-fetch daily bars for all candidate tickers.
        Only tickers passing the basic price filter are retained.

        Returns dict mapping symbol -> DataFrame with columns
        [open, high, low, close, volume] and a DatetimeIndex in ET.

        Feed note: universe scoring only needs daily OHLCV for volume/volatility
        metrics — IEX coverage is sufficient for this purpose and works on all
        Alpaca subscription tiers. The strategies themselves always use feed="sip"
        for minute bars where afterhours coverage matters; that distinction does
        not apply here.

        Date note: end_dt is set 5 calendar days before today to stay within the
        free-tier "recent SIP data" restriction. The 30-day scoring window is
        unaffected — it still captures a full month of history.
        """
        end_dt   = datetime.datetime.now(ET) - datetime.timedelta(days=5)
        start_dt = end_dt - datetime.timedelta(days=self.cfg.volume_lookback_days + 10)

        bars_data: dict[str, pd.DataFrame] = {}
        BATCH = 100  # Alpaca handles ~100 symbols comfortably per request

        # Tickers that Alpaca has explicitly rejected as invalid (e.g. BRK.B
        # on some subscription tiers) are tracked so they don't poison retries.
        known_bad: set[str] = set()

        def _extract_bars(resp, batch):
            """Parse a bars response and populate bars_data in place."""
            try:
                full_df = resp.df
                full_df.columns = [c.lower() for c in full_df.columns]
                full_df.index = pd.MultiIndex.from_tuples(
                    [(sym, pd.Timestamp(ts).tz_convert(ET))
                     for sym, ts in full_df.index],
                    names=["symbol", "timestamp"],
                )
            except Exception:
                full_df = None

            for sym in batch:
                if sym in known_bad:
                    continue
                try:
                    if full_df is not None and sym in full_df.index.get_level_values("symbol"):
                        df = full_df.loc[sym].copy()
                    else:
                        raw = resp[sym]
                        df  = pd.DataFrame([
                            {"open": b.open, "high": b.high, "low": b.low,
                             "close": b.close, "volume": b.volume,
                             "timestamp": b.timestamp}
                            for b in raw
                        ]).set_index("timestamp")
                        df.columns = [c.lower() for c in df.columns]

                    if df.empty or len(df) < self.cfg.hv_period:
                        continue

                    last_close = float(df["close"].iloc[-1])
                    if not (self.cfg.min_price <= last_close <= self.cfg.max_price):
                        continue

                    bars_data[sym] = df

                except (KeyError, AttributeError, IndexError):
                    pass  # symbol had no data in this period

        for i in range(0, len(tickers), BATCH):
            batch = [t for t in tickers[i : i + BATCH] if t not in known_bad]
            if not batch:
                continue

            batch_n = i // BATCH + 1
            total_n = (len(tickers) - 1) // BATCH + 1

            try:
                resp = self.dc.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols = batch,
                    timeframe         = TimeFrame.Day,
                    start             = start_dt,
                    end               = end_dt,
                    feed              = "iex",
                ))
                _extract_bars(resp, batch)
                self.log.info(
                    "  Bars batch %d/%d: %d tickers retained.",
                    batch_n, total_n,
                    len([s for s in batch if s in bars_data]),
                )

            except Exception as exc:
                err_str = str(exc)
                # Alpaca returns "invalid symbol: XYZ" when a single ticker in
                # the batch is unrecognised. Identify it, blacklist it, and
                # retry the rest of the batch rather than discarding all 100.
                if "invalid symbol" in err_str.lower():
                    import re as _re
                    bad_match = _re.search(r"invalid symbol[:\s]+([A-Z.\-]+)", err_str, _re.I)
                    if bad_match:
                        bad_sym = bad_match.group(1).strip().upper()
                        known_bad.add(bad_sym)
                        self.log.warning(
                            "  Batch %d/%d: invalid symbol '%s' blacklisted, retrying batch.",
                            batch_n, total_n, bad_sym,
                        )
                        retry_batch = [t for t in batch if t != bad_sym]
                        if retry_batch:
                            try:
                                resp = self.dc.get_stock_bars(StockBarsRequest(
                                    symbol_or_symbols = retry_batch,
                                    timeframe         = TimeFrame.Day,
                                    start             = start_dt,
                                    end               = end_dt,
                                    feed              = "iex",
                                ))
                                _extract_bars(resp, retry_batch)
                                self.log.info(
                                    "  Bars batch %d/%d (retry): %d tickers retained.",
                                    batch_n, total_n,
                                    len([s for s in retry_batch if s in bars_data]),
                                )
                            except Exception as exc2:
                                self.log.warning(
                                    "  Bars batch %d/%d retry failed: %s",
                                    batch_n, total_n, exc2,
                                )
                    else:
                        self.log.warning(
                            "  Bars batch %d/%d failed (invalid symbol, unparseable): %s",
                            batch_n, total_n, exc,
                        )
                else:
                    self.log.warning(
                        "  Bars batch %d/%d failed: %s", batch_n, total_n, exc,
                    )

        self.log.info("Total tickers with usable bar data: %d", len(bars_data))
        return bars_data

    # =========================================================================
    # SCORING
    # =========================================================================

    def _score_tickers(self, bars_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate four metrics for every ticker in bars_data, normalise each
        to [0, 1] within the universe, combine with weights, and sort.

        Returns a DataFrame with columns:
            symbol, avg_volume, annualized_hv, momentum_20d, rel_volume,
            score_volume, score_volatility, score_momentum, score_rel_volume,
            composite_score
        Sorted by composite_score descending.
        """
        rows: list[dict] = []

        for sym, df in bars_data.items():
            try:
                # ── Metric 1: Average daily volume ────────────────────────────
                avg_vol = float(df["volume"].tail(self.cfg.volume_lookback_days).mean())
                if avg_vol < self.cfg.min_avg_daily_volume:
                    continue  # eliminate illiquid stocks before scoring

                # ── Metric 2: Annualised historical volatility ─────────────────
                hv_series = calculate_historical_volatility(df, period=self.cfg.hv_period)
                ann_hv    = float(hv_series.iloc[-1]) if not hv_series.isna().all() else np.nan
                if np.isnan(ann_hv) or ann_hv <= 0:
                    continue

                # ── Metric 3: Price momentum (simple return) ───────────────────
                if len(df) >= self.cfg.momentum_period + 1:
                    momentum = float(
                        (df["close"].iloc[-1] - df["close"].iloc[-self.cfg.momentum_period - 1])
                        / df["close"].iloc[-self.cfg.momentum_period - 1]
                    )
                else:
                    momentum = 0.0

                # ── Metric 4: Relative volume (recent vs baseline) ─────────────
                recent_vol   = float(df["volume"].tail(self.cfg.rel_volume_recent).mean())
                baseline_vol = float(df["volume"].tail(self.cfg.volume_lookback_days).mean())
                rel_vol = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

                rows.append({
                    "symbol":        sym,
                    "avg_volume":    avg_vol,
                    "annualized_hv": ann_hv,
                    "momentum_20d":  momentum,
                    "rel_volume":    rel_vol,
                })

            except Exception as exc:
                self.log.debug("Scoring failed for %s: %s", sym, exc)

        if not rows:
            self.log.error("No tickers survived the scoring filters.")
            return pd.DataFrame()

        df_scores = pd.DataFrame(rows)

        # ── Normalise metrics to [0, 1] within the universe ───────────────────

        # Volume: log-normalise so mega-caps don't dominate, then min-max scale
        log_vol = np.log1p(df_scores["avg_volume"])
        df_scores["score_volume"] = _minmax(log_vol)

        # Volatility: Gaussian centred on target_hv — peaks at sweet spot
        df_scores["score_volatility"] = np.exp(
            -0.5 * ((df_scores["annualized_hv"] - self.cfg.target_annualized_hv)
                    / self.cfg.hv_score_sigma) ** 2
        )
        # Already [0,1] by definition of the Gaussian with peak=1

        # Momentum: normalise raw returns, but cap extreme outliers first
        clipped_mom = df_scores["momentum_20d"].clip(
            lower=df_scores["momentum_20d"].quantile(0.05),
            upper=df_scores["momentum_20d"].quantile(0.95),
        )
        df_scores["score_momentum"] = _minmax(clipped_mom)

        # Relative volume: cap at 3x to prevent a single spike dominating
        capped_rv = df_scores["rel_volume"].clip(upper=3.0)
        df_scores["score_rel_volume"] = _minmax(capped_rv)

        # ── Composite weighted score ───────────────────────────────────────────
        df_scores["composite_score"] = (
            self.cfg.weight_volume       * df_scores["score_volume"]
            + self.cfg.weight_volatility * df_scores["score_volatility"]
            + self.cfg.weight_momentum   * df_scores["score_momentum"]
            + self.cfg.weight_rel_volume * df_scores["score_rel_volume"]
        )

        return df_scores.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # =========================================================================
    # CACHE
    # =========================================================================

    def _cache_path(self) -> Path:
        return Path(self.cfg.cache_path)

    def _save_cache(self, tickers: list[str], sector_map: Optional[dict] = None) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve an existing sector_map in the file if one isn't provided
        existing_sector_map: dict = {}
        if sector_map is None and path.exists():
            try:
                with open(path) as f:
                    old = json.load(f)
                existing_sector_map = old.get("sector_map", {})
            except Exception:
                pass

        payload = {
            "built_at":   datetime.datetime.now(ET).isoformat(),
            "ttl_days":   self.cfg.cache_ttl_days,
            "tickers":    tickers,
            "sector_map": sector_map if sector_map is not None else existing_sector_map,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.log.info("Universe cached to '%s' (%d tickers).", path, len(tickers))

    def _load_cache(self) -> Optional[list[str]]:
        """
        Load cached tickers if the cache file exists and is within TTL.
        Returns None if cache is missing, corrupt, or expired.
        """
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
            built_at = datetime.datetime.fromisoformat(payload["built_at"])
            age_days = (datetime.datetime.now(ET) - built_at).days
            if age_days > self.cfg.cache_ttl_days:
                self.log.info(
                    "Cache expired (%d days old, TTL=%d).",
                    age_days, self.cfg.cache_ttl_days,
                )
                return None
            self.log.info(
                "Cache valid (built %dd ago): %d tickers.",
                age_days, len(payload["tickers"]),
            )
            return payload["tickers"]
        except Exception as exc:
            self.log.warning("Cache load failed: %s", exc)
            return None

    def clear_cache(self) -> None:
        """Delete the cache file, forcing a rebuild on next get_universe() call."""
        path = self._cache_path()
        if path.exists():
            path.unlink()
            self.log.info("Universe cache cleared.")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def plot_scores(self, scored_df: pd.DataFrame, top_n: int = 40) -> None:
        """
        Four-panel chart showing the score breakdown for the top-N stocks.

        Panels:
          1. Composite score bar chart (top 40 stocks, colour = composite score)
          2. Volatility distribution with target HV marked
          3. Volume vs volatility scatter, points coloured by composite score
          4. Score component breakdown heatmap for top 20 stocks
        """
        if scored_df.empty:
            print("No scored data to plot.")
            return

        top   = scored_df.head(top_n).copy()
        top20 = scored_df.head(20).copy()

        fig = plt.figure(figsize=(16, 12))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        # ── 1. Composite score bar chart ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        colours = plt.cm.RdYlGn(top["composite_score"].values)
        ax1.bar(top["symbol"], top["composite_score"], color=colours, edgecolor="white")
        ax1.axhline(
            scored_df["composite_score"].median(),
            color="gray", linewidth=0.8, linestyle="--", label="Median score"
        )
        ax1.set_title(f"Top {top_n} Stocks by Composite Score", fontsize=12)
        ax1.set_ylabel("Composite Score")
        ax1.tick_params(axis="x", rotation=70, labelsize=7)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # ── 2. HV distribution ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(
            scored_df["annualized_hv"] * 100,
            bins=30, color="#1f77b4", edgecolor="white", alpha=0.8,
        )
        ax2.axvline(
            self.cfg.target_annualized_hv * 100,
            color="#d62728", linewidth=1.5, linestyle="--",
            label=f"Target HV {self.cfg.target_annualized_hv*100:.0f}%",
        )
        ax2.set_title("Historical Volatility Distribution (Full Universe)")
        ax2.set_xlabel("Annualised HV (%)")
        ax2.set_ylabel("Number of Stocks")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # ── 3. Volume vs volatility scatter ───────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        sc = ax3.scatter(
            scored_df["annualized_hv"] * 100,
            np.log10(scored_df["avg_volume"]),
            c=scored_df["composite_score"],
            cmap="RdYlGn", s=20, alpha=0.7,
        )
        ax3.axvline(
            self.cfg.target_annualized_hv * 100,
            color="#d62728", linewidth=1.0, linestyle="--", alpha=0.6,
        )
        plt.colorbar(sc, ax=ax3, label="Composite Score")
        ax3.set_title("Volume vs Volatility (colour = score)")
        ax3.set_xlabel("Annualised HV (%)")
        ax3.set_ylabel("Log10(Avg Daily Volume)")
        ax3.grid(alpha=0.3)

        plt.suptitle("Universe Selector — Score Analysis", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

        # ── 4. Score component heatmap (separate figure for clarity) ──────────
        heatmap_cols = [
            "score_volume", "score_volatility",
            "score_momentum", "score_rel_volume", "composite_score",
        ]
        fig2, ax4 = plt.subplots(figsize=(10, 7))
        hm_data = top20[heatmap_cols].set_index(top20["symbol"]).T
        im = ax4.imshow(hm_data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax4.set_xticks(range(len(top20)))
        ax4.set_xticklabels(top20["symbol"], rotation=60, ha="right", fontsize=8)
        ax4.set_yticks(range(len(heatmap_cols)))
        ax4.set_yticklabels(
            ["Volume", "Volatility", "Momentum", "Rel. Volume", "COMPOSITE"],
            fontsize=9,
        )
        plt.colorbar(im, ax=ax4, label="Score [0-1]")
        ax4.set_title("Score Component Heatmap — Top 20 Stocks", fontsize=11)
        plt.tight_layout()
        plt.show()

    def print_top_n(self, scored_df: pd.DataFrame, n: int = 20) -> None:
        """Print a formatted table of the top-N scored stocks."""
        if scored_df.empty:
            print("No scored data.")
            return
        top = scored_df.head(n)
        print(f"\n{'='*76}")
        print(f"  Universe Top {n}  —  Composite Score Breakdown")
        print(f"{'='*76}")
        print(f"  {'#':>3}  {'Ticker':<7}  {'Comp':>6}  {'Vol':>6}  "
              f"{'HV%':>6}  {'Momt':>6}  {'RelVol':>6}  {'AvgVol':>12}  {'AnnHV':>8}")
        print(f"  {'─'*3}  {'─'*7}  {'─'*6}  {'─'*6}  "
              f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*8}")
        for i, row in top.iterrows():
            print(
                f"  {i+1:>3}  {row['symbol']:<7}  "
                f"{row['composite_score']:>6.3f}  "
                f"{row['score_volume']:>6.3f}  "
                f"{row['score_volatility']:>6.3f}  "
                f"{row['score_momentum']:>6.3f}  "
                f"{row['score_rel_volume']:>6.3f}  "
                f"{row['avg_volume']:>12,.0f}  "
                f"{row['annualized_hv']*100:>7.1f}%"
            )
        print(f"{'='*76}\n")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]. Returns 0.5 if all values equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# =============================================================================
# QUICK-START
# =============================================================================

if __name__ == "__main__":
    API_KEY    = "YOUR_API_KEY"
    SECRET_KEY = "YOUR_SECRET_KEY"

    from alpaca.data.historical import StockHistoricalDataClient

    dc       = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    selector = UniverseSelector(dc, UniverseConfig())

    # Build and inspect (skip cache so we see the full scoring output)
    scored_df = selector.build_universe(return_scores=True)

    selector.print_top_n(scored_df, n=30)
    selector.plot_scores(scored_df, top_n=40)

    # Save to cache for the strategy to consume
    selector._save_cache(scored_df["symbol"].tolist())