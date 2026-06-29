"""
rebuild_universe.py
-------------------
Daily scheduled task to force-rebuild the scored stock universe and refresh
the sector map in the cache.

PythonAnywhere scheduled task: Every day at 7:00 AM ET (12:00 UTC summer /
13:00 UTC winter). This runs before the trading session (run_session.py starts
at 7:55 AM). The MIN_REBUILD_AGE_DAYS constant controls how often a full
rebuild actually happens — set to 1 for daily, 7 for weekly.

PYTHONANYWHERE SETUP
--------------------
  Tasks → Scheduled → Add:
    Time: 12:00 UTC (every day)
    Command: python /home/<username>/Stock-Trading-Framework/execution/rebuild_universe.py

What this script does
---------------------
  1. Force-rebuilds the universe ignoring any cached data
     (UniverseSelector.get_universe(use_cache=False)).
  2. Calls build_and_cache_sector_map() to write fresh GICS sector data
     into the same cache file — RiskManager reads sector_map directly
     from that JSON without an additional Wikipedia scrape.
  3. Logs how many tickers were scored and cached.
  4. Prints the top 20 scored stocks so you can review quality weekly.
"""

import datetime
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")

# How many days old the cache must be before a rebuild is triggered.
# PythonAnywhere is scheduled daily — set to 1 for true daily, 7 for weekly.
MIN_REBUILD_AGE_DAYS = 1


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _setup_logging() -> None:
    log_dir = Path(__file__).parent.parent / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rebuild_{datetime.date.today().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _setup_logging()
    log = logging.getLogger("rebuild_universe")

    log.info("=" * 60)
    log.info(f"  Universe Rebuild — {datetime.date.today()}")
    log.info("=" * 60)

    # ── Load API credentials ──────────────────────────────────────────────────
    try:
        import config
        API_KEY    = config.API_KEY
        SECRET_KEY = config.SECRET_KEY
    except ImportError:
        import os
        API_KEY    = os.environ.get("ALPACA_API_KEY", "")
        SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set. Edit config.py or set environment variables.")
        sys.exit(1)

    # ── Initialise data client ────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    from risk.risk_manager import build_and_cache_sector_map

    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # top_n=150 gives the strategies a large pool to choose from
    ucfg     = UniverseConfig(top_n=150)
    selector = UniverseSelector(dc, ucfg)

    # ── Check cache freshness before doing the expensive rebuild ──────────────
    _cache = Path(ucfg.cache_path)
    if _cache.exists():
        age_days = (datetime.datetime.now().timestamp() - _cache.stat().st_mtime) / 86400
        if age_days < MIN_REBUILD_AGE_DAYS:
            log.info(
                f"Cache is {age_days:.1f}d old (threshold={MIN_REBUILD_AGE_DAYS}d) — skipping rebuild."
            )
            sys.exit(0)
        log.info(f"Cache is {age_days:.1f}d old — proceeding with rebuild.")
    else:
        log.info("No cache found — running initial build.")

    # ── Force full rebuild (ignores cache TTL) ────────────────────────────────
    log.info("Starting full universe rebuild (use_cache=False)...")
    log.info("This fetches ~30 days of daily bars for ~600 S&P500 + NDX100 tickers.")
    log.info("Expect 1–3 minutes depending on network latency.")

    scored_df = selector.build_universe(return_scores=True)

    if scored_df is None or (hasattr(scored_df, "empty") and scored_df.empty):
        log.error("Universe build returned empty results — aborting.")
        sys.exit(1)

    tickers = scored_df["symbol"].tolist()
    log.info(f"Scored {len(scored_df)} candidates → selected top {len(tickers)} tickers.")

    # Save to cache (tickers only — sector_map added in next step)
    selector._save_cache(tickers)
    log.info(f"Universe cached to '{ucfg.cache_path}'.")

    # ── Refresh sector map in cache ───────────────────────────────────────────
    log.info("Building and caching sector map from Wikipedia S&P 500 table...")
    sector_map = build_and_cache_sector_map(cache_path=ucfg.cache_path)
    if sector_map:
        log.info(f"Sector map written: {len(sector_map)} stocks classified.")
    else:
        log.warning("Sector map empty — Wikipedia scrape may have failed.")

    # ── Print top 20 for weekly review ────────────────────────────────────────
    log.info("Top 20 scored stocks for this week:")
    selector.print_top_n(scored_df, n=20)

    print("\n" + "=" * 60)
    print(f"  Rebuild complete: {len(tickers)} tickers cached")
    print(f"  Sector map:       {len(sector_map)} stocks")
    print(f"  Cache path:       {ucfg.cache_path}")
    print("=" * 60)

    log.info("rebuild_universe.py complete.")


if __name__ == "__main__":
    main()
