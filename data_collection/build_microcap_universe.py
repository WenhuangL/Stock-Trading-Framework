"""
build_microcap_universe.py
--------------------------
Offline builder for the Point-In-Time (PIT) MICRO-CAP universe consumed by
strategies/strategy_microcap_reversion.py.

Why this is separate from build_historical_universes.py
-------------------------------------------------------
The existing builder screens the S&P 500 / 400 index membership with a $10 price
FLOOR and a *share*-volume filter — the exact opposite of what a micro-cap
strategy needs.  This builder instead:

  * draws its candidate pool from the full set of tradeable US equities
    (Alpaca get_all_assets), not an index;
  * screens on a DOLLAR-volume floor (close x volume) — the only liquidity metric
    that means anything in a thin name — and a micro-cap PRICE BAND;
  * ranks each week's survivors by dollar volume and keeps the most liquid top_n.

Output is the same ISO-week -> [tickers] JSON format every other PIT universe
uses, so the strategy loads it exactly like the RSI-2 universe.

SURVIVORSHIP BIAS — READ THIS
-----------------------------
yfinance (like Alpaca) returns data only for names that still exist, so a pool
enumerated today is survivorship-biased: the delisted corpses are missing.  Two
mitigations:
  1. The strategy itself runs a Monte-Carlo survivorship stress-test that injects
     the missing corpses into its reported metrics.
  2. This builder accepts --constituents-file / --delisting-file so a real
     point-in-time / delisting dataset (Polygon, Nasdaq Data Link / Sharadar, or
     a hand-built CSV of delisted tickers) can be unioned into the candidate pool
     later WITHOUT any strategy change.  Adding such a source is the single
     highest-value upgrade to this whole niche.

USAGE
-----
    # Small, fast proof-of-pipeline on an explicit ticker list:
    python data_collection/build_microcap_universe.py \
        --start 2022-06-01 --end 2023-12-31 \
        --tickers GPRO PLUG FCEL SENS BBIG CLOV WKHS RIOT \
        --min-dollar-volume 250000 --min-price 1 --max-price 15

    # Full Alpaca-enumerated pool (heavy offline job — thousands of names):
    python data_collection/build_microcap_universe.py \
        --start 2022-06-01 --end 2023-12-31 --universe alpaca \
        --max-candidates 1500 --out output/historical_universes_microcap.json

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy yfinance
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the proven yfinance bulk-download + normalization scaffold.
from data_collection.build_historical_universes import fetch_bulk_daily_data

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MicrocapUniverseBuilder")


# =============================================================================
# CANDIDATE POOL
# =============================================================================

def get_alpaca_equities(max_candidates: Optional[int] = None) -> List[str]:
    """Enumerate active, tradeable US common-stock symbols from Alpaca.

    Replaces the index-membership pool with the full listable universe — the
    only way to reach names small enough to matter here.
    """
    try:
        import config
        api_key, secret_key = config.API_KEY, config.SECRET_KEY
    except Exception:
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        log.error("Alpaca credentials not found (config.py or ALPACA_API_KEY env).")
        return []

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    tc  = TradingClient(api_key, secret_key, paper=True)
    req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = tc.get_all_assets(req)

    # Major exchanges only; skip OTC (no reliable data / not tradeable via Alpaca).
    ok_exchanges = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}
    symbols = [
        a.symbol for a in assets
        if a.tradable and str(getattr(a.exchange, "value", a.exchange)) in ok_exchanges
        and "." not in a.symbol and "-" not in a.symbol  # skip preferreds / warrants / units
    ]
    log.info("Alpaca returned %d active tradeable US-equity symbols.", len(symbols))

    if max_candidates and len(symbols) > max_candidates:
        # Deterministic sample keeps the offline job manageable while staying
        # reproducible; a full run simply omits --max-candidates.
        symbols = sorted(symbols)[:max_candidates]
        log.info("Capped candidate pool to first %d symbols (--max-candidates).", max_candidates)
    return symbols


def load_ticker_file(path: str) -> List[str]:
    """Load tickers from a .txt (one per line) or .csv (a 'symbol'/'ticker' column).
    This is the seam for an external PIT / delisting dataset."""
    p = Path(path)
    if not p.exists():
        log.warning("Ticker file not found: %s", path)
        return []
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        for col in ("symbol", "ticker", "Symbol", "Ticker"):
            if col in df.columns:
                return df[col].astype(str).str.upper().str.strip().tolist()
        # Fall back to the first column.
        return df.iloc[:, 0].astype(str).str.upper().str.strip().tolist()
    return [ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()]


# =============================================================================
# WEEKLY MICRO-CAP SCORING
# =============================================================================

def calculate_weekly_microcap(
    df: pd.DataFrame,
    top_n: int,
    min_price: float,
    max_price: float,
    min_dollar_volume: float,
    dollar_vol_period: int = 20,
) -> dict:
    """Build the PIT micro-cap universe: ISO-week -> [tickers].

    Each week samples the last trading day, keeps names inside the price band whose
    trailing dollar volume clears the floor, and takes the top_n by dollar volume.
    """
    log.info("Computing rolling dollar volume and screening micro-caps...")
    df = df.sort_index(level=["symbol", "date"])

    close  = df["close"]
    volume = df["volume"]
    dollar_vol = (close * volume).groupby(level="symbol").rolling(dollar_vol_period).mean()
    dollar_vol.index = dollar_vol.index.droplevel(0)

    scored = pd.DataFrame({
        "close":      close,
        "dollar_vol": dollar_vol,
    }).dropna()

    scored = scored.reset_index()
    scored["iso_week"] = scored["date"].dt.strftime("%G-W%V")

    # Last observation per (week, symbol) — no lookahead.
    weekly = scored.groupby(["iso_week", "symbol"]).last().reset_index()

    historical_map: dict = {}
    weeks = sorted(weekly["iso_week"].unique())

    for week in weeks:
        wk = weekly[weekly["iso_week"] == week]
        wk = wk[
            (wk["close"] >= min_price)
            & (wk["close"] <= max_price)
            & (wk["dollar_vol"] >= min_dollar_volume)
        ]
        if wk.empty:
            historical_map[week] = []
            continue
        top = wk.sort_values("dollar_vol", ascending=False).head(top_n)
        historical_map[week] = top["symbol"].tolist()

    non_empty = sum(1 for v in historical_map.values() if v)
    log.info("Micro-cap universe: %d weeks, %d non-empty.", len(historical_map), non_empty)
    return historical_map


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build the PIT micro-cap universe.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", help="Explicit candidate tickers (wins over --universe).")
    parser.add_argument("--universe", choices=["alpaca"], default="alpaca",
                        help="Candidate pool source when --tickers is omitted.")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Cap the Alpaca candidate pool (keeps the offline job manageable).")
    parser.add_argument("--constituents-file", default=None,
                        help="Extra tickers (.txt/.csv) unioned into the pool. Seam for an "
                             "external point-in-time constituent dataset.")
    parser.add_argument("--delisting-file", default=None,
                        help="Delisted tickers (.txt/.csv) unioned into the pool to fight "
                             "survivorship bias. Seam for a real delisting dataset.")
    parser.add_argument("--top-n", type=int, default=200, help="Names kept per week (default 200).")
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--max-price", type=float, default=15.0)
    parser.add_argument("--min-dollar-volume", type=float, default=250_000.0,
                        help="20-day average dollar-volume floor (default $250k/day).")
    parser.add_argument("--out", default="output/historical_universes_microcap.json")
    args = parser.parse_args()

    # 1. Candidate pool
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = get_alpaca_equities(max_candidates=args.max_candidates)
    for extra in (args.constituents_file, args.delisting_file):
        if extra:
            pool = load_ticker_file(extra)
            if pool:
                tickers = sorted(set(tickers) | set(pool))
                log.info("Unioned %d tickers from %s (pool now %d).", len(pool), extra, len(tickers))
    if not tickers:
        log.error("Empty candidate pool. Exiting.")
        return

    # 2. Fetch adjusted daily data (split/dividend-clean membership).
    df = fetch_bulk_daily_data(tickers, args.start, args.end, pad_days=60)
    if df.empty:
        log.error("No data fetched. Check dates / tickers.")
        return

    # 3. Weekly PIT micro-cap screen.
    universe_map = calculate_weekly_microcap(
        df, top_n=args.top_n,
        min_price=args.min_price, max_price=args.max_price,
        min_dollar_volume=args.min_dollar_volume,
    )

    # 4. Persist.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(universe_map, f, indent=2)
    log.info("Saved micro-cap PIT universe to '%s'.", out_path)
    non_empty = [w for w, v in universe_map.items() if v]
    if non_empty:
        w0 = non_empty[len(non_empty) // 2]
        log.info("Sample week %s: %s ...", w0, universe_map[w0][:8])


if __name__ == "__main__":
    main()
