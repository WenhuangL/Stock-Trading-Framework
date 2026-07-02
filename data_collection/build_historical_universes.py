"""
build_historical_universes.py
-----------------------------
Offline data engineering script to pre-compute Point-In-Time (PIT) stock
universes for highly accurate, lookahead-free backtesting.

Instead of dynamically calculating the universe during a backtest (which is
incredibly slow and prone to API limits), this script fetches years of daily
data all at once, calculates the exact universe scores for every single Friday
in history, and saves a map of ISO-Week strings to Ticker lists.

USAGE
-----
    # Default: Run on S&P 500 from 2022 to 2023, save top 100 per week
    python build_historical_universes.py --start 2022-01-01 --end 2023-12-31

    # Custom tickers and custom Top N
    python build_historical_universes.py --tickers AAPL MSFT NVDA TSLA --top-n 2

    # Save to a specific file
    python build_historical_universes.py --out data/historical_universes.json

REQUIREMENTS
------------
    pip install pandas numpy yfinance
"""

import argparse
import datetime
import json
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# Suppress yfinance warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("UniverseBuilder")

# S&P 500 members removed during 2022-2024, absent from the current Wikipedia list.
# Including them eliminates survivorship bias for backtests over that period.
# Tickers whose data ends before the backtest start (e.g. CTXS taken private Sep 2022)
# return empty DataFrames from yfinance and are silently dropped downstream.
_HISTORICALLY_REMOVED_SP500: frozenset = frozenset({
    "SIVB",  # SVB Financial Group  — removed Mar 2023 (bank failure)
    "FRC",   # First Republic Bank  — removed May 2023 (bank failure)
    "ATVI",  # Activision Blizzard  — removed Oct 2023 (Microsoft acq.)
    "TWTR",  # Twitter              — removed Nov 2022 (privatized; data through Oct 27 2022)
    "CTXS",  # Citrix Systems       — removed Sep 2022 (taken private)
    "CERN",  # Cerner Corp          — removed Jun 2022 (Oracle acq.)
    "INFO",  # IHS Markit           — removed Feb 2022 (S&P Global merger)
    "XLNX",  # Xilinx               — removed Feb 2022 (AMD acq.)
    "PBCT",  # People's United      — removed Feb 2022 (M&T Bank acq.)
})


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia, then union with
    historically removed members to eliminate survivorship bias for 2022-2024 backtests."""
    log.info("Fetching S&P 500 ticker list from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Disguise the Python script as a standard Chrome web browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

        # Fetch the webpage using requests instead of pandas directly
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Feed the raw HTML text into pandas
        tables = pd.read_html(response.text)

        # yfinance uses '-' instead of '.' for classes (e.g. BRK-B)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
    except Exception as exc:
        log.error(f"Failed to fetch S&P 500 list: {exc}")
        tickers = []

    combined = list(set(tickers) | _HISTORICALLY_REMOVED_SP500)
    log.info(
        "Added %d historically removed tickers to candidate pool (survivorship-bias fix).",
        len(_HISTORICALLY_REMOVED_SP500),
    )
    return combined

def get_sp400_tickers() -> List[str]:
    """Fetch S&P 400 mid-cap constituents from Wikipedia."""
    log.info("Fetching S&P 400 ticker list...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Fetched {len(tickers)} S&P 400 tickers.")
        return tickers
    except Exception as exc:
        log.error(f"Failed to fetch S&P 400 list: {exc}")
        return []

def fetch_bulk_daily_data(tickers: List[str], start: str, end: str, pad_days: int = 60) -> pd.DataFrame:
    """
    Fetch daily bars for all tickers using yfinance.
    We pad the start date by pad_days to ensure the very first week in the
    requested range has enough historical data to calculate moving averages.
    Use pad_days=250 when sma200_filter=True so SMA(200) has adequate history.
    """
    start_dt = pd.to_datetime(start) - pd.Timedelta(days=pad_days)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)  # Inclusive end

    log.info(f"Downloading daily data for {len(tickers)} tickers ({start_dt.date()} to {end_dt.date()})...")

    # Download bulk data
    data = yf.download(
        tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        group_by="ticker",
        progress=True,
        auto_adjust=True,  # Automatically adjust for splits/dividends
        threads=True
    )

    # If only one ticker is passed, yfinance doesn't add the top-level ticker index.
    # We standardize it to a stacked format: MultiIndex (date, symbol) -> OHLCV
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])

    # Stack the columns so 'ticker' becomes part of the index
    stacked = data.stack(level=0, future_stack=True).rename_axis(['date', 'symbol'])
    stacked.columns = [c.lower() for c in stacked.columns]

    # Clean up
    stacked = stacked.dropna(subset=['close', 'volume'])
    log.info(f"Download complete. Processed {len(stacked):,} total daily bars.")

    return stacked


# =============================================================================
# VECTORIZED SCORING (Replicates stock_universe.py exactly)
# =============================================================================

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1] within a specific cross-section."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def calculate_weekly_scores(df: pd.DataFrame, top_n: int, sma200_filter: bool = False) -> dict:
    """
    Calculate the universe scores vector-style across all dates and tickers,
    sample the scores every Friday, and extract the top N tickers per week.

    When sma200_filter=True, only stocks trading above their 200-day SMA are
    eligible for selection each week.  Requires fetch_bulk_daily_data to be
    called with pad_days >= 250 so the SMA has adequate warm-up history.
    """
    log.info("Calculating rolling indicators (Volume, Volatility, Momentum)...")

    # Sort for rolling operations
    df = df.sort_index(level=['symbol', 'date'])

    # 1. Rolling Average Volume (30 days)
    vol_30 = df.groupby('symbol')['volume'].rolling(30).mean()
    vol_30.index = vol_30.index.droplevel(0)

    # 2. Rolling Relative Volume (5d avg / 30d avg)
    vol_5 = df.groupby('symbol')['volume'].rolling(5).mean()
    vol_5.index = vol_5.index.droplevel(0)
    rvol = vol_5 / vol_30

    # 3. Historical Volatility (20-day annualized log returns)
    close = df['close']
    log_ret = np.log(close / df.groupby('symbol')['close'].shift(1))
    hv_20 = log_ret.groupby('symbol').rolling(20).std() * np.sqrt(252)
    hv_20.index = hv_20.index.droplevel(0)

    # 4. Momentum (20-day return)
    mom_20 = (close / df.groupby('symbol')['close'].shift(20)) - 1

    # 5. SMA(200) filter flag (only computed when requested)
    score_dict = {'vol_30': vol_30, 'rvol': rvol, 'hv_20': hv_20, 'mom_20': mom_20}
    if sma200_filter:
        sma200_raw = df.groupby('symbol')['close'].rolling(200).mean()
        sma200_raw.index = sma200_raw.index.droplevel(0)
        # Positional comparison: both arrays derived from same df in same row order.
        # close > NaN evaluates to False in numpy, so early rows (no SMA yet) are
        # automatically excluded from the filter.
        above_arr = (df['close'].values > sma200_raw.values)
        score_dict['above_sma200'] = pd.Series(above_arr, index=vol_30.index)
        log.info("SMA(200) filter enabled — stocks below 200-day MA will be excluded.")

    # Combine into a single scoring DataFrame
    scores = pd.DataFrame(score_dict).dropna()

    log.info("Sampling scores on week-ending days and ranking...")

    # Reset index to make 'date' a column we can group by easily
    scores = scores.reset_index()

    # Create an ISO Week string (e.g., "2023-W04")
    scores['iso_week'] = scores['date'].dt.strftime('%G-W%V')

    # To avoid lookahead bias, we only want the scores as they stood on the
    # *last trading day* of that specific week (usually Friday).
    # We group by iso_week and symbol, and take the last row.
    weekly_final_scores = scores.groupby(['iso_week', 'symbol']).last().reset_index()

    # Now, group by week and rank the cross-section of stocks
    historical_map = {}

    weeks = weekly_final_scores['iso_week'].unique()
    weeks.sort()

    for week in weeks:
        week_data = weekly_final_scores[weekly_final_scores['iso_week'] == week].copy()

        # Apply SMA(200) filter BEFORE ranking so only qualifying stocks compete.
        if sma200_filter and 'above_sma200' in week_data.columns:
            week_data = week_data[week_data['above_sma200'] == True]
            if week_data.empty:
                historical_map[week] = []
                continue

        # Apply Min-Max normalization cross-sectionally for this specific week
        score_vol = _minmax(week_data['vol_30'])
        score_mom = _minmax(week_data['mom_20'])
        score_rvol = _minmax(week_data['rvol'])

        # Apply Gaussian curve to Volatility (Targeting 40% annualized)
        # Stocks near 40% get 1.0, stocks at 10% or 80% get near 0
        target_hv = 0.40
        decay = 0.15
        score_hv = np.exp(-0.5 * ((week_data['hv_20'] - target_hv) / decay) ** 2)

        # Composite score (Matching stock_universe.py weights)
        week_data = week_data.copy()
        week_data['composite'] = (
                0.35 * score_vol +
                0.30 * score_hv +
                0.20 * score_mom +
                0.15 * score_rvol
        )

        # Sort and take Top N
        top_stocks = week_data.sort_values('composite', ascending=False).head(top_n)
        historical_map[week] = top_stocks['symbol'].tolist()

    log.info(f"Processed {len(historical_map)} distinct trading weeks.")
    return historical_map


def calculate_weekly_scores_short(df: pd.DataFrame, top_n: int) -> dict:
    """
    Build a short-optimised PIT universe.

    Pre-filter: confirmed death cross (SMA(50) < SMA(200)) AND price below SMA(200).
    Score weights:
      40% — most negative 20-day momentum
      30% — steepest declining SMA(200) slope (20-day change)
      20% — volatility Gaussian centred at 40% annualised
      10% — liquidity (30-day average volume)

    Requires pad_days >= 250 so SMA(200) has adequate warm-up history.
    """
    log.info("Calculating rolling indicators for short universe...")

    df = df.sort_index(level=['symbol', 'date'])

    close = df['close']

    # 30-day average volume — liquidity floor
    vol_30 = df.groupby('symbol')['volume'].rolling(30).mean()
    vol_30.index = vol_30.index.droplevel(0)

    # 20-day annualised historical volatility
    log_ret = np.log(close / df.groupby('symbol')['close'].shift(1))
    hv_20 = log_ret.groupby('symbol').rolling(20).std() * np.sqrt(252)
    hv_20.index = hv_20.index.droplevel(0)

    # 20-day momentum
    mom_20 = (close / df.groupby('symbol')['close'].shift(20)) - 1

    # SMA(200) and SMA(50) — for pre-filter and slope
    sma200_raw = df.groupby('symbol')['close'].rolling(200).mean()
    sma200_raw.index = sma200_raw.index.droplevel(0)

    sma50_raw = df.groupby('symbol')['close'].rolling(50).mean()
    sma50_raw.index = sma50_raw.index.droplevel(0)

    # 20-trading-day slope of SMA(200): (current - 20 bars ago) / |20 bars ago|
    sma200_lag20 = sma200_raw.groupby(level='symbol').shift(20)
    sma200_slope = (sma200_raw - sma200_lag20) / sma200_lag20.abs()

    # Death-cross flag and below-SMA(200) flag
    death_cross  = pd.Series((sma50_raw.values < sma200_raw.values), index=vol_30.index)
    below_sma200 = pd.Series((df['close'].values < sma200_raw.values), index=vol_30.index)

    scores = pd.DataFrame({
        'vol_30':        vol_30,
        'hv_20':         hv_20,
        'mom_20':        mom_20,
        'sma200_slope':  sma200_slope,
        'death_cross':   death_cross,
        'below_sma200':  below_sma200,
    }).dropna()

    log.info("Sampling short scores on week-ending days and ranking...")

    scores = scores.reset_index()
    scores['iso_week'] = scores['date'].dt.strftime('%G-W%V')

    weekly_final_scores = scores.groupby(['iso_week', 'symbol']).last().reset_index()

    historical_map: dict = {}
    weeks = weekly_final_scores['iso_week'].unique()
    weeks.sort()

    for week in weeks:
        week_data = weekly_final_scores[weekly_final_scores['iso_week'] == week].copy()

        # Pre-filter: death cross AND price below SMA(200)
        week_data = week_data[
            (week_data['death_cross'] == True) & (week_data['below_sma200'] == True)
        ]
        if week_data.empty:
            historical_map[week] = []
            continue

        score_neg_mom   = _minmax(-week_data['mom_20'])
        score_sma_slope = _minmax(-week_data['sma200_slope'])
        score_hv        = np.exp(-0.5 * ((week_data['hv_20'] - 0.40) / 0.15) ** 2)
        score_vol       = _minmax(week_data['vol_30'])

        week_data = week_data.copy()
        week_data['composite'] = (
            0.40 * score_neg_mom +
            0.30 * score_sma_slope +
            0.20 * score_hv +
            0.10 * score_vol
        )

        top_stocks = week_data.sort_values('composite', ascending=False).head(top_n)
        historical_map[week] = top_stocks['symbol'].tolist()

    non_empty = sum(1 for v in historical_map.values() if v)
    log.info(f"Short universe: {len(historical_map)} weeks, {non_empty} non-empty.")
    return historical_map


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Historical PIT Universes")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", type=str, nargs="+",
                        help="Specific list of tickers (e.g. AAPL MSFT). If omitted, uses S&P 500.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of stocks to select per week (default 100)")
    parser.add_argument("--out", type=str, default="output/historical_universes.json", help="Output JSON path")
    parser.add_argument("--universe", choices=["sp500", "sp400", "combined"], default="sp500")
    parser.add_argument(
        "--sma200-filter", action="store_true",
        help=(
            "Only include stocks trading above their 200-day SMA in each week's "
            "selection.  Intended for the RSI-2 strategy universe.  Use with "
            "--out output/historical_universes_rsi2.json."
        ),
    )
    parser.add_argument(
        "--short-universe", action="store_true",
        help=(
            "Build a short-optimised universe: pre-filtered by death cross "
            "(SMA50 < SMA200) and scored on negative momentum + declining "
            "SMA(200) slope.  Use with "
            "--out output/historical_universes_rsi2_short.json."
        ),
    )

    args = parser.parse_args()
    # 1. Determine the ticker list
    if args.tickers:
        # Explicit --tickers list always wins
        tickers = args.tickers
    else:
        # Use --universe to determine the fetch function
        if args.universe == "sp400":
            tickers = get_sp400_tickers()
        elif args.universe == "combined":
            tickers = list(set(get_sp500_tickers() + get_sp400_tickers()))
        else:  # default: sp500
            tickers = get_sp500_tickers()

        if not tickers:
            log.error("No tickers provided and universe fetch failed. Exiting.")
            return

    # 2. Fetch the data (pad extra history for SMA(200) warm-up when needed)
    # Short universe needs 250 days for SMA(200) + 20-day slope lag; same for sma200_filter.
    pad_days = 250 if (args.sma200_filter or args.short_universe) else 60
    df = fetch_bulk_daily_data(tickers, args.start, args.end, pad_days=pad_days)
    if df.empty:
        log.error("No data fetched. Check date range or ticker symbols.")
        return

    # 3. Calculate scores and build the map
    if args.short_universe:
        universe_map = calculate_weekly_scores_short(df, top_n=args.top_n)
    else:
        universe_map = calculate_weekly_scores(df, top_n=args.top_n, sma200_filter=args.sma200_filter)

    # 4. Save to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(universe_map, f, indent=2)

    log.info(f"Success! Historical universe cache saved to '{out_path}'.")
    log.info(f"Sample mapping for {list(universe_map.keys())[0]}: {universe_map[list(universe_map.keys())[0]][:5]}...")


if __name__ == "__main__":
    main()