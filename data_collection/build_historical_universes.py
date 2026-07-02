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


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia."""
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
        return tickers
    except Exception as exc:
        log.error(f"Failed to fetch S&P 500 list: {exc}")
        return []

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

def fetch_bulk_daily_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily bars for all tickers using yfinance.
    We pad the start date by 60 days to ensure the very first week in the
    requested range has enough historical data to calculate 30-day moving averages.
    """
    start_dt = pd.to_datetime(start) - pd.Timedelta(days=60)
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


def calculate_weekly_scores(df: pd.DataFrame, top_n: int, target_hv: float = 0.40) -> dict:
    """
    Calculate the universe scores vector-style across all dates and tickers,
    sample the scores every Friday, and extract the top N tickers per week.

    target_hv: center of the volatility Gaussian (annualized HV). 0.40 is the
    default reversion universe. Raise it (e.g. 0.55) to bias toward more volatile
    names that extend further from VWAP — tested as a lever for VWAP reversion.
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

    # Combine into a single scoring DataFrame
    scores = pd.DataFrame({
        'vol_30': vol_30,
        'rvol': rvol,
        'hv_20': hv_20,
        'mom_20': mom_20
    }).dropna()

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

        # Apply Min-Max normalization cross-sectionally for this specific week
        score_vol = _minmax(week_data['vol_30'])
        score_mom = _minmax(week_data['mom_20'])
        score_rvol = _minmax(week_data['rvol'])

        # Apply Gaussian curve to Volatility (center = target_hv, default 40%)
        # Stocks near target_hv get 1.0, stocks far from it get near 0
        decay = 0.15
        score_hv = np.exp(-0.5 * ((week_data['hv_20'] - target_hv) / decay) ** 2)

        # Composite score (Matching stock_universe.py weights)
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
    parser.add_argument("--universe", choices=["sp500", "sp400", "combined"],default="sp500")
    parser.add_argument("--target-hv", type=float, default=0.40,
                        help="Center of the volatility Gaussian, annualized HV "
                             "(default 0.40). Raise (e.g. 0.55) for a higher-volatility "
                             "reversion universe.")

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

    # 2. Fetch the data
    df = fetch_bulk_daily_data(tickers, args.start, args.end)
    if df.empty:
        log.error("No data fetched. Check date range or ticker symbols.")
        return

    # 3. Calculate scores and build the map
    log.info(f"Volatility target (Gaussian center): {args.target_hv:.2f} annualized HV")
    universe_map = calculate_weekly_scores(df, top_n=args.top_n, target_hv=args.target_hv)

    # 4. Save to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(universe_map, f, indent=2)

    log.info(f"Success! Historical universe cache saved to '{out_path}'.")
    log.info(f"Sample mapping for {list(universe_map.keys())[0]}: {universe_map[list(universe_map.keys())[0]][:5]}...")


if __name__ == "__main__":
    main()