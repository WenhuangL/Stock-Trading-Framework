"""
data_cache.py
-------------
A local caching layer for Alpaca historical data.
Saves downloaded minute and daily bars to disk as compressed .parquet files
so subsequent backtests load in seconds instead of hours.
"""
import logging
from pathlib import Path
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

log = logging.getLogger("DataCache")


def normalize_symbol(symbol: str) -> str:
    """Map Wikipedia/universe class-share notation to Alpaca's market-data format.

    The universe is built from Wikipedia, which renders share classes with a dash
    (BRK-B, BF-B, MOG-A). Alpaca's historical data API expects a dot (BRK.B,
    BF.B, MOG.A) and returns ``invalid symbol`` for the dash form. US equity
    tickers never contain a legitimate dash, so converting every dash to a dot is
    safe.
    """
    return symbol.replace("-", ".")


class LocalDataCache:
    def __init__(self, data_client: StockHistoricalDataClient, cache_dir: str = "data_collection/cache"):
        self.dc = data_client
        self.cache_dir = Path(cache_dir)

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "minute").mkdir(exist_ok=True)
        (self.cache_dir / "day").mkdir(exist_ok=True)

    def get_bars_df(
            self,
            symbol: str,
            timeframe: TimeFrame,
            start,
            end,
            feed: str = "sip"
    ) -> pd.DataFrame:
        """
        Fetch bars for a single symbol. Checks local parquet cache first.
        If data is missing or incomplete, fetches from Alpaca, updates cache, and returns.

        Daily bars: one file per symbol (AAPL.parquet), 7-day buffer covers weekends.
        Minute bars: one file per symbol per date (AAPL_20240103.parquet) because
        each backtest day fetches a narrow intraday window and the 7-day buffer
        would otherwise cause a cache hit that returns the wrong day's bars.
        """
        api_symbol = normalize_symbol(symbol)
        tf_folder = "minute" if timeframe.unit == TimeFrameUnit.Minute else "day"

        # Convert to UTC pandas timestamps for easy, bulletproof comparison
        start_pd = pd.to_datetime(start, utc=True)
        end_pd = pd.to_datetime(end, utc=True)

        # Minute bars use date+feed-keyed files so SIP and IEX data are stored
        # separately and switching feeds never silently returns wrong-feed data.
        if timeframe.unit == TimeFrameUnit.Minute:
            date_str = start_pd.strftime("%Y%m%d")
            file_path = self.cache_dir / tf_folder / f"{api_symbol}_{date_str}_{feed}.parquet"
        else:
            # Cache filename includes feed so SIP and IEX daily bars are stored separately.
            file_path = self.cache_dir / tf_folder / f"{api_symbol}_{feed}.parquet"

        df = pd.DataFrame()
        needs_fetch = True

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    if timeframe.unit == TimeFrameUnit.Minute:
                        # Date-keyed file: exact match guaranteed, no buffer needed.
                        needs_fetch = False
                    else:
                        # Check if our cached data covers the requested date range.
                        # Add a 7-day buffer to safely cover weekends and long holidays.
                        min_date = pd.to_datetime(df.index.min(), utc=True)
                        max_date = pd.to_datetime(df.index.max(), utc=True)
                        if start_pd >= (min_date - pd.Timedelta(days=7)) and end_pd <= (max_date + pd.Timedelta(days=7)):
                            needs_fetch = False
            except Exception as exc:
                log.warning(f"Cache read failed for {symbol}, will re-fetch. Error: {exc}")

        if needs_fetch:
            log.info(f"[{symbol}] Fetching from Alpaca API (Cache miss/incomplete)...")
            try:
                resp = self.dc.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=[api_symbol],
                    timeframe=timeframe,
                    start=start_pd,
                    end=end_pd,
                    feed=feed
                ))

                if resp.df is not None and not resp.df.empty:
                    # Extract single symbol DataFrame
                    if hasattr(resp.df.index, 'levels'):
                        new_df = resp.df.loc[api_symbol].copy()
                    else:
                        new_df = resp.df.copy()

                    new_df.columns = [c.lower() for c in new_df.columns]
                    new_df.index = pd.to_datetime(new_df.index, utc=True)

                    if timeframe.unit == TimeFrameUnit.Minute:
                        # Date-keyed: no merging needed, just save the fetched slice.
                        df = new_df.sort_index()
                    else:
                        # Merge with existing data
                        if not df.empty:
                            df = pd.concat([df, new_df])
                            # Drop duplicates keeping the latest fetched data, sort by time
                            df = df[~df.index.duplicated(keep='last')].sort_index()
                        else:
                            df = new_df.sort_index()

                    # Save back to cache
                    df.to_parquet(file_path)
                else:
                    log.debug(f"[{symbol}] Alpaca returned empty dataframe.")
                    return pd.DataFrame()

            except Exception as exc:
                log.error(f"[{symbol}] API fetch failed: {exc}")
                return pd.DataFrame()

        # Final step: Slice the dataframe to exactly what was requested
        if df.empty:
            return df

        mask = (df.index >= start_pd) & (df.index <= end_pd)
        return df.loc[mask].copy()