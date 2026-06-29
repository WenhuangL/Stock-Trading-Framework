"""
run_backtest.py
---------------
Standalone backtest runner for both IntradayStrategy and EodReversionStrategy.

Run this manually (locally or on PythonAnywhere via a console — NOT as a
scheduled task) to validate strategy parameters against historical data before
going live.

COMBINED SUMMARY
----------------
Because the intraday strategy operates 9:30 AM – 3:55 PM and the EOD strategy
operates 3:55 PM – 7:50 PM, they operate on NON-OVERLAPPING time windows each
day. Capital is therefore shared across the full day. The combined P&L is the
sum of both strategies' P&L on the same starting portfolio, adjusted so the EOD
strategy only uses capital not currently deployed by the intraday strategy at
3:55 PM.

For simplicity, this runner runs both backtests independently on the same
initial_cash figure and reports:
  - Individual results for each strategy
  - Estimated combined daily return (additive, since time windows don't overlap)

USAGE
-----
    python run_backtest.py

    # Or with custom tickers / date range:
    python run_backtest.py --start 2023-06-01 --end 2023-12-31

REQUIREMENTS
------------
    pip install alpaca-py pandas numpy matplotlib
"""

import argparse
import logging
import sys
from zoneinfo import ZoneInfo
import sys, os
import json  # Added for loading historical_universes.json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_backtest")


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

# DEFAULT_TICKERS removed — universe is now loaded from UniverseSelector.
# Pass --tickers on the CLI to override with a specific list.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA",
    "AMD",  "JPM",  "V",    "UNH",  "XOM",  "LLY",  "JNJ",  "PG",
    "MA",   "HD",   "MRK",  "ABBV", "CVX",
]

DEFAULT_START      = "2023-01-01"
DEFAULT_END        = "2023-12-31"
DEFAULT_CASH       = 100_000.0


# =============================================================================
# COMBINED SUMMARY PRINTER
# =============================================================================

def _print_combined_summary(
    intraday_results: dict,
    eod_results:      dict,
    initial_cash:     float,
) -> None:
    id_summary  = intraday_results.get("summary", {})
    eod_summary = eod_results.get("summary", {})

    # Intraday returns: final_value, initial_cash, num_trades
    id_trades   = id_summary.get("num_trades", 0)
    id_final    = id_summary.get("final_value", initial_cash)
    id_start    = id_summary.get("initial_cash", initial_cash)
    id_pnl      = id_final - id_start

    # EOD returns: num_trades, total_pnl (or final_value if available)
    eod_trades  = eod_summary.get("num_trades", 0)
    eod_pnl     = eod_summary.get("total_pnl",
                    eod_summary.get("final_value", initial_cash) - initial_cash)

    combined    = id_pnl + eod_pnl
    combined_pct = (combined / initial_cash) * 100 if initial_cash else 0

    print("\n" + "=" * 70)
    print("  COMBINED BACKTEST SUMMARY")
    print("=" * 70)
    print(f"  {'Strategy':<25}  {'Trades':>7}  {'P&L':>12}  {'Return':>8}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*12}  {'-'*8}")
    print(f"  {'Intraday (ORB+VWAP+PH)':<25}  {id_trades:>7}  "
          f"${id_pnl:>+11,.2f}  {id_pnl/initial_cash*100:>+7.2f}%")
    print(f"  {'EOD Reversion':<25}  {eod_trades:>7}  "
          f"${eod_pnl:>+11,.2f}  {eod_pnl/initial_cash*100:>+7.2f}%")
    print(f"  {'-'*25}  {'-'*7}  {'-'*12}  {'-'*8}")
    print(f"  {'COMBINED':<25}  {id_trades+eod_trades:>7}  "
          f"${combined:>+11,.2f}  {combined_pct:>+7.2f}%")
    print(f"\n  Starting capital: ${initial_cash:,.0f}")
    print(f"  Ending estimate:  ${initial_cash + combined:,.0f}")
    print("=" * 70 + "\n")

    for label, summary in [("INTRADAY DETAIL", id_summary),
                            ("EOD DETAIL", eod_summary)]:
        if not summary:
            continue
        print(f"  {label}")
        print(f"  {'-'*40}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"    {k:<28} {v:>10.4f}")
            else:
                print(f"    {k:<28} {str(v):>10}")
        print()


# =============================================================================
# SPY BUY-AND-HOLD BENCHMARK
# =============================================================================

def _spy_benchmark_pct(dc, start_date: str, end_date: str):
    """Return SPY buy-and-hold % return over the period, or None on failure.

    Context for the strategy numbers: a backtest that loses money while SPY
    rallied is failing twice over. This makes the benchmark explicit.
    """
    try:
        import datetime as _dt
        from alpaca.data.timeframe import TimeFrame
        from data_collection.data_cache import LocalDataCache
        cache = LocalDataCache(dc)
        s = _dt.datetime.fromisoformat(start_date)
        e = _dt.datetime.fromisoformat(end_date)
        df = cache.get_bars_df("SPY", TimeFrame.Day, s, e, feed="sip")
        if df is None or df.empty or len(df) < 2:
            return None
        first = float(df["close"].iloc[0])
        last  = float(df["close"].iloc[-1])
        return (last / first - 1.0) * 100.0
    except Exception as exc:
        log.warning(f"SPY benchmark unavailable: {exc}")
        return None


# =============================================================================
# SINGLE-PERIOD RUNNER
# =============================================================================

def _run_period(
    label, start_date, end_date, *,
    tc, dc, rm, ucfg, tickers,
    hu_intraday, hu_eod, cash, show_charts,
):
    """Run intraday + EOD backtests for one date range and print the results."""
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig
    from strategies.strategy_eod_reversion import EodReversionStrategy, EodReversionConfig

    log.info("=" * 60)
    log.info(f"  Backtest Runner - {label}")
    log.info("=" * 60)
    log.info(f"  Tickers:  {len(tickers)} stocks")
    log.info(f"  Period:   {start_date} -> {end_date}")
    log.info(f"  Capital:  ${cash:,.0f}")
    log.info("")

    # ── Intraday ──────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("  Running IntradayStrategy.backtest()...")
    log.info("-" * 60)
    intraday = IntradayStrategy(
        trading_client=tc, data_client=dc, risk_manager=rm,
        config=IntradayConfig(), universe_config=ucfg,
    )
    intraday_results: dict = {}
    try:
        intraday_results = intraday.backtest(
            tickers=tickers, start_date=start_date, end_date=end_date,
            initial_cash=cash, historical_universes=hu_intraday,
        )
        intraday.print_summary(intraday_results)
    except Exception as exc:
        log.exception(f"Intraday backtest failed: {exc}")

    # ── EOD ───────────────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("  Running EodReversionStrategy.backtest()...")
    log.info("-" * 60)
    eod = EodReversionStrategy(
        trading_client=tc, data_client=dc, config=EodReversionConfig(),
        universe_config=ucfg, risk_manager=rm,
    )
    eod_results: dict = {}
    try:
        eod_results = eod.backtest(
            tickers=tickers, start_date=start_date, end_date=end_date,
            initial_cash=cash, historical_universes=hu_eod,
        )
        eod.print_summary(eod_results)
    except Exception as exc:
        log.exception(f"EOD backtest failed: {exc}")

    # ── Combined + benchmark ──────────────────────────────────────────────────
    _print_combined_summary(intraday_results, eod_results, cash)

    spy_pct = _spy_benchmark_pct(dc, start_date, end_date)
    if spy_pct is not None:
        id_pct  = intraday_results.get("summary", {}).get("total_return_pct", 0.0)
        eod_pct = eod_results.get("summary", {}).get("total_return_pct", 0.0)
        print(f"  BENCHMARK — SPY buy & hold ({label}): {spy_pct:+.2f}%")
        print(f"    vs Intraday {id_pct:+.2f}%  |  vs EOD {eod_pct:+.2f}%")
        print("=" * 70 + "\n")

    # ── Persist trade records to SQLite ──────────────────────────────────────
    try:
        from data_collection.trade_db import save_run
        run_id = save_run(
            intraday_results=intraday_results,
            eod_results=eod_results,
            start_date=start_date,
            end_date=end_date,
            spy_return_pct=spy_pct,
        )
        log.info(f"Trades saved to output/trades.db  (run_id={run_id})")
    except Exception as exc:
        log.warning(f"Failed to save trades to database: {exc}")

    if show_charts:
        log.info("Generating charts (close each window to continue)...")
        if intraday_results:
            try:
                intraday.plot_backtest(intraday_results)
            except Exception as exc:
                log.warning(f"Intraday plot failed: {exc}")
        if eod_results:
            try:
                eod.plot_backtest(eod_results)
            except Exception as exc:
                log.warning(f"EOD plot failed: {exc}")

    return intraday_results, eod_results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # ── CLI argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Run intraday + EOD backtests on the same ticker list and date range."
    )
    parser.add_argument("--start",  default=DEFAULT_START,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=DEFAULT_END,    help="End date   YYYY-MM-DD")
    parser.add_argument("--cash",   default=DEFAULT_CASH,   type=float, help="Starting cash $")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Optional: space-separated ticker list. "
                             "If omitted, uses the scored universe from UniverseSelector.")
    parser.add_argument("--split", action="store_true",
                        help="Split the date range in half and report in-sample vs "
                             "out-of-sample separately (guards against overfitting).")
    args = parser.parse_args()

    start_date = args.start
    end_date   = args.end
    cash       = args.cash

    # ── Load credentials ──────────────────────────────────────────────────────
    try:
        import config
        API_KEY = config.API_KEY
        SECRET_KEY = config.SECRET_KEY
    except ImportError:
        # Removed the local 'import os' since it's already imported at the top
        API_KEY = os.environ.get("ALPACA_API_KEY", "")
        SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set. Edit config.py or set environment variables.")
        sys.exit(1)

    # ── Initialise clients ────────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from risk.risk_manager import RiskManager, RiskConfig
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig
    from strategies.strategy_eod_reversion import EodReversionStrategy, EodReversionConfig

    tc = TradingClient(API_KEY, SECRET_KEY, paper=True)
    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())

    ucfg = UniverseConfig()

    # ── Load ticker universe ──────────────────────────────────────────────────
    if args.tickers:
        # CLI override: user passed explicit tickers — use them as-is
        tickers = args.tickers
        log.info(f"Using {len(tickers)} CLI-supplied tickers.")
    else:
        # Load scored universe from UniverseSelector (rebuilds if cache stale)
        log.info("Loading scored universe from UniverseSelector...")
        try:
            selector = UniverseSelector(dc, ucfg)
            tickers  = selector.get_universe(use_cache=True)
            if not tickers:
                raise ValueError("Universe returned empty list.")
            log.info(f"Universe loaded: {len(tickers)} tickers.")
        except Exception as exc:
            log.warning(
                f"UniverseSelector failed ({exc}). "
                f"Falling back to {len(FALLBACK_TICKERS)}-stock default list."
            )
            tickers = FALLBACK_TICKERS

    # ── Load historical universes for Point-In-Time backtesting ───────────────
    historical_universes_intraday = None
    historical_universes_eod = None

    intraday_universe_path = os.path.join("output", "historical_universes.json")
    if os.path.exists(intraday_universe_path):
        with open(intraday_universe_path, "r") as f:
            historical_universes_intraday = json.load(f)
        log.info(f"Loaded intraday PIT universes from {intraday_universe_path}")
    else:
        log.warning(f"Could not find {intraday_universe_path}. Intraday backtest will use static universe.")

    eod_universe_path = os.path.join("output", "historical_universes_eod.json")
    if os.path.exists(eod_universe_path):
        with open(eod_universe_path, "r") as f:
            historical_universes_eod = json.load(f)
        log.info(f"Loaded EOD PIT universes from {eod_universe_path}")
    else:
        log.warning(f"Could not find {eod_universe_path}. EOD backtest will use static universe.")

    # TEMPORARY VALIDATION — remove after confirming
    '''if historical_universes_intraday:
        weeks = sorted(historical_universes_intraday.keys())
        log.info(f"Intraday universe: {len(weeks)} weeks, "
                 f"first={weeks[0]} ({len(historical_universes_intraday[weeks[0]])} tickers), "
                 f"last={weeks[-1]} ({len(historical_universes_intraday[weeks[-1]])} tickers)")
    else:
        log.warning("Intraday universe is None — will use static ticker list")

    if historical_universes_eod:
        weeks = sorted(historical_universes_eod.keys())
        log.info(f"EOD universe: {len(weeks)} weeks, "
                 f"first={weeks[0]} ({len(historical_universes_eod[weeks[0]])} tickers), "
                 f"last={weeks[-1]} ({len(historical_universes_eod[weeks[-1]])} tickers)")
    else:
        log.warning("EOD universe is None — will use static ticker list")

    log.info(f"Combined data download list: {len(tickers)} unique tickers")'''

    # ── Expand data download list to cover all historical tickers ─────────────
    if not args.tickers:
        unique_tickers = set()
        for universe in [historical_universes_intraday, historical_universes_eod]:
            if universe:
                for week_list in universe.values():
                    unique_tickers.update(week_list)
        if unique_tickers:
            tickers = list(unique_tickers)
            log.info(f"Expanded data download list to {len(tickers)} unique historical tickers.")

    # ── Build the list of periods to run ──────────────────────────────────────
    periods = [("FULL", start_date, end_date)]
    if args.split:
        import datetime as _dt
        s = _dt.date.fromisoformat(start_date)
        e = _dt.date.fromisoformat(end_date)
        mid = s + (e - s) / 2
        periods = [
            ("IN-SAMPLE",      start_date,            mid.isoformat()),
            ("OUT-OF-SAMPLE",  (mid + _dt.timedelta(days=1)).isoformat(), end_date),
        ]
        log.info(f"Split mode: IN-SAMPLE {start_date}->{mid.isoformat()} | "
                 f"OUT-OF-SAMPLE {(mid + _dt.timedelta(days=1)).isoformat()}->{end_date}")

    # Only show blocking chart windows for a single full-period run.
    show_charts = not args.split

    for label, p_start, p_end in periods:
        _run_period(
            label, p_start, p_end,
            tc=tc, dc=dc, rm=rm, ucfg=ucfg, tickers=tickers,
            hu_intraday=historical_universes_intraday,
            hu_eod=historical_universes_eod,
            cash=cash, show_charts=show_charts,
        )

    log.info("run_backtest.py complete.")


if __name__ == "__main__":
    main()