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
    python run_backtest.py                              # all three strategies (default)
    python run_backtest.py --strategies intraday        # intraday only
    python run_backtest.py --strategies eod swing       # skip intraday
    python run_backtest.py --strategies swing --split   # swing, in/out-of-sample

    # Custom date range:
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

_ALL_STRATEGIES = frozenset({"intraday", "eod", "swing", "rsi2", "microcap"})


def _print_combined_summary(
    intraday_results: dict,
    eod_results:      dict,
    initial_cash:     float,
    swing_results:    dict = None,
    rsi2_results:     dict = None,
    microcap_results: dict = None,
    strategies:       frozenset = _ALL_STRATEGIES,
) -> None:
    id_summary       = intraday_results.get("summary", {}) if "intraday" in strategies else {}
    eod_summary      = eod_results.get("summary", {})      if "eod"      in strategies else {}
    swing_summary    = (swing_results or {}).get("summary", {}) if "swing" in strategies else {}
    rsi2_summary     = (rsi2_results  or {}).get("summary", {}) if "rsi2"  in strategies else {}
    microcap_summary = (microcap_results or {}).get("summary", {}) if "microcap" in strategies else {}

    id_trades = id_summary.get("num_trades", 0)
    id_pnl    = id_summary.get("final_value", initial_cash) - id_summary.get("initial_cash", initial_cash)

    eod_trades = eod_summary.get("num_trades", 0)
    eod_pnl    = eod_summary.get("total_pnl",
                   eod_summary.get("final_value", initial_cash) - initial_cash)

    sw_trades = swing_summary.get("num_trades", 0)
    sw_pnl    = swing_summary.get("final_value", initial_cash) - initial_cash if swing_summary else 0.0

    rsi2_trades = rsi2_summary.get("num_trades", 0)
    rsi2_pnl    = rsi2_summary.get("final_value", initial_cash) - initial_cash \
                  if rsi2_summary else 0.0

    mc_trades = microcap_summary.get("num_trades", 0)
    mc_pnl    = microcap_summary.get("final_value", initial_cash) - initial_cash \
                if microcap_summary else 0.0

    combined     = id_pnl + eod_pnl + sw_pnl + rsi2_pnl + mc_pnl
    total_trades = id_trades + eod_trades + sw_trades + rsi2_trades + mc_trades
    combined_pct = (combined / initial_cash) * 100 if initial_cash else 0

    title = "BACKTEST SUMMARY" if len(strategies) == 1 else "COMBINED BACKTEST SUMMARY"
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  {'Strategy':<25}  {'Trades':>7}  {'P&L':>12}  {'Return':>8}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*12}  {'-'*8}")
    if "intraday" in strategies:
        print(f"  {'Intraday (ORB+VWAP+PH)':<25}  {id_trades:>7}  "
              f"${id_pnl:>+11,.2f}  {id_pnl/initial_cash*100:>+7.2f}%")
    if "eod" in strategies:
        print(f"  {'EOD Reversion':<25}  {eod_trades:>7}  "
              f"${eod_pnl:>+11,.2f}  {eod_pnl/initial_cash*100:>+7.2f}%")
    if "swing" in strategies:
        print(f"  {'Swing Breakout':<25}  {sw_trades:>7}  "
              f"${sw_pnl:>+11,.2f}  {sw_pnl/initial_cash*100:>+7.2f}%")
    if "rsi2" in strategies and rsi2_summary:
        print(f"  {'RSI-2 Mean Reversion':<25}  {rsi2_trades:>7}  "
              f"${rsi2_pnl:>+11,.2f}  {rsi2_pnl/initial_cash*100:>+7.2f}%")
    if "microcap" in strategies and microcap_summary:
        print(f"  {'Micro-Cap Reversion':<25}  {mc_trades:>7}  "
              f"${mc_pnl:>+11,.2f}  {mc_pnl/initial_cash*100:>+7.2f}%")
    if len(strategies) > 1:
        print(f"  {'-'*25}  {'-'*7}  {'-'*12}  {'-'*8}")
        print(f"  {'COMBINED':<25}  {total_trades:>7}  "
              f"${combined:>+11,.2f}  {combined_pct:>+7.2f}%")
    print(f"\n  Starting capital: ${initial_cash:,.0f}")
    print(f"  Ending estimate:  ${initial_cash + combined:,.0f}")
    print("=" * 70 + "\n")

    detail_rows = []
    if "intraday" in strategies:
        detail_rows.append(("INTRADAY DETAIL", id_summary))
    if "eod" in strategies:
        detail_rows.append(("EOD DETAIL", eod_summary))
    if "swing" in strategies and swing_summary:
        detail_rows.append(("SWING DETAIL", swing_summary))
    if rsi2_summary:
        detail_rows.append(("RSI-2 DETAIL", rsi2_summary))
    if microcap_summary:
        detail_rows.append(("MICRO-CAP DETAIL", microcap_summary))

    for label, summary in detail_rows:
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
    hu_intraday, hu_eod, hu_rsi2=None, hu_rsi2_short=None, hu_microcap=None,
    cash, show_charts,
    strategies: frozenset = _ALL_STRATEGIES,
):
    """Run selected strategy backtests for one date range and print results."""
    log.info("=" * 60)
    log.info(f"  Backtest Runner - {label}  [{', '.join(sorted(strategies))}]")
    log.info("=" * 60)
    log.info(f"  Tickers:  {len(tickers)} stocks")
    log.info(f"  Period:   {start_date} -> {end_date}")
    log.info(f"  Capital:  ${cash:,.0f}")
    log.info(f"  Strategies: {', '.join(sorted(strategies))}")
    log.info("")

    intraday_results: dict = {}
    eod_results:      dict = {}
    swing_results:    dict = {}
    rsi2_results:     dict = {}
    microcap_results: dict = {}
    intraday = eod = swing = rsi2 = microcap = None

    # ── Intraday ──────────────────────────────────────────────────────────────
    if "intraday" in strategies:
        from strategies.strategy_intraday import IntradayStrategy, IntradayConfig
        log.info("-" * 60)
        log.info("  Running IntradayStrategy.backtest()...")
        log.info("-" * 60)
        intraday = IntradayStrategy(
            trading_client=tc, data_client=dc, risk_manager=rm,
            config=IntradayConfig(), universe_config=ucfg,
        )
        try:
            intraday_results = intraday.backtest(
                tickers=tickers, start_date=start_date, end_date=end_date,
                initial_cash=cash, historical_universes=hu_intraday,
            )
            intraday.print_summary(intraday_results)
        except Exception as exc:
            log.exception(f"Intraday backtest failed: {exc}")

    # ── EOD ───────────────────────────────────────────────────────────────────
    if "eod" in strategies:
        from strategies.strategy_eod_reversion import EodReversionStrategy, EodReversionConfig
        log.info("-" * 60)
        log.info("  Running EodReversionStrategy.backtest()...")
        log.info("-" * 60)
        eod = EodReversionStrategy(
            trading_client=tc, data_client=dc, config=EodReversionConfig(),
            universe_config=ucfg, risk_manager=rm,
        )
        try:
            eod_results = eod.backtest(
                tickers=tickers, start_date=start_date, end_date=end_date,
                initial_cash=cash, historical_universes=hu_eod,
            )
            eod.print_summary(eod_results)
        except Exception as exc:
            log.exception(f"EOD backtest failed: {exc}")

    # ── Swing ─────────────────────────────────────────────────────────────────
    if "swing" in strategies:
        from strategies.strategy_swing import SwingStrategy, SwingConfig
        log.info("-" * 60)
        log.info("  Running SwingStrategy.backtest()...")
        log.info("-" * 60)
        swing = SwingStrategy(
            trading_client=tc, data_client=dc, config=SwingConfig(),
            risk_manager=rm, universe_config=ucfg,
        )
        try:
            swing_results = swing.backtest(
                tickers=tickers, start_date=start_date, end_date=end_date,
                initial_cash=cash, historical_universes=hu_intraday,
            )
            swing.print_summary(swing_results)
        except Exception as exc:
            log.exception(f"Swing backtest failed: {exc}")

    # ── RSI-2 ─────────────────────────────────────────────────────────────────
    if "rsi2" in strategies:
        from strategies.strategy_rsi2 import Rsi2Strategy, Rsi2Config
        from data_collection.stock_universe import Rsi2UniverseConfig
        log.info("-" * 60)
        log.info("  Running Rsi2Strategy.backtest()...")
        log.info("-" * 60)
        rsi2 = Rsi2Strategy(
            trading_client=tc, data_client=dc, config=Rsi2Config(),
            risk_manager=rm, universe_config=Rsi2UniverseConfig(),
        )
        try:
            rsi2_results = rsi2.backtest(
                tickers=tickers, start_date=start_date, end_date=end_date,
                initial_cash=cash, historical_universes=hu_rsi2,
                historical_universes_short=hu_rsi2_short,
            )
            rsi2.print_summary(rsi2_results)
        except Exception as exc:
            log.exception(f"RSI-2 backtest failed: {exc}")

    # ── Micro-cap mean reversion ──────────────────────────────────────────────
    if "microcap" in strategies:
        from strategies.strategy_microcap_reversion import (
            MicrocapReversionStrategy, MicrocapReversionConfig,
        )
        log.info("-" * 60)
        log.info("  Running MicrocapReversionStrategy.backtest()...")
        log.info("-" * 60)
        microcap = MicrocapReversionStrategy(
            trading_client=tc, data_client=dc, config=MicrocapReversionConfig(),
            risk_manager=rm, universe_config=ucfg,
        )
        try:
            microcap_results = microcap.backtest(
                tickers=tickers, start_date=start_date, end_date=end_date,
                initial_cash=cash, historical_universes=hu_microcap,
            )
            microcap.print_summary(microcap_results)
        except Exception as exc:
            log.exception(f"Micro-cap backtest failed: {exc}")

    # ── Combined + benchmark ──────────────────────────────────────────────────
    _print_combined_summary(
        intraday_results, eod_results, cash, swing_results,
        rsi2_results=rsi2_results, microcap_results=microcap_results,
        strategies=strategies,
    )

    spy_pct = _spy_benchmark_pct(dc, start_date, end_date)
    if spy_pct is not None:
        parts = []
        if "intraday" in strategies:
            parts.append(f"Intraday {intraday_results.get('summary', {}).get('total_return_pct', 0.0):+.2f}%")
        if "eod" in strategies:
            parts.append(f"EOD {eod_results.get('summary', {}).get('total_return_pct', 0.0):+.2f}%")
        if "swing" in strategies:
            parts.append(f"Swing {swing_results.get('summary', {}).get('total_return_pct', 0.0):+.2f}%")
        if "rsi2" in strategies:
            parts.append(f"RSI-2 {rsi2_results.get('summary', {}).get('total_return_pct', 0.0):+.2f}%")
        if "microcap" in strategies:
            parts.append(f"Micro-cap {microcap_results.get('summary', {}).get('total_return_pct', 0.0):+.2f}%")
        print(f"  BENCHMARK — SPY buy & hold ({label}): {spy_pct:+.2f}%")
        print(f"    vs {' | vs '.join(parts)}")
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
            rsi2_results=rsi2_results,
            microcap_results=microcap_results,
        )
        log.info(f"Trades saved to output/trades.db  (run_id={run_id})")
    except Exception as exc:
        log.warning(f"Failed to save trades to database: {exc}")

    if show_charts:
        log.info("Generating charts (close each window to continue)...")
        if intraday and intraday_results:
            try:
                intraday.plot_backtest(intraday_results)
            except Exception as exc:
                log.warning(f"Intraday plot failed: {exc}")
        if eod and eod_results:
            try:
                eod.plot_backtest(eod_results)
            except Exception as exc:
                log.warning(f"EOD plot failed: {exc}")
        if swing and swing_results:
            try:
                swing.plot_backtest(swing_results)
            except Exception as exc:
                log.warning(f"Swing plot failed: {exc}")
        if rsi2 and rsi2_results:
            try:
                rsi2.plot_backtest(rsi2_results)
            except Exception as exc:
                log.warning(f"RSI-2 plot failed: {exc}")
        if microcap and microcap_results:
            try:
                microcap.plot_backtest(microcap_results)
            except Exception as exc:
                log.warning(f"Micro-cap plot failed: {exc}")

    return intraday_results, eod_results, rsi2_results, microcap_results


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # ── CLI argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Run intraday + EOD + swing backtests on the same ticker list and date range."
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
    parser.add_argument(
        "--strategies", nargs="+",
        choices=["intraday", "eod", "swing", "rsi2", "microcap"],
        default=["intraday", "eod", "swing", "rsi2", "microcap"],
        help="Which strategies to backtest (default: all). "
             "Use --strategies rsi2 to run only RSI-2 (fast, no minute-bar download); "
             "--strategies microcap for the micro-cap mean-reversion strategy.",
    )
    parser.add_argument(
        "--intraday-universe", default=None,
        help="Override path to the intraday PIT universe JSON (default: "
             "output/historical_universes.json). Used to A/B alternate universes.",
    )
    args = parser.parse_args()

    strategies = frozenset(args.strategies)

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

    intraday_universe_path = args.intraday_universe or os.path.join("output", "historical_universes.json")
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

    historical_universes_rsi2 = None
    for rsi2_universe_path in [
        os.path.join("output", "historical_universes_rsi2_bidir.json"),
        os.path.join("output", "historical_universes_rsi2.json"),
    ]:
        if os.path.exists(rsi2_universe_path):
            with open(rsi2_universe_path, "r") as f:
                historical_universes_rsi2 = json.load(f)
            log.info(f"Loaded RSI-2 PIT universes from {rsi2_universe_path}")
            break
    if historical_universes_rsi2 is None:
        log.warning(
            "Could not find RSI-2 PIT universe. "
            "RSI-2 backtest will use static universe. "
            "Run: python data_collection/build_historical_universes.py "
            "--universe sp500 --out output/historical_universes_rsi2_bidir.json "
            "--start <date> --end <date>"
        )

    historical_universes_rsi2_short = None
    rsi2_short_path = os.path.join("output", "historical_universes_rsi2_short.json")
    if os.path.exists(rsi2_short_path):
        with open(rsi2_short_path) as f:
            historical_universes_rsi2_short = json.load(f)
        log.info(f"Loaded RSI-2 short universe from {rsi2_short_path}")

    historical_universes_microcap = None
    microcap_universe_path = os.path.join("output", "historical_universes_microcap.json")
    if os.path.exists(microcap_universe_path):
        with open(microcap_universe_path) as f:
            historical_universes_microcap = json.load(f)
        log.info(f"Loaded micro-cap PIT universes from {microcap_universe_path}")
    elif "microcap" in strategies:
        log.warning(
            "Could not find micro-cap PIT universe. Micro-cap backtest will use the "
            "static ticker list. Build it with: python data_collection/"
            "build_microcap_universe.py --start <date> --end <date>"
        )

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
        for universe in [
            historical_universes_intraday, historical_universes_eod,
            historical_universes_rsi2, historical_universes_rsi2_short,
            historical_universes_microcap,
        ]:
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
            hu_rsi2=historical_universes_rsi2,
            hu_rsi2_short=historical_universes_rsi2_short,
            hu_microcap=historical_universes_microcap,
            cash=cash, show_charts=show_charts,
            strategies=strategies,
        )

    log.info("run_backtest.py complete.")


if __name__ == "__main__":
    main()