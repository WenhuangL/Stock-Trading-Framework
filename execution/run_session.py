"""
run_session.py
--------------
Master orchestrator for the full trading day.
Scheduled on PythonAnywhere as an always-on task starting at 7:55 AM ET.

SCHEDULE (all times ET)
-----------------------
  7:55 AM   Script starts, clients initialised, universe loaded
  8:00 AM   Premarket watchlist preparation begins
  9:25 AM   Pre-session risk check — aborts if macro conditions unsafe
  9:30 AM   Phase 1: Opening Range Breakout (ORB)
 11:00 AM   Phase 2: VWAP Reversion  ─┐  managed internally
  3:05 PM   Phase 3: Power Hour       ─┘  by IntradayStrategy.run()
  3:55 PM   Intraday session ends — EOD reversion script takes over
  8:00 PM   EOD reversion script (run_eod.py) terminates independently

PYTHONANYWHERE SETUP
--------------------
  1. Upload all project files to your PythonAnywhere home directory.
  2. Go to Tasks → Always-on tasks → add:
         python /home/<username>/run_session.py
  3. Separately, add a scheduled task at 15:55 ET (20:55 UTC in summer):
         python /home/<username>/run_eod.py
  4. Add a daily task at 12:00 UTC to rebuild the universe:
         python /home/<username>/Stock-Trading-Framework/execution/rebuild_universe.py
         (MIN_REBUILD_AGE_DAYS in rebuild_universe.py controls actual frequency)
  5. Set your API keys in config.py (never hardcode them here).

FILE LAYOUT
-----------
  run_session.py       ← this file (intraday session)
  run_eod.py           ← EOD reversion (separate PythonAnywhere task)
  rebuild_universe.py  ← daily universe rebuild (MIN_REBUILD_AGE_DAYS controls frequency)
  config.py            ← API keys and shared settings (not in version control)
  strategy_intraday.py
  strategy_eod_reversion.py
  risk_manager.py
  stock_universe.py
  analyze.py
  webscraper.py
  analysis/            ← SQLite DB, universe cache, risk event log

LOGGING
-------
All output goes to stdout (PythonAnywhere captures this to the task log)
and to analysis/session_YYYYMMDD.log for persistent review.
"""

import datetime
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Try loading config; fall back to environment variables ────────────────────
try:
    import config
    API_KEY    = config.API_KEY
    SECRET_KEY = config.SECRET_KEY
    PAPER      = getattr(config, "PAPER", True)
except ImportError:
    import os
    API_KEY    = os.environ.get("ALPACA_API_KEY", "")
    SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    PAPER      = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

ET = ZoneInfo("America/New_York")


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _setup_logging() -> None:
    log_dir  = Path(__file__).parent.parent / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"session_{datetime.date.today().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


log = logging.getLogger("run_session")


# =============================================================================
# TIME UTILITIES
# =============================================================================

def _now_et() -> datetime.datetime:
    return datetime.datetime.now(ET)


def _wait_until(hhmm: str, label: str = "") -> None:
    """Block until HH:MM ET today. Logs a message every 15 minutes."""
    h, m = map(int, hhmm.split(":"))
    target = _now_et().replace(hour=h, minute=m, second=0, microsecond=0)
    while _now_et() < target:
        remaining = (target - _now_et()).total_seconds()
        if remaining > 0:
            msg = f"Waiting for {hhmm} ET"
            if label:
                msg += f" ({label})"
            msg += f" — {remaining/60:.0f} min remaining"
            log.info(msg)
            time.sleep(min(remaining, 900))  # wake at most every 15 min


def _is_trading_day() -> bool:
    """Return False on weekends. (Holidays are handled by Alpaca rejecting orders.)"""
    return _now_et().weekday() < 5  # Mon-Fri


# =============================================================================
# MAIN SESSION
# =============================================================================

def main() -> None:
    _setup_logging()
    log.info("=" * 60)
    log.info(f"  Trading Session — {datetime.date.today()}  {'[PAPER]' if PAPER else '[LIVE]'}")
    log.info("=" * 60)

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set. Edit config.py or set environment variables.")
        sys.exit(1)

    if not _is_trading_day():
        log.info("Weekend — session aborted.")
        sys.exit(0)

    # ── Initialise clients ────────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from risk.risk_manager import RiskManager, RiskConfig, build_and_cache_sector_map
    from data_collection.stock_universe import UniverseConfig
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig

    tc = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    ucfg     = UniverseConfig(top_n=100)
    rcfg     = RiskConfig()
    icfg     = IntradayConfig()

    rm       = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=rcfg)
    strategy = IntradayStrategy(tc, dc, rm, config=icfg, universe_config=ucfg)

    # Ensure sector map is in the universe cache (used by risk manager)
    build_and_cache_sector_map()

    # ── 8:00 AM: Premarket preparation ───────────────────────────────────────
    _wait_until("08:00", "premarket prep")
    log.info("Starting premarket watchlist preparation...")
    watchlist = strategy.prepare_watchlist()
    log.info(f"Watchlist complete: {len(watchlist)} candidates.")

    # ── 9:25 AM: Pre-session risk gate ────────────────────────────────────────
    _wait_until("09:25", "pre-session check")
    ok, reason = rm.pre_session_check()
    if not ok:
        log.warning(f"SESSION ABORTED by risk manager: {reason}")
        _send_notification(f"Session aborted: {reason}")
        sys.exit(0)

    log.info("Pre-session checks passed. Starting live session.")

    # ── 9:30 AM → 3:55 PM: Run all intraday phases ───────────────────────────
    _wait_until("09:30", "market open")
    try:
        session_result = strategy.run()
        total_pnl      = session_result.get("total_pnl", 0.0)
        num_trades     = len(session_result.get("trades", []))
        log.info(f"Intraday session complete: {num_trades} trades | P&L ${total_pnl:+.2f}")
        _send_notification(f"Intraday done: {num_trades} trades | P&L ${total_pnl:+.2f}")
    except Exception as exc:
        log.exception(f"Unhandled exception in intraday session: {exc}")
        # Emergency close on any uncaught error
        try:
            rm.emergency_close_all(strategy._positions, f"unhandled exception: {exc}")
        except Exception:
            pass
        _send_notification(f"Session ERROR: {exc}")

    # ── 3:55 PM: EOD reversion strategy takes over (separate scheduled task) ──
    log.info(
        "Intraday session ended. EOD reversion strategy (run_eod.py) "
        "is scheduled separately at 3:55 PM ET."
    )
    log.info("run_session.py complete.")


# =============================================================================
# OPTIONAL: PythonAnywhere email notification
# (requires SMTP config in config.py — delete this function if not needed)
# =============================================================================

def _send_notification(message: str) -> None:
    """
    Send a session summary notification.
    Configure SMTP settings in config.py to enable:
        SMTP_HOST  = "smtp.gmail.com"
        SMTP_PORT  = 587
        SMTP_USER  = "your@gmail.com"
        SMTP_PASS  = "your_app_password"
        NOTIFY_TO  = "your@gmail.com"
    """
    try:
        host  = getattr(config, "SMTP_HOST",  None)
        port  = getattr(config, "SMTP_PORT",  587)
        user  = getattr(config, "SMTP_USER",  None)
        pw    = getattr(config, "SMTP_PASS",  None)
        to    = getattr(config, "NOTIFY_TO",  None)
        if not all([host, user, pw, to]):
            return
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"{datetime.date.today()} — {message}")
        msg["Subject"] = f"Trading Session: {message[:50]}"
        msg["From"]    = user
        msg["To"]      = to
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pw)
            smtp.send_message(msg)
    except Exception as exc:
        log.debug(f"Notification failed: {exc}")


# =============================================================================
# EOD REVERSION RUNNER  (run_eod.py — separate PythonAnywhere scheduled task)
# =============================================================================
# Paste this into a separate file called run_eod.py:
#
# import sys, logging
# from pathlib import Path
# logging.basicConfig(level=logging.INFO,
#     format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
# try:
#     import config
#     API_KEY, SECRET_KEY = config.API_KEY, config.SECRET_KEY
#     PAPER = getattr(config, "PAPER", True)
# except ImportError:
#     import os
#     API_KEY    = os.environ.get("ALPACA_API_KEY", "")
#     SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
#     PAPER      = True
# from alpaca.trading.client import TradingClient
# from alpaca.data.historical import StockHistoricalDataClient
# from risk_manager import RiskManager, RiskConfig, build_and_cache_sector_map
# from stock_universe import UniverseSelector, UniverseConfig
# from strategy_eod_reversion import EodReversionStrategy, EodReversionConfig
# tc = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
# dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
# rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
# ok, reason = rm.pre_session_check()
# if not ok:
#     logging.warning(f"EOD session aborted: {reason}")
#     sys.exit(0)
# ucfg = UniverseConfig(top_n=100)
# tickers = UniverseSelector(dc, ucfg).get_universe(use_cache=True)
# strategy = EodReversionStrategy(tc, dc, config=EodReversionConfig(),
#                                 universe_config=ucfg)
# strategy.run(tickers=tickers)


# =============================================================================
# UNIVERSE REBUILD RUNNER  (rebuild_universe.py — weekly Monday 7 AM task)
# =============================================================================
# Paste this into a separate file called rebuild_universe.py:
#
# import logging
# logging.basicConfig(level=logging.INFO)
# try:
#     import config
#     API_KEY, SECRET_KEY = config.API_KEY, config.SECRET_KEY
# except ImportError:
#     import os
#     API_KEY    = os.environ.get("ALPACA_API_KEY", "")
#     SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
# from alpaca.data.historical import StockHistoricalDataClient
# from stock_universe import UniverseSelector, UniverseConfig
# from risk_manager import build_and_cache_sector_map
# dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
# selector = UniverseSelector(dc, UniverseConfig(top_n=150))
# selector.get_universe(use_cache=False)   # force full rebuild
# build_and_cache_sector_map()             # refresh sector data
# print("Universe rebuild complete.")


# =============================================================================
# BACKTEST RUNNER  (run_backtest.py — run manually, not scheduled)
# =============================================================================
# Paste this into run_backtest.py and run locally or on PythonAnywhere manually:
#
# import logging
# logging.basicConfig(level=logging.INFO)
# try:
#     import config
#     API_KEY, SECRET_KEY = config.API_KEY, config.SECRET_KEY
# except ImportError:
#     import os
#     API_KEY    = os.environ.get("ALPACA_API_KEY", "")
#     SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
# from alpaca.trading.client import TradingClient
# from alpaca.data.historical import StockHistoricalDataClient
# from risk_manager import RiskManager, RiskConfig
# from stock_universe import UniverseSelector, UniverseConfig
# from strategy_intraday import IntradayStrategy, IntradayConfig
# from strategy_eod_reversion import EodReversionStrategy, EodReversionConfig
#
# tc = TradingClient(API_KEY, SECRET_KEY, paper=True)
# dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
# rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
#
# # Select a small universe for initial testing
# tickers = ["NVDA","TSLA","META","AMZN","MSFT","AAPL","GOOG","AMD",
#            "CRM","PANW","CRWD","DDOG","MRVL","SMCI","MU",
#            "NFLX","ADBE","INTC","QCOM","AVGO"]
#
# # ── Intraday backtest ──────────────────────────────────────────────────────
# intraday = IntradayStrategy(tc, dc, rm,
#     config=IntradayConfig(), universe_config=UniverseConfig())
# intraday_results = intraday.backtest(
#     tickers     = tickers,
#     start_date  = "2023-01-01",
#     end_date    = "2023-12-31",
#     initial_cash= 100_000,
# )
# intraday.print_summary(intraday_results)
# intraday.plot_backtest(intraday_results)
#
# # ── EOD backtest ───────────────────────────────────────────────────────────
# eod = EodReversionStrategy(tc, dc,
#     config=EodReversionConfig(), universe_config=UniverseConfig())
# eod_results = eod.backtest(
#     tickers     = tickers,
#     start_date  = "2023-01-01",
#     end_date    = "2023-12-31",
#     initial_cash= 100_000,
# )
# eod.print_summary(eod_results)
# eod.plot_backtest(eod_results)


if __name__ == "__main__":
    main()
