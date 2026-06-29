"""
run_eod.py
----------
Scheduled task for the End-of-Day Mean-Reversion strategy.

PythonAnywhere scheduled task: 3:55 PM ET (20:55 UTC summer / 21:55 UTC winter).
This script is self-contained — it loads keys, runs the pre-session risk check,
loads the scored universe from cache, and hands off to EodReversionStrategy.run().

Logging appends to the same session log file created by run_session.py so the
full trading day is captured in a single file.

PYTHONANYWHERE SETUP
--------------------
  Tasks → Scheduled → Add:
    Time: 20:55 UTC (adjust for DST: 20:55 Jun–Oct, 21:55 Nov–Mar)
    Command: python /home/<username>/run_eod.py
"""

import datetime
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")

# =============================================================================
# LOGGING  (append to today's session log, same as run_session.py)
# =============================================================================

def _setup_logging() -> None:
    log_dir  = Path("../output")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"session_{datetime.date.today().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if somehow re-imported
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    else:
        # Ensure file handler for today is present even if console handler exists
        has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
        if not has_file:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)


# =============================================================================
# HELPERS
# =============================================================================

def _now_et() -> datetime.datetime:
    return datetime.datetime.now(ET)


def _is_trading_day() -> bool:
    """Return False on weekends."""
    return _now_et().weekday() < 5  # Mon–Fri


def _send_notification(message: str) -> None:
    """Send email notification if SMTP is configured in config.py."""
    try:
        import config as _cfg
        host = getattr(_cfg, "SMTP_HOST", None)
        port = getattr(_cfg, "SMTP_PORT", 587)
        user = getattr(_cfg, "SMTP_USER", None)
        pw   = getattr(_cfg, "SMTP_PASS", None)
        to   = getattr(_cfg, "NOTIFY_TO", None)
        if not all([host, user, pw, to]):
            return
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"{datetime.date.today()} — {message}")
        msg["Subject"] = f"EOD Session: {message[:50]}"
        msg["From"]    = user
        msg["To"]      = to
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pw)
            smtp.send_message(msg)
    except Exception as exc:
        logging.getLogger("run_eod").debug(f"Notification failed: {exc}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _setup_logging()
    log = logging.getLogger("run_eod")

    log.info("=" * 60)
    log.info(f"  EOD Reversion Session — {datetime.date.today()}")
    log.info("=" * 60)

    # ── Weekend guard ─────────────────────────────────────────────────────────
    if not _is_trading_day():
        log.info("Weekend — EOD session aborted.")
        sys.exit(0)

    # ── Load API credentials ──────────────────────────────────────────────────
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

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set. Edit config.py or set environment variables.")
        sys.exit(1)

    log.info(f"Mode: {'PAPER' if PAPER else 'LIVE'}")

    # ── Initialise Alpaca clients ─────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from risk.risk_manager import RiskManager, RiskConfig
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    from strategies.strategy_eod_reversion import EodReversionStrategy, EodReversionConfig

    tc = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # ── Pre-session risk gate ─────────────────────────────────────────────────
    rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
    log.info("Running pre-session risk check...")
    ok, reason = rm.pre_session_check()
    if not ok:
        log.warning(f"EOD session aborted by risk manager: {reason}")
        _send_notification(f"EOD aborted: {reason}")
        sys.exit(0)
    log.info("Pre-session checks passed.")

    # ── Load scored universe from cache ───────────────────────────────────────
    ucfg     = UniverseConfig(top_n=100)
    selector = UniverseSelector(dc, ucfg)
    log.info("Loading ticker universe from cache...")
    tickers  = selector.get_universe(use_cache=True)
    log.info(f"Universe loaded: {len(tickers)} tickers.")

    if not tickers:
        log.error("Empty universe — aborting EOD session.")
        sys.exit(1)

    # ── Run strategy ──────────────────────────────────────────────────────────
    ecfg     = EodReversionConfig()
    strategy = EodReversionStrategy(
        trading_client  = tc,
        data_client     = dc,
        config          = ecfg,
        universe_config = ucfg,
        risk_manager    = rm,
    )

    log.info("Starting EodReversionStrategy.run()...")
    try:
        strategy.run(tickers=tickers)
        log.info("EOD reversion strategy completed successfully.")
        _send_notification("EOD session completed.")
    except Exception as exc:
        log.exception(f"Unhandled exception in EOD session: {exc}")
        try:
            rm.emergency_close_all([], f"unhandled exception: {exc}")
        except Exception:
            pass
        _send_notification(f"EOD ERROR: {exc}")

    log.info("run_eod.py complete.")


if __name__ == "__main__":
    main()
