"""
preflight_check.py
------------------
Runs a pre-flight sanity check of the live paper trading setup.
Run this before the first session and after any config changes.

    python execution/preflight_check.py

Exit codes: 0 = all checks passed, 1 = one or more failures.
"""

import sys
import os
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
ET = ZoneInfo("America/New_York")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = 0


def check(label: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    global failures
    tag = PASS if ok else (WARN if warn else FAIL)
    print(f"  {tag}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok and not warn:
        failures += 1


# =============================================================================
# Credentials
# =============================================================================
print("\n=== Credentials ===")
try:
    import config
    API_KEY    = config.API_KEY
    SECRET_KEY = config.SECRET_KEY
    PAPER      = getattr(config, "PAPER", True)
    check("config.py loads",        True)
    check("API_KEY set",            bool(API_KEY))
    check("SECRET_KEY set",         bool(SECRET_KEY))
    check("PAPER=True (safe mode)", PAPER, f"PAPER={PAPER}")
except ImportError as exc:
    check("config.py loads", False, str(exc))
    sys.exit(1)

# =============================================================================
# Alpaca API connectivity
# =============================================================================
print("\n=== Alpaca API ===")
tc = None
dc = None
try:
    from alpaca.trading.client import TradingClient
    tc   = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    acct = tc.get_account()
    cash = float(acct.cash)
    pv   = float(acct.portfolio_value)
    check("TradingClient connects",  True)
    check("Account cash > $0",       cash > 0,                f"${cash:,.2f}")
    check("Portfolio value",         True,                    f"${pv:,.2f}")
    check("Account not restricted",  not acct.trading_blocked,
          "trading_blocked=True" if acct.trading_blocked else "")
except Exception as exc:
    check("TradingClient connects", False, str(exc))

try:
    from alpaca.data.historical import StockHistoricalDataClient
    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    check("DataClient connects", True)
except Exception as exc:
    check("DataClient connects", False, str(exc))

# =============================================================================
# Universe cache
# =============================================================================
print("\n=== Universe Cache ===")
try:
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    ucfg       = UniverseConfig()
    selector   = UniverseSelector(dc, ucfg)
    tickers    = selector.get_universe(use_cache=True)
    cache_path = Path(ucfg.cache_path)
    if cache_path.exists():
        age_days = (
            datetime.datetime.now()
            - datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)
        ).days
        check("Universe cache exists", True,
              f"{len(tickers)} tickers, {age_days}d old")
        check("Cache < 7 days old",    age_days < 7,
              f"{age_days}d", warn=(age_days >= 7))
    else:
        check("Universe cache exists", False,
              "run rebuild_universe.py first")
except Exception as exc:
    check("Universe loads", False, str(exc))

# =============================================================================
# Historical universes (point-in-time candidate selection)
# =============================================================================
print("\n=== Historical Universes ===")
hu_path = Path(__file__).parent.parent / "output" / "historical_universes.json"
if hu_path.exists():
    with open(hu_path) as f:
        hu = json.load(f)
    weeks = sorted(hu.keys())
    check("historical_universes.json exists", True,
          f"{len(weeks)} weeks, latest={weeks[-1] if weeks else 'none'}")
else:
    check("historical_universes.json exists", False,
          "run data_collection/build_historical_universes.py first", warn=True)

# =============================================================================
# Risk manager
# =============================================================================
print("\n=== Risk Manager ===")
try:
    from risk.risk_manager import RiskManager, RiskConfig
    rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
    check("RiskManager initializes", True)
    # Don't call pre_session_check() — it hits VIX/news APIs and could abort
    # the preflight on a macro blowup day. Just verify instantiation.
except Exception as exc:
    check("RiskManager initializes", False, str(exc))

# =============================================================================
# Strategy instantiation
# =============================================================================
print("\n=== Strategy ===")
try:
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig
    icfg = IntradayConfig()
    strat = IntradayStrategy(tc, dc, rm, config=icfg, universe_config=ucfg)
    check("IntradayStrategy initializes", True,
          f"SL={icfg.vwap_sl_atr} TP={icfg.vwap_tp_extension_atr} max_pos={icfg.vwap_max_positions}")
except Exception as exc:
    check("IntradayStrategy initializes", False, str(exc))

# =============================================================================
# Output directory
# =============================================================================
print("\n=== Output Directory ===")
out_dir = Path(__file__).parent.parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)
check("output/ directory writable", os.access(out_dir, os.W_OK), str(out_dir))

# =============================================================================
# Result
# =============================================================================
print()
if failures == 0:
    print("  All checks passed. Ready for live paper trading.")
else:
    print(f"  {failures} check(s) failed. Fix the issues above before going live.")
sys.exit(0 if failures == 0 else 1)
