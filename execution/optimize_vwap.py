"""
optimize_vwap.py
----------------
Grid search optimizer for VWAP strategy parameters.

Systematically tests combinations of parameters over a short representative period
to find the optimal configuration without running a full-year backtest for each.

USAGE
-----
    python execution/optimize_vwap.py

WHAT IT SWEEPS
--------------
Configure which dimensions to sweep by setting the *_RANGE variables below:
  - Single-element list  -> fixes that param at that value (not swept, not shown in table)
  - Multi-element list   -> sweeps that dimension (shown in progress + results table)

Phase 1 (SL/TP):   SL_ATR_RANGE x TP_EXT_RANGE
Phase 2 (max_pos): set MAX_POS_RANGE = [4, 5, 6, 7, 8]
Phase 3 (current): BREAKEVEN_ATR_RANGE x EXT_ATR_RANGE x VOL_DECAY_RANGE

AVOIDING EXPONENTIAL BLOWUP
----------------------------
Don't add more than two new dimensions at a time. Recommended workflow:
  1. Find optimal SL+TP pair (Phase 1).
  2. Fix SL/TP, sweep max_positions (Phase 2).
  3. Fix all above, sweep breakeven/extension/vol_decay (Phase 3).
  4. Fix all above, test dead zone variants via run_backtest.py manually.

RESULTS
-------
- Console: progress line per combo + top-15 table sorted by expectancy
- CSV: all results saved to output/optimize_<YYYYMMDD_HHMM>.csv
"""

import itertools
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("optimize_vwap")

# =============================================================================
# GRID CONFIGURATION — edit these to change what gets swept
# =============================================================================

TEST_START = "2023-10-01"
TEST_END   = "2023-10-31"   # ~23 trading days; use "2023-07-01"/"2023-09-29" for Q3

# ── Phase 1: SL/TP sweep ──────────────────────────────────────────────────────
SL_ATR_RANGE  = [1.40]                                  # current optimal: 1.40
TP_EXT_RANGE  = [1.00]                                  # current optimal: 1.00

# ── Phase 2: optional max_pos sweep ──────────────────────────────────────────
MAX_POS_RANGE = []   # [] = fixed at IntradayConfig default; e.g. [4, 5, 6, 7, 8]

# ── Phase 3: breakeven / entry threshold / vol filter ────────────────────────
BREAKEVEN_ATR_RANGE = [0.50, 0.75, 1.00, 1.25, 1.50]       # current: 1.00
EXT_ATR_RANGE       = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25]  # current: 2.50
VOL_DECAY_RANGE     = [0.60, 0.70, 0.80]                    # current: 0.80

# Current (baseline) config values — used to mark the baseline row in the output table
CURRENT_SL_ATR        = 1.40
CURRENT_TP_EXT        = 1.00
CURRENT_MAX_POS       = 6
CURRENT_BREAKEVEN_ATR = 1.00
CURRENT_EXT_ATR       = 2.50
CURRENT_VOL_DECAY     = 0.80

# =============================================================================
# RESULT EXTRACTION
# =============================================================================

def _extract_metrics(
    result: dict,
    sl_atr: float,
    tp_ext: float,
    max_pos,
    breakeven_atr: float,
    ext_atr: float,
    vol_decay: float,
) -> dict:
    trades = result.get("trades", [])
    vwap   = [t for t in trades if t.get("phase") == "vwap"]

    tp_n = sum(1 for t in vwap if t.get("exit_reason") == "take_profit")
    sl_n = sum(1 for t in vwap if t.get("exit_reason") == "stop_loss")
    mh_n = sum(1 for t in vwap if t.get("exit_reason") == "max_hold")

    # Compute all metrics from VWAP trades only (not the whole-strategy summary,
    # which would mix in ORB trades and skew avg_win/avg_loss/expectancy).
    pnls   = [t.get("pnl", 0.0) for t in vwap]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl  = sum(pnls)
    num_trades = len(pnls)
    win_rate   = len(wins) / num_trades * 100 if num_trades else 0.0
    avg_pnl    = total_pnl / num_trades if num_trades else 0.0
    avg_win    = sum(wins) / len(wins) if wins else 0.0
    avg_loss   = sum(losses) / len(losses) if losses else 0.0

    p   = len(wins) / num_trades if num_trades else 0.0
    q   = len(losses) / num_trades if num_trades else 0.0
    exp = p * avg_win + q * avg_loss  # expected P&L per trade

    return {
        "sl_atr":     sl_atr,
        "tp_ext":     tp_ext,
        "max_pos":    max_pos if max_pos is not None else CURRENT_MAX_POS,
        "be_atr":     breakeven_atr,
        "ext_atr":    ext_atr,
        "vol_decay":  vol_decay,
        "trades":     num_trades,
        "win_pct":    round(win_rate, 1),
        "total_pnl":  round(total_pnl, 2),
        "avg_pnl":    round(avg_pnl, 2),
        "avg_win":    round(avg_win, 2),
        "avg_loss":   round(avg_loss, 2),
        "expectancy": round(exp, 2),
        "tp_n":       tp_n,
        "sl_n":       sl_n,
        "mh_n":       mh_n,
    }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def _print_table(rows: list[dict], show_flags: dict) -> None:
    """Print top-15 combinations sorted by expectancy.

    show_flags: dict with keys sl, tp, pos, be, ext, vd — True if that dimension
    was swept (i.e. has more than one value in its range).
    """
    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows).sort_values("expectancy", ascending=False)

    # Mark the current-config baseline row
    is_baseline = (
        (df["sl_atr"]    == CURRENT_SL_ATR) &
        (df["tp_ext"]    == CURRENT_TP_EXT) &
        (df["max_pos"]   == CURRENT_MAX_POS) &
        (df["be_atr"]    == CURRENT_BREAKEVEN_ATR) &
        (df["ext_atr"]   == CURRENT_EXT_ATR) &
        (df["vol_decay"] == CURRENT_VOL_DECAY)
    )

    print()
    print("=" * 100)
    print("  TOP 15 COMBINATIONS -- sorted by expectancy (expected P&L per trade)")
    print("=" * 100)

    # Build header dynamically: only show columns for dimensions that were swept
    hdr = f"{'#':>5}"
    if show_flags.get("sl"):  hdr += f"  {'SL_ATR':>7}"
    if show_flags.get("tp"):  hdr += f"  {'TP_EXT':>7}"
    if show_flags.get("pos"): hdr += f"  {'MaxPos':>7}"
    if show_flags.get("be"):  hdr += f"  {'BE_ATR':>7}"
    if show_flags.get("ext"): hdr += f"  {'EXT_ATR':>8}"
    if show_flags.get("vd"):  hdr += f"  {'VD':>5}"
    hdr += (
        f"  {'Trades':>7} {'Win%':>6}  {'TotalPnL':>10}"
        f"  {'AvgPnL':>8}  {'Expect':>8}  {'TP_n':>5} {'SL_n':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        marker = " *" if is_baseline[row.name] else "  "
        if rank > 15 and not is_baseline[row.name]:
            continue

        line = f"{marker}{rank:>3}"
        if show_flags.get("sl"):  line += f"  {row['sl_atr']:>7.2f}"
        if show_flags.get("tp"):  line += f"  {row['tp_ext']:>7.2f}"
        if show_flags.get("pos"): line += f"  {int(row['max_pos']):>7}"
        if show_flags.get("be"):  line += f"  {row['be_atr']:>7.2f}"
        if show_flags.get("ext"): line += f"  {row['ext_atr']:>8.2f}"
        if show_flags.get("vd"):  line += f"  {row['vol_decay']:>5.2f}"
        line += (
            f"  {int(row['trades']):>7} {row['win_pct']:>5.1f}%  "
            f"${row['total_pnl']:>9,.2f}  ${row['avg_pnl']:>7.2f}"
            f"  ${row['expectancy']:>7.2f}  {row['tp_n']:>5} {row['sl_n']:>5}"
        )
        print(line)

    print()
    if not any(is_baseline):
        print("  (* = current config not in top 15; see full CSV)")
    else:
        print("  (* = current config)")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # ── Load credentials ──────────────────────────────────────────────────────
    try:
        import config
        API_KEY    = config.API_KEY
        SECRET_KEY = config.SECRET_KEY
    except ImportError:
        API_KEY    = os.environ.get("ALPACA_API_KEY", "")
        SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set. Edit config.py or set ALPACA_API_KEY env var.")
        sys.exit(1)

    # ── Init clients ──────────────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from risk.risk_manager import RiskManager, RiskConfig
    from data_collection.stock_universe import UniverseSelector, UniverseConfig
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig

    tc  = TradingClient(API_KEY, SECRET_KEY, paper=True)
    dc  = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    rm  = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
    ucfg = UniverseConfig()

    # ── Load tickers ──────────────────────────────────────────────────────────
    log.info("Loading universe...")
    try:
        selector = UniverseSelector(dc, ucfg)
        tickers  = selector.get_universe(use_cache=True)
        if not tickers:
            raise ValueError("Empty universe.")
        log.info(f"Universe: {len(tickers)} tickers.")
    except Exception as exc:
        log.warning(f"UniverseSelector failed ({exc}). Using fallback 20-stock list.")
        tickers = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA",
            "AMD",  "JPM",  "V",    "UNH",  "XOM",  "LLY",  "JNJ",
            "PG",   "MA",   "HD",   "MRK",  "ABBV", "CVX",
        ]

    # ── Load historical universes ──────────────────────────────────────────────
    hu_intraday = None
    hu_path = os.path.join("output", "historical_universes.json")
    if os.path.exists(hu_path):
        with open(hu_path) as f:
            hu_intraday = json.load(f)
        log.info(f"Loaded PIT universes from {hu_path}")

        # Only load tickers active during the test period -- not all 800+ historical ones.
        # Keys are ISO week format: "2023-W40". A week overlaps the test period if its
        # Monday is on or before test_end AND its Sunday is on or after test_start.
        from datetime import timedelta as _td
        test_start_d = datetime.strptime(TEST_START, "%Y-%m-%d").date()
        test_end_d   = datetime.strptime(TEST_END,   "%Y-%m-%d").date()
        unique_tickers: set = set()
        for week_str, week_list in hu_intraday.items():
            try:
                year, week = week_str.split("-W")
                week_start = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w").date()
                week_end   = week_start + _td(days=6)
                if week_start <= test_end_d and week_end >= test_start_d:
                    unique_tickers.update(week_list)
            except (ValueError, AttributeError):
                pass
        if unique_tickers:
            tickers = list(unique_tickers)
            log.info(f"Filtered to {len(tickers)} tickers active in test period.")
        else:
            log.warning("No matching universe weeks found; using full static universe.")
    else:
        log.warning(f"No {hu_path} found. Using static ticker list.")

    # ── Build grid ────────────────────────────────────────────────────────────
    max_pos_dim = MAX_POS_RANGE if MAX_POS_RANGE else [None]
    combos = list(itertools.product(
        SL_ATR_RANGE, TP_EXT_RANGE, max_pos_dim,
        BREAKEVEN_ATR_RANGE, EXT_ATR_RANGE, VOL_DECAY_RANGE,
    ))
    n_combos = len(combos)

    show_flags = {
        "sl":  len(SL_ATR_RANGE) > 1,
        "tp":  len(TP_EXT_RANGE) > 1,
        "pos": bool(MAX_POS_RANGE),
        "be":  len(BREAKEVEN_ATR_RANGE) > 1,
        "ext": len(EXT_ATR_RANGE) > 1,
        "vd":  len(VOL_DECAY_RANGE) > 1,
    }

    dim_labels = []
    if show_flags["sl"]:  dim_labels.append(f"{len(SL_ATR_RANGE)} SL")
    if show_flags["tp"]:  dim_labels.append(f"{len(TP_EXT_RANGE)} TP")
    if show_flags["pos"]: dim_labels.append(f"{len(MAX_POS_RANGE)} MaxPos")
    if show_flags["be"]:  dim_labels.append(f"{len(BREAKEVEN_ATR_RANGE)} BE")
    if show_flags["ext"]: dim_labels.append(f"{len(EXT_ATR_RANGE)} EXT")
    if show_flags["vd"]:  dim_labels.append(f"{len(VOL_DECAY_RANGE)} VD")
    swept_dims = " x ".join(dim_labels) if dim_labels else "(no dims swept — single run)"

    log.info("")
    log.info(f"Test period : {TEST_START} to {TEST_END}")
    log.info(f"Grid        : {swept_dims}")
    log.info(f"Combinations: {n_combos}")
    log.info(f"ETA         : ~{n_combos * 5 // 60}m{n_combos * 5 % 60:02d}s (~5s/run after preloading)")

    if n_combos > 200:
        ans = input(f"\n  {n_combos} combinations is a large grid. Continue? [y/N] ").strip().lower()
        if ans != "y":
            log.info("Aborted.")
            return

    # ── Build strategy and load minute data ONCE ──────────────────────────────
    base_cfg = IntradayConfig()
    strat = IntradayStrategy(
        trading_client=tc, data_client=dc, risk_manager=rm,
        config=base_cfg, universe_config=ucfg,
    )

    log.info("")
    log.info(f"Loading minute bars for {TEST_START} to {TEST_END}...")
    ms = strat._load_minute_store(tickers, TEST_START, TEST_END)
    if not ms:
        log.error("No minute data loaded. Aborting.")
        sys.exit(1)
    log.info(f"Loaded {len(ms)} tickers. Starting grid search...")
    log.info("")

    # Suppress verbose per-run logging during the grid loop
    strat.log.setLevel(logging.WARNING)
    logging.getLogger("IntradayStrategy").setLevel(logging.WARNING)

    # ── Grid loop ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    t_start = time.time()

    for i, (sl_atr, tp_ext, max_pos, breakeven_atr, ext_atr, vol_decay) in enumerate(combos, 1):
        cfg = deepcopy(base_cfg)
        cfg.vwap_sl_atr             = sl_atr
        cfg.vwap_tp_extension_atr   = tp_ext
        cfg.vwap_breakeven_atr      = breakeven_atr
        cfg.vwap_extension_atr      = ext_atr
        cfg.vwap_vol_decay          = vol_decay
        if max_pos is not None:
            cfg.vwap_max_positions  = max_pos
        strat.cfg = cfg

        result = strat.backtest(
            tickers=tickers,
            start_date=TEST_START,
            end_date=TEST_END,
            historical_universes=hu_intraday,
            _minute_store=ms,
        )

        row = _extract_metrics(result, sl_atr, tp_ext, max_pos, breakeven_atr, ext_atr, vol_decay)
        results.append(row)

        elapsed = time.time() - t_start
        avg_sec = elapsed / i
        eta_sec = avg_sec * (n_combos - i)
        eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"

        # Progress line: only show swept dimensions to keep it readable
        dim_str = ""
        if show_flags["sl"]:  dim_str += f"  SL={sl_atr:.2f}"
        if show_flags["tp"]:  dim_str += f"  TP={tp_ext:.2f}"
        if show_flags["pos"] and max_pos is not None: dim_str += f"  MaxPos={int(max_pos)}"
        if show_flags["be"]:  dim_str += f"  BE={breakeven_atr:.2f}"
        if show_flags["ext"]: dim_str += f"  EXT={ext_atr:.2f}"
        if show_flags["vd"]:  dim_str += f"  VD={vol_decay:.2f}"

        print(
            f"[{i:>3}/{n_combos}]{dim_str}"
            f"  =>  trades={row['trades']:>3}  win={row['win_pct']:>5.1f}%"
            f"  PnL=${row['total_pnl']:>8,.2f}  exp=${row['expectancy']:>6.2f}"
            f"  ETA {eta_str}",
            flush=True,
        )

    # ── Restore logging ───────────────────────────────────────────────────────
    strat.log.setLevel(logging.INFO)

    # ── Print ranked table ────────────────────────────────────────────────────
    _print_table(results, show_flags)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join("output", f"optimize_{ts}.csv")
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(results).sort_values("expectancy", ascending=False).to_csv(out, index=False)
    log.info(f"All results saved to {out}")

    total_time = time.time() - t_start
    log.info(f"Done in {int(total_time // 60)}m{int(total_time % 60):02d}s.")


if __name__ == "__main__":
    main()
