"""
optimize_vwap_segments.py
-------------------------
Time-segmented, train/test-validated grid search for VWAP reversion parameters.

Unlike optimize_vwap.py (which tunes one param set for the whole session on a
single period), this script:
  1. Restricts VWAP ENTRIES to one time segment at a time (exits stay natural —
     phase_end is left at 15:55 so a trade entered in the segment still holds
     for its normal duration up to vwap_max_hold_bars).
  2. Sweeps the two most impactful reversion params (extension_atr, sl_atr).
  3. Runs every combo on a TRAIN period and a TEST period, so the winner can be
     validated out-of-sample. A change is only worth adopting if the best-on-
     train combo ALSO beats the baseline on the unseen TEST period.

Only 2023 minute data is cached, so TRAIN = H1 2023, TEST = H2 2023 (the same
split used for the earlier overfitting check).

USAGE
-----
    python execution/optimize_vwap_segments.py

Edit SEGMENTS, EXT_ATR_RANGE, SL_ATR_RANGE, TRAIN_*, TEST_* below to change scope.
Results print per-segment and are saved to output/optimize_segments_<ts>.csv.
"""

import itertools
import json
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("optimize_segments")

# =============================================================================
# CONFIGURATION — edit to change scope
# =============================================================================

TRAIN_START, TRAIN_END = "2023-01-01", "2023-06-30"
TEST_START,  TEST_END  = "2023-07-01", "2023-12-31"

# Segments to optimize. entry_start/entry_cutoff restrict ENTRIES to the window;
# phase_end stays late so exits are natural (not truncated at the boundary).
# Start with the two data-rich segments; extend to afternoon/close if these pay off.
SEGMENTS = {
    "morning": {"entry_start": "09:45", "entry_cutoff": "11:00"},
    "midday":  {"entry_start": "11:00", "entry_cutoff": "14:00"},
    # "afternoon": {"entry_start": "14:00", "entry_cutoff": "15:00"},  # sparse (~127 tr/yr)
    # "close":     {"entry_start": "15:00", "entry_cutoff": "15:45"},  # too sparse (~39 tr/yr)
}
PHASE_END = "15:55"   # exits allowed through here for every segment

# Grid (baseline is included: ext=2.50, sl=1.40)
EXT_ATR_RANGE = [2.00, 2.25, 2.50, 2.75, 3.00]
SL_ATR_RANGE  = [1.00, 1.20, 1.40, 1.60, 1.80]

BASELINE_EXT = 2.50
BASELINE_SL  = 1.40


# =============================================================================
# METRIC EXTRACTION
# =============================================================================

def _metrics(result: dict) -> dict:
    vwap = [t for t in result.get("trades", []) if t.get("phase") == "vwap"]
    pnls   = [t.get("pnl", 0.0) for t in vwap]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n      = len(pnls)
    avg_win  = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    p = len(wins) / n if n else 0.0
    q = len(losses) / n if n else 0.0
    return {
        "trades":     n,
        "total_pnl":  round(sum(pnls), 2),
        "win_pct":    round(p * 100, 1),
        "expectancy": round(p * avg_win + q * avg_loss, 2),
    }


def _tickers_active(hu: dict, start: str, end: str, fallback: list) -> list:
    """Return the union of PIT-universe tickers whose ISO week overlaps [start, end]."""
    if not hu:
        return fallback
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out: set = set()
    for week_str, week_list in hu.items():
        try:
            yr, wk = week_str.split("-W")
            wk_start = datetime.strptime(f"{yr}-W{wk}-1", "%Y-W%W-%w").date()
            if wk_start <= e and (wk_start + timedelta(days=6)) >= s:
                out.update(week_list)
        except (ValueError, AttributeError):
            pass
    return list(out) if out else fallback


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    try:
        import config
        API_KEY, SECRET_KEY = config.API_KEY, config.SECRET_KEY
    except ImportError:
        API_KEY  = os.environ.get("ALPACA_API_KEY", "")
        SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    if not API_KEY or not SECRET_KEY:
        log.error("API keys not set.")
        sys.exit(1)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from risk.risk_manager import RiskManager, RiskConfig
    from data_collection.stock_universe import UniverseConfig
    from strategies.strategy_intraday import IntradayStrategy, IntradayConfig

    tc = TradingClient(API_KEY, SECRET_KEY, paper=True)
    dc = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    rm = RiskManager(tc, dc, API_KEY, SECRET_KEY, config=RiskConfig())
    ucfg = UniverseConfig()

    hu = None
    hu_path = os.path.join("output", "historical_universes.json")
    if os.path.exists(hu_path):
        with open(hu_path) as f:
            hu = json.load(f)
        log.info(f"Loaded PIT universes from {hu_path}")
    else:
        log.warning("No historical_universes.json — using a static fallback list.")

    fallback = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA", "AMD"]
    base_cfg = IntradayConfig()
    strat = IntradayStrategy(trading_client=tc, data_client=dc, risk_manager=rm,
                             config=base_cfg, universe_config=ucfg)
    strat.log.setLevel(logging.WARNING)
    logging.getLogger("IntradayStrategy").setLevel(logging.WARNING)

    combos = list(itertools.product(EXT_ATR_RANGE, SL_ATR_RANGE))
    log.info(f"Grid: {len(EXT_ATR_RANGE)} EXT x {len(SL_ATR_RANGE)} SL = {len(combos)} combos "
             f"per segment per period.")

    # ── Load minute stores once per period ────────────────────────────────────
    stores = {}
    for label, (start, end) in {"TRAIN": (TRAIN_START, TRAIN_END),
                                 "TEST": (TEST_START, TEST_END)}.items():
        tickers = _tickers_active(hu, start, end, fallback)
        log.info(f"[{label}] loading minute bars for {len(tickers)} tickers "
                 f"({start} -> {end})...")
        t0 = time.time()
        stores[label] = strat._load_minute_store(tickers, start, end)
        log.info(f"[{label}] loaded {len(stores[label])} tickers in {time.time()-t0:.0f}s.")

    # ── Run grid per segment per period ───────────────────────────────────────
    rows: list[dict] = []
    for seg_name, seg in SEGMENTS.items():
        for period_label, (start, end) in {"TRAIN": (TRAIN_START, TRAIN_END),
                                           "TEST": (TEST_START, TEST_END)}.items():
            ms = stores[period_label]
            for i, (ext, sl) in enumerate(combos, 1):
                cfg = deepcopy(base_cfg)
                cfg.vwap_entry_start   = seg["entry_start"]
                cfg.vwap_entry_cutoff  = seg["entry_cutoff"]
                cfg.vwap_phase_end     = PHASE_END
                cfg.vwap_extension_atr = ext
                cfg.vwap_sl_atr        = sl
                strat.cfg = cfg

                res = strat.backtest(tickers=list(ms.keys()), start_date=start,
                                     end_date=end, historical_universes=hu,
                                     _minute_store=ms)
                m = _metrics(res)
                rows.append({"segment": seg_name, "period": period_label,
                             "ext_atr": ext, "sl_atr": sl, **m})
                print(f"  [{seg_name:>7}/{period_label:<5}] "
                      f"ext={ext:.2f} sl={sl:.2f}  "
                      f"trades={m['trades']:>4}  win={m['win_pct']:>5.1f}%  "
                      f"pnl=${m['total_pnl']:>9,.2f}  exp=${m['expectancy']:>6.2f}",
                      flush=True)

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join("output", f"optimize_segments_{ts}.csv")
    df.to_csv(out, index=False)

    # ── Per-segment train/test comparison ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SEGMENT OPTIMIZATION — baseline vs best-on-train, validated on test")
    print("=" * 78)
    for seg_name in SEGMENTS:
        tr = df[(df.segment == seg_name) & (df.period == "TRAIN")]
        te = df[(df.segment == seg_name) & (df.period == "TEST")]

        base_tr = tr[(tr.ext_atr == BASELINE_EXT) & (tr.sl_atr == BASELINE_SL)].iloc[0]
        base_te = te[(te.ext_atr == BASELINE_EXT) & (te.sl_atr == BASELINE_SL)].iloc[0]

        best_tr = tr.sort_values("total_pnl", ascending=False).iloc[0]
        best_te = te[(te.ext_atr == best_tr.ext_atr) & (te.sl_atr == best_tr.sl_atr)].iloc[0]

        print(f"\n  {seg_name.upper()}")
        print(f"    {'':<18}{'ext':>5}{'sl':>6}{'  train_pnl':>13}{'  test_pnl':>12}{'  test_win':>10}")
        print(f"    {'baseline':<18}{base_tr.ext_atr:>5.2f}{base_tr.sl_atr:>6.2f}"
              f"{base_tr.total_pnl:>13,.0f}{base_te.total_pnl:>12,.0f}{base_te.win_pct:>9.1f}%")
        print(f"    {'best-on-train':<18}{best_tr.ext_atr:>5.2f}{best_tr.sl_atr:>6.2f}"
              f"{best_tr.total_pnl:>13,.0f}{best_te.total_pnl:>12,.0f}{best_te.win_pct:>9.1f}%")

        beats_train = best_tr.total_pnl > base_tr.total_pnl
        beats_test  = best_te.total_pnl > base_te.total_pnl
        verdict = ("ADOPT — beats baseline on BOTH train and test"
                   if beats_train and beats_test
                   else "REJECT — best-on-train does NOT beat baseline out-of-sample")
        print(f"    -> {verdict}")

    print("\n" + "=" * 78)
    log.info(f"Full grid saved to {out}")


if __name__ == "__main__":
    main()
