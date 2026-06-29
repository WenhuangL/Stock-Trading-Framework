"""
trade_db.py
-----------
SQLite persistence for backtest trade records.

Every backtest run writes to output/trades.db automatically.
Query with pandas or any SQLite viewer (DB Browser for SQLite, DBeaver, etc.):

    import pandas as pd, sqlite3
    conn = sqlite3.connect("output/trades.db")

    # Worst symbols by P&L this run
    pd.read_sql(
        "SELECT symbol, SUM(pnl) as total_pnl, COUNT(*) as trades "
        "FROM trades WHERE run_id=? GROUP BY symbol ORDER BY total_pnl",
        conn, params=["<run_id>"]
    )

    # Exit reason breakdown by phase
    pd.read_sql(
        "SELECT phase, exit_reason, COUNT(*) as n, ROUND(AVG(pnl),2) as avg_pnl "
        "FROM trades WHERE strategy='intraday' GROUP BY phase, exit_reason",
        conn
    )

Schema
------
backtest_runs   one row per run with high-level metrics
trades          one row per trade with all fields
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("output/trades.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and indexes if they don't exist yet."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id              TEXT PRIMARY KEY,
                run_timestamp       TEXT NOT NULL,
                start_date          TEXT,
                end_date            TEXT,
                initial_cash        REAL,
                intraday_final      REAL,
                intraday_return_pct REAL,
                intraday_trades     INTEGER,
                intraday_win_rate   REAL,
                eod_final           REAL,
                eod_return_pct      REAL,
                eod_trades          INTEGER,
                eod_win_rate        REAL,
                spy_return_pct      REAL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                strategy          TEXT NOT NULL,
                phase             TEXT,
                date              TEXT,
                symbol            TEXT,
                direction         TEXT,
                entry_price       REAL,
                exit_price        REAL,
                tp_price          REAL,
                sl_price          REAL,
                qty               INTEGER,
                pnl               REAL,
                pnl_pct           REAL,
                exit_reason       TEXT,
                bars_held         INTEGER,
                entry_time        TEXT,
                exit_time         TEXT,
                spy_day_return    REAL,
                daily_decline_pct REAL,
                had_afterhours    INTEGER,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_trades_run_id     ON trades(run_id);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy   ON trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_trades_phase      ON trades(phase);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol     ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_date       ON trades(date);
            CREATE INDEX IF NOT EXISTS idx_trades_exit       ON trades(exit_reason);

            CREATE TABLE IF NOT EXISTS scan_candidates (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT    NOT NULL,
                date             TEXT    NOT NULL,
                symbol           TEXT    NOT NULL,
                signal_time      TEXT,
                direction        TEXT,
                bar_close        REAL,
                vwap_val         REAL,
                atr_val          REAL,
                dist_atr         REAL,
                vol_decay_ratio  REAL,
                signal_vol_ratio REAL,
                rank_score       REAL,
                rank_position    INTEGER,
                spy_regime       TEXT,
                was_traded       INTEGER,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_run  ON scan_candidates(run_id);
            CREATE INDEX IF NOT EXISTS idx_candidates_date ON scan_candidates(date);
            CREATE INDEX IF NOT EXISTS idx_candidates_sym  ON scan_candidates(symbol);
        """)

        # Migrate trades table — add indicator columns if not present (backward compat)
        cursor = conn.cursor()
        for col, coltype in [
            ("vwap_at_entry",    "REAL"),
            ("atr_at_entry",     "REAL"),
            ("dist_atr",         "REAL"),
            ("vol_decay_ratio",  "REAL"),
            ("signal_vol_ratio", "REAL"),
            ("spy_regime",       "TEXT"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists


def save_run(
    intraday_results: dict,
    eod_results: dict,
    start_date: str,
    end_date: str,
    spy_return_pct: Optional[float] = None,
) -> str:
    """
    Persist a complete backtest run to the database.
    Returns the 8-char run_id so the caller can log it.
    """
    init_db()

    run_id = str(uuid.uuid4())[:8]
    run_ts = datetime.now().isoformat(timespec="seconds")

    intra_s      = intraday_results.get("summary", {})
    eod_s        = eod_results.get("summary", {}) if eod_results else {}
    intra_trades = intraday_results.get("trades", [])
    eod_trades   = eod_results.get("trades", [])

    with _connect() as conn:
        conn.execute(
            "INSERT INTO backtest_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, run_ts, start_date, end_date,
                intra_s.get("initial_cash"),
                intra_s.get("final_value"),
                intra_s.get("total_return_pct"),
                intra_s.get("num_trades"),
                intra_s.get("win_rate_pct"),
                eod_s.get("final_value"),
                eod_s.get("total_return_pct"),
                eod_s.get("num_trades"),
                eod_s.get("win_rate_pct"),
                spy_return_pct,
            ),
        )

        rows = []

        for t in intra_trades:
            entry_p = float(t.get("entry_price") or 0)
            exit_p  = float(t.get("exit_price") or 0)
            dirn    = t.get("direction", "long")
            if entry_p > 0:
                sign    = 1 if dirn == "long" else -1
                pnl_pct = round(sign * (exit_p - entry_p) / entry_p * 100, 4)
            else:
                pnl_pct = None
            rows.append((
                run_id, "intraday",
                t.get("phase"),
                str(t.get("date", "")),
                t.get("symbol"),
                dirn,
                entry_p, exit_p,
                t.get("tp_price"), t.get("sl_price"),
                t.get("qty"),
                t.get("pnl"),
                pnl_pct,
                t.get("exit_reason"),
                t.get("bars_held"),
                t.get("entry_time"), t.get("exit_time"),
                t.get("spy_day_return"),
                None, None,
                t.get("vwap_at_entry"),
                t.get("atr_at_entry"),
                t.get("dist_atr"),
                t.get("vol_decay_ratio"),
                t.get("signal_vol_ratio"),
                t.get("spy_regime"),
            ))

        for t in eod_trades:
            entry_p = float(t.get("entry_price") or 0)
            exit_p  = float(t.get("exit_price") or 0)
            pnl_pct = round((exit_p - entry_p) / entry_p * 100, 4) if entry_p > 0 else None
            rows.append((
                run_id, "eod",
                "eod",
                str(t.get("date", "")),
                t.get("symbol"),
                None,
                entry_p, exit_p,
                None, None,
                t.get("qty"),
                t.get("pnl"),
                pnl_pct,
                t.get("exit_reason"),
                t.get("bars_held"),
                None, None,
                None,
                t.get("pct_change_at_entry"),
                1 if t.get("had_afterhours") else 0,
                None, None, None, None, None, None,  # indicator cols (intraday only)
            ))

        conn.executemany("""
            INSERT INTO trades (
                run_id, strategy, phase, date, symbol, direction,
                entry_price, exit_price, tp_price, sl_price, qty,
                pnl, pnl_pct, exit_reason, bars_held,
                entry_time, exit_time, spy_day_return,
                daily_decline_pct, had_afterhours,
                vwap_at_entry, atr_at_entry, dist_atr,
                vol_decay_ratio, signal_vol_ratio, spy_regime
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

        candidates = intraday_results.get("candidates", [])
        if candidates:
            conn.executemany("""
                INSERT INTO scan_candidates (
                    run_id, date, symbol, signal_time, direction,
                    bar_close, vwap_val, atr_val, dist_atr,
                    vol_decay_ratio, signal_vol_ratio,
                    rank_score, rank_position, spy_regime, was_traded
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                (
                    run_id,
                    str(c.get("date", "")),
                    c.get("symbol"),
                    c.get("signal_time"),
                    c.get("direction"),
                    c.get("bar_close"),
                    c.get("vwap_val"),
                    c.get("atr_val"),
                    c.get("dist_atr"),
                    c.get("vol_decay_ratio"),
                    c.get("signal_vol_ratio"),
                    c.get("rank_score"),
                    c.get("rank_position"),
                    c.get("spy_regime"),
                    c.get("was_traded", 0),
                )
                for c in candidates
            ])

    return run_id
