"""SQLite archive of runs — the substrate the automated scientist loop queries.

One row per run: the law text (so a result is always reproducible from the
archive alone), its hash (to group re-runs of the same law), and the full
observer report as JSON. Deliberately schema-light: query metrics out of the
JSON with SQLite's json_extract when needed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL,
    law_hash   TEXT,
    law_name   TEXT,
    law        TEXT,
    size       INTEGER,
    seed       INTEGER,
    steps      INTEGER,
    verdict    TEXT,
    hypothesis TEXT DEFAULT '',
    report     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_law_hash ON runs(law_hash);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    # migrate archives created before the hypothesis column existed
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN hypothesis TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    return conn


def law_hash(law_text: str) -> str:
    return hashlib.sha256(law_text.encode("utf-8")).hexdigest()[:12]


def record(db_path: str, report: dict, law_text: str, law_name: str = "",
           hypothesis: str = "") -> int:
    """Append a run; returns its row id."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO runs (ts, law_hash, law_name, law, size, seed, steps,"
            " verdict, hypothesis, report) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), law_hash(law_text), law_name, law_text,
             report["size"][0], report["seed"], report["steps"],
             report.get("verdict", ""), hypothesis, json.dumps(report)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def recent(db_path: str, limit: int = 20) -> list[dict]:
    """Most recent runs as {id, ts, law_name, seed, verdict}."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, ts, law_name, seed, verdict FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "ts": r[1], "law_name": r[2], "seed": r[3], "verdict": r[4]}
                for r in rows]
    finally:
        conn.close()


def memory(db_path: str, limit: int = 10) -> list[dict]:
    """The scientist's long-term memory: the latest run of each distinct law,
    newest first. Returns {id, law_name, law, hypothesis, verdict, report(dict)}.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, law_name, law, hypothesis, verdict, report FROM runs"
            " WHERE id IN (SELECT MAX(id) FROM runs GROUP BY law_hash)"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "law_name": r[1], "law": r[2], "hypothesis": r[3] or "",
                 "verdict": r[4], "report": json.loads(r[5])}
                for r in rows]
    finally:
        conn.close()
