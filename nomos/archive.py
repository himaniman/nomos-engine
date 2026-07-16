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
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL,
    law_hash  TEXT,
    law_name  TEXT,
    law       TEXT,
    size      INTEGER,
    seed      INTEGER,
    steps     INTEGER,
    verdict   TEXT,
    report    TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_law_hash ON runs(law_hash);
"""


def law_hash(law_text: str) -> str:
    return hashlib.sha256(law_text.encode("utf-8")).hexdigest()[:12]


def record(db_path: str, report: dict, law_text: str, law_name: str = "") -> int:
    """Append a run; returns its row id."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        cur = conn.execute(
            "INSERT INTO runs (ts, law_hash, law_name, law, size, seed, steps, verdict, report)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), law_hash(law_text), law_name, law_text,
             report["size"][0], report["seed"], report["steps"],
             report.get("verdict", ""), json.dumps(report)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def recent(db_path: str, limit: int = 20) -> list[dict]:
    """Most recent runs as {id, ts, law_name, seed, verdict}."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT id, ts, law_name, seed, verdict FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "ts": r[1], "law_name": r[2], "seed": r[3], "verdict": r[4]}
                for r in rows]
    finally:
        conn.close()
