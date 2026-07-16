import os
import sqlite3
import tempfile

from nomos import archive


def _report():
    return {"size": [32, 32], "seed": 7, "steps": 100, "verdict": "alive: test",
            "activity": {"mean": 0.5, "final": 0.4, "trend": "steady"}}


def test_record_and_recent_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "runs.db")
        rid1 = archive.record(db, _report(), "channel a\nrule r: when 1 add a = 1", "lawA")
        rid2 = archive.record(db, _report(), "channel b\nrule r: when 1 add b = 1", "lawB")
        assert rid2 == rid1 + 1
        rows = archive.recent(db)
        assert [r["law_name"] for r in rows] == ["lawB", "lawA"]   # newest first
        assert rows[0]["verdict"].startswith("alive")


def test_law_hash_stable_and_distinct():
    assert archive.law_hash("x") == archive.law_hash("x")
    assert archive.law_hash("x") != archive.law_hash("y")


def test_memory_dedupes_by_law_and_keeps_hypothesis():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "runs.db")
        law_a = "channel a\nrule r: when 1 add a = 1"
        archive.record(db, _report(), law_a, "lawA-v1", hypothesis="old idea")
        archive.record(db, _report(), law_a, "lawA-v2", hypothesis="new idea")
        archive.record(db, _report(), "channel b\nrule r: when 1 add b = 1", "lawB")
        mem = archive.memory(db)
        assert [m["law_name"] for m in mem] == ["lawB", "lawA-v2"]  # deduped, newest first
        assert mem[1]["hypothesis"] == "new idea"
        assert mem[0]["report"]["activity"]["trend"] == "steady"


def test_memory_migrates_pre_hypothesis_archive():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "runs.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     " ts REAL, law_hash TEXT, law_name TEXT, law TEXT, size INTEGER,"
                     " seed INTEGER, steps INTEGER, verdict TEXT, report TEXT)")
        conn.execute("INSERT INTO runs (ts, law_hash, law_name, law, size, seed,"
                     " steps, verdict, report) VALUES (0, 'h', 'old', 'channel a',"
                     " 32, 1, 10, 'static', '{}')")
        conn.commit()
        conn.close()
        mem = archive.memory(db)                      # must not raise
        assert mem[0]["hypothesis"] == ""
        archive.record(db, _report(), "channel c", "new", hypothesis="h2")
        assert archive.memory(db)[0]["hypothesis"] == "h2"
