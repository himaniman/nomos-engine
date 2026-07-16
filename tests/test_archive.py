import os
import tempfile

from nomos import archive


def _report():
    return {"size": [32, 32], "seed": 7, "steps": 100, "verdict": "alive: test"}


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
