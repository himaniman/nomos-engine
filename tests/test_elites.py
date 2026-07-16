import os
import tempfile

from nomos import archive, elites


def _report(act_final=0.05, gzip=0.4, rec=0.9):
    return {
        "size": [32, 32], "seed": 1, "steps": 100, "verdict": "alive: test",
        "activity": {"mean": act_final, "final": act_final, "trend": "steady"},
        "channels": {"a": {"gzip_ratio": gzip}},
        "dynamics": {"a": {"recurrence_lag": 2, "recurrence_corr": rec}},
    }


def test_descriptor_and_cell_binning():
    c = elites.cell(_report(act_final=0.05, gzip=0.4, rec=0.9))
    assert elites.cell_label(c) == "activity=alive structure=structured order=locked"
    c = elites.cell(_report(act_final=0.5, gzip=0.95, rec=0.1))
    assert elites.cell_label(c) == "activity=boiling structure=noisy order=chaotic"


def test_frozen_world_counts_as_locked():
    rep = _report(act_final=0.0, gzip=0.0)
    del rep["dynamics"]                      # constant fields yield no recurrence
    d = elites.descriptor(rep)
    assert d["order"] == 1.0
    assert elites.cell_label(elites.cell(rep)).endswith("order=locked")


def test_grid_and_coverage_from_archive():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "runs.db")
        archive.record(db, _report(act_final=0.05), "channel a", "first")
        archive.record(db, _report(act_final=0.5, gzip=0.95, rec=0.1), "channel b", "second")
        archive.record(db, _report(act_final=0.05), "channel c", "same-cell")
        cells = elites.grid(db)
        assert len(cells) == 2
        assert sum(len(v) for v in cells.values()) == 3
        text = elites.coverage_summary(db)
        assert f"2/{elites.TOTAL_CELLS} regimes visited" in text
        assert "first" in text               # oldest occupant credited
