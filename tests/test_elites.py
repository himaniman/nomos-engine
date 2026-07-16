import os
import tempfile

from nomos import archive, elites


def _report(act_final=0.05, coherence=0.7, rec=0.9):
    ch = {"gzip_ratio": 0.4, "std": 0.2}
    if coherence is not None:
        ch["coherence"] = coherence
    return {
        "size": [32, 32], "seed": 1, "steps": 100, "verdict": "alive: test",
        "activity": {"mean": act_final, "final": act_final, "trend": "steady"},
        "channels": {"a": ch},
        "dynamics": {"a": {"recurrence_lag": 2, "recurrence_corr": rec}},
    }


def test_structure_axis_uses_coherence():
    # high coherence -> ordered, mid -> structured, low -> noisy
    assert elites.cell_label(elites.cell(_report(coherence=0.95))).split()[1] == "structure=ordered"
    assert elites.cell_label(elites.cell(_report(coherence=0.7))).split()[1] == "structure=structured"
    assert elites.cell_label(elites.cell(_report(coherence=0.45))).split()[1] == "structure=textured"
    assert elites.cell_label(elites.cell(_report(coherence=0.1))).split()[1] == "structure=noisy"


def test_descriptor_and_cell_binning():
    c = elites.cell(_report(act_final=0.05, coherence=0.7, rec=0.9))
    assert elites.cell_label(c) == "activity=alive structure=structured order=locked"
    c = elites.cell(_report(act_final=0.5, coherence=0.1, rec=0.1))
    assert elites.cell_label(c) == "activity=boiling structure=noisy order=chaotic"


def test_flat_field_is_uniform():
    rep = _report(coherence=None)
    rep["channels"] = {"a": {"gzip_ratio": 0.0, "std": 0.0}}   # constant field
    assert elites.cell_label(elites.cell(rep)).split()[1] == "structure=uniform"


def test_legacy_run_without_coherence_falls_back_to_gzip():
    rep = _report(coherence=None)
    rep["channels"] = {"a": {"gzip_ratio": 0.95, "std": 0.3}}  # noisy, no coherence stored
    # 1 - 0.95 = 0.05 coherence proxy -> noisy, not a crash
    assert elites.cell_label(elites.cell(rep)).split()[1] == "structure=noisy"


def test_frozen_world_counts_as_locked():
    rep = _report(act_final=0.0, coherence=0.9)
    del rep["dynamics"]                      # constant fields yield no recurrence
    d = elites.descriptor(rep)
    assert d["order"] == 1.0
    assert elites.cell_label(elites.cell(rep)).endswith("order=locked")


def test_grid_and_coverage_from_archive():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "runs.db")
        archive.record(db, _report(act_final=0.05, coherence=0.7), "channel a", "first")
        archive.record(db, _report(act_final=0.5, coherence=0.1, rec=0.1), "channel b", "second")
        archive.record(db, _report(act_final=0.05, coherence=0.7), "channel c", "same-cell")
        cells = elites.grid(db)
        assert len(cells) == 2
        assert sum(len(v) for v in cells.values()) == 3
        text = elites.coverage_summary(db)
        assert f"2/{elites.TOTAL_CELLS} regimes visited" in text
        assert "first" in text               # oldest occupant credited
