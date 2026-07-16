import numpy as np

from nomos import metrics
from nomos.dsl import parse_program
from nomos.engine import World


def test_components_counts_toroidal_blobs():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True          # one 2x2 blob
    mask[7, 7] = True              # a singleton
    r = metrics.components(mask)
    assert r["count"] == 2
    assert r["max_size"] == 4


def test_components_wrap_join():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 3] = True
    mask[7, 3] = True              # touches the top across the wrap
    assert metrics.components(mask)["count"] == 1


def test_dominant_period_finds_oscillation():
    series = [1.0 if t % 4 < 2 else 0.0 for t in range(64)]   # period 4
    p = metrics.dominant_period(series)
    assert p is not None
    assert abs(p["period"] - 4.0) < 0.5


def test_dominant_period_none_on_constant():
    assert metrics.dominant_period([0.5] * 32) is None


def test_temporal_signature_blinker_recurs_at_lag_2():
    a = np.zeros((5, 5)); a[2, 1:4] = 1.0     # horizontal
    b = np.zeros((5, 5)); b[1:4, 2] = 1.0     # vertical
    frames = [a, b] * 8                        # period-2 flip
    sig = metrics.temporal_signature(frames)
    assert sig["recurrence_lag"] == 2
    assert sig["recurrence_corr"] > 0.9
    assert sig["persistence"] < 0.5            # consecutive frames anti-correlate


def test_axis_alignment_detects_gradient():
    ramp = np.broadcast_to(np.arange(16) / 16, (16, 16)).copy()
    rep = metrics.channel_report(ramp)
    assert rep["gradient"]["axis"] == "x"
    assert rep["gradient"]["corr"] > 0.99


def test_pearson_none_on_constant():
    assert metrics.pearson(np.ones((4, 4)), np.arange(16.0).reshape(4, 4)) is None


def test_summarize_with_history_has_dynamics():
    prog = parse_program(open("laws/game_of_life.nomos", encoding="utf-8").read())
    world = World(prog, size=32, seed=3)
    activity = []
    history = [{"alive": world.state["alive"].copy()}]
    for _ in range(40):
        activity.append(world.step())
        history.append({"alive": world.state["alive"].copy()})
    report = metrics.summarize(world, activity, history=history)
    assert "dynamics" in report
    assert "alive" in report["dynamics"]
    assert "persistence" in report["dynamics"]["alive"]
    assert "correlations" in report
