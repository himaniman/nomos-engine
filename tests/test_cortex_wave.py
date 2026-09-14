"""The rule that can be found (cortex/wave.py) and the atlas genome (cortex/atlas_lineage).

Pinned: the perceptron architecture CAN hold the wave (the built target reproduces the
hand rule's field to the last digit and takes the ladder like it), a rule is a pure
function of its weights, and the teacher's loss is what it claims — zero for the target,
positive for anything else."""

import numpy as np

from cortex import atlas as A
from cortex import atlas_lineage as L
from cortex import wave as W


def test_exact_rule_is_the_wave():
    t = W.Teacher(n_maps=6, seed=3)
    assert t.loss(W.exact()) < 1e-12
    assert t.loss(W.NeuralRule(seed=1)) > 1e-3


def test_target_wave_routes_and_blocks():
    src = np.zeros((1, 15, 15))
    solid = np.zeros((1, 15, 15))
    solid[0, :, 8] = 1.0
    solid[0, 2, 8] = 0.0                    # one gap
    src[0, 7, 12] = 1.0
    V = W.target_wave(src, solid, 30)[0]
    assert V[7, 8] == 0.0                   # a wall holds nothing
    assert V[2, 7] > V[7, 7]                # this side is reached through the gap


def test_exact_rule_takes_the_ladder_like_the_hand_rule():
    from cortex import probes
    hand = probes._suite(lambda: A.Driver(A.Config()), 400)["profile"]
    neural = probes._suite(lambda: A.Driver(A.Config(), rule=W.exact()), 400)["profile"]
    for k in ("near", "side", "behind", "corner"):
        assert neural[k] >= 0.9 and hand[k] >= 0.9, (k, hand[k], neural[k])


def test_rule_is_a_pure_function_of_theta():
    theta = np.random.default_rng(5).normal(0, 0.1, W.NeuralRule().n_params)
    t = W.Teacher(n_maps=3)
    assert t.loss(W.NeuralRule(theta)) == t.loss(W.NeuralRule(theta.copy()))


def test_teaching_lowers_the_loss():
    res = W.teach(generations=15, lam=8, seed=0, log=lambda s: None)
    assert res["history"][-1] < res["history"][0]


def test_program_clamps_and_steps():
    p = L.Program({"radius": 99, "decay": -1, "nonsense": 3})
    assert p.values["radius"] == 24 and p.values["decay"] == 0.01
    assert "nonsense" not in p.values
    rng = np.random.default_rng(0)
    kid = p.mutate(rng, prob=1.0)
    assert kid.diff(p)                         # something moved
    assert all(L.FIELDS[k][1] <= v <= L.FIELDS[k][2] for k, v in kid.values.items())
    assert isinstance(kid.values["radius"], int)


def test_program_config_runs_a_life():
    cfg = L.Program({"radius": 5, "subticks": 2}).config()
    w, mind = A.live(1, 120, cfg)
    assert mind.atlas.side == 11 and w.tick == 120
