"""The brain: a sheet that thinks by relaxing, and the wiring that lets it play.

The claims worth pinning down are the ones that make the SHAPE do work — organs hear
by position, signal has to cross the tissue to reach an effector, state persists
between ticks — plus the invariants that stop mutation from producing a brain that
cannot run at all."""

import numpy as np
import pytest

from cortex import brain as B
from cortex.agent import Mind, fresh_genome, live
from cortex.senses import RAY_CH, layout, sensor_size
from cortex.world import Config as WorldConfig
from cortex.world import World

N_IN = sensor_size(64)
LAYOUT = layout(64)


def _genome(seed=0, cfg=None):
    return B.random_genome(N_IN, np.random.default_rng(seed), cfg or B.Config())


def _pheno(g, cfg=None):
    return B.Phenotype(g, cfg or B.Config(), LAYOUT)


# --- the seed brain -----------------------------------------------------------------

def test_the_seed_is_wired_as_an_organism():
    """Sense at the edge, act from the core: so signal must cross the tissue, and the
    shape is doing a job on the very first tick rather than after some mutation."""
    g = _genome()
    ys, xs = np.nonzero(g.mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sens, eff = g.organs()
    assert sens > 0 and eff == 1
    for y, x in zip(*np.nonzero(g.in_mask & g.mask)):
        assert y in (y0, y1) or x in (x0, x1), "a sensory organ off the perimeter"
    ey, ex = np.argwhere(g.out_mask & g.mask)[0]
    assert (y0 < ey < y1) and (x0 < ex < x1), "the effector should sit inside"


def test_the_lattice_is_wide_enough_for_the_eye():
    """Retinotopy hands out ray slots by COLUMN, so a sheet narrower than the ray count
    physically cannot resolve them — the reason grid is 64."""
    assert B.Config().grid >= LAYOUT["n_rays"]


def test_commands_are_in_the_range_the_world_takes():
    g = _genome()
    p = _pheno(g)
    rng = np.random.default_rng(1)
    for _ in range(12):
        thrust, turn, act = p.act(rng.normal(0, 0.5, N_IN))
        assert 0.0 <= thrust <= 1.0
        assert -1.0 <= turn <= 1.0
        assert act in (0.0, 1.0)


# --- what the shape buys ------------------------------------------------------------

def test_an_organs_column_picks_which_ray_it_hears():
    g = _genome()
    ys, xs = np.nonzero(g.mask)
    g.organ_mod[:] = 0
    g.in_mask[:] = False
    left = (int(ys.min()), int(xs.min()))
    right = (int(ys.min()), int(xs.max()))
    for cell in (left, right):
        g.in_mask[cell] = True
        g.organ_mod[cell] = 1                       # modality 1 = a per-ray channel
    M = B.build_listen(g, ys, xs, LAYOUT)
    rows = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    lo = int(np.nonzero(M[rows[left]])[0][0]) // RAY_CH
    hi = int(np.nonzero(M[rows[right]])[0][0]) // RAY_CH
    assert lo == 0 and hi == LAYOUT["n_rays"] - 1, f"columns mapped to rays {lo}/{hi}"


def test_a_brain_with_no_organs_is_never_deaf():
    g = _genome()
    g.in_mask[:] = False
    ys, xs = np.nonzero(g.mask)
    M = B.build_listen(g, ys, xs, LAYOUT)
    assert M.sum() > 0, "mutation must not be able to blind a brain outright"


def test_the_medium_is_the_memory():
    """State persists between acts, so the same input twice does not give the same
    inner state — that persistence IS the brain's short-term memory."""
    g = _genome()
    p = _pheno(g)
    v = np.random.default_rng(2).normal(0, 0.5, N_IN)
    p.act(v)
    first = p.state.copy()
    p.act(v)
    assert not np.allclose(first, p.state)


def test_thinking_longer_changes_what_is_thought():
    """subticks is how far a signal can travel before the command is read — if it made
    no difference the sheet would not be computing anything."""
    g = _genome()
    v = np.random.default_rng(3).normal(0, 0.5, N_IN)
    short, long = g.copy(), g.copy()
    short.subticks, long.subticks = 2, 40
    a = _pheno(short).act(v)
    b = _pheno(long).act(v)
    assert a != b


def test_a_brain_is_deterministic():
    g = _genome()
    v = np.random.default_rng(4).normal(0, 0.5, N_IN)
    p, q = _pheno(g), _pheno(g)
    assert [p.act(v) for _ in range(5)] == [q.act(v) for _ in range(5)]


# --- learning -----------------------------------------------------------------------

def test_reward_moves_the_readout_and_silence_does_not():
    g = _genome()
    p = _pheno(g)
    v = np.random.default_rng(5).normal(0, 0.5, N_IN)
    p.act(v)
    p.learn(0.0)
    quiet = p.plastic_norm()
    p.act(v)
    p.learn(1.0)
    assert p.plastic_norm() > quiet + 1e-6, "a reward must leave a mark"


def test_plastic_deltas_stay_bounded():
    cfg = B.Config()
    p = _pheno(_genome(), cfg)
    v = np.random.default_rng(6).normal(0, 0.5, N_IN)
    for _ in range(200):
        p.act(v)
        p.learn(5.0)
    assert np.abs(p.dW).max() <= cfg.plastic_clip + 1e-9
    for d in p._dW_rule:
        assert np.abs(d).max() <= cfg.rule_clip + 1e-9


def test_a_frozen_brain_does_not_drift():
    cfg = B.Config(plastic=False, rule_plastic=False)
    p = _pheno(_genome(), cfg)
    v = np.random.default_rng(7).normal(0, 0.5, N_IN)
    for _ in range(20):
        p.act(v)
        p.learn(3.0)
    assert p.plastic_norm() == 0.0, "the A/B control arm must really be frozen"


# --- mutation -----------------------------------------------------------------------

def test_mutation_keeps_the_brain_runnable_and_bounded():
    cfg = B.Config(max_cells=200, max_types=4, max_layers=4, max_hidden=30)
    g = _genome(cfg=cfg)
    rng = np.random.default_rng(8)
    v = rng.normal(0, 0.5, N_IN)
    for _ in range(120):
        g = B.mutate(g, rng, cfg)
        assert g.cells() <= cfg.max_cells
        assert g.n_types <= cfg.max_types
        assert g.max_depth() <= cfg.max_layers
        assert cfg.min_hidden <= g.max_width() <= cfg.max_hidden
        assert cfg.subticks_min <= g.subticks <= cfg.subticks_max
    out = _pheno(g, cfg).act(v)
    assert all(np.isfinite(q) for q in out)


def test_mutation_grows_structure_over_time():
    cfg = B.Config()
    g = _genome(cfg=cfg)
    rng = np.random.default_rng(9)
    before = (g.cells(), g.n_types, g.params())
    for _ in range(150):
        g = B.mutate(g, rng, cfg)
    assert g.cells() > before[0], "appearance should be biased over disappearance"
    assert g.params() > before[2]


def test_a_genome_survives_a_round_trip():
    g = _genome()
    rng = np.random.default_rng(10)
    for _ in range(20):
        g = B.mutate(g, rng)
    back = B.Genome.from_dict(g.to_dict())
    v = rng.normal(0, 0.5, N_IN)
    assert _pheno(g).act(v) == _pheno(back).act(v)


# --- playing ------------------------------------------------------------------------

def test_a_brain_plays_the_world_and_the_life_is_reproducible():
    g = fresh_genome(seed=0)
    a, ma = live(g, seed=2, max_ticks=400)
    b, mb = live(g, seed=2, max_ticks=400)
    assert (a.tick, a.agent.y, a.agent.x, a.agent.energy) == \
           (b.tick, b.agent.y, b.agent.x, b.agent.energy)
    assert a.agent.ate == b.agent.ate
    assert ma.thoughts == mb.thoughts


def test_thinking_is_rarer_than_the_world_ticks():
    g = fresh_genome(seed=0)
    w, mind = live(g, seed=2, max_ticks=400, think_every=4)
    assert mind.thoughts == 100, "the command should be held between thoughts"


def test_the_brain_only_ever_sees_the_sensor_vector():
    """Same terms as the baselines and the human: no back door to the world."""
    g = fresh_genome(seed=0)
    w = World(seed=2, cfg=WorldConfig())
    mind = Mind(g, think_every=1)
    mind.attach(w)
    mind.step(w)
    assert mind.sensors is not None and len(mind.sensors) == g.n_in
    p = _pheno(g)
    assert p.act(mind.sensors) == _pheno(g).act(mind.sensors)


def test_energy_change_is_the_reward():
    g = fresh_genome(seed=0)
    w = World(seed=2, cfg=WorldConfig())
    mind = Mind(g, think_every=1)
    mind.attach(w)
    for _ in range(30):
        before = w.agent.energy
        mind.step(w)
        assert abs(mind.last_reward - (w.agent.energy - before) / mind.reward_scale) < 1e-12
    assert mind.last_reward < 0, "walking about should cost, and be felt as a cost"
