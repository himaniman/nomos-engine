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

def _border(mask, y, x):
    G = mask.shape[0]
    return any(not (0 <= y + dy < G and 0 <= x + dx < G and mask[y + dy, x + dx])
               for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def test_the_seed_is_a_grown_shape_not_a_rectangle():
    """A rectangle has no morphology to select on, and its organs would inherit its
    symmetry. The seed is a blob grown by accretion, and every seed is a different one."""
    g = _genome()
    ys, xs = np.nonzero(g.mask)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    fill = g.cells() / (h * w)
    assert fill < 0.85, f"the seed fills {fill:.0%} of its box — that is a rectangle"
    a = _genome(seed=1)
    b = _genome(seed=2)
    assert not np.array_equal(a.mask, b.mask), "every seed should grow a different body"


def test_the_genome_is_a_seed_not_a_body():
    """The whole point of development: what is inherited is small enough to be a
    PROGRAM, and the body is what the program builds."""
    cfg = B.Config()
    g = _genome(cfg=cfg)
    assert g.cells() == cfg.seed_cells, "the genome is carrying a body, not a seed"
    p = _pheno(g, cfg)
    assert p.cells() == g.cells(), "a phenotype starts as its seed"
    frozen = B.Config(develop=False)          # ...and with development OFF the genome
    whole = _genome(cfg=frozen).cells()       # still carries a whole brain (lacunae
    assert whole > 5 * cfg.seed_cells, whole  # mean it is not exactly start_cells)


def test_organs_are_placed_morphologically():
    """Senses on the BORDER, commands read from DEEP tissue — so a signal has to cross
    the sheet — and only SOME border cells are organs; the rest are skin."""
    g = _genome()
    sens, eff = g.organs()
    assert sens > 0 and eff > 0
    border = [(int(y), int(x)) for y, x in np.argwhere(g.mask) if _border(g.mask, y, x)]
    for y, x in np.argwhere(g.in_mask & g.mask):
        assert _border(g.mask, y, x), "a sensory organ buried in the tissue"
    for y, x in np.argwhere(g.out_mask & g.mask):
        assert not _border(g.mask, y, x), "an effector exposed on the border"
    assert sens < len(border), "every border cell became an organ — no skin at all"


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
    order = np.argsort(xs)                              # real cells, leftmost/rightmost
    left = (int(ys[order[0]]), int(xs[order[0]]))
    right = (int(ys[order[-1]]), int(xs[order[-1]]))
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
    cfg = B.Config(start_cells=90, max_cells=200, max_types=4, max_layers=4,
                   max_hidden=30)
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


def test_energy_change_is_what_reward_is_made_of():
    """`last_delta` is the body's ledger: energy in, energy out, every tick."""
    g = fresh_genome(seed=0)
    w = World(seed=2, cfg=WorldConfig())
    mind = Mind(g, think_every=1)
    mind.attach(w)
    for _ in range(30):
        before = w.agent.energy
        mind.step(w)
        assert abs(mind.last_delta - (w.agent.energy - before) / mind.reward_scale) < 1e-12
    assert mind.last_delta < 0, "walking about should cost, and be felt as a cost"


def test_but_reward_is_the_SURPRISE_in_it():
    """...and `last_reward` is what the brain learns from, which is not the same thing.

    Measured 2026-07-24: over 4000 ticks a brain receives zero positive reward events
    and four thousand negative ones, because living costs energy every tick whatever
    you do. Learning from the raw ledger means learning 'everything I do is bad',
    uniformly and forever, with no positive example anywhere. Subtracting a running
    mean leaves only what was not already expected."""
    g = fresh_genome(seed=0)
    g.program.update({"reward_baseline": 0.05})
    w = World(seed=2, cfg=WorldConfig())
    mind = Mind(g, think_every=1)
    mind.attach(w)
    for _ in range(300):
        mind.step(w)
    assert mind.last_delta < 0                      # the body is still paying...
    assert abs(mind.last_reward) < abs(mind.last_delta), \
        "a steady, predictable drain must stop being news"


# --- the three signal layers ----------------------------------------------------------

def _chem_slice(g, cfg):
    return g.channel_map(cfg)["chemical"]


def test_the_channel_map_partitions_every_channel():
    g = _genome()
    cfg = B.Config()
    cm = g.channel_map(cfg)
    covered = []
    for lo, hi in cm.values():
        covered += list(range(lo, hi))
    assert sorted(covered) == list(range(g.C)), cm
    assert cm["hormone"][1] - cm["hormone"][0] == 1


def test_chemistry_spreads_and_electricity_does_not():
    """The chemical channels are the only ones with a spatial GRADIENT: put a spike in
    one cell and it should be smeared over the neighbourhood, while a spike in an
    electric channel stays where it was put."""
    cfg = B.Config(chem_mix=0.5, chem_every=1, chem_decay=0.0)
    g = _genome(cfg=cfg)
    for rule in g.rules:                              # a rule that outputs nothing, so
        for L in rule:                                # only the physics moves anything
            L["W"][:] = 0.0
            L["b"][:] = 0.0
    g.P_in[:] = 0.0
    g.P_horm[:] = 0.0
    c0, c1 = _chem_slice(g, cfg)
    p = _pheno(g, cfg)
    mid = len(p._ys) // 2
    cy, cx = int(p._ys[mid]), int(p._xs[mid])
    p.state[cy, cx, c0] = 1.0                         # a chemical spike...
    p.state[cy, cx, 0] = 1.0                          # ...and an electric one
    p.act(np.zeros(N_IN))
    neigh = [(cy + dy, cx + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
             if g.mask[cy + dy, cx + dx]]
    assert neigh, "need a neighbour to diffuse into"
    spread = max(abs(float(p.state[ny, nx, c0])) for ny, nx in neigh)
    stayed = max(abs(float(p.state[ny, nx, 0])) for ny, nx in neigh)
    # a relative claim, not a magic constant: the chemical moved and the electric
    # did not, whatever the sheet's size happens to make the absolute numbers
    assert spread > 1e-3, f"the chemical channel did not diffuse ({spread:.5f})"
    assert stayed < spread / 5, f"the electric channel leaked ({stayed:.5f})"


def test_chemistry_fades():
    cfg = B.Config(chem_mix=0.0, chem_every=1, chem_decay=0.2)
    g = _genome(cfg=cfg)
    for rule in g.rules:
        for L in rule:
            L["W"][:] = 0.0
            L["b"][:] = 0.0
    g.P_in[:] = 0.0
    g.P_horm[:] = 0.0
    c0, _ = _chem_slice(g, cfg)
    p = _pheno(g, cfg)
    cy, cx = int(p._ys[0]), int(p._xs[0])
    p.state[cy, cx, c0] = 1.0
    p.act(np.zeros(N_IN))
    assert abs(float(p.state[cy, cx, c0])) < 0.5, "a plume that never fades piles up"


def test_the_hormone_reaches_cells_no_organ_can():
    """A bloodstream, not a wire: it must arrive even at a brain with no organs at all
    and no sensory input whatsoever."""
    cfg = B.Config()
    g = _genome(cfg=cfg)
    g.in_mask[:] = False
    g.P_in[:] = 0.0                                   # nothing can come in by sense
    zero = np.zeros(N_IN)
    calm = _pheno(g, cfg)
    calm.act(zero, hormone=[0.0, 0.0, 0.0])
    starving = _pheno(g, cfg)
    starving.act(zero, hormone=[-1.0, 0.0, 0.0])
    assert not np.allclose(calm.state, starving.state), "the hormone never arrived"


def test_the_hormone_changes_what_the_same_senses_mean():
    """The point of the layer: the same input in a different internal state is a
    different computation, not merely a different input."""
    cfg = B.Config()
    g = _genome(cfg=cfg)
    v = np.random.default_rng(12).normal(0, 0.5, N_IN)
    fed, hungry = _pheno(g, cfg), _pheno(g, cfg)
    a = [fed.act(v, hormone=[1.0, 0.0, 0.0]) for _ in range(3)][-1]
    b = [hungry.act(v, hormone=[-1.0, 0.0, 0.0]) for _ in range(3)][-1]
    assert a != b


def test_each_layer_can_be_switched_off_for_the_ab():
    """Every one of the three has to be removable on its own, or we cannot ever say
    whether it was worth having."""
    g = _genome()
    v = np.random.default_rng(13).normal(0, 0.5, N_IN)
    full = _pheno(g, B.Config()).act(v, hormone=[0.7, 0.1, 0.0])
    no_chem = _pheno(g, B.Config(chem_channels=0)).act(v, hormone=[0.7, 0.1, 0.0])
    no_horm = _pheno(g, B.Config(hormone=False)).act(v, hormone=[0.7, 0.1, 0.0])
    no_local = _pheno(g, B.Config(neuromod=False)).act(v, hormone=[0.7, 0.1, 0.0])
    assert full != no_chem and full != no_horm and full != no_local


def test_the_body_talks_to_the_brain_through_the_hormone():
    from cortex.agent import Mind
    from cortex.world import Config as WCfg
    g = fresh_genome(seed=0)
    w = World(seed=2, cfg=WCfg())
    mind = Mind(g, think_every=1)
    mind.attach(w)
    full = mind.hormone(w)
    assert -1.0 <= full[0] <= 1.0 and full[1] == 0.0
    w.agent.energy = 1.0                               # nearly dead
    w.agent.poison = 30.0
    starving = mind.hormone(w)
    assert starving[0] < full[0] and starving[1] == 1.0
