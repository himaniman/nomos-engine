"""Morphogenesis: the genome stops being a body and becomes a program for growing one.

The claims that matter, and that were each hard-won during the build:

  * the tissue GROWS from a seed, by its own cells' requests, not by anything placing
    cells;
  * the GENOME IS NEVER TOUCHED during a life — otherwise the phenotype quietly
    becomes the heredity and the growth program rots unselected
    (docs/cortex-design.md rejects inheriting a grown shape for exactly this reason);
  * growth is PAID FOR — in a resource fed at the border, and in the agent's own
    energy — so a bigger brain has to earn its keep;
  * growth is RATE-LIMITED, because an unlimited sheet is not a body but a boil;
  * a cell's FATE is read from the chemistry where it stands (concentration decides
    what an organ becomes: the French-flag model);
  * and all of it is deterministic.
"""

import numpy as np

from cortex import brain as B
from cortex.agent import Mind, fresh_genome, live
from cortex.senses import layout, sensor_size
from cortex.world import Config as WorldConfig
from cortex.world import World

N_IN = sensor_size(64)
LAYOUT = layout(64)


def _pheno(cfg=None, seed=0):
    cfg = cfg or B.Config()
    g = B.random_genome(N_IN, np.random.default_rng(seed), cfg)
    return g, B.Phenotype(g, cfg, LAYOUT)


def _grow(p, rounds=40, sensors=None, energy=1.0):
    v = np.zeros(p.genome.n_in) if sensors is None else sensors
    for _ in range(rounds):
        p.act(v, hormone=[0.5, 0.0, 0.0])
        p.develop(energy_frac=energy, embryo=True)


# --- it grows -----------------------------------------------------------------------

def test_a_seed_grows_into_a_body():
    g, p = _pheno()
    start = p.cells()
    _grow(p, 40)
    assert p.cells() > start, f"the tissue never grew ({start} -> {p.cells()})"
    assert p.births > 0
    assert p.cells() == int(p.mask.sum())


def test_the_genome_is_never_touched_by_a_life():
    """If a life could edit its genome, the phenotype would become the heredity and the
    growth program would stop being selected. This is the load-bearing invariant."""
    g, p = _pheno()
    before = (g.mask.copy(), g.type_map.copy(), g.in_mask.copy(),
              g.out_mask.copy(), g.organ_mod.copy())
    _grow(p, 40)
    assert p.cells() > int(before[0].sum()), "nothing happened, so nothing is proved"
    for name, was in zip(("mask", "type_map", "in_mask", "out_mask", "organ_mod"), before):
        assert np.array_equal(getattr(g, name), was), f"a life edited its genome ({name})"


def test_growth_is_reproducible():
    ga, pa = _pheno(seed=3)
    gb, pb = _pheno(seed=3)
    _grow(pa, 30)
    _grow(pb, 30)
    assert np.array_equal(pa.mask, pb.mask)
    assert np.array_equal(pa.organ_mod, pb.organ_mod)
    assert (pa.births, pa.deaths) == (pb.births, pb.deaths)


def test_two_genomes_grow_into_different_bodies():
    _, a = _pheno(seed=1)
    _, b = _pheno(seed=2)
    _grow(a, 30)
    _grow(b, 30)
    assert not np.array_equal(a.mask, b.mask)


# --- it is paid for -----------------------------------------------------------------

def test_a_starving_body_grows_less_than_a_fed_one():
    """Resource is fed in proportion to how full the agent is, so growth is bought with
    food that was actually eaten."""
    _, fed = _pheno(seed=5)
    _, starved = _pheno(seed=5)
    _grow(fed, 30, energy=1.0)
    _grow(starved, 30, energy=0.0)
    assert fed.cells() > starved.cells(), (fed.cells(), starved.cells())


def test_division_spends_the_resource_it_needs():
    cfg = B.Config()
    _, p = _pheno(cfg)
    _grow(p, 20)
    alive = np.nonzero(p.mask)
    assert p.resource[alive].max() > 0, "no supply reached the tissue at all"
    # nothing may be sitting on more than the cap, and nothing may go negative
    assert p.resource.min() >= 0.0
    assert p.resource.max() <= cfg.resource_cap + 1e-9


def test_carrying_a_brain_costs_the_agent_energy():
    """The counterweight to growth: without it nothing in the system argues for a
    small brain."""
    g = fresh_genome(seed=0)
    w = World(seed=2, cfg=WorldConfig())
    mind = Mind(g, B.Config(embryo_ticks=0, dev_every=10), think_every=2)
    mind.attach(w)
    w.agent.energy = 100.0
    for _ in range(50):
        mind.step(w)
    spent = 100.0 - w.agent.energy
    floor = 50 * w.cfg.stand_cost
    assert spent > floor, "a brain that costs nothing cannot be selected against"


def test_growth_is_rate_limited():
    """An early version turned over thousands of cells in one life and settled nothing:
    a body has to change slowly enough to mean something."""
    cfg = B.Config(max_born_per_dev=3)
    _, p = _pheno(cfg)
    for _ in range(10):
        p.act(np.zeros(p.genome.n_in), hormone=[1.0, 0.0, 0.0])
        before = p.cells()
        out = p.develop(energy_frac=1.0, embryo=True)
        assert out["born"] <= cfg.max_born_per_dev, out
        assert p.cells() - before <= cfg.max_born_per_dev


def test_a_body_never_vanishes():
    cfg = B.Config()
    _, p = _pheno(cfg)
    for _ in range(30):                       # a rule screaming DIE at every cell
        p.act(np.zeros(p.genome.n_in))
        p.morph[:, B.DIE] = 10.0
        p.develop(energy_frac=0.0)
    assert p.cells() >= 4, "the tissue dissolved completely"
    assert p.act(np.zeros(p.genome.n_in)) is not None


# --- fate ----------------------------------------------------------------------------

def test_organ_fate_is_read_from_the_chemistry_where_a_cell_stands():
    """Concentration decides fate — the French-flag model — so `organ_map` says which
    chemical means which sense, and an eye region is something that can FORM."""
    cfg = B.Config()
    g, p = _pheno(cfg, seed=7)
    c0, c1 = p._chem0, p._chem1
    ys, xs = np.nonzero(p.mask)
    g.organ_map[:] = np.array([2, 5, 8])[:len(g.organ_map)]
    p.act(np.zeros(g.n_in))
    for j in range(c1 - c0):                  # make chemical j dominate everywhere
        p.state[ys, xs, c0:c1] = 0.0
        p.state[ys, xs, c0 + j] = 1.0
        p.morph[:, B.BE_SENSE] = 10.0         # every cell asks to become a sense
        p.morph[:, B.BE_ACT] = -10.0
        p.morph[:, B.GROW] = -10.0
        p.morph[:, B.DIE] = -10.0
        p.develop(energy_frac=0.0)
        mods = set(int(m) for m in p.organ_mod[p.mask & p.in_mask])
        assert mods == {int(g.organ_map[j])}, f"chemical {j} gave fates {mods}"


def test_a_cell_can_stop_being_an_organ():
    cfg = B.Config()
    _, p = _pheno(cfg)
    p.act(np.zeros(p.genome.n_in))
    p.morph[:, B.BE_SENSE] = 10.0
    p.morph[:, B.GROW] = p.morph[:, B.DIE] = -10.0
    p.develop(energy_frac=0.0)
    assert p.organs()[0] > 0
    p.act(np.zeros(p.genome.n_in))
    p.morph[:, B.BE_SENSE] = -10.0
    p.morph[:, B.GROW] = p.morph[:, B.DIE] = -10.0
    p.develop(energy_frac=0.0)
    assert p.organs()[0] == 0, "an organ could never be given up again"


def test_a_daughter_inherits_half_of_her_mother():
    """Which is also what makes MIGRATION free: divide toward, apoptose behind, and the
    cell has moved with what it was holding."""
    cfg = B.Config()
    _, p = _pheno(cfg)
    ys, xs = np.nonzero(p.mask)
    p.act(np.zeros(p.genome.n_in))
    p.state[ys, xs, 0] = 0.8
    p.resource[ys, xs] = 3.0
    p.morph[:, B.GROW] = 10.0
    p.morph[:, B.DIE] = -10.0
    before = set(map(tuple, np.argwhere(p.mask)))
    p.develop(energy_frac=1.0)
    fresh = [c for c in map(tuple, np.argwhere(p.mask)) if c not in before]
    assert fresh, "no daughter to check"
    for cy, cx in fresh:
        assert abs(float(p.state[cy, cx, 0]) - 0.4) < 1e-9


# --- switching it off -----------------------------------------------------------------

def test_development_can_be_switched_off_for_the_ab():
    cfg = B.Config(develop=False)
    g, p = _pheno(cfg)
    start = p.cells()
    _grow(p, 30)
    assert p.cells() == start and p.births == 0
    assert start > 100, "with development off the genome should carry a whole brain"


def test_a_life_with_development_is_still_reproducible():
    g = fresh_genome(seed=0)
    a, ma = live(g, seed=2, max_ticks=600, think_every=3)
    b, mb = live(g, seed=2, max_ticks=600, think_every=3)
    assert (a.agent.y, a.agent.x, a.agent.energy) == (b.agent.y, b.agent.x, b.agent.energy)
    assert ma.phenotype.cells() == mb.phenotype.cells()
    assert np.array_equal(ma.phenotype.mask, mb.phenotype.mask)


def test_the_seed_already_has_several_cell_types_laid_out_as_regions():
    """One type meant every cell ran the same network until a 4%-per-mutation split
    happened to fire. Several, and laid out as REGIONS rather than salt-and-pepper,
    because a daughter inherits her mother's type — so patches grow with the body."""
    cfg = B.Config()
    g, p = _pheno(cfg, seed=4)
    assert g.n_types == cfg.seed_types > 1
    counts = np.bincount(g.type_map[g.mask], minlength=g.n_types)
    assert (counts > 0).all(), f"a type with no cells in the seed: {counts}"
    # regions, not speckle — measured AGAINST speckle rather than against a magic
    # threshold: in a 24-cell seed split three ways most cells are boundary cells, so
    # what matters is that neighbours agree far more often than chance
    ys, xs = np.nonzero(g.mask)

    def agreement(tmap):
        hits = total = 0
        for y, x in zip(ys, xs):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if g.mask[y + dy, x + dx]:
                    total += 1
                    hits += int(tmap[y + dy, x + dx] == tmap[y, x])
        return hits / max(1, total)

    speckle = g.type_map.copy()
    rng = np.random.default_rng(0)
    speckle[ys, xs] = rng.permutation(g.type_map[ys, xs])
    laid_out, scattered = agreement(g.type_map), agreement(speckle)
    assert laid_out > scattered + 0.2,         f"neighbours agree {laid_out:.0%} vs {scattered:.0%} shuffled — that is speckle"


def test_type_regions_grow_with_the_tissue():
    cfg = B.Config()
    g, p = _pheno(cfg, seed=4)
    before = np.bincount(g.type_map[g.mask], minlength=g.n_types)
    _grow(p, 60)
    after = np.bincount(p.type_map[p.mask], minlength=g.n_types)
    assert after.sum() > before.sum(), "nothing grew"
    assert (after >= 1).sum() >= 2, f"development collapsed to one type: {after}"
