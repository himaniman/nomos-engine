"""The growth program: the named, heritable half of a genome.

What is worth pinning down is the CONTRACT rather than any particular value — that a
program cannot leave its declared ranges whoever wrote it, that it survives a round
trip through JSON, that it actually reaches development (which is the whole reason it
exists: before this, the constants governing how a body grows were fixed in Config and
selection could not touch them), and that the two read-out knobs do what the
diagnostic said they do."""

import json

import numpy as np
import pytest

from cortex import brain as B
from cortex import program as P
from cortex.agent import Mind, fresh_genome
from cortex.senses import layout, sensor_size
from cortex.world import Config as WorldConfig
from cortex.world import World

N_IN = sensor_size(64)
LAYOUT = layout(64)


def _genome(seed=0, cfg=None):
    return B.random_genome(N_IN, np.random.default_rng(seed), cfg or B.Config())


# --- the contract --------------------------------------------------------------------

def test_defaults_cover_every_field():
    prog = P.Program()
    assert set(prog.to_dict()) == set(P.FIELDS)


def test_values_are_clamped_into_range():
    prog = P.Program({"grow_thresh": 99.0, "chem_decay": -5.0})
    lo_g, hi_g = P.FIELDS["grow_thresh"][1], P.FIELDS["grow_thresh"][2]
    assert prog["grow_thresh"] == hi_g
    assert prog["chem_decay"] == P.FIELDS["chem_decay"][1]
    assert lo_g <= prog["grow_thresh"] <= hi_g


def test_integer_fields_stay_integers():
    prog = P.Program({"dev_every": 33.7, "chem_every": 2.2})
    for k in P.INT_FIELDS:
        assert isinstance(prog[k], int), k


def test_nonsense_values_fall_back_to_the_default():
    """A model writing null, a string or a NaN must lose the FIELD, not the run."""
    prog = P.Program({"grow_bias": None, "chem_mix": "fast", "axon_lr": float("nan")})
    assert prog["grow_bias"] == P.DEFAULTS["grow_bias"]
    assert prog["chem_mix"] == P.DEFAULTS["chem_mix"]
    assert prog["axon_lr"] == P.DEFAULTS["axon_lr"]


def test_unknown_keys_are_reported_not_raised():
    prog = P.Program()
    unknown = prog.update({"grow_thresh": 0.4, "elan_vital": 1.0})
    assert unknown == ["elan_vital"]
    assert prog["grow_thresh"] == 0.4
    assert "elan_vital" not in prog


def test_mutation_stays_in_range_and_moves_something():
    rng = np.random.default_rng(0)
    prog = P.Program()
    child = prog.mutate(rng, prob=1.0, scale=0.5)
    assert prog.diff(child), "a full-probability mutation changed nothing"
    for k, (default, lo, hi, is_int, _) in P.FIELDS.items():
        assert lo <= child[k] <= hi, k


def test_mutation_can_lift_a_field_off_its_floor():
    """Jitter is relative to the RANGE, not the value — an absolute-fraction jitter
    would pin any knob that ever reached zero, which is how a search space quietly
    collapses."""
    rng = np.random.default_rng(1)
    prog = P.Program({"axon_lr": 0.0})
    moved = any(prog.mutate(rng, prob=1.0, scale=0.3)["axon_lr"] > 0.0
                for _ in range(20))
    assert moved


def test_prompt_view_names_every_field_with_its_range():
    text = P.Program().as_prompt()
    for k in P.FIELDS:
        assert k in text
    assert ".." in text                       # ranges are shown, not just values


# --- it reaches the genome, and the genome reaches the phenotype ----------------------

def test_genome_carries_a_program_and_survives_json():
    g = _genome(0)
    g.program.update({"grow_thresh": 0.42, "dev_every": 77})
    back = B.Genome.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.program["grow_thresh"] == pytest.approx(0.42)
    assert back.program["dev_every"] == 77


def test_old_genomes_without_a_program_still_load():
    """Genomes saved before the program existed must not become unreadable."""
    d = _genome(0).to_dict()
    del d["program"]
    assert B.Genome.from_dict(d).program["grow_thresh"] == P.DEFAULTS["grow_thresh"]


def test_program_overrides_the_ambient_config():
    g = _genome(0)
    g.program.update({"chem_decay": 0.31})
    p = B.Phenotype(g, B.Config(chem_decay=0.02), LAYOUT)
    assert p.cfg.chem_decay == pytest.approx(0.31)


def test_copy_does_not_share_the_program():
    g = _genome(0)
    h = g.copy()
    h.program.update({"grow_thresh": 0.9})
    assert g.program["grow_thresh"] != 0.9


def test_mutate_moves_the_growth_program():
    """The point of the whole module: before this, development ran on constants that
    were identical in every generation of every lineage, so selection had no purchase
    on HOW a body grows — only on the weights it grew with."""
    g = _genome(0)
    rng = np.random.default_rng(0)
    changed = any(g.program.diff(B.mutate(g, rng).program) for _ in range(10))
    assert changed


def test_think_every_comes_from_the_program():
    g = _genome(0)
    g.program.update({"think_every": 7})
    assert Mind(g).think_every == 7
    assert Mind(g, think_every=3).think_every == 3      # an explicit call still wins


# --- the two read-out knobs the diagnostic argued for --------------------------------

def test_pooling_reads_from_the_loudest_effectors():
    # a SEED carries two dozen cells and barely any effectors; pooling is about what a
    # GROWN body does, so build the pre-development full-size brain to test it on
    cfg = B.Config(develop=False, pool_frac=1.0, pool_min=1)
    g = _genome(0, cfg)
    p = B.Phenotype(g, cfg, LAYOUT)
    n = len(p._out_rows)
    assert n >= 4
    st = np.zeros((len(p._ys), g.C))
    st[p._out_rows[0], :] = 1.0                    # exactly one cell is saying anything
    p.cfg.pool_frac, p.cfg.pool_min = 1.0, 1
    quiet = p._pool(st)
    p.cfg.pool_frac = 1.0 / n                      # ...pool only that one
    loud = p._pool(st)
    assert loud.mean() > quiet.mean() * 2


def test_pooling_everything_is_still_the_plain_mean():
    g = _genome(0)
    p = B.Phenotype(g, B.Config(pool_frac=1.0), LAYOUT)
    st = np.random.default_rng(0).normal(size=(len(p._ys), g.C))
    assert np.allclose(p._pool(st), st[p._out_rows].mean(axis=0))


def test_pooling_is_deterministic_under_ties():
    g = _genome(0)
    p = B.Phenotype(g, B.Config(pool_frac=0.25, pool_min=2), LAYOUT)
    st = np.ones((len(p._ys), g.C))                # every effector equally loud
    assert np.allclose(p._pool(st), p._pool(st))


def test_thrust_rest_sets_what_a_silent_sheet_commands():
    """A brain holding nothing used to sprint, which is free marks on any probe whose
    food is straight ahead."""
    for rest in (0.15, 0.5, 0.9):
        g = _genome(0)
        g.program.update({"thrust_rest": rest})
        p = B.Phenotype(g, B.Config(), LAYOUT)
        p.genome.P_out[:] = 0.0                    # nothing the state says can matter
        p.genome.b_out[:] = 0.0
        thrust, _, _ = p.act(np.zeros(g.n_in))
        assert thrust == pytest.approx(rest, abs=0.02)


# --- development actually runs on it -------------------------------------------------

def test_grow_threshold_in_the_program_changes_the_body():
    """The end-to-end claim: a knob with a name, written into a genome, changes how
    much tissue a life grows."""
    sizes = []
    for thresh in (0.05, 1.2):
        g = fresh_genome(seed=0)
        g.program.update({"grow_thresh": thresh})
        w = World(seed=0, cfg=WorldConfig())
        mind = Mind(g)
        mind.attach(w)
        for _ in range(400):
            mind.step(w)
        sizes.append(mind.phenotype.cells())
    assert sizes[0] > sizes[1]


def test_innate_urges_moved_out_of_the_weights():
    """They used to be baked into every rule's output bias at birth. Same arithmetic,
    but as a weight they were unnamed and unreadable."""
    g = _genome(0)
    C = g.C
    for r in g.rules:
        assert r[-1]["b"][C + B.GROW] == pytest.approx(0.0)
        assert r[-1]["b"][C + B.SPROUT] == pytest.approx(0.0)
