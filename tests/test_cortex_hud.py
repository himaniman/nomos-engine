"""The brain column of the HUD.

The claim worth a test is the one that is easy to get wrong and impossible to notice:
WATCHING an agent think must not change what it thinks."""

from collections import defaultdict

import numpy as np
import pygame
import pytest

from cortex import brain as B
from cortex.agent import Mind, fresh_genome
from cortex.app import App
from cortex.brainpanel import BRAIN_W, VIEWS


def _app(mutations=40, **kw):
    pygame.init()
    rng = np.random.default_rng(3)
    g = fresh_genome(seed=0)
    for _ in range(mutations):
        g = B.mutate(g, rng)
    return App(seed=2, view=25, cell=8, mind=Mind(g, think_every=2), **kw)


def test_the_brain_gets_its_own_column():
    app = _app()
    assert app.brain_panel is not None
    assert app.w == app.view * app.cell + 360 + BRAIN_W
    surf = app.draw()
    assert surf.get_size() == (app.w, app.h)
    col = pygame.surfarray.array3d(surf)[app.w - BRAIN_W:]
    assert col.std() > 0, "the brain column is blank"
    pygame.quit()


def test_the_brain_column_stays_in_its_column():
    app = _app()
    before = pygame.surfarray.array3d(app.draw())[:app.w - BRAIN_W].copy()
    app.brain_panel.cycle_view()
    after = pygame.surfarray.array3d(app.draw())[:app.w - BRAIN_W]
    assert np.array_equal(before, after), "the brain panel painted outside its column"
    pygame.quit()


def test_every_view_renders():
    app = _app()
    seen = set()
    for _ in range(len(VIEWS)):
        name = app.brain_panel.cycle_view()
        seen.add(name)
        arr = pygame.surfarray.array3d(app.draw())[app.w - BRAIN_W:]
        assert arr.std() > 0, f"view {name} drew nothing"
    assert seen == set(VIEWS)
    pygame.quit()


def test_the_sheet_shows_the_LIVING_body_not_the_seed():
    """Under development the genome is a seed and the body is grown, so the panel has
    to read the phenotype — showing the genome would show a 24-cell blob forever."""
    app = _app()
    app.brain_panel.view = VIEWS.index("organs")
    before = pygame.surfarray.array3d(app.draw())[app.w - BRAIN_W:].copy()
    p = app.mind.phenotype
    ys, xs = np.nonzero(p.mask)
    p.in_mask[ys, xs] = True                       # every living cell becomes an organ
    p.organ_mod[ys, xs] = 3
    after = pygame.surfarray.array3d(app.draw())[app.w - BRAIN_W:]
    assert not np.array_equal(before, after), "the sheet ignored the living body"
    # and the genome must be untouched by any of it
    assert int(app.mind.genome.mask.sum()) <= int(p.mask.sum())
    pygame.quit()


def test_watching_it_think_does_not_change_what_it_thinks():
    """The trace is one frame per sub-tick, produced by the same arithmetic as an
    ordinary act — so turning the animation on must not perturb the trajectory."""
    g = fresh_genome(seed=0)
    v = np.random.default_rng(11).normal(0, 0.5, g.n_in)
    from cortex.senses import layout
    quiet = B.Phenotype(g, B.Config(), layout(64))
    watched = B.Phenotype(g, B.Config(), layout(64))
    watched.record = True
    for _ in range(6):
        a = quiet.act(v)
        b = watched.act(v)
        assert a == b
    assert np.allclose(quiet.state, watched.state)
    assert len(watched.trace) == g.subticks, "one frame per sub-tick"


def test_the_wheel_can_change_hands():
    app = _app()
    assert app.auto
    y0 = app.world.agent.y
    for _ in range(20):
        app.step(defaultdict(bool))
    assert app.mind.thoughts > 0
    app.auto = False                               # the human takes over
    thoughts = app.mind.thoughts
    keys = defaultdict(bool)
    for _ in range(20):
        app.step(keys)
    assert app.mind.thoughts == thoughts, "the brain kept thinking after being unplugged"
    assert app.world.tick == 40
    pygame.quit()
