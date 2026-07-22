"""Display rate and world speed are different things, and the layout must not twitch."""

from collections import defaultdict

import numpy as np
import pygame

from cortex.agent import Mind, fresh_genome
from cortex.app import App
from cortex.brainpanel import BRAIN_W


def _app(**kw):
    pygame.init()
    return App(seed=2, view=41, cell=10, mind=Mind(fresh_genome(seed=0), think_every=3),
               **kw)


def test_world_speed_is_independent_of_the_frame_rate():
    """A frame is a frame; how much WORLD passes inside it is the speed. Below 1x a
    frame usually advances nothing, which is what lets you watch a thought."""
    app = _app()
    keys = defaultdict(bool)
    app.speed_i = app.speeds.index(1.0)
    assert sum(app.frame(keys) for _ in range(20)) == 20
    app.speed_i = app.speeds.index(4.0)
    assert sum(app.frame(keys) for _ in range(20)) == 80
    app.speed_i = app.speeds.index(0.25)
    assert sum(app.frame(keys) for _ in range(20)) == 5
    pygame.quit()


def test_speed_is_a_bounded_ladder():
    app = _app()
    for _ in range(30):
        app.slower()
    assert app.speed == min(app.speeds)
    for _ in range(30):
        app.faster()
    assert app.speed == max(app.speeds)
    pygame.quit()


def test_pausing_stops_the_world_but_not_the_frames():
    app = _app()
    keys = defaultdict(bool)
    app.paused = True
    assert sum(app.frame(keys) for _ in range(10)) == 0
    assert app.draw() is not None
    pygame.quit()


def test_the_layout_does_not_move_as_the_brain_grows():
    """The whole point of the fixed box: a cell being born must not shuffle every panel
    underneath it while you are reading them."""
    app = _app()
    keys = defaultdict(bool)
    app.draw()
    small = app.mind.phenotype.cells()
    boxes = [tuple(app.brain_panel._box_rect)]
    for _ in range(600):
        app.frame(keys)
    app.draw()
    boxes.append(tuple(app.brain_panel._box_rect))
    grown = app.mind.phenotype.cells()
    assert grown > small, "the brain did not grow, so nothing is proved"
    assert len(set(boxes)) == 1, f"the sheet's box moved as the body grew: {boxes}"
    assert app.w == app.view * app.cell + 360 + BRAIN_W, "the window changed size"
    pygame.quit()


def test_a_bigger_body_is_drawn_smaller_not_wider():
    app = _app()
    keys = defaultdict(bool)
    app.draw()
    first = app.brain_panel._sheet_scale
    for _ in range(600):
        app.frame(keys)
    app.draw()
    assert app.brain_panel._sheet_scale <= first, "the sheet failed to scale down"
    assert app.brain_panel._sheet_rect.width <= BRAIN_W
    pygame.quit()
