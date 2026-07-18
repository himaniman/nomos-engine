"""Sanity for the CONTINUOUS aqua world: it must live, stay deterministic, move
agents in float space, read a continuous motor command, and reseed on extinction —
the vivarium ecology, made smooth."""

from __future__ import annotations

import math
import os

import numpy as np

from aqua.world import INTERO, RAY_CH, World, _logits, sensor_size
from vivarium.config import Config


def _cfg(**kw):
    base = dict(size=48, seed=1, init_population=18, brain="nca")
    base.update(kw)
    return Config(**base)


def test_world_lives_and_is_deterministic():
    a = World(_cfg())
    b = World(_cfg())
    for _ in range(200):
        a.step()
        b.step()
    assert a.population() > 0                        # did not go extinct instantly
    assert a.snapshot()["births"] > 0               # food was eaten -> reproduction
    assert a.population() == b.population()          # reproducible
    assert a.snapshot()["births"] == b.snapshot()["births"]


def test_agents_live_in_float_space():
    w = World(_cfg())
    for _ in range(50):
        w.step()
    a = next(iter(w.agents.values()))
    assert isinstance(a.y, float) and isinstance(a.x, float)
    assert -math.pi <= a.heading < math.pi          # heading kept wrapped
    # a non-integer position proves motion is continuous, not cell-snapped
    assert any(abs(ag.y - round(ag.y)) > 1e-6 or abs(ag.x - round(ag.x)) > 1e-6
               for ag in w.agents.values())


def test_command_is_continuous_and_monotonic():
    """Forward thrust rises with the forward logit; turn follows right-minus-left."""
    w = World(_cfg())
    slow, _ = w._command(np.array([0.0, 0.0, 0.0, 0.0]))
    fast, _ = w._command(np.array([5.0, 0.0, 0.0, 0.0]))
    assert 0.0 <= slow < fast <= w.max_speed
    _, left = w._command(np.array([0.0, 3.0, 0.0, 0.0]))
    _, right = w._command(np.array([0.0, 0.0, 3.0, 0.0]))
    assert left < 0.0 < right


def test_sensor_vector_width_matches_brain():
    w = World(_cfg())
    a = next(iter(w.agents.values()))
    assert w.n_in == sensor_size(w.cfg.n_rays, w.smell)
    assert w._sense(a).shape == (w.n_in,)
    # dropping smell narrows the input by exactly the smell channels
    assert World(_cfg(), smell=False).n_in == w.n_in - 2


def test_proprioception_reports_own_motion():
    """The agent's own last speed shows up in the interoception block."""
    w = World(_cfg(init_population=0))
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)
    a.last_speed = w.max_speed
    o = w.cfg.n_rays * RAY_CH
    assert w._sense(a)[o + 1] == 1.0                # speed channel = full speed


def test_smell_points_toward_food():
    """With food dead ahead and none elsewhere, the smell FORWARD channel is
    strongly positive and the lateral one near zero."""
    w = World(_cfg(init_population=0))
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)  # facing +x
    w.food[:] = 0
    w.food[24, 30] = 1                              # straight ahead (+x)
    w._food_xy = np.argwhere(w.food > 0)
    o = w.cfg.n_rays * RAY_CH
    vec = w._sense(a)
    fwd, lat = vec[o + INTERO], vec[o + INTERO + 1]
    assert fwd > 0.3 and abs(lat) < 1e-6


def test_opacity_blocks_a_move_into_another_agent():
    w = World(_cfg(init_population=0))
    g = w._seed_genome()
    a = w._spawn_agent(g, pos=(24.0, 24.0), heading=0.0)
    b = w._spawn_agent(w._seed_genome(), pos=(24.0, 25.2), heading=0.0)
    reward = w._advance(a, 24.0, 25.0)              # would land on top of b
    assert reward == 0.0
    assert (a.y, a.x) == (24.0, 24.0)               # move was refused


def test_reseed_on_extinction():
    w = World(_cfg())
    for _ in range(120):                            # build a hall of long-lived genomes
        w.step()
    w.agents.clear()                                # force total extinction
    w.step()                                        # step() must repopulate from the hall
    assert w.population() > 0
    assert w.epoch >= 1


def test_neat_brain_also_runs():
    w = World(_cfg(brain="neat"))
    for _ in range(40):
        w.step()
    a = next(iter(w.agents.values()))
    assert _logits(a.brain).shape == (w.n_out,)     # continuous read-out works for NEAT too


def test_app_draws_headless():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame

    from aqua import app

    w = World(_cfg())
    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    for _ in range(60):
        w.step()
    sel = next(iter(w.agents)) if w.agents else None
    surf = app.draw(w, 10, 560, font, small, selected=sel,
                    hud={"fps": 60, "speed": 1, "sps": 60, "think": False, "msg": ""},
                    hitboxes=[])
    assert surf.get_size() == (w.w * 10 + 560, w.h * 10)
    pygame.quit()
