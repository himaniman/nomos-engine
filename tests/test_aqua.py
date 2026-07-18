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
    # the brain input is sized for the MAX ray count (vision is heritable per agent)
    assert w.n_in == sensor_size(w.cfg.n_rays_max, w.smell)
    assert w._sense(a).shape == (w.n_in,)
    # dropping smell narrows the input by exactly the smell channels
    assert World(_cfg(), smell=False).n_in == w.n_in - 2


def test_proprioception_reports_own_motion():
    """The agent's own last speed shows up in the interoception block."""
    w = World(_cfg(init_population=0))
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)
    a.last_speed = w.max_speed
    o = w.cfg.n_rays_max * RAY_CH
    assert w._sense(a)[o + 1] == 1.0                # speed channel = full speed


def test_smell_points_toward_food():
    """With food dead ahead and none elsewhere, the smell FORWARD channel is
    strongly positive and the lateral one near zero."""
    w = World(_cfg(init_population=0))
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)  # facing +x
    a.brain.genome.diet = 0.0                       # fully digests food type 1 (edible)
    w.food_xy = np.array([[24.0, 30.0]])           # one morsel straight ahead (+x)
    w.food_type = np.array([1])
    o = w.cfg.n_rays_max * RAY_CH
    vec = w._sense(a)
    fwd, lat = vec[o + INTERO], vec[o + INTERO + 1]
    assert fwd > 0.3 and abs(lat) < 1e-6


def test_opacity_blocks_a_move_into_another_agent():
    w = World(_cfg(init_population=0))
    g = w._seed_genome()
    a = w._spawn_agent(g, pos=(24.0, 24.0), heading=0.0)
    b = w._spawn_agent(w._seed_genome(), pos=(24.0, 25.2), heading=0.0)
    reward = w._advance(a, 24.0, 25.0)              # marches toward b, must not overlap it
    assert reward == 0.0
    assert a.x < b.x                                # did not pass through b
    assert math.hypot(a.y - b.y, a.x - b.x) >= 2 * w.radius - 1e-9   # bodies never overlap


# -- Stage 1: continuous WALLS (float rectangles, analytic collision + vision) ------

def test_wall_rects_agree_with_placement_cache():
    """The analytic wall rectangles and the grid `self.walls` (still backing food
    placement) both derive from the strokes, so a cell centre is inside a rect iff
    that cell is marked wall — the two views can never drift apart."""
    w = World(_cfg(init_population=0))
    assert w.wall_rects.shape[1] == 4 and len(w.wall_rects) >= 4   # at least the border
    for cy in range(w.h):
        for cx in range(w.w):
            inside = any(y0 < cy < y1 and x0 < cx < x1 for y0, y1, x0, x1 in w.wall_rects)
            assert inside == bool(w.walls[cy, cx])


def test_ray_hits_wall_at_analytic_distance():
    """A ray fired at a border wall returns its exact geometric distance (the left
    face of the right border column), not a ray-marched approximation."""
    w = World(_cfg(init_population=0))
    cy, cx = w.h / 2.0, w.w / 2.0
    d = w._ray_wall_dist(cy, cx, 1.0, 0.0, 1e3)     # +x ray toward the right border
    assert abs(d - ((w.w - 1 - 0.5) - cx)) < 1e-9


def test_wall_blocks_body_and_slides_not_tunnels():
    """A body driven straight at a wall is stopped as a CIRCLE (never overlaps the
    rectangle) and does not tunnel through it, however fast it is pushed."""
    w = World(_cfg(init_population=0))
    # a vertical wall bar at column 24, spanning several rows around the centre-ish
    w.strokes = [{"axis": "v", "y": 20, "x": 24, "length": 8, "door": 0, "dir": 1}]
    w._rasterize_walls()
    w._build_wall_rects()
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 22.0), heading=0.0)  # facing +x wall
    w._advance(a, 40.0, 22.0)                        # shove far past the wall in one call
    assert a.x < 24 - w.radius + 1e-9               # stopped before the wall face
    assert not w._wall_blocks_circle(a.y, a.x)      # body does not overlap the wall


def test_no_body_overlaps_walls_over_a_run():
    w = World(_cfg(size=40, seed=2))               # small enough to stay fast; 250 ticks
    for _ in range(250):                           # spans two wall-drift events
        w.step()
    assert w.population() > 0
    assert not any(w._wall_blocks_circle(a.y, a.x) for a in w.agents.values())


# -- Stage 2: continuous FOOD (float morsels, eaten by circle overlap) --------------

def test_food_is_float_morsels_not_a_grid():
    """Food lives as float positions + types, never snapped to cell centres, and it
    never grows inside a wall."""
    w = World(_cfg(seed=1))
    assert not hasattr(w, "food")                   # the grid is gone
    assert w.food_xy.shape[1] == 2 and len(w.food_xy) == len(w.food_type)
    for _ in range(60):
        w.step()
    assert len(w.food_type) > 0
    assert any(abs(y - round(y)) > 1e-6 or abs(x - round(x)) > 1e-6
               for y, x in w.food_xy)               # genuinely off the grid
    assert not any(w._point_in_walls(y, x) for y, x in w.food_xy)


def test_eating_needs_body_overlap_and_removes_the_morsel():
    w = World(_cfg(init_population=0))
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)
    a.brain.genome.diet, a.brain.genome.carnivore = 0.0, 0.0   # herbivore, digests type 1
    far = w.radius + w.food_radius + 0.5
    w.food_xy = np.array([[24.0, 24.0 + far]])      # out of reach
    w.food_type = np.array([1])
    assert w._eat_food_here(a) == 0.0 and len(w.food_type) == 1
    w.food_xy = np.array([[24.0, 24.0 + 0.5 * (w.radius + w.food_radius)]])  # overlapping
    w.food_type = np.array([1])
    gain = w._eat_food_here(a)
    assert gain > 0.0 and len(w.food_type) == 0      # eaten, morsel removed


# -- Stage 3: continuous CORPSES (float obstacle circles that decay) -----------------

def test_corpses_are_float_circles_that_block_and_decay():
    w = World(_cfg(init_population=0))
    assert not hasattr(w, "corpse")                 # the grid is gone
    a = w._spawn_agent(w._seed_genome(), pos=(24.0, 24.0), heading=0.0)
    a.energy = -1.0                                 # doomed; step() will kill it
    w.step()
    assert len(w.corpse_decay) == 1                 # death dropped a float body
    cy, cx = w.corpse_xy[0]
    assert w._corpse_blocks_circle(cy, cx)          # it blocks a body on top of it
    assert not w._corpse_blocks_circle(cy + 5, cx + 5)   # but not far away
    start = int(w.corpse_decay[0])
    for _ in range(start + 2):                       # let it decay away
        w.step()
    assert len(w.corpse_decay) == 0


def test_no_body_overlaps_a_corpse_over_a_run():
    w = World(_cfg(size=40, seed=3))
    for _ in range(250):
        w.step()
    assert not any(w._corpse_blocks_circle(a.y, a.x) for a in w.agents.values())


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
