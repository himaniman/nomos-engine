"""The atlas: the sheet as a map, a CA that plans by wave.

What is worth pinning: the map is an honest projection of the senses (a wall seen is a
wall painted, where it is), the wave routes AROUND what the map knows, the read-out at
the centre follows the wave, and the whole thing is deterministic — a life is a pure
function of (config, seed, ticks) like everything else here."""

import numpy as np

from cortex import atlas as A
from cortex import probes
from cortex import tiles as T
from cortex.senses import ScentField, sense
from cortex.world import World


def _arena_atlas(name="open", radius=16):
    w, spots = probes.build(name)
    at = A.Atlas(A.Config(radius=radius), w.cfg)
    field = ScentField(w.cfg.smell_radius)
    return w, spots, at, field


# --- painting: the senses land where they refer to -----------------------------------

def test_a_seen_wall_is_painted_where_it_is():
    w, _, at, field = _arena_atlas("corner")
    a = w.agent                                  # at (16, 6) facing east; wall at x=16
    at.act(sense(w, field), a.y, a.x, a.heading)
    R = at.R
    col = at.state[A.SOLID, :, R + 10]           # map column of world x = 16
    assert col[R - 3:R + 4].max() > 0.9, "the wall dead ahead must be painted solid"
    # and the ground between here and there is known free
    assert at.state[A.FREE, R, R + 1:R + 9].min() > 0.9


def test_the_map_scrolls_with_the_body():
    w, _, at, field = _arena_atlas("corner")
    a = w.agent
    at.act(sense(w, field), a.y, a.x, a.heading)
    R = at.R
    before = at.state[A.SOLID, R, R + 10]
    assert before > 0.9
    a.x += 2.0                                    # teleport two cells east
    at.act(sense(w, field), a.y, a.x, a.heading)
    # the wall is now eight cells ahead on the map, not ten
    assert at.state[A.SOLID, R, R + 8] > 0.9


def test_a_seen_morsel_is_a_thing_and_a_source():
    w, spots, at, field = _arena_atlas("near")   # berry four cells dead ahead
    a = w.agent
    at.act(sense(w, field), a.y, a.x, a.heading)
    R = at.R
    assert at.state[A.THING, R, R + 4] > 0.5
    assert at.sources[R, R + 4] > 0.5


# --- the rule: the wave goes round what it knows ---------------------------------------

def test_the_wave_routes_around_a_known_wall():
    at = A.Atlas(A.Config(radius=8, subticks=40))
    R = at.R
    at.cell = (0, 0)
    # a wall across the whole map at column R+3, with one gap at row R-4
    at.state[A.SOLID, :, R + 3] = 1.0
    at.state[A.SOLID, R - 4, R + 3] = 0.0
    src = np.zeros((at.side, at.side))
    src[R, R + 6] = 1.0                           # source behind the wall
    at._wave(src, 40)
    V = at.state[A.VALUE]
    # the value climbs toward the gap, not toward the wall in front of the source
    assert V[R - 4, R + 2] > V[R, R + 2]
    # ...and the leak through the wall is a trickle: the cell right behind the wall
    # from the source is worth far less than the cell right behind the gap
    assert V[R - 4, R + 2] > 3 * V[R, R + 4] * A.Config().leak
    # the read-out steps toward the gap (up and right), not into the wall
    at.last_target = None
    cmd = at._read(0.0, np.zeros(at.layout["size"]))
    dy, dx = at.last_target
    assert dy < 0, f"expected a step toward the gap, got {(dy, dx)}"


def test_no_source_means_stand_still():
    at = A.Atlas(A.Config(radius=6, frontier=0.0))
    at.cell = (0, 0)
    at._wave(np.zeros((at.side, at.side)), 5)
    thrust, turn, act = at._read(0.0, np.zeros(at.layout["size"]))
    assert (thrust, turn, act) == (0.0, 0.0, 0.0)


# --- behaviour ---------------------------------------------------------------------

def test_it_takes_the_foraging_ladder():
    out = probes._suite(lambda: A.Driver(A.Config()), 600)
    p = out["profile"]
    for name in ("near", "open", "side", "behind", "corner"):
        assert p[name] >= 0.9, f"{name}: {p[name]}"


def test_a_life_is_deterministic():
    r1 = A.lives([3], ticks=1500)[0]
    r2 = A.lives([3], ticks=1500)[0]
    assert r1 == r2


def test_a_life_pays_for_its_map():
    paid = A.lives([3], ticks=800, cfg=A.Config(cell_cost=3e-5))[0]
    free = A.lives([3], ticks=800, cfg=A.Config(cell_cost=0.0))[0]
    # same behaviour (the cost does not steer), less energy
    assert paid["walked"] == free["walked"]
    w_paid, _ = A.live(3, 800, A.Config(cell_cost=3e-5))
    w_free, _ = A.live(3, 800, A.Config(cell_cost=0.0))
    assert w_paid.agent.energy < w_free.agent.energy


def test_frame_has_world_and_map_side_by_side():
    w, mind = A.live(0, 60)
    img = A.frame(w, mind.atlas, scale=2)
    S = mind.atlas.side * 2
    assert img.shape == (S, 2 * S + 4, 3)
    assert img.dtype == np.uint8


# --- learning ----------------------------------------------------------------------

def test_one_burn_teaches_the_hue():
    at = A.Atlas(A.Config(radius=6))
    import math
    from cortex.world import BERRY, FOOD_HSV, NIGHTCAP
    def hue(kind):
        h = FOOD_HSV[kind][0] * 2 * math.pi
        return math.cos(h), math.sin(h)
    assert at.appetite(*hue(NIGHTCAP)) > 0.5 and at.appetite(*hue(BERRY)) > 0.5
    at.learn(at._feat(*hue(NIGHTCAP)), -1.0)
    assert at.appetite(*hue(NIGHTCAP)) < 0.5, "one burn must be enough"
    assert at.appetite(*hue(BERRY)) > 0.5, "...and must not spoil the berry"
    at.learn(at._feat(*hue(BERRY)), +1.0)
    assert at.appetite(*hue(NIGHTCAP)) < 0.5, "a good berry must not forgive the nightcap"


def test_the_poison_lesson_is_learned():
    first, second = A.poison_lesson(ticks=900)
    assert first["bad"] == 1, "never burned, it eats the purple one"
    assert second["good"] == 1 and second["bad"] == 0, second
    naive = A.poison_lesson(A.Config(appetite_lr=0.0), ticks=900)
    assert naive[1]["bad"] == 1, "without learning the second visit repeats the first"
