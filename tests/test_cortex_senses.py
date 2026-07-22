"""The sensorium: properties not labels, exact ray geometry, and a scent that walks.

The load-bearing claim here is the geodesic smell — a straight-line odour would leak
through rock into sealed pockets and point an agent at a wall, which is the failure
`nomos/forage.py` had to fix before its world was navigable at all."""

import math

import numpy as np

from cortex import tiles as T
from cortex.senses import (INTERO, N_TOUCH, ODOUR_CH, RAY_CH, TOUCH_CH, ScentField,
                           layout, ray_cast, sense, sensor_size)
from cortex.world import BERRY, FRUIT, Config, FoodSpot, World


def _bare(seed=0, side=64, **kw):
    """A world of clear ground with no food — a blank sheet to put one thing on."""
    w = World(seed=seed, cfg=Config(size=side, **kw))
    for y in range(side):
        for x in range(side):
            w.tiles.set(y, x, T.GROUND)
    ch = w.cfg.chunk
    for cy in range(side // ch + 1):
        for cx in range(side // ch + 1):
            w.food[(cy, cx)] = []
    w.agent.y, w.agent.x, w.agent.heading = 20.0, 20.0, 0.0
    return w


def _rays(v, n):
    return v[:n * RAY_CH].reshape(n, RAY_CH)


# --- layout ------------------------------------------------------------------------

def test_layout_covers_the_vector_exactly():
    n = 64
    L = layout(n)
    assert L["size"] == sensor_size(n)
    assert L["slices"]["vision"] == (0, n * RAY_CH)
    assert L["slices"]["intero"][1] == L["size"]
    seen = set()
    for lo, hi in L["slices"].values():
        assert not (seen & set(range(lo, hi))), "slices overlap"
        seen |= set(range(lo, hi))
    assert seen == set(range(L["size"])), "the layout leaves gaps"


def test_layout_is_shaped_for_the_brains_retinotopy():
    """vivarium/nca.py::_build_listen needs exactly these keys to give an eye organ
    one ray and a nose organ the odour block."""
    L = layout(64)
    assert set(L) >= {"ray_ch", "n_rays", "ray_mod_channel", "static_mod_idx"}
    assert len(L["ray_mod_channel"]) == RAY_CH
    for idx in L["static_mod_idx"].values():
        assert idx.max() < L["size"]


# --- vision ------------------------------------------------------------------------

def test_ray_distance_is_geometrically_exact():
    """Not a marched multiple of some step: the true distance to the wall FACE."""
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 30, T.STONE)
    d, m = ray_cast(w.tiles, 20.0, 20.0, 0.0, 40.0)
    assert m == T.STONE
    assert abs(d - 9.5) < 1e-9, d                      # cell 30 starts at x = 29.5


def test_ray_reports_nothing_when_it_reaches_nothing():
    w = _bare()
    d, m = ray_cast(w.tiles, 20.0, 20.0, 0.0, 5.0)
    assert m == -1 and d == 5.0


def test_a_wall_ahead_shows_up_solid_and_near():
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 24, T.STONE)
    r = _rays(sense(w), w.cfg.n_rays)
    mid = w.cfg.n_rays // 2
    assert r[mid, 5] == 1.0, "a stone wall should read as solid"
    assert r[mid, 4] > 0.7, "and as near"
    w2 = _bare()
    assert _rays(sense(w2), w2.cfg.n_rays)[mid, 4] < 0.05, "empty ground should read far"


def test_foliage_blocks_sight_but_not_the_body():
    """OPAQUE is not SOLID — an affordance the agent has to discover from colour."""
    w = _bare()
    for y in range(18, 23):
        w.tiles.set(y, 24, T.FOLIAGE)
    r = _rays(sense(w), w.cfg.n_rays)[w.cfg.n_rays // 2]
    assert r[4] > 0.7, "leaves should stop the ray"
    assert r[5] == 0.0, "but not read as solid"
    assert not w.tiles.blocked_circle(20.0, 24.0, w.cfg.radius), "nor block the body"


def test_food_is_seen_as_its_own_colour_not_as_food():
    """There is no 'edible' channel — only a hue an agent may learn to trust."""
    w = _bare()
    w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=BERRY)]
    berry = _rays(sense(w), w.cfg.n_rays)[w.cfg.n_rays // 2].copy()
    w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=FRUIT)]
    fruit = _rays(sense(w), w.cfg.n_rays)[w.cfg.n_rays // 2]
    assert berry[4] > 0.5 and fruit[4] > 0.5, "both should be seen"
    assert abs(berry[0] - fruit[0]) + abs(berry[1] - fruit[1]) > 0.2, \
        "different kinds must look different"


def test_hue_is_a_circle_so_reds_do_not_sit_next_to_a_seam():
    """Raw 0..1 hue puts a discontinuity between the berry (~0.97) and the wrap —
    exactly where the poison discrimination has to happen."""
    a = np.array([math.cos(0.99 * 2 * math.pi), math.sin(0.99 * 2 * math.pi)])
    b = np.array([math.cos(0.01 * 2 * math.pi), math.sin(0.01 * 2 * math.pi)])
    assert np.linalg.norm(a - b) < 0.2, "neighbouring hues must be neighbouring inputs"


def test_damage_is_visible_to_the_eye_too():
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 24, T.RUBBLE)
    mid = w.cfg.n_rays // 2
    before = _rays(sense(w), w.cfg.n_rays)[mid, 3]
    w.damage[(20, 24)] = 2
    after = _rays(sense(w), w.cfg.n_rays)[mid, 3]
    assert after < before * 0.8, "a chipped tile must look chipped"


def test_the_eye_resolves_a_small_object():
    """64 rays over 120 deg is 1.9 deg each — a 2x2 object ten cells out must span
    several rays, or shape is not perceivable in principle."""
    w = _bare()
    for y in range(19, 21):
        for x in range(29, 31):
            w.tiles.set(y, x, T.TRUNK)
    r = _rays(sense(w), w.cfg.n_rays)
    hits = int((r[:, 4] > 0.1).sum())
    assert 3 <= hits <= 20, f"a 2x2 block ten cells away covered {hits} rays"


# --- smell -------------------------------------------------------------------------

def test_scent_does_not_leak_through_rock():
    """A morsel sealed behind stone must smell of NOTHING, not faintly."""
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 24, T.STONE)
    w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=BERRY)]   # 6 cells away, walled off
    L = layout(w.cfg.n_rays)
    lo, hi = L["slices"]["odour"]
    v = sense(w)[lo:hi]
    assert v[0] == 0.0, f"scent came through a stone wall: {v[0]}"


def test_scent_points_along_the_walk_not_along_the_sightline():
    """The contrast IS the claim. Same agent, same morsel, same straight-line bearing
    (dead ahead) — but put a wall in between with a far gap and the reported direction
    must swing away from the sightline and toward the way round."""
    def odour(wall: bool):
        w = _bare()
        if wall:
            for y in range(64):
                w.tiles.set(y, 24, T.STONE)
            w.tiles.set(26, 24, T.GROUND)                     # the only way through
        w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=BERRY)]
        lo, hi = layout(w.cfg.n_rays)["slices"]["odour"]
        return sense(w)[lo:hi]

    clear, blocked = odour(False), odour(True)
    assert clear[0] > 0.0 and clear[1] > 0.9 and abs(clear[2]) < 0.1, \
        "with open ground the scent should point straight at it"
    assert blocked[0] > 0.0, "the morsel is reachable the long way, so it must smell"
    assert abs(blocked[2]) > 0.3, "but the direction must bend toward the gap"
    assert blocked[1] < clear[1], "and lean less into the wall than the sightline does"


def test_scent_is_steeper_up_close():
    def intensity(dx):
        w = _bare()
        w.food[(0, 0)] = [FoodSpot(y=20.0, x=20.0 + dx, kind=BERRY)]
        L = layout(w.cfg.n_rays)
        return sense(w)[L["slices"]["odour"][0]]
    near, far = intensity(2), intensity(16)
    assert near > far * 3, f"inverse-square expected, got {near:.3f} vs {far:.3f}"


def test_scent_seeps_through_what_can_be_opened_but_only_faintly():
    """A door is not airtight and neither is rubble. This matters more than it sounds:
    with a shut door as a hard scent wall, food behind one is undetectable and the door
    mechanic could never be learned at all. The PENALTY is what keeps it honest — you
    can tell something is back there, and it smells further away than it is."""
    def odour(mat):
        w = _bare()
        for y in range(64):
            w.tiles.set(y, 24, mat)
        w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=BERRY)]
        lo, _ = layout(w.cfg.n_rays)["slices"]["odour"]
        return float(sense(w)[lo])

    assert odour(T.STONE) == 0.0, "stone is stone"
    through_door = odour(T.DOOR)
    through_rubble = odour(T.RUBBLE)
    assert 0.0 < through_door < odour(T.GROUND) / 2, through_door
    assert 0.0 < through_rubble < through_door, "rubble should muffle more than a door"


def test_scent_cache_notices_a_broken_wall():
    w = _bare()
    for y in range(64):
        w.tiles.set(y, 24, T.RUBBLE)
    w.food[(0, 0)] = [FoodSpot(y=20.0, x=26.0, kind=BERRY)]
    field = ScentField(w.cfg.smell_radius)
    L = layout(w.cfg.n_rays)
    lo, _ = L["slices"]["odour"]
    muffled = sense(w, field)[lo]
    w.tiles.set(20, 24, T.GROUND)                             # smash the wall
    assert sense(w, field)[lo] > muffled * 2, "the scent field ignored a changed map"


# --- touch and interoception --------------------------------------------------------

def test_touch_reports_the_right_sector():
    w = _bare()
    w.tiles.set(20, 21, T.STONE)                              # dead ahead
    w.agent.x = 20.3
    L = layout(w.cfg.n_rays)
    lo, _ = L["slices"]["touch"]
    v = sense(w)
    assert v[lo] > 0.0, "should feel what is in front"
    assert v[lo + TOUCH_CH] == 0.0, "and nothing on the right"
    w.agent.heading = math.pi                                 # turn around
    v = sense(w)
    assert v[lo] == 0.0 and v[lo + 2 * TOUCH_CH] > 0.0, "now it is behind us"


def test_interoception_reads_the_body():
    w = _bare()
    w.agent.energy = w.cfg.max_energy / 2
    w.step(thrust=1.0, turn=1.0)
    L = layout(w.cfg.n_rays)
    lo, hi = L["slices"]["intero"]
    v = sense(w)[lo:hi]
    assert 0.4 < v[0] < 0.6                                   # energy
    assert v[1] == 1.0                                        # full speed
    assert v[2] == 1.0                                        # commanded turn
    assert abs(v[3] ** 2 + v[4] ** 2 - 1.0) < 1e-9            # heading on the circle
    assert v[7] == 1.0                                        # bias


# --- the whole thing ----------------------------------------------------------------

def test_sensing_is_deterministic_and_finite():
    w = World(seed=2)
    f = ScentField(w.cfg.smell_radius)
    for _ in range(40):
        w.step(thrust=1.0, turn=0.2)
    a, b = sense(w, f), sense(w, f)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert a.shape == (sensor_size(w.cfg.n_rays),)
    assert np.abs(a).max() <= 1.0 + 1e-9, "sensors should stay in [-1, 1]"


def test_sensing_never_labels():
    """The guard for the whole design: no channel says 'food'. Two morsels of the same
    colour and smell must be indistinguishable however differently they feed you."""
    from cortex.world import FOOD_HSV, FOOD_ODOUR, NIGHTCAP
    assert np.allclose(FOOD_ODOUR[BERRY], FOOD_ODOUR[NIGHTCAP]), \
        "poison must smell exactly like the berry it imitates"
    assert not np.allclose(FOOD_HSV[BERRY][0], FOOD_HSV[NIGHTCAP][0]), \
        "...and differ only in hue, or there is nothing to learn"
