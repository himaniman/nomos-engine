"""What the agent perceives — properties, never labels.

The rule this whole module exists to enforce (docs/cortex-world.md): the world tells
the agent "greenish, smells sweet, four cells away, and it stops light", NEVER "food".
Edibility is a conclusion, not an input. That is the only reason poison can ever be a
learnable problem rather than a lookup.

Four modalities, all egocentric:

  VISION   a fan of `n_rays` rays over `fov`, marched by exact grid DDA. Per ray:
           hue as (cos, sin) — hue is a CIRCLE, and a raw 0..1 hue puts a seam right
           between the berry (0.97) and the wrap point, which is precisely the
           discrimination the poison experiment needs — plus saturation, brightness,
           distance and whether the hit was solid. Brightness is multiplied by
           `World.damage_frac`, the same call the renderer uses, so a chipped wall
           looks chipped to the player and to the agent alike.

  SMELL    GEODESIC, not straight-line: a BFS from the agent's four neighbours flows
           AROUND walls, so a morsel in a sealed pocket cannot smell through the rock
           and a scent in a cave leads along the tunnel. (nomos/forage.py learned this
           the hard way; the journal calls it the decisive fix.) Each odorant reports
           an intensity plus the egocentric direction of the shortest walkable path,
           so smell tells you which way to GO, not merely that something is near.

  TOUCH    four sectors around the body, relative to heading: how hard it is pressed
           against something and what that something looks like.

  INTERO   energy, motion, heading, age, poisoning, bias.

`sense()` returns a flat float vector; `layout()` describes it. The layout is shaped
for the NCA brain's retinotopy (`vivarium/nca.py::_build_listen`, which maps a brain
cell's POSITION onto one ray slot), so sensory organs can specialise without any
rewiring when the brain arrives.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from cortex.tiles import BREAKABLE, DOOR, DOOR_OPEN, HSV, OPAQUE, SOLID
from cortex.world import FOOD_HSV, FOOD_ODOUR, N_ODOUR, World

RAY_CH = 6          # hue_cos, hue_sin, saturation, brightness, nearness, solid
ODOUR_CH = 3        # intensity, forward, lateral  (per odorant)
N_TOUCH = 4         # front, right, back, left (egocentric)
TOUCH_CH = 2        # pressure, hue of what is touched
INTERO = 8          # energy, speed, turn, sin(head), cos(head), age, poison, bias

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))          # N, S, W, E — fixed BFS order
WAYPOINT = 6.0      # when the walk bends, the smell points at the path this far out


def sensor_size(n_rays: int) -> int:
    return n_rays * RAY_CH + N_ODOUR * ODOUR_CH + N_TOUCH * TOUCH_CH + INTERO


def rays_from_size(n_in: int) -> int:
    """The inverse: how many rays a sensor vector of this width implies. So a genome
    and a world can never silently disagree about which slot means what, wherever the
    question is asked from."""
    static = N_ODOUR * ODOUR_CH + N_TOUCH * TOUCH_CH + INTERO
    return max(1, (int(n_in) - static) // RAY_CH)


def layout(n_rays: int) -> dict:
    """Index map of the sensor vector. `ray_mod_channel` / `static_mod_idx` are in the
    exact shape `vivarium/nca.py::_build_listen` expects, so a brain cell that becomes
    an eye organ hears ONE ray (chosen by its lattice position) and a nose organ hears
    the odour block — retinotopy with no adapter layer."""
    o = n_rays * RAY_CH
    odour = np.arange(o, o + N_ODOUR * ODOUR_CH, dtype=np.int64)
    touch = np.arange(odour[-1] + 1, odour[-1] + 1 + N_TOUCH * TOUCH_CH, dtype=np.int64)
    intero = np.arange(touch[-1] + 1, touch[-1] + 1 + INTERO, dtype=np.int64)
    return {"ray_ch": RAY_CH, "n_rays": n_rays,
            # modality ids: 1..6 are the per-ray channels, 7/8/9 the static blocks
            "ray_mod_channel": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
            "static_mod_idx": {7: odour, 8: touch, 9: intero},
            "slices": {"vision": (0, o), "odour": (o, o + N_ODOUR * ODOUR_CH),
                       "touch": (int(touch[0]), int(touch[-1]) + 1),
                       "intero": (int(intero[0]), int(intero[-1]) + 1)},
            "size": sensor_size(n_rays)}


# --- vision ------------------------------------------------------------------------

def ray_cast(tiles, y: float, x: float, ang: float, reach: float) -> tuple[float, int]:
    """March a ray from (y, x) at world angle `ang` until it meets an OPAQUE tile.
    Returns (distance, material) — material -1 when nothing was hit within `reach`.

    Exact grid traversal (Amanatides-Woo): the ray jumps boundary to boundary instead
    of sampling at a step size, so the hit distance is geometrically true and no thin
    wall is ever missed. Cells are unit squares centred on integers, so the traversal
    runs in u = y + 0.5 where cell k owns [k, k+1)."""
    dy, dx = math.sin(ang), math.cos(ang)
    u, v = y + 0.5, x + 0.5
    cy, cx = math.floor(u), math.floor(v)
    inf = float("inf")
    step_y, step_x = (1 if dy > 0 else -1), (1 if dx > 0 else -1)
    t_dy = abs(1.0 / dy) if dy else inf                  # distance per full cell
    t_dx = abs(1.0 / dx) if dx else inf
    t_y = ((cy + 1 - u) if dy > 0 else (u - cy)) * t_dy if dy else inf
    t_x = ((cx + 1 - v) if dx > 0 else (v - cx)) * t_dx if dx else inf
    t = 0.0
    while t < reach:
        if t_y < t_x:
            t, t_y, cy = t_y, t_y + t_dy, cy + step_y
        else:
            t, t_x, cx = t_x, t_x + t_dx, cx + step_x
        m = tiles.material(cy, cx)
        if OPAQUE[m]:
            return min(t, reach), int(m)
    return reach, -1


def _vision(w: World, out: np.ndarray) -> None:
    c, a = w.cfg, w.agent
    n, fov, reach = c.n_rays, c.fov, c.ray_reach
    step = fov / n
    base = a.heading - fov / 2 + step / 2
    dist = np.full(n, reach)
    hsv = np.zeros((n, 3))
    solid = np.zeros(n)
    for i in range(n):
        d, m = ray_cast(w.tiles, a.y, a.x, base + i * step, reach)
        dist[i] = d
        if m >= 0:
            hsv[i] = HSV[m]
            solid[i] = 1.0 if SOLID[m] else 0.0
            # a chipped tile LOOKS chipped: same damage_frac the renderer darkens by
            hy, hx = a.y + math.sin(base + i * step) * (d + 0.01), \
                     a.x + math.cos(base + i * step) * (d + 0.01)
            hsv[i, 2] *= 1.0 - 0.55 * w.damage_frac(int(round(hy)), int(round(hx)))

    # FOOD is a float entity, so it is projected onto the fan rather than marched into:
    # each morsel claims the rays its angular width covers, and wins only if it is
    # nearer than whatever that ray already hit.
    for s in w.nearby_food(a.y, a.x, reach):
        dy, dx = s.y - a.y, s.x - a.x
        d = math.hypot(dy, dx)
        if d < 1e-9:
            d = 1e-9
        rel = (math.atan2(dy, dx) - a.heading + math.pi) % (2 * math.pi) - math.pi
        half = math.atan2(c.food_radius, d) + step / 2
        lo = int(math.floor((rel - half + fov / 2) / step))
        hi = int(math.ceil((rel + half + fov / 2) / step))
        for i in range(max(0, lo), min(n, hi + 1)):
            if d < dist[i]:
                dist[i] = d
                hsv[i] = FOOD_HSV[s.kind]
                solid[i] = 0.0

    hue = hsv[:, 0] * 2 * math.pi
    out[0:n * RAY_CH:RAY_CH] = np.cos(hue)
    out[1:n * RAY_CH:RAY_CH] = np.sin(hue)
    out[2:n * RAY_CH:RAY_CH] = hsv[:, 1]
    out[3:n * RAY_CH:RAY_CH] = hsv[:, 2]
    out[4:n * RAY_CH:RAY_CH] = 1.0 - dist / reach        # NEARNESS: strong up close
    out[5:n * RAY_CH:RAY_CH] = solid


# --- smell -------------------------------------------------------------------------

def _passing(mat) -> float | None:
    """What it costs a SMELL to get past this tile, or None if it cannot.

    A door is not airtight and neither is a pile of rubble: scent seeps through, just
    faintly. This matters more than it sounds — with a shut door as a hard scent wall,
    food behind one is undetectable and the only way to find it is blind luck, so the
    door mechanic could never be learned at all. The penalty is what keeps it honest:
    you can tell there is something back there, and it smells further away than it is."""
    if not SOLID[mat]:
        return 1.0
    if mat in (DOOR, DOOR_OPEN):
        return 7.0
    if BREAKABLE[mat]:
        return 11.0
    return None                                          # stone is stone


class ScentField:
    """Geodesic distance from the agent, plus which of its four steps each cell is
    reached through. Recomputed only when the agent changes CELL or the map changes
    (breaking a wall opens a new path), which is what keeps a BFS affordable per tick."""

    def __init__(self, radius: int = 24):
        self.radius = int(radius)
        self._key = None
        self.dist: dict[tuple[int, int], float] = {}
        self.first: dict[tuple[int, int], int] = {}
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def update(self, w: World) -> None:
        ay, ax = int(round(w.agent.y)), int(round(w.agent.x))
        key = (ay, ax, w.tiles.version)
        if key == self._key:
            return
        self._key = key
        R = self.radius
        dist = {(ay, ax): 0.0}
        first = {(ay, ax): -1}
        parent = {}
        queue = deque()
        for k, (dy, dx) in enumerate(_DIRS):
            n = (ay + dy, ax + dx)
            if not SOLID[w.tiles.material(*n)]:
                dist[n], first[n], parent[n] = 1.0, k, (ay, ax)
                queue.append(n)
        while queue:
            qy, qx = queue.popleft()
            base = dist[(qy, qx)]
            if base + 1.0 > R:
                continue
            tag = first[(qy, qx)]
            for dy, dx in _DIRS:
                n = (qy + dy, qx + dx)
                if n in dist or abs(n[0] - ay) > R or abs(n[1] - ax) > R:
                    continue
                cost = _passing(w.tiles.material(*n))
                if cost is None:
                    continue
                d = base + cost
                if d > R:
                    continue
                dist[n], first[n], parent[n] = d, tag, (qy, qx)
                queue.append(n)
        self.dist, self.first, self.parent = dist, first, parent

    def waypoint(self, cell: tuple[int, int], ahead: float) -> tuple[int, int]:
        """The cell on the shortest walkable path to `cell` that is about `ahead`
        steps out from the agent — what to steer at when the straight line is not the
        way. Walks the BFS parents back from the target."""
        p = cell
        while p in self.parent and self.dist.get(p, 0.0) > ahead:
            p = self.parent[p]
        return p


def _smell(w: World, field: ScentField, out: np.ndarray, off: int) -> None:
    """Each odorant: how strong, and the egocentric direction of the WALK to it.

    The direction is a bearing, not a compass point. The first version reported the
    first cardinal step of the BFS, which is right far away in a corridor and wrong
    almost everywhere else: for a morsel two rows down and sixteen columns over the
    tag said SOUTH, because a south-first path is as short as an east-first one and the
    search happened to grow that tree first. A nose that says "south" for something
    that is east is a nose you cannot paint a map from. So: if the walk is about as
    long as the straight line, the bearing IS the straight line (there is nothing worth
    walking round); if the walk is longer, the bearing is to the point on the walk a
    few cells out — the way round, as a direction."""
    c, a = w.cfg, w.agent
    field.update(w)
    total = np.zeros(N_ODOUR)
    vec = np.zeros((N_ODOUR, 2))                          # summed world-frame bearings
    for s in w.nearby_food(a.y, a.x, field.radius):
        cell = (int(round(s.y)), int(round(s.x)))
        d = field.dist.get(cell)
        if d is None:                                    # walled off: no scent at all,
            continue                                     # not a faint one through rock
        amp = 1.0 / (1.0 + (d / c.smell_sharp) ** 2)
        mix = FOOD_ODOUR[s.kind] * amp
        total += mix
        dy, dx = s.y - a.y, s.x - a.x
        straight = math.hypot(dy, dx)
        if straight < 1e-9:
            continue
        if d > 1.15 * straight + 0.5 and straight > 2.5:
            wy, wx = field.waypoint(cell, WAYPOINT)      # the way round
            dy, dx = wy - a.y, wx - a.x
            norm = math.hypot(dy, dx)
            if norm < 1e-9:
                continue
            dy, dx = dy / norm, dx / norm
        else:
            dy, dx = dy / straight, dx / straight        # the straight line IS the walk
        vec += np.outer(mix, (dy, dx))
    sh, ch = math.sin(a.heading), math.cos(a.heading)
    for j in range(N_ODOUR):
        norm = max(1e-9, total[j])
        fwd = (vec[j, 0] * sh + vec[j, 1] * ch) / norm   # heading unit = (sin, cos)
        lat = (vec[j, 0] * ch - vec[j, 1] * sh) / norm   # its right-hand normal
        out[off + j * ODOUR_CH + 0] = min(1.0, total[j])
        out[off + j * ODOUR_CH + 1] = fwd
        out[off + j * ODOUR_CH + 2] = lat


# --- touch and interoception --------------------------------------------------------

def _touch(w: World, out: np.ndarray, off: int) -> None:
    """Four egocentric probes just outside the body: what is pressed against us, and
    what colour it is. Free information — the collision code already knows."""
    a, c = w.agent, w.cfg
    for k in range(N_TOUCH):
        ang = a.heading + k * math.pi / 2                # front, right, back, left
        press, hue = 0.0, 0.0
        for reach, weight in ((c.radius + 0.06, 1.0), (c.radius + 0.30, 0.45)):
            py = int(round(a.y + math.sin(ang) * reach))
            px = int(round(a.x + math.cos(ang) * reach))
            m = w.tiles.material(py, px)
            if SOLID[m]:
                press, hue = weight, HSV[m][0]
                break
        out[off + k * TOUCH_CH + 0] = press
        out[off + k * TOUCH_CH + 1] = hue


def _intero(w: World, out: np.ndarray, off: int) -> None:
    a, c = w.agent, w.cfg
    out[off + 0] = a.energy / c.max_energy
    out[off + 1] = a.speed / c.max_speed
    out[off + 2] = a.last_turn
    out[off + 3] = math.sin(a.heading)
    out[off + 4] = math.cos(a.heading)
    out[off + 5] = math.tanh(a.age / 5000.0)
    out[off + 6] = min(1.0, a.poison / 30.0)             # something is wrong inside
    out[off + 7] = 1.0                                   # bias


# --- the whole vector ---------------------------------------------------------------

def sense(w: World, field: ScentField | None = None) -> np.ndarray:
    """The agent's whole egocentric view of the world, as one flat vector."""
    n = w.cfg.n_rays
    out = np.zeros(sensor_size(n))
    _vision(w, out)
    o = n * RAY_CH
    _smell(w, field if field is not None else ScentField(w.cfg.smell_radius), out, o)
    o += N_ODOUR * ODOUR_CH
    _touch(w, out, o)
    _intero(w, out, o + N_TOUCH * TOUCH_CH)
    return out
