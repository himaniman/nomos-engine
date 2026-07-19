"""The continuous vivarium world.

A float-space fork of `vivarium.world`: agents have continuous (y, x) positions,
a continuous heading (radians) and a continuous speed, and their motor command is
read continuously from the brain's output logits — forward THRUST from the
forward/stay tension, TURN-RATE from the left/right tension — instead of a 4-way
argmax. Everything else is vivarium's ecology, kept as-is: energy metabolism,
ray vision (same input width, so evolved brains/pools transfer), food types with
drifting halos, diet, predation, speciation by colour, corpses, hall-of-fame and
extinction reseed from the best. Food and walls remain on a grid; agents move in
float space over them, collide as circles (opacity is no longer free — it's a
distance test), and see by ray-marching the grid + nearby agents.

Deterministic in (Config, seed): agents update in a fixed id order and all
sampling draws from one seeded rng, so a run reproduces exactly (per-environment,
as with the numba brain path).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from aqua.space import Buckets
from vivarium import nca, neat
from vivarium.config import Config
from vivarium.world import _hue_dist

_BRAINS = {"neat": neat, "nca": nca}
_TWO_PI = 2.0 * math.pi

# Rich sensor layout (see World._sense). The continuous world affords more, and
# smoother, senses than the grid one — this is where that pays off. Exteroception
# is a per-ray fan; interoception is the agent's own body state; smell is an
# omnidirectional gradient to the nearest food (a modality vision doesn't give).
RAY_CH = 4     # per vision ray: edible food / obstacle / KIN agent / NON-KIN agent
INTERO = 7     # energy, own speed, own turn-rate, sin(heading), cos(heading), age, bias
SMELL = 2      # egocentric (forward, lateral) direction to the nearest food


def sensor_size(n_rays: int, smell: bool = True) -> int:
    """Width of the brain's input vector for this sensor configuration."""
    return n_rays * RAY_CH + INTERO + (SMELL if smell else 0)


def _logits(brain) -> np.ndarray:
    """The brain's raw action outputs (4,). NCA stores them as `last_logits`;
    NEAT exposes them as the output nodes of its relaxed state."""
    ll = getattr(brain, "last_logits", None)
    if ll is not None:
        return np.asarray(ll, dtype=np.float64)
    return np.asarray(brain.state[brain.out_idx], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return (a + math.pi) % _TWO_PI - math.pi


class Agent:
    __slots__ = ("id", "y", "x", "heading", "energy", "age", "offspring", "brain",
                 "last_speed", "last_turn", "food_eaten", "kills")

    def __init__(self, aid, y, x, heading, energy, brain):
        self.id = aid
        self.y = float(y)
        self.x = float(x)
        self.heading = float(heading)      # radians; 0 = +x, pi/2 = +y (down)
        self.energy = energy
        self.age = 0
        self.offspring = 0                  # children bred (HUD + hall success score)
        self.food_eaten = 0                 # food morsels eaten in this life (HUD)
        self.kills = 0                      # prey caught by predation in this life (HUD)
        self.brain = brain
        self.last_speed = 0.0              # proprioception: last commanded speed/turn,
        self.last_turn = 0.0               # fed back as senses next tick


class World:
    """A continuous bounded world of evolving brains. Grid-backed food/walls, but
    float agents that move and see smoothly."""

    def __init__(self, config: Config | None = None, genome_pool=None,
                 max_speed: float = 1.0, max_turn: float = 0.45,
                 radius: float = 0.45, mate_radius: float = 1.8,
                 ray_step: float = 0.6, smell: bool = True,
                 smell_range: float | None = None, substep: float = 0.5,
                 smell_sharpness: float = 6.0):
        self.cfg = config or Config()
        c = self.cfg
        self.max_speed = max_speed          # cells per tick at full forward thrust
        self.max_turn = max_turn            # radians per tick at full turn
        # body radius ~= wall thickness/2 (a creature about as wide as a wall bar) yet
        # under half a cell so it still threads a one-cell doorway (0.5 wedges in doors).
        self.radius = radius                # agent body radius (collision/eat)
        # A move is marched in hops of <= `substep` cells and the body is collision-
        # tested at every hop, so no matter how fast an agent goes it can never JUMP
        # over a wall or a food cell — the world always recomputes what it hit. With
        # substep (0.5) < 2*radius (0.7) consecutive body circles overlap, so nothing
        # thin can slip between two samples. This is what makes max_speed safe to raise.
        self.substep = substep
        self.mate_radius = mate_radius      # how close two agents breed
        self.ray_step = ray_step            # vision ray-march increment (cells)
        self.smell = smell                  # omnidirectional food-gradient sense on?
        self.smell_range = (smell_range if smell_range is not None
                            else 2.0 * c.ray_length_max)
        # smell is an INVERSE-SQUARE field: faint across the whole map (a long tail out
        # to smell_range) but rising steeply near a source — half strength at
        # smell_sharpness cells, ~1 on top of the food. So a creature always has a weak
        # bearing to the nearest food from anywhere, sharpening fast as it closes in.
        self.smell_sharpness = smell_sharpness

        self.genome_pool = list(genome_pool) if genome_pool else None
        self.brain = _BRAINS[c.brain]
        self.rng = np.random.default_rng(c.seed)
        self.h = self.w = c.size
        self.tick = 0
        self.epoch = 0
        self.epoch_start = 0
        self.food_rate = float(c.food_regrow)
        self.food_cap = float(c.food_target)
        self._next_id = 0
        # The brain input is sized for the MAX ray count; each agent fans its own
        # (heritable) n_rays into the first slots and leaves the rest zero, so a
        # variable eye plugs into a fixed-width brain.
        self.n_in = sensor_size(c.n_rays_max, self.smell)
        self.n_out = self.brain.N_ACTIONS
        self._next_node_id = self.n_in + self.n_out
        if self.genome_pool:
            self._next_node_id = max([self._next_node_id]
                                     + [g.max_node_id() + 1 for g in self.genome_pool])

        self.strokes = self._generate_strokes()
        self.walls = np.zeros((self.h, self.w), dtype=bool)
        self._rasterize_walls()        # aggregate cache: open-area count for food density
        self._build_wall_rects()       # the real geometry: float rectangles (collision/vision)
        self.food_radius = c.food_radius
        # food is now a cloud of FLOAT morsels (stage 2): parallel arrays of positions
        # and types, queried/eaten by true distance — no longer a grid.
        self.food_xy = np.empty((0, 2), dtype=np.float64)
        self.food_type = np.empty(0, dtype=np.int64)
        self.halos = np.array([[self.rng.integers(2, self.h - 2),
                                self.rng.integers(2, self.w - 2)]
                               for _ in range(c.n_food_types)], dtype=np.float64)
        # corpses are float obstacle CIRCLES now (stage 3): position + ticks-left arrays,
        # no grid. A dead body blocks like a wall until it decays.
        self.corpse_radius = c.corpse_radius
        self.corpse_xy = np.empty((0, 2), dtype=np.float64)
        self.corpse_decay = np.empty(0, dtype=np.int64)

        self.agents: dict[int, Agent] = {}
        # spatial index of agent ids for the O(1)-ish proximity queries (collision,
        # mate search, vision) — kept live as agents spawn/move/die (see aqua.space).
        self.abuckets = Buckets(cell=max(2.0, 2.0 * radius, mate_radius))
        self.hall: list[tuple[int, object]] = []           # all-time best (success score)
        self.hall_epoch: list[tuple[int, object]] = []      # best of the CURRENT epoch only
        self.stats = {"births": 0, "deaths": 0}
        self.deaths_log: deque = deque()
        # the ray fan is now PER-AGENT (heritable count), so relative angles are
        # computed per genome on demand and cached by count.
        self._ray_angle_cache: dict[int, np.ndarray] = {}

        for _ in range(c.init_population):
            self._spawn_agent(self._seed_genome())
        self._regrow_food()

    # -- brains -------------------------------------------------------------

    def _mutate(self, g, leap=False):
        child, self._next_node_id = self.brain.mutate(g, self.rng, self.cfg,
                                                      self._next_node_id, leap=leap)
        return child

    def _phenotype(self, genome):
        c = self.cfg
        if c.brain == "nca":
            return self.brain.Phenotype(genome, subticks=c.nca_subticks,
                                        plastic=c.nca_plastic, lr=c.nca_plastic_lr,
                                        forget=c.nca_plastic_forget,
                                        elig_decay=c.nca_plastic_elig,
                                        clip=c.nca_plastic_clip)
        return self.brain.Phenotype(genome)

    def _seed_genome(self):
        if self.genome_pool:
            g = self.genome_pool[self.rng.integers(len(self.genome_pool))]
            return self._mutate(g)
        return self.brain.random_genome(self.n_in, self.n_out, self.rng,
                                        forward_bias=self.cfg.init_forward_bias,
                                        cfg=self.cfg)

    def hall_genomes(self):
        return [g.copy() for _, g in self.hall]

    # -- map (grid, copied from vivarium) -----------------------------------

    def _generate_strokes(self) -> list[dict]:
        target = int(self.cfg.wall_density * (self.h - 2) * (self.w - 2))
        lo = max(3, min(self.h, self.w) // 10)
        hi = max(lo + 1, min(self.h, self.w) // 3)
        strokes, placed = [], 0
        for _ in range(20_000):
            if placed >= target:
                break
            length = int(self.rng.integers(lo, hi + 1))
            axis = "h" if self.rng.random() < 0.5 else "v"
            if axis == "h":
                y = int(self.rng.integers(2, self.h - 2))
                x = int(self.rng.integers(1, max(2, self.w - length)))
            else:
                x = int(self.rng.integers(2, self.w - 2))
                y = int(self.rng.integers(1, max(2, self.h - length)))
            strokes.append({"axis": axis, "y": y, "x": x, "length": length,
                            "door": int(self.rng.integers(length)),
                            "dir": 1 if self.rng.random() < 0.5 else -1})
            placed += length - 1
        return strokes

    def _stroke_cells(self, s: dict):
        for i in range(s["length"]):
            if i == s["door"]:
                continue
            yy = s["y"] + (i if s["axis"] == "v" else 0)
            xx = s["x"] + (i if s["axis"] == "h" else 0)
            if 0 < yy < self.h - 1 and 0 < xx < self.w - 1:
                yield yy, xx

    def _rasterize_walls(self) -> None:
        self.walls[:] = False
        self.walls[0, :] = self.walls[-1, :] = True
        self.walls[:, 0] = self.walls[:, -1] = True
        for s in self.strokes:
            for yy, xx in self._stroke_cells(s):
                self.walls[yy, xx] = True
        cy, cx = self.h // 2, self.w // 2
        self.walls[cy - 2:cy + 3, cx - 2:cx + 3] = False

    def _build_wall_rects(self) -> None:
        """The CONTINUOUS wall geometry: each stroke becomes one or more axis-aligned
        float RECTANGLES `wall_thickness` cells thick (a run of cells split by the door
        / interior clip / centre safe-zone into contiguous rectangles), plus four border
        bars. Collision (`_wall_blocks_circle`) and vision (`_ray_wall_dist`) run on
        these analytically — no per-cell blockiness, exact ray hits. Derived from the
        same strokes as the `self.walls` cache, so the two never disagree."""
        h, w, ht = self.h, self.w, 0.5 * self.cfg.wall_thickness
        rects = [(-0.5, 0.5, -0.5, w - 0.5),                  # top border row
                 (h - 1.5, h - 0.5, -0.5, w - 0.5),           # bottom border row
                 (-0.5, h - 0.5, -0.5, 0.5),                  # left border col
                 (-0.5, h - 0.5, w - 1.5, w - 0.5)]           # right border col
        ccy, ccx = h // 2, w // 2

        def clear(yy, xx):                                    # the spawn safe-zone hole
            return ccy - 2 <= yy <= ccy + 2 and ccx - 2 <= xx <= ccx + 2

        for s in self.strokes:
            runs, run = [], []                                # contiguous along-axis cells
            for i in range(s["length"]):
                yy = s["y"] + (i if s["axis"] == "v" else 0)
                xx = s["x"] + (i if s["axis"] == "h" else 0)
                solid = (i != s["door"] and 0 < yy < h - 1 and 0 < xx < w - 1
                         and not clear(yy, xx))
                if solid:
                    run.append(i)
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for grp in runs:
                i0, i1 = grp[0], grp[-1]
                if s["axis"] == "h":
                    y = s["y"]
                    rects.append((y - ht, y + ht, s["x"] + i0 - 0.5, s["x"] + i1 + 0.5))
                else:
                    x = s["x"]
                    rects.append((s["y"] + i0 - 0.5, s["y"] + i1 + 0.5, x - ht, x + ht))
        self.wall_rects = np.array(rects, dtype=np.float64)   # (N, 4) = y0, y1, x0, x1

    def _wall_blocks_circle(self, y: float, x: float) -> bool:
        """Does a body circle of `self.radius` at (y, x) overlap any wall rectangle?
        (circle-vs-AABB: distance to the rect's closest point < radius)."""
        r2 = self.radius * self.radius
        for y0, y1, x0, x1 in self.wall_rects:
            ny = y0 if y < y0 else (y1 if y > y1 else y)
            nx = x0 if x < x0 else (x1 if x > x1 else x)
            dy, dx = y - ny, x - nx
            if dy * dy + dx * dx < r2:
                return True
        return False

    def _corpse_blocks_circle(self, y: float, x: float) -> bool:
        """Does a body circle at (y, x) overlap any corpse circle? (float obstacles:
        centres closer than radius + corpse_radius)."""
        if len(self.corpse_decay) == 0:
            return False
        dy = self.corpse_xy[:, 0] - y
        dx = self.corpse_xy[:, 1] - x
        reach = self.radius + self.corpse_radius
        return bool(np.any(dy * dy + dx * dx < reach * reach))

    def _blocked_circle(self, y: float, x: float) -> bool:
        """Would an agent's BODY (a circle of `self.radius`) at (y, x) overlap a wall
        or a corpse? Walls are analytic float rectangles; corpses are still grid cells."""
        return self._wall_blocks_circle(y, x) or self._corpse_blocks_circle(y, x)

    def _rects_within(self, y: float, x: float, r: float) -> np.ndarray:
        """The wall rectangles whose closest point is within `r` of (y, x) — the only
        ones a ray of reach `r` from here could possibly hit (a per-agent prune so the
        analytic ray test loops a handful of walls, not all of them)."""
        wr = self.wall_rects
        if len(wr) == 0:
            return wr
        ny = np.clip(y, wr[:, 0], wr[:, 1])
        nx = np.clip(x, wr[:, 2], wr[:, 3])
        return wr[(y - ny) ** 2 + (x - nx) ** 2 <= r * r]

    def _ray_wall_dist(self, ay: float, ax: float, dx: float, dy: float,
                       maxd: float, rects=None) -> float:
        """Distance from (ay, ax) along the unit ray (dx, dy) to the nearest wall
        rectangle, capped at `maxd` (ray-vs-AABB slab test, analytic). Used by vision
        and the app's FOV cone — one source of truth for wall geometry. `rects`
        defaults to ALL walls; vision passes a per-agent within-reach subset."""
        if rects is None:
            rects = self.wall_rects
        eps = 1e-9
        if abs(dx) < eps:
            dx = eps if dx >= 0 else -eps
        if abs(dy) < eps:
            dy = eps if dy >= 0 else -eps
        ix, iy = 1.0 / dx, 1.0 / dy
        best = maxd
        for y0, y1, x0, x1 in rects:
            tx0, tx1 = (x0 - ax) * ix, (x1 - ax) * ix
            lox, hix = (tx0, tx1) if tx0 < tx1 else (tx1, tx0)
            ty0, ty1 = (y0 - ay) * iy, (y1 - ay) * iy
            loy, hiy = (ty0, ty1) if ty0 < ty1 else (ty1, ty0)
            t_enter = lox if lox > loy else loy
            t_exit = hix if hix < hiy else hiy
            if t_enter <= t_exit and t_exit >= 0.0:
                t = t_enter if t_enter >= 0.0 else 0.0
                if t < best:
                    best = t
        return best

    def _blocked(self, y: float, x: float, exclude: int) -> bool:
        """A position is blocked if the body overlaps a wall/corpse or another
        agent's body (opacity). Used for tangential SLIDING, where predation never
        applies — sliding just stops at a solid neighbour."""
        return (self._blocked_circle(y, x)
                or self._agent_near(y, x, 2 * self.radius, exclude=exclude) is not None)

    # -- food (float morsels; the grid is gone) -----------------------------

    def _point_in_walls(self, y: float, x: float) -> bool:
        """Is the point (y, x) inside any wall rectangle? (food placement test)."""
        for y0, y1, x0, x1 in self.wall_rects:
            if y0 <= y <= y1 and x0 <= x <= x1:
                return True
        return False

    def _nearest_food_dist(self, y: float, x: float) -> float:
        """Distance to the nearest existing food morsel (inf if none)."""
        if len(self.food_type) == 0:
            return math.inf
        dy = self.food_xy[:, 0] - y
        dx = self.food_xy[:, 1] - x
        return float(np.sqrt(np.min(dy * dy + dx * dx)))

    def _valid_food_pos(self, y: float, x: float) -> bool:
        """A float spot is plantable if it's inside the map, clear of walls, corpses,
        other food (no stacking) and agents."""
        return (1.0 <= y <= self.h - 1.0 and 1.0 <= x <= self.w - 1.0
                and not self._point_in_walls(y, x)
                and not self._corpse_near_food(y, x)
                and self._nearest_food_dist(y, x) > 2.0 * self.food_radius
                and self._agent_near(y, x, self.radius + self.food_radius) is None)

    def _corpse_near_food(self, y: float, x: float) -> bool:
        """Is a corpse close enough to forbid planting food here?"""
        if len(self.corpse_decay) == 0:
            return False
        dy = self.corpse_xy[:, 0] - y
        dx = self.corpse_xy[:, 1] - x
        reach = self.corpse_radius + self.food_radius
        return bool(np.any(dy * dy + dx * dx < reach * reach))

    def _random_free_food_pos(self):
        for _ in range(20):
            y = float(self.rng.uniform(1.0, self.h - 1.0))
            x = float(self.rng.uniform(1.0, self.w - 1.0))
            if self._valid_food_pos(y, x):
                return y, x
        return None

    def _add_food(self, pos, ftype: int) -> None:
        self.food_xy = np.vstack([self.food_xy, [pos[0], pos[1]]])
        self.food_type = np.append(self.food_type, ftype)

    def _remove_food(self, j: int) -> None:
        self.food_xy = np.delete(self.food_xy, j, axis=0)
        self.food_type = np.delete(self.food_type, j)

    def _effective_regrow(self) -> float:
        c = self.cfg
        if c.food_pop_target <= 0:
            return self.food_rate
        over = len(self.agents) - c.food_pop_target
        return int(round(min(c.food_regrow_boot,
                             max(0.0, c.food_regrow_boot - c.food_regrow_gain * over))))

    def _regrow_food(self) -> None:
        c = self.cfg
        open_cells = int((~self.walls).sum())          # walls cache: coarse open-area count
        target = int(self.food_cap * open_cells)
        deficit = target - len(self.food_type)
        sigma = c.food_halo_sigma * min(self.h, self.w)
        rate = self._effective_regrow()
        count = int(rate)
        frac = rate - count
        if frac and self.rng.random() < frac:
            count += 1
        for _ in range(min(count, max(0, deficit))):
            k = int(self.rng.integers(c.n_food_types))
            if self.rng.random() < c.food_scatter_prob:
                pos = self._random_free_food_pos()
                if pos is not None:
                    self._add_food(pos, k + 1)
                continue
            cy, cx = self.halos[k]
            for _try in range(6):
                y = float(cy + self.rng.normal(0.0, sigma))
                x = float(cx + self.rng.normal(0.0, sigma))
                if self._valid_food_pos(y, x):
                    self._add_food((y, x), k + 1)
                    break

    def _diet_eff(self, diet: float, food_type: int) -> float:
        k = self.cfg.n_food_types
        if k <= 1:
            return 1.0
        target = (food_type - 1) / (k - 1)
        return max(self.cfg.diet_min_eff, 1.0 - abs(diet - target))

    def _drift_halos(self) -> None:
        c = self.cfg
        if c.food_halo_drift_every <= 0 or self.tick % c.food_halo_drift_every != 0:
            return
        step = self.rng.uniform(-c.food_halo_drift, c.food_halo_drift, self.halos.shape)
        self.halos = np.clip(self.halos + step, 2, [self.h - 2, self.w - 2])

    # -- agents / geometry --------------------------------------------------

    def _rebuild_buckets(self) -> None:
        """Rebuild the agent index from scratch (O(n), once per tick) so it can't drift."""
        self.abuckets.clear()
        for a in self.agents.values():
            self.abuckets.insert(a.id, a.y, a.x)

    def _agents_within(self, y: float, x: float, r: float, exclude: int = -1):
        """Yield agents in the bucket-neighbourhood of (y, x) (a SUPERSET of those
        within r; callers filter by exact distance)."""
        for aid in self.abuckets.neighbors(y, x, r):
            if aid == exclude:
                continue
            b = self.agents.get(aid)
            if b is not None:
                yield b

    def _agent_near(self, y: float, x: float, r: float, exclude: int = -1):
        """The closest agent whose centre is within `r` of (y, x), or None — via the
        spatial bucket, so it's local work, not a scan of the whole population."""
        best, best_d = None, r
        for a in self._agents_within(y, x, r, exclude):
            d = math.hypot(a.y - y, a.x - x)
            if d < best_d:
                best, best_d = a, d
        return best

    def _random_free_pos(self):
        for _ in range(40):
            y = float(self.rng.uniform(1.5, self.h - 1.5))
            x = float(self.rng.uniform(1.5, self.w - 1.5))
            if not self._blocked_circle(y, x) and self._agent_near(y, x, 2 * self.radius) is None:
                return y, x
        return None

    def _free_pos_near(self, y: float, x: float):
        """A free spot ~one body away from (y, x) to drop a child."""
        d = 2.2 * self.radius
        for k in range(8):
            ang = k / 8.0 * _TWO_PI
            ny, nx = y + math.sin(ang) * d, x + math.cos(ang) * d
            if (1.0 <= ny < self.h - 1.0 and 1.0 <= nx < self.w - 1.0
                    and not self._blocked_circle(ny, nx)
                    and self._agent_near(ny, nx, 2 * self.radius) is None):
                return ny, nx
        return None

    def _spawn_agent(self, genome, pos=None, heading=None, energy=None):
        pos = pos or self._random_free_pos()
        if pos is None:
            return None
        y, x = pos
        head = float(self.rng.uniform(0, _TWO_PI)) if heading is None else heading
        a = Agent(self._next_id, y, x, head,
                  self.cfg.start_energy if energy is None else energy,
                  self._phenotype(genome))
        self.agents[a.id] = a
        self.abuckets.insert(a.id, y, x)
        self._next_id += 1
        return a

    # -- vision (continuous ray-march) --------------------------------------

    def ray_angles(self, n: int) -> np.ndarray:
        """Relative angles of an `n`-ray fan spread across the FOV (cached by count).
        More rays = finer angular resolution within the same field of view."""
        n = max(1, int(n))
        a = self._ray_angle_cache.get(n)
        if a is None:
            fov = self.cfg.fov
            a = (np.array([0.0]) if n == 1
                 else np.linspace(-fov / 2, fov / 2, n))
            self._ray_angle_cache[n] = a
        return a

    def _sense(self, a: Agent) -> np.ndarray:
        """The rich input vector: a per-ray vision fan (edible food / obstacle /
        KIN / NON-KIN agent, each a smooth ray-march proximity), the agent's own
        body state (energy, speed, turn-rate, a heading compass, age), and an
        omnidirectional SMELL gradient toward the nearest food — a modality vision
        doesn't give (it reaches past the rays and through walls). Diet dims
        indigestible food, as in the grid world."""
        c = self.cfg
        vec = np.zeros(self.n_in)
        g = a.brain.genome
        nr = max(1, min(int(g.n_rays), c.n_rays_max))         # this agent's heritable eye
        rl = max(1.0, float(g.ray_len))
        diet, hue = g.diet, g.hue
        effs = [self._diet_eff(diet, t) for t in range(1, c.n_food_types + 1)]
        cone = 0.5 * c.fov + 0.6                              # a ray can only see this arc
        # nearby agents once (bucket-local, FOV-pruned), tagged KIN / NON-KIN
        near = []
        for b in self._agents_within(a.y, a.x, rl + self.radius, exclude=a.id):
            d = math.hypot(b.y - a.y, b.x - a.x)
            if d <= rl + self.radius:
                bearing = math.atan2(b.y - a.y, b.x - a.x)
                if abs(_wrap_angle(bearing - a.heading)) <= cone:
                    kin = _hue_dist(hue, b.brain.genome.hue) <= c.species_hue
                    near.append((d, bearing, kin))
        near_food = []                                        # food morsels within the CONE
        if len(self.food_type):                               # (only what a ray could see:
            fdy = self.food_xy[:, 0] - a.y                    #  vectorised, FOV-pruned)
            fdx = self.food_xy[:, 1] - a.x
            fdd = np.hypot(fdy, fdx)
            fb = np.arctan2(fdy, fdx)
            rel_b = np.abs((fb - a.heading + math.pi) % _TWO_PI - math.pi)
            in_view = (fdd <= rl + self.food_radius) & (rel_b <= cone)
            for idx in np.nonzero(in_view)[0]:
                near_food.append((float(fdd[idx]), float(fb[idx]), int(self.food_type[idx])))
        near_corpse = []                                      # corpse obstacles in the CONE
        if len(self.corpse_decay):
            cdy = self.corpse_xy[:, 0] - a.y
            cdx = self.corpse_xy[:, 1] - a.x
            cdd = np.hypot(cdy, cdx)
            cb = np.arctan2(cdy, cdx)
            crel = np.abs((cb - a.heading + math.pi) % _TWO_PI - math.pi)
            cview = (cdd <= rl + self.corpse_radius) & (crel <= cone)
            for idx in np.nonzero(cview)[0]:
                near_corpse.append((float(cdd[idx]), float(cb[idx])))
        vis_rects = self._rects_within(a.y, a.x, rl)         # walls a ray could reach
        half_spacing = 0.5 * c.fov / max(1, nr - 1)
        half_ray = half_spacing + self.radius / rl
        for i, rel in enumerate(self.ray_angles(nr)):
            ang = a.heading + rel
            dx, dy = math.cos(ang), math.sin(ang)
            dw = self._ray_wall_dist(a.y, a.x, dx, dy, rl, vis_rects)   # analytic WALL hit
            if dw < rl:
                hit_d, hit_ch, hit_val = dw, 1, 1.0 - dw / rl
            else:
                hit_d, hit_ch, hit_val = rl, -1, 0.0
            for cd, cbear in near_corpse:                    # a corpse along this ray? (obstacle)
                if cd < hit_d and abs(_wrap_angle(cbear - ang)) <= half_spacing + self.corpse_radius / cd:
                    hit_d, hit_ch, hit_val = cd, 1, 1.0 - cd / rl
            for fd, fbear, ft in near_food:                  # a food morsel along this ray?
                if fd < hit_d and abs(_wrap_angle(fbear - ang)) <= half_spacing + self.food_radius / fd:
                    hit_d, hit_ch, hit_val = fd, 0, effs[ft - 1] * (1.0 - fd / rl)
            for d, bearing, kin in near:                     # an agent nearer than that?
                if d < hit_d and abs(_wrap_angle(bearing - ang)) <= half_ray:
                    hit_d, hit_ch = d, (2 if kin else 3)     # KIN / NON-KIN channel
                    hit_val = 1.0 - d / rl
            if hit_ch >= 0:
                vec[i * RAY_CH + hit_ch] = hit_val

        o = c.n_rays_max * RAY_CH                              # interoception block
                                                              # (fixed offset: rays live in
                                                              # the first n_rays_max slots)
        vec[o + 0] = a.energy / c.max_energy
        vec[o + 1] = a.last_speed / self.max_speed
        vec[o + 2] = a.last_turn / self.max_turn             # signed turn-rate
        vec[o + 3] = math.sin(a.heading)                     # heading compass (absolute)
        vec[o + 4] = math.cos(a.heading)
        vec[o + 5] = math.tanh(a.age / 500.0)                # maturity
        vec[o + 6] = 1.0                                     # bias

        if self.smell and len(self.food_type):               # nearest-food gradient
            dy = self.food_xy[:, 0] - a.y
            dx = self.food_xy[:, 1] - a.x
            dd = np.hypot(dy, dx)
            j = int(np.argmin(dd))
            dist = float(dd[j])
            if 1e-9 < dist <= self.smell_range:
                ft = int(self.food_type[j])
                amp = self._diet_eff(diet, ft) / (1.0 + (dist / self.smell_sharpness) ** 2)
                rel = _wrap_angle(math.atan2(dy[j], dx[j]) - a.heading)
                so = o + INTERO
                vec[so + 0] = amp * math.cos(rel)            # food ahead (+) / behind (-)
                vec[so + 1] = amp * math.sin(rel)            # food to one side (signed)
        return vec

    # -- continuous motor read-out ------------------------------------------

    def _command(self, logits: np.ndarray) -> tuple[float, float]:
        """Continuous (speed, turn) from the four action logits: forward THRUST is
        the forward-vs-stay tension through a sigmoid (0..max_speed), TURN-RATE is
        the right-vs-left tension through a tanh (-max_turn..max_turn). So the same
        4-output brain that argmaxed a discrete action now drives smoothly."""
        fwd, left, right, stay = (float(logits[nca.FORWARD]), float(logits[nca.TURN_LEFT]),
                                  float(logits[nca.TURN_RIGHT]), float(logits[nca.STAY]))
        speed = self.max_speed / (1.0 + math.exp(-(fwd - stay)))
        turn = self.max_turn * math.tanh(right - left)
        return speed, turn

    # -- one tick -----------------------------------------------------------

    def step(self) -> None:
        c = self.cfg
        plastic = c.brain == "nca" and c.nca_plastic
        self._regrow_food()
        self._rebuild_buckets()                             # resync the index to tick-start
        dead: list[Agent] = []
        bred: set[int] = set()
        for aid in sorted(self.agents):
            a = self.agents.get(aid)
            if a is None:
                continue
            if c.brain == "nca":
                a.brain.act(self._sense(a), rng=self.rng, temp=c.nca_explore_temp)
            else:
                a.brain.act(self._sense(a))
            speed, turn = self._command(_logits(a.brain))
            a.last_speed, a.last_turn = speed, turn         # proprioception for next tick
            a.energy -= c.base_cost
            if c.brain == "nca":
                a.energy -= c.nca_cell_cost * int(a.brain.genome.mask.sum())
            reward = 0.0
            if abs(turn) > 1e-6:
                a.energy -= c.turn_cost * (abs(turn) / self.max_turn)
                a.heading = _wrap_angle(a.heading + turn)
            if speed > 1e-6:
                a.energy -= c.move_cost * (speed / self.max_speed)
                ny = a.y + math.sin(a.heading) * speed
                nx = a.x + math.cos(a.heading) * speed
                reward = self._advance(a, ny, nx)
                self.abuckets.move(a.id, a.y, a.x)          # reindex to its new cell
            if plastic:
                a.brain.learn(reward / c.food_energy)
            if a.energy >= c.repro_min_energy and a.id not in bred:
                self._try_reproduce(a, bred)
            a.age += 1
            if a.energy <= 0:
                dead.append(a)

        for a in dead:
            self._kill(a)
        if len(self.corpse_decay):                           # corpses decay, expired ones go
            self.corpse_decay -= 1
            live = self.corpse_decay > 0
            self.corpse_xy, self.corpse_decay = self.corpse_xy[live], self.corpse_decay[live]
        self._drift_walls()
        self._drift_halos()
        self._immigrate()
        cutoff = self.tick - self.cfg.lifespan_window
        while self.deaths_log and self.deaths_log[0][0] < cutoff:
            self.deaths_log.popleft()
        self.tick += 1
        if not self.agents:
            self._reseed()

    def _advance(self, a: Agent, ny: float, nx: float) -> float:
        """March agent `a` from its current spot toward (ny, nx) in body-sized
        sub-steps, colliding as a CIRCLE the whole way: walls/corpses and other
        agents (opacity) block it, and it EATS every food cell it enters — the whole
        path is resolved, not just the endpoint, so a fast agent can neither tunnel
        through a wall nor skip food. When a hop hits a wall it SLIDES along it
        (keeps the free axis) instead of dead-stopping, so agents don't wedge into
        walls. A carnivore that reaches another agent eats it. Returns total energy
        gained (the learn signal)."""
        c = self.cfg
        dy, dx = ny - a.y, nx - a.x
        dist = math.hypot(dy, dx)
        if dist < 1e-9:
            return 0.0
        n = max(1, int(math.ceil(dist / self.substep)))
        sy, sx = dy / n, dx / n                               # per-hop delta
        reward = 0.0
        for _ in range(n):
            ty, tx = a.y + sy, a.x + sx
            other = self._agent_near(ty, tx, 2 * self.radius, exclude=a.id)
            if other is not None:
                if (c.predation and self._can_predate(a, other)
                        and not self._wall_blocks_circle(ty, tx)):
                    reward += self._predate(a, other, ty, tx)  # eat prey, move onto it
                    reward += self._eat_food_here(a)
                    continue
                break                                          # opaque agent: stop
            if not self._blocked_circle(ty, tx):
                a.y, a.x = ty, tx
            elif abs(sx) > 1e-9 and not self._blocked(a.y, tx, a.id):
                a.x = tx                                       # slide along +/- x
            elif abs(sy) > 1e-9 and not self._blocked(ty, a.x, a.id):
                a.y = ty                                       # slide along +/- y
            else:
                break                                          # wedged: stop
            reward += self._eat_food_here(a)
        return reward

    def _eat_food_here(self, a: Agent) -> float:
        """Eat the nearest food morsel whose CIRCLE the agent's body overlaps; return
        the energy gained (0 if none in reach)."""
        c = self.cfg
        if len(self.food_type) == 0:
            return 0.0
        dy = self.food_xy[:, 0] - a.y
        dx = self.food_xy[:, 1] - a.x
        d2 = dy * dy + dx * dx
        j = int(np.argmin(d2))
        reach = self.radius + self.food_radius
        if d2[j] >= reach * reach:
            return 0.0
        ft = int(self.food_type[j])
        self._remove_food(j)
        g = a.brain.genome
        gain = c.food_energy * self._diet_eff(g.diet, ft) * (1 - g.carnivore)
        a.energy = min(c.max_energy, a.energy + gain)
        a.food_eaten += 1
        return gain

    # -- reproduction -------------------------------------------------------

    def _try_reproduce(self, a: Agent, bred: set) -> None:
        """Sexual reproduction by PROXIMITY: a fed, same-species agent within
        `mate_radius` mates (crossover of the two, fitter leads); a very-full lone
        agent clones asexually. The child drops at a free spot beside the parent."""
        c = self.cfg
        mate, mate_d = None, self.mate_radius              # NEAREST eligible same-species mate
        for b in self._agents_within(a.y, a.x, self.mate_radius, exclude=a.id):
            if b.id in bred or b.energy < c.repro_min_energy:
                continue
            d = math.hypot(b.y - a.y, b.x - a.x)
            if d < mate_d and _hue_dist(a.brain.genome.hue,
                                        b.brain.genome.hue) <= c.species_hue:
                mate, mate_d = b, d
        cell = self._free_pos_near(a.y, a.x)
        if cell is None:
            return
        if mate is not None:
            fit, other = (a, mate) if a.energy >= mate.energy else (mate, a)
            child_g = self._mutate(self.brain.crossover(fit.brain.genome,
                                                        other.brain.genome, self.rng))
            a.energy -= c.repro_cost / 2
            mate.energy -= c.repro_cost / 2
            bred.add(a.id)
            bred.add(mate.id)
            parents = (a, mate)
        elif a.energy >= c.asexual_energy:
            child_g = self._mutate(a.brain.genome)
            a.energy -= c.repro_cost
            bred.add(a.id)
            parents = (a,)
        else:
            return
        self._lamarck(child_g, parents)
        if self._spawn_agent(child_g, pos=cell, energy=c.repro_cost * 0.8) is not None:
            self.stats["births"] += 1
            for p in parents:
                p.offspring += 1

    def _lamarck(self, child_g, parents) -> None:
        lam = self.cfg.nca_lamarck
        if self.cfg.brain != "nca" or lam <= 0.0:
            return
        dWs = [p.brain.dW for p in parents if getattr(p.brain, "dW", None) is not None]
        if dWs:
            child_g.P_out = child_g.P_out + lam * np.mean(dWs, axis=0)

    # -- predation ----------------------------------------------------------

    def _can_predate(self, pred: Agent, prey: Agent | None) -> bool:
        c = self.cfg
        return (prey is not None
                and pred.brain.genome.carnivore >= c.carnivore_threshold
                and _hue_dist(pred.brain.genome.hue, prey.brain.genome.hue) > c.species_hue)

    def _predate(self, pred: Agent, prey: Agent, ny: float, nx: float) -> float:
        g = pred.brain.genome
        self.agents.pop(prey.id, None)
        self.abuckets.remove(prey.id)
        self.stats["deaths"] += 1
        self._remember(prey)
        gain = min(self.cfg.max_energy, pred.energy + self.cfg.meat_energy * g.carnivore) \
            - pred.energy
        pred.energy += gain
        pred.kills += 1
        pred.y, pred.x = ny, nx
        return gain

    # -- death / hall -------------------------------------------------------

    def _kill(self, a: Agent) -> None:
        self._add_corpse(a.y, a.x)                            # drop a float body at death
        self.agents.pop(a.id, None)
        self.abuckets.remove(a.id)
        self.stats["deaths"] += 1
        self._remember(a)

    def _add_corpse(self, y: float, x: float) -> None:
        self.corpse_xy = np.vstack([self.corpse_xy, [y, x]])
        self.corpse_decay = np.append(self.corpse_decay, self.cfg.corpse_decay)

    def _remember(self, a: Agent) -> None:
        success = a.age + self.cfg.hall_offspring_weight * a.offspring
        g = a.brain.genome.copy()
        for hall in (self.hall, self.hall_epoch):           # all-time + current-epoch boards
            hall.append((success, g))
            hall.sort(key=lambda t: t[0], reverse=True)
            del hall[max(self.cfg.reseed_count, 12):]
        self.deaths_log.append((self.tick, a.age))

    def lifespan_stats(self) -> tuple[float, float]:
        cutoff = self.tick - self.cfg.lifespan_window
        ages = [age for t, age in self.deaths_log if t >= cutoff]
        if not ages:
            return 0.0, 0.0
        return float(np.mean(ages)), float(np.median(ages))

    def top_genomes(self, k: int):
        return [g.copy() for _, g in self.hall[:k]]

    # -- landscape drift ----------------------------------------------------

    def _drift_walls(self) -> None:
        c = self.cfg
        if c.wall_drift_every <= 0 or self.tick % c.wall_drift_every != 0:
            return
        moved = False
        for s in self.strokes:
            if self.rng.random() >= c.wall_step_prob:
                continue
            ny = s["y"] + (s["dir"] if s["axis"] == "v" else 0)
            nx = s["x"] + (s["dir"] if s["axis"] == "h" else 0)
            end = (ny + (s["length"] - 1 if s["axis"] == "v" else 0),
                   nx + (s["length"] - 1 if s["axis"] == "h" else 0))
            if 1 <= ny and 1 <= nx and 0 < end[0] < self.h - 1 and 0 < end[1] < self.w - 1:
                s["y"], s["x"] = ny, nx
                moved = True
            else:
                s["dir"] *= -1
        if moved:
            self._rasterize_walls()        # refresh the food-placement cache and
            self._build_wall_rects()       # the continuous geometry together
            self._clear_under_walls()

    def _clear_under_walls(self) -> None:
        if len(self.food_type):                              # drop food a wall moved onto
            keep = np.array([not self._point_in_walls(y, x) for y, x in self.food_xy])
            self.food_xy, self.food_type = self.food_xy[keep], self.food_type[keep]
        for a in [ag for ag in self.agents.values()
                  if self._wall_blocks_circle(ag.y, ag.x)]:  # body caught by a moved wall
            self.agents.pop(a.id, None)
            self.abuckets.remove(a.id)
            self.stats["deaths"] += 1
            self._remember(a)

    # -- immigration / extinction reseed ------------------------------------

    def _immigrate(self) -> None:
        c = self.cfg
        if (c.immigrate_every <= 0 or self.tick == 0 or self.tick % c.immigrate_every
                or not self.agents):
            return
        donors = [g for _, g in self.hall[:6]] or \
            [a.brain.genome for a in self.agents.values()]
        for _ in range(c.immigrate_count):
            leap = self.rng.random() < c.nca_leap_prob
            g = self._mutate(donors[int(self.rng.integers(len(donors)))].copy(), leap=leap)
            g.hue = float(self.rng.random())
            g.diet = float(self.rng.random())
            self._spawn_agent(g)

    def _strong_mutate(self, g):
        leap_pass = int(self.rng.integers(self.cfg.reseed_mutate_passes)) \
            if self.rng.random() < self.cfg.nca_leap_prob else -1
        for i in range(self.cfg.reseed_mutate_passes):
            g = self._mutate(g, leap=(i == leap_pass))
        return g

    def _reseed(self) -> None:
        self.epoch += 1
        self.epoch_start = self.tick
        self.hall_epoch = []                                # fresh leaderboard per epoch
        elites = [g for _, g in self.hall[:self.cfg.reseed_elites]]
        if not elites:
            elites = [self._seed_genome()
                      for _ in range(min(self.cfg.reseed_elites, self.cfg.reseed_count))]
        for i in range(self.cfg.reseed_count):
            if i < len(elites):
                self._spawn_agent(elites[i].copy())
            else:
                parent = elites[int(self.rng.integers(len(elites)))]
                self._spawn_agent(self._strong_mutate(parent))

    # -- observation (mirrors vivarium so the app panel is reusable) --------

    def population(self) -> int:
        return len(self.agents)

    def _species_groups_raw(self) -> list[list[Agent]]:
        agents = sorted(self.agents.values(), key=lambda a: a.brain.genome.hue)
        if not agents:
            return []
        groups, cur = [], [agents[0]]
        for a in agents[1:]:
            if a.brain.genome.hue - cur[-1].brain.genome.hue > self.cfg.species_hue:
                groups.append(cur)
                cur = [a]
            else:
                cur.append(a)
        groups.append(cur)
        return groups

    def species_count(self) -> int:
        hues = sorted(a.brain.genome.hue for a in self.agents.values())
        if not hues:
            return 0
        gaps = [hues[i] - hues[i - 1] for i in range(1, len(hues))]
        gaps.append(1.0 - hues[-1] + hues[0])
        return max(1, sum(1 for g in gaps if g > self.cfg.species_hue))

    def species_groups(self) -> list[dict]:
        thr = self.cfg.carnivore_threshold
        out = []
        for grp in self._species_groups_raw():
            gs = [a.brain.genome for a in grp]
            out.append({"n": len(grp),
                        "hue": float(np.mean([g.hue for g in gs])),
                        "diet": float(np.mean([g.diet for g in gs])),
                        "carn": sum(1 for g in gs if g.carnivore >= thr) / len(gs),
                        "age": float(np.mean([a.age for a in grp])),
                        "cells": float(np.mean([g.complexity()[0] for g in gs])),
                        "params": float(np.mean([g.param_count() for g in gs]))})
        out.sort(key=lambda s: s["n"], reverse=True)
        return out

    def population_genomes(self, per: int = 3):
        out = []
        for grp in self._species_groups_raw():
            for a in sorted(grp, key=lambda a: a.age, reverse=True)[:per]:
                out.append(a.brain.genome.copy())
        return out

    def carnivore_fraction(self) -> float:
        n = len(self.agents)
        if not n:
            return 0.0
        thr = self.cfg.carnivore_threshold
        return sum(1 for a in self.agents.values()
                   if a.brain.genome.carnivore >= thr) / n

    def snapshot(self) -> dict:
        life_mean, life_median = self.lifespan_stats()
        energies = [a.energy for a in self.agents.values()]
        return {"tick": self.tick, "population": self.population(),
                "energy_bodies": float(sum(energies)),
                "energy_food": float(len(self.food_type)) * self.cfg.food_energy,
                "energy_mean": float(np.mean(energies)) if energies else 0.0,
                "food": int(len(self.food_type)), "corpses": int(len(self.corpse_decay)),
                "births": self.stats["births"], "deaths": self.stats["deaths"],
                "species": self.species_count(), "carnivores": self.carnivore_fraction(),
                "life_mean": life_mean, "life_median": life_median,
                "epoch": self.epoch, "epoch_age": self.tick - self.epoch_start,
                "food_rate": self.food_rate,
                "food_cap": int(self.food_cap * int((~self.walls).sum()))}
