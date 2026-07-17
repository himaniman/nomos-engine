"""The vivarium world: a bounded grid where agents live, eat, collide and breed.

Deterministic in (Config.seed, Config). One agent per cell (opacity), heading +
turn movement, ray vision, energy metabolism, asexual clone+mutate reproduction,
corpses as decaying obstacles, and reseed-from-the-longest-lived on extinction.
Agents update in a fixed id order so a dynamic population stays reproducible.
"""

from __future__ import annotations

import numpy as np

from vivarium import neat
from vivarium.config import Config
from vivarium.neat import FORWARD, TURN_LEFT, TURN_RIGHT, Phenotype


def input_size(n_rays: int) -> int:
    """Sensor vector width: 3 typed channels (food, obstacle, agent) per ray,
    plus interoceptive energy and a constant bias input."""
    return n_rays * 3 + 2

def _hue_dist(a: float, b: float) -> float:
    """Circular distance between two hues in [0, 0.5]."""
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


# heading -> (dy, dx): 0=N(up), 1=E, 2=S(down), 3=W
_HEAD_VEC = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
# heading -> base ray angle (x=cos, y=sin, y points down)
_HEAD_ANG = {0: -np.pi / 2, 1: 0.0, 2: np.pi / 2, 3: np.pi}


class Agent:
    __slots__ = ("id", "y", "x", "heading", "energy", "age", "brain", "offspring")

    def __init__(self, aid, y, x, heading, energy, brain):
        self.id = aid
        self.y, self.x = y, x
        self.heading = heading
        self.energy = energy
        self.age = 0
        self.offspring = 0            # children produced (the fecundity signal)
        self.brain = brain


class World:
    def __init__(self, config: Config | None = None, genome_pool=None):
        self.cfg = config or Config()
        c = self.cfg
        # Optional seed pool of pre-evolved genomes: initial agents and extinction
        # reseeds draw from it (+ mutation) instead of random. This lets an
        # already-competent population be dropped into a HARD world (e.g. a maze,
        # where random brains can't bootstrap) and start living at once.
        self.genome_pool = list(genome_pool) if genome_pool else None
        self.rng = np.random.default_rng(c.seed)
        self.h = self.w = c.size
        self.tick = 0
        self._next_id = 0
        self.n_in = input_size(c.n_rays)
        self.n_out = neat.N_ACTIONS
        # unique-id counter for hidden nodes added by structural mutation; starts
        # above any id in the seed pool so inherited nodes never clash with new ones
        self._next_node_id = self.n_in + self.n_out
        if self.genome_pool:
            self._next_node_id = max([self._next_node_id]
                                     + [g.max_node_id() + 1 for g in self.genome_pool])

        self.strokes = self._generate_strokes()
        self.walls = np.zeros((self.h, self.w), dtype=bool)
        self._rasterize_walls()
        self.food = np.zeros((self.h, self.w), dtype=bool)
        self.occ = np.full((self.h, self.w), -1, dtype=np.int64)   # agent id or -1
        self.corpse = np.zeros((self.h, self.w), dtype=np.int64)   # ticks remaining

        self.agents: dict[int, Agent] = {}
        self.hall: list[tuple[int, neat.Genome]] = []   # (age, genome), longest-lived
        self.stats = {"births": 0, "deaths": 0}
        self._ray_offsets = self._build_ray_offsets()   # precomputed vision rays

        for _ in range(c.init_population):
            self._spawn_agent(self._seed_genome())
        self._regrow_food()

    def _mutate(self, g: neat.Genome) -> neat.Genome:
        child, self._next_node_id = neat.mutate(g, self.rng, self.cfg,
                                                self._next_node_id)
        return child

    def _seed_genome(self) -> neat.Genome:
        """A fresh starting genome: a mutated draw from the seed pool if one was
        supplied, otherwise a random minimal net."""
        if self.genome_pool:
            g = self.genome_pool[self.rng.integers(len(self.genome_pool))]
            return self._mutate(g)
        return neat.random_genome(self.n_in, self.n_out, self.rng,
                                  forward_bias=self.cfg.init_forward_bias)

    def hall_genomes(self) -> list[neat.Genome]:
        """The longest-lived genomes recorded so far (for saving / reseeding)."""
        return [g.copy() for _, g in self.hall]

    # -- map ----------------------------------------------------------------

    def _generate_strokes(self) -> list[dict]:
        """Inner wall strokes as OBJECTS — each a line with one doorway and a walk
        direction (+1/-1 along its own axis). Laid until ~`wall_density` of the
        interior is covered. Seeded -> the map is part of the run's determinism."""
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
            s = {"axis": axis, "y": y, "x": x, "length": length,
                 "door": int(self.rng.integers(length)),
                 "dir": 1 if self.rng.random() < 0.5 else -1}
            strokes.append(s)
            placed += length - 1                            # minus the doorway
        return strokes

    def _stroke_cells(self, s: dict):
        """The wall cells of a stroke (doorway excluded), clipped to the interior."""
        for i in range(s["length"]):
            if i == s["door"]:
                continue
            yy = s["y"] + (i if s["axis"] == "v" else 0)
            xx = s["x"] + (i if s["axis"] == "h" else 0)
            if 0 < yy < self.h - 1 and 0 < xx < self.w - 1:
                yield yy, xx

    def _rasterize_walls(self) -> None:
        """Rebuild the wall grid from the border + current stroke positions, with
        a clear centre patch for (re)spawns."""
        self.walls[:] = False
        self.walls[0, :] = self.walls[-1, :] = True
        self.walls[:, 0] = self.walls[:, -1] = True
        for s in self.strokes:
            for yy, xx in self._stroke_cells(s):
                self.walls[yy, xx] = True
        cy, cx = self.h // 2, self.w // 2
        self.walls[cy - 2:cy + 3, cx - 2:cx + 3] = False

    # -- helpers ------------------------------------------------------------

    def _free(self, y: int, x: int) -> bool:
        """A cell an agent may occupy: in bounds, not wall/corpse/agent."""
        return (0 <= y < self.h and 0 <= x < self.w and not self.walls[y, x]
                and self.corpse[y, x] == 0 and self.occ[y, x] < 0)

    def _random_free_cell(self):
        free = np.argwhere((~self.walls) & (self.occ < 0) & (self.corpse == 0)
                           & (~self.food))
        if len(free) == 0:
            return None
        return tuple(free[self.rng.integers(len(free))])

    def _spawn_agent(self, genome: neat.Genome, cell=None, heading=None,
                     energy=None) -> Agent | None:
        cell = cell or self._random_free_cell()
        if cell is None:
            return None
        y, x = cell
        h = self.rng.integers(4) if heading is None else heading
        a = Agent(self._next_id, y, x, int(h),
                  self.cfg.start_energy if energy is None else energy,
                  Phenotype(genome))
        self.agents[a.id] = a
        self.occ[y, x] = a.id
        self._next_id += 1
        return a

    def _regrow_food(self) -> None:
        open_cells = int((~self.walls).sum())
        target = int(self.cfg.food_target * open_cells)
        deficit = target - int(self.food.sum())
        for _ in range(min(self.cfg.food_regrow, max(0, deficit))):
            cell = self._random_free_cell()
            if cell is None:
                break
            self.food[cell] = True

    # -- vision -------------------------------------------------------------

    def _build_ray_offsets(self):
        """Precompute, per heading, the integer (dy, dx) cell offsets along each
        vision ray. These depend only on (heading, fov, n_rays, ray_length) — all
        fixed — so casting a ray at run time is just walking a precomputed list,
        with no per-step round()/linspace (which the profiler showed were ~half of
        the whole simulation's time)."""
        c = self.cfg
        offsets = {}
        for heading, base in _HEAD_ANG.items():
            rays = []
            for ang in base + np.linspace(-c.fov / 2, c.fov / 2, c.n_rays):
                dx, dy = np.cos(ang), np.sin(ang)
                rays.append([(int(round(dy * k)), int(round(dx * k)))
                             for k in range(1, c.ray_length + 1)])
            offsets[heading] = rays
        return offsets

    def _sense(self, a: Agent) -> np.ndarray:
        """Ray vision fanned around the heading + interoception, as the brain's
        input vector. Each ray reports the first thing it hits (food / obstacle /
        agent) and how near it is — walked over precomputed ray offsets."""
        c = self.cfg
        walls, corpse, occ, food = self.walls, self.corpse, self.occ, self.food
        H, W, rl = self.h, self.w, c.ray_length
        vec = np.zeros(self.n_in)
        ay, ax, aid = a.y, a.x, a.id
        for i, ray in enumerate(self._ray_offsets[a.heading]):
            for k, (dy, dx) in enumerate(ray, start=1):
                cy, cx = ay + dy, ax + dx
                if not (0 <= cy < H and 0 <= cx < W) or walls[cy, cx] or corpse[cy, cx]:
                    vec[i * 3 + 1] = 1.0 - k / rl       # obstacle
                    break
                o = occ[cy, cx]
                if o >= 0 and o != aid:
                    vec[i * 3 + 2] = 1.0 - k / rl       # another agent
                    break
                if food[cy, cx]:
                    vec[i * 3] = 1.0 - k / rl           # food
                    break
        vec[-2] = a.energy / c.max_energy               # interoception
        vec[-1] = 1.0                                    # bias
        return vec

    # -- one tick -----------------------------------------------------------

    def step(self) -> None:
        c = self.cfg
        self._regrow_food()
        dead: list[Agent] = []
        bred: set[int] = set()                     # at most one child per agent/tick
        for aid in sorted(self.agents):            # fixed order -> reproducible
            a = self.agents.get(aid)
            if a is None:
                continue
            action = a.brain.act(self._sense(a))
            a.energy -= c.base_cost
            if action == FORWARD:
                a.energy -= c.move_cost
                dy, dx = _HEAD_VEC[a.heading]
                ny, nx = a.y + dy, a.x + dx
                if self._free(ny, nx):
                    self.occ[a.y, a.x] = -1
                    a.y, a.x = ny, nx
                    self.occ[ny, nx] = a.id
                    if self.food[ny, nx]:               # step onto food = eat
                        self.food[ny, nx] = False
                        a.energy = min(c.max_energy, a.energy + c.food_energy)
            elif action == TURN_LEFT:
                a.energy -= c.turn_cost
                a.heading = (a.heading - 1) % 4
            elif action == TURN_RIGHT:
                a.energy -= c.turn_cost
                a.heading = (a.heading + 1) % 4
            if a.energy >= c.repro_min_energy and a.id not in bred:
                self._try_reproduce(a, bred)           # innate drive: breed when able
            a.age += 1
            if a.energy <= 0:
                dead.append(a)

        for a in dead:
            self._kill(a)
        self.corpse[self.corpse > 0] -= 1              # corpses decay away
        self._drift_walls()                            # the living landscape shifts
        self.tick += 1
        if not self.agents:
            self._reseed()

    def _drift_walls(self) -> None:
        """The landscape lives: every `wall_drift_every` ticks each stroke may WALK
        one cell along its own axis (leading with either end, like a snake),
        keeping its shape. A stroke that would leave the interior, or whose new
        cells are occupied by an agent, reverses instead of moving — so walking
        walls never bury a creature. Topology stays roughly constant; positions
        drift slowly."""
        c = self.cfg
        if c.wall_drift_every <= 0 or self.tick % c.wall_drift_every != 0:
            return
        moved = False
        for s in self.strokes:
            if self.rng.random() >= c.wall_step_prob:
                continue
            ny = s["y"] + (s["dir"] if s["axis"] == "v" else 0)
            nx = s["x"] + (s["dir"] if s["axis"] == "h" else 0)
            cand = dict(s, y=ny, x=nx)
            cells = list(self._stroke_cells(cand))
            end = (ny + (cand["length"] - 1 if cand["axis"] == "v" else 0),
                   nx + (cand["length"] - 1 if cand["axis"] == "h" else 0))
            in_bounds = 1 <= ny and 1 <= nx and 0 < end[0] < self.h - 1 \
                and 0 < end[1] < self.w - 1
            blocked = any(self.occ[yy, xx] >= 0 for yy, xx in cells)
            if in_bounds and not blocked:
                s["y"], s["x"] = ny, nx
                moved = True
            else:
                s["dir"] *= -1                          # hit a wall/agent: turn back
        if moved:
            self._rasterize_walls()

    def _try_reproduce(self, a: Agent, bred: set) -> None:
        """Sexual reproduction by PROXIMITY: if a fed neighbour is adjacent, the
        child's genome is a crossover of the two (fitter = higher energy) plus
        mutation, and both parents pay. A lone but very-full agent falls back to
        asexual clone+mutate so pioneers can still colonize empty ground."""
        c = self.cfg
        neigh = ((-1, 0), (0, 1), (1, 0), (0, -1))
        mate = None
        for dy, dx in neigh:
            oid = self.occ[a.y + dy, a.x + dx] if 0 <= a.y + dy < self.h \
                and 0 <= a.x + dx < self.w else -1
            if oid >= 0 and oid != a.id and oid not in bred:
                b = self.agents.get(oid)
                # reproductive isolation by colour: mate only within the species
                if (b is not None and b.energy >= c.repro_min_energy
                        and _hue_dist(a.brain.genome.hue, b.brain.genome.hue)
                        <= c.species_hue):
                    mate = b
                    break
        cell = next(((a.y + dy, a.x + dx) for dy, dx in neigh
                     if self._free(a.y + dy, a.x + dx)), None)
        if cell is None:
            return
        if mate is not None:
            fit, other = (a, mate) if a.energy >= mate.energy else (mate, a)
            child_g = self._mutate(neat.crossover(fit.brain.genome,
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
        if self._spawn_agent(child_g, cell=cell, energy=c.repro_cost * 0.8) is not None:
            self.stats["births"] += 1
            for p in parents:
                p.offspring += 1

    def _kill(self, a: Agent) -> None:
        self.occ[a.y, a.x] = -1
        self.corpse[a.y, a.x] = self.cfg.corpse_decay      # body blocks, then decays
        self.agents.pop(a.id, None)
        self.stats["deaths"] += 1
        self._remember(a)

    def _remember(self, a: Agent) -> None:
        """Hall-of-fame of the most SUCCESSFUL genomes = long-lived AND fecund
        (age + a weight per child), so the saved pool selects for genomes that
        both survive and reproduce, not just old survivors."""
        success = a.age + self.cfg.hall_offspring_weight * a.offspring
        self.hall.append((success, a.brain.genome.copy()))
        self.hall.sort(key=lambda t: t[0], reverse=True)
        del self.hall[max(self.cfg.reseed_count, 12):]

    def top_genomes(self, k: int) -> list[neat.Genome]:
        """The k most successful genomes recorded (for island migration)."""
        return [g.copy() for _, g in self.hall[:k]]

    def _reseed(self) -> None:
        """Extinction: repopulate from the longest-lived genomes, else the seed
        pool, else random."""
        genomes = [g for _, g in self.hall[:self.cfg.reseed_count]]
        if not genomes:
            genomes = [self._seed_genome() for _ in range(self.cfg.reseed_count)]
        for g in genomes:
            self._spawn_agent(g.copy())

    # -- observation --------------------------------------------------------

    def population(self) -> int:
        return len(self.agents)

    def species_count(self) -> int:
        """How many distinct colour-species are alive: sort the lineage hues and
        split wherever the gap exceeds the mating tolerance (with wrap-around)."""
        hues = sorted(a.brain.genome.hue for a in self.agents.values())
        if not hues:
            return 0
        gaps = [hues[i] - hues[i - 1] for i in range(1, len(hues))]
        gaps.append(1.0 - hues[-1] + hues[0])          # wrap seam
        return max(1, sum(1 for g in gaps if g > self.cfg.species_hue))

    def snapshot(self) -> dict:
        return {"tick": self.tick, "population": self.population(),
                "food": int(self.food.sum()), "corpses": int((self.corpse > 0).sum()),
                "births": self.stats["births"], "deaths": self.stats["deaths"],
                "species": self.species_count()}
