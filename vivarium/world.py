"""The vivarium world: a bounded grid where agents live, eat, collide and breed.

Deterministic in (Config.seed, Config). One agent per cell (opacity), heading +
turn movement, ray vision, energy metabolism, asexual clone+mutate reproduction,
corpses as decaying obstacles, and reseed-from-the-longest-lived on extinction.
Agents update in a fixed id order so a dynamic population stays reproducible.
"""

from __future__ import annotations

import numpy as np

from vivarium.brain import (FORWARD, REPRODUCE, TURN_LEFT, TURN_RIGHT, Brain,
                            input_size)
from vivarium.config import Config

# heading -> (dy, dx): 0=N(up), 1=E, 2=S(down), 3=W
_HEAD_VEC = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
# heading -> base ray angle (x=cos, y=sin, y points down)
_HEAD_ANG = {0: -np.pi / 2, 1: 0.0, 2: np.pi / 2, 3: np.pi}


class Agent:
    __slots__ = ("id", "y", "x", "heading", "energy", "age", "brain")

    def __init__(self, aid, y, x, heading, energy, brain):
        self.id = aid
        self.y, self.x = y, x
        self.heading = heading
        self.energy = energy
        self.age = 0
        self.brain = brain


class World:
    def __init__(self, config: Config | None = None, genome_pool=None):
        self.cfg = config or Config()
        c = self.cfg
        # Optional seed pool of pre-evolved genomes: initial agents and extinction
        # reseeds draw from it (+ mutation) instead of random. This lets an
        # already-competent population be dropped into a HARD world (e.g. a maze,
        # where random brains can't bootstrap) and start living at once.
        self.genome_pool = [np.asarray(g, dtype=np.float64) for g in genome_pool] \
            if genome_pool else None
        self.rng = np.random.default_rng(c.seed)
        self.h = self.w = c.size
        self.tick = 0
        self._next_id = 0

        self.walls = self._generate_walls()
        self.food = np.zeros((self.h, self.w), dtype=bool)
        self.occ = np.full((self.h, self.w), -1, dtype=np.int64)   # agent id or -1
        self.corpse = np.zeros((self.h, self.w), dtype=np.int64)   # ticks remaining

        self.agents: dict[int, Agent] = {}
        self.hall: list[tuple[int, np.ndarray]] = []   # (age, genome), longest-lived
        self.n_in = input_size(c.n_rays)
        self.stats = {"births": 0, "deaths": 0}

        for _ in range(c.init_population):
            self._spawn_agent(self._seed_genome())
        self._regrow_food()

    def _seed_genome(self) -> np.ndarray:
        """A fresh starting genome: drawn (with mutation) from the seed pool if
        one was supplied, otherwise random."""
        if self.genome_pool:
            g = self.genome_pool[self.rng.integers(len(self.genome_pool))]
            return g + self.rng.normal(0.0, self.cfg.mutation, size=g.shape)
        return Brain.random(self.n_in, self.cfg.hidden, self.rng,
                            forward_bias=self.cfg.init_forward_bias).genome

    def hall_genomes(self) -> list[np.ndarray]:
        """The longest-lived genomes recorded so far (for saving / reseeding)."""
        return [g.copy() for _, g in self.hall]

    # -- map ----------------------------------------------------------------

    def _generate_walls(self) -> np.ndarray:
        """Border walls + inner strokes, each pierced by one doorway so the maze
        partitions the space into rooms/corridors without sealing it. Strokes are
        laid until ~`wall_density` of the interior is wall; a clear centre patch
        guarantees room to (re)spawn. Seeded -> the map is part of the run's
        determinism."""
        walls = np.zeros((self.h, self.w), dtype=bool)
        walls[0, :] = walls[-1, :] = True
        walls[:, 0] = walls[:, -1] = True
        target = int(self.cfg.wall_density * (self.h - 2) * (self.w - 2))
        lo = max(3, min(self.h, self.w) // 10)
        hi = max(lo + 1, min(self.h, self.w) // 3)
        placed = 0
        for _ in range(20_000):
            if placed >= target:
                break
            length = int(self.rng.integers(lo, hi + 1))
            door = int(self.rng.integers(length))
            if self.rng.random() < 0.5:                     # horizontal stroke
                y = int(self.rng.integers(2, self.h - 2))
                x0 = int(self.rng.integers(1, max(2, self.w - length)))
                cells = [(y, x0 + i) for i in range(length)]
            else:                                           # vertical stroke
                x = int(self.rng.integers(2, self.w - 2))
                y0 = int(self.rng.integers(1, max(2, self.h - length)))
                cells = [(y0 + i, x) for i in range(length)]
            for i, (yy, xx) in enumerate(cells):
                if i != door and 0 < yy < self.h - 1 and 0 < xx < self.w - 1:
                    placed += int(not walls[yy, xx])
                    walls[yy, xx] = True
        cy, cx = self.h // 2, self.w // 2
        walls[cy - 2:cy + 3, cx - 2:cx + 3] = False         # clear centre for spawns
        return walls

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

    def _spawn_agent(self, genome: np.ndarray, cell=None, heading=None) -> Agent | None:
        cell = cell or self._random_free_cell()
        if cell is None:
            return None
        y, x = cell
        h = self.rng.integers(4) if heading is None else heading
        a = Agent(self._next_id, y, x, int(h), self.cfg.start_energy,
                  Brain(self.n_in, self.cfg.hidden, genome))
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

    def _sense(self, a: Agent) -> np.ndarray:
        """Ray vision fanned around the heading + interoception, as the brain's
        input vector. Each ray reports the first thing it hits (food / obstacle /
        agent) and how near it is."""
        c = self.cfg
        vec = np.zeros(self.n_in)
        base = _HEAD_ANG[a.heading]
        angles = base + np.linspace(-c.fov / 2, c.fov / 2, c.n_rays)
        for i, ang in enumerate(angles):
            dx, dy = np.cos(ang), np.sin(ang)
            hit_type, hit_dist = -1, c.ray_length
            for k in range(1, c.ray_length + 1):
                cy = a.y + int(round(dy * k))
                cx = a.x + int(round(dx * k))
                if not (0 <= cy < self.h and 0 <= cx < self.w) or self.walls[cy, cx] \
                        or self.corpse[cy, cx] > 0:
                    hit_type, hit_dist = 1, k          # obstacle
                    break
                if self.occ[cy, cx] >= 0 and self.occ[cy, cx] != a.id:
                    hit_type, hit_dist = 2, k          # another agent
                    break
                if self.food[cy, cx]:
                    hit_type, hit_dist = 0, k          # food
                    break
            if hit_type >= 0:
                vec[i * 3 + hit_type] = 1.0 - hit_dist / c.ray_length
        vec[-2] = a.energy / c.max_energy               # interoception
        vec[-1] = 1.0                                    # bias
        return vec

    # -- one tick -----------------------------------------------------------

    def step(self) -> None:
        c = self.cfg
        self._regrow_food()
        dead: list[Agent] = []
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
            elif action == REPRODUCE:
                self._try_reproduce(a)
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
        """The landscape lives: every `wall_drift_every` ticks erode some inner
        walls and grow as many new ones elsewhere, so the maze slowly reshapes
        while keeping roughly constant density. Never walls an occupied/food cell
        (uses `_random_free_cell`), so it can't bury an agent or a meal."""
        c = self.cfg
        if c.wall_drift_every <= 0 or self.tick % c.wall_drift_every != 0:
            return
        inner = np.argwhere(self.walls)
        inner = [(y, x) for y, x in inner if 0 < y < self.h - 1 and 0 < x < self.w - 1]
        if not inner:
            return
        k = max(1, int(len(inner) * c.wall_drift_frac))
        for i in self.rng.choice(len(inner), size=min(k, len(inner)), replace=False):
            y, x = inner[i]
            self.walls[y, x] = False                    # erode
        for _ in range(k):
            cell = self._random_free_cell()
            if cell is not None:
                self.walls[cell] = True                 # regrow elsewhere

    def _try_reproduce(self, a: Agent) -> None:
        c = self.cfg
        if a.energy < c.repro_min_energy:
            return
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):   # fixed order = reproducible
            ny, nx = a.y + dy, a.x + dx
            if self._free(ny, nx):
                a.energy -= c.repro_cost
                child = self._spawn_agent(a.brain.child_genome(self.rng, c.mutation),
                                          cell=(ny, nx))
                if child is not None:
                    child.energy = c.repro_cost * 0.8
                    self.stats["births"] += 1
                return

    def _kill(self, a: Agent) -> None:
        self.occ[a.y, a.x] = -1
        self.corpse[a.y, a.x] = self.cfg.corpse_decay      # body blocks, then decays
        self.agents.pop(a.id, None)
        self.stats["deaths"] += 1
        self._remember(a)

    def _remember(self, a: Agent) -> None:
        """Keep a small hall-of-fame of the longest-lived genomes, for reseeding."""
        self.hall.append((a.age, a.brain.genome.copy()))
        self.hall.sort(key=lambda t: t[0], reverse=True)
        del self.hall[max(self.cfg.reseed_count, 12):]

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

    def snapshot(self) -> dict:
        return {"tick": self.tick, "population": self.population(),
                "food": int(self.food.sum()), "corpses": int((self.corpse > 0).sum()),
                "births": self.stats["births"], "deaths": self.stats["deaths"]}
