"""The vivarium world: a bounded grid where agents live, eat, collide and breed.

Deterministic in (Config.seed, Config). One agent per cell (opacity), heading +
turn movement, ray vision, energy metabolism, asexual clone+mutate reproduction,
corpses as decaying obstacles, and reseed-from-the-longest-lived on extinction.
Agents update in a fixed id order so a dynamic population stays reproducible.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from vivarium import nca, neat
from vivarium.config import Config
from vivarium.neat import FORWARD, TURN_LEFT, TURN_RIGHT, STAY

_BRAINS = {"neat": neat, "nca": nca}


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
        self.brain = _BRAINS[c.brain]              # neat or nca — the pluggable brain
        self.rng = np.random.default_rng(c.seed)
        self.h = self.w = c.size
        self.tick = 0
        self.epoch = 0                # how many times the world has gone fully extinct
        self.epoch_start = 0          # tick the current epoch (population) began
        self.food_rate = float(c.food_regrow)  # live-tunable fixed food/tick, may be < 1
                                               # (fractional -> probabilistic; app [ ] keys)
        self.food_cap = float(c.food_target)   # live-tunable STANDING food ceiling (frac
                                               # of open cells); the app , . keys move it
        self._next_id = 0
        self.n_in = input_size(c.n_rays)
        self.n_out = self.brain.N_ACTIONS
        # unique-id counter for hidden nodes added by structural mutation; starts
        # above any id in the seed pool so inherited nodes never clash with new ones
        self._next_node_id = self.n_in + self.n_out
        if self.genome_pool:
            self._next_node_id = max([self._next_node_id]
                                     + [g.max_node_id() + 1 for g in self.genome_pool])

        self.strokes = self._generate_strokes()
        self.walls = np.zeros((self.h, self.w), dtype=bool)
        self._rasterize_walls()
        self.food = np.zeros((self.h, self.w), dtype=np.int8)      # 0 none, 1..K type
        # each food type has a HALO centre it spawns around; the centres DRIFT, so
        # food is a moving resource patch (seeded -> part of the run's determinism)
        self.halos = np.array([[self.rng.integers(2, self.h - 2),
                                self.rng.integers(2, self.w - 2)]
                               for _ in range(c.n_food_types)], dtype=np.float64)
        self.occ = np.full((self.h, self.w), -1, dtype=np.int64)   # agent id or -1
        self.corpse = np.zeros((self.h, self.w), dtype=np.int64)   # ticks remaining

        self.agents: dict[int, Agent] = {}
        self.hall: list[tuple[int, neat.Genome]] = []   # (age, genome), longest-lived
        self.stats = {"births": 0, "deaths": 0}
        self.deaths_log: deque = deque()                 # (tick, age) of recent deaths
        self._ray_offsets = self._build_ray_offsets()   # precomputed vision rays

        for _ in range(c.init_population):
            self._spawn_agent(self._seed_genome())
        self._regrow_food()

    def _mutate(self, g, leap=False):
        child, self._next_node_id = self.brain.mutate(g, self.rng, self.cfg,
                                                      self._next_node_id, leap=leap)
        return child

    def _phenotype(self, genome):
        """Build a runnable brain from a genome (NCA needs the think-time, and carries
        the lifetime-learning plasticity params when enabled)."""
        c = self.cfg
        if c.brain == "nca":
            return self.brain.Phenotype(genome, subticks=c.nca_subticks,
                                        plastic=c.nca_plastic, lr=c.nca_plastic_lr,
                                        forget=c.nca_plastic_forget,
                                        elig_decay=c.nca_plastic_elig,
                                        clip=c.nca_plastic_clip,
                                        neuromod=c.nca_neuromod)
        return self.brain.Phenotype(genome)

    def _seed_genome(self):
        """A fresh starting genome: a mutated draw from the seed pool if one was
        supplied, otherwise a random minimal net."""
        if self.genome_pool:
            g = self.genome_pool[self.rng.integers(len(self.genome_pool))]
            return self._mutate(g)
        return self.brain.random_genome(self.n_in, self.n_out, self.rng,
                                        forward_bias=self.cfg.init_forward_bias,
                                        cfg=self.cfg)

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
                           & (self.food == 0))
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
                  self._phenotype(genome))
        self.agents[a.id] = a
        self.occ[y, x] = a.id
        self._next_id += 1
        return a

    def _free_food_cell(self, y: int, x: int) -> bool:
        return (0 < y < self.h - 1 and 0 < x < self.w - 1 and not self.walls[y, x]
                and self.corpse[y, x] == 0 and self.occ[y, x] < 0 and self.food[y, x] == 0)

    def _effective_regrow(self) -> float:
        """Food throughput this tick. With the adaptive controller on, it's generous
        while the population is at/under target (so learners have plenty) and throttled
        proportionally as the population climbs past it — homeostasis around the
        target. Off (food_pop_target <= 0) it's the fixed food_regrow."""
        c = self.cfg
        if c.food_pop_target <= 0:
            return self.food_rate                          # live-tunable fixed rate
        over = len(self.agents) - c.food_pop_target
        return int(round(min(c.food_regrow_boot,
                             max(0.0, c.food_regrow_boot - c.food_regrow_gain * over))))

    def _regrow_food(self) -> None:
        c = self.cfg
        open_cells = int((~self.walls).sum())
        target = int(self.food_cap * open_cells)           # live-tunable standing ceiling
        deficit = target - int((self.food > 0).sum())
        sigma = c.food_halo_sigma * min(self.h, self.w)
        # a FRACTIONAL rate (< 1 food/tick) is resolved probabilistically: e.g. 0.3 sows
        # one food on ~30% of ticks. Only draws rng when there's a fraction, so integer
        # rates stay bit-for-bit reproducible.
        rate = self._effective_regrow()
        count = int(rate)
        frac = rate - count
        if frac and self.rng.random() < frac:
            count += 1
        for _ in range(min(count, max(0, deficit))):
            k = int(self.rng.integers(c.n_food_types))     # a random food type...
            if self.rng.random() < c.food_scatter_prob:    # rare background dispersal:
                cell = self._random_free_cell()            # scatter anywhere on the map
                if cell is not None:
                    self.food[cell] = k + 1
                continue
            cy, cx = self.halos[k]                          # otherwise spawn near its halo
            for _try in range(6):                           # keep food NEAR its halo:
                y = int(round(cy + self.rng.normal(0.0, sigma)))   # retry LOCALLY instead
                x = int(round(cx + self.rng.normal(0.0, sigma)))   # of scattering food
                if self._free_food_cell(y, x):                     # anywhere on the map
                    self.food[y, x] = k + 1                  # type stored as 1..K
                    break
            # no free cell near the halo after a few tries -> skip (patch stays tight)

    def _diet_eff(self, diet: float, food_type: int) -> float:
        """How well a genome's diet digests a food type: 1.0 on its preferred type,
        down to `diet_min_eff` on the farthest — the dietary-niche trade-off."""
        k = self.cfg.n_food_types
        if k <= 1:
            return 1.0
        target = (food_type - 1) / (k - 1)
        return max(self.cfg.diet_min_eff, 1.0 - abs(diet - target))

    def _drift_halos(self) -> None:
        """Every `food_halo_drift_every` ticks each food halo centre takes a small
        random step, so the resource patches wander the map."""
        c = self.cfg
        if c.food_halo_drift_every <= 0 or self.tick % c.food_halo_drift_every != 0:
            return
        step = self.rng.uniform(-c.food_halo_drift, c.food_halo_drift, self.halos.shape)
        self.halos = np.clip(self.halos + step, 2, [self.h - 2, self.w - 2])

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
        # per-type digestion efficiency for this agent's diet, so food it CAN eat
        # looks brighter than food it can't (diet-aware vision)
        diet = a.brain.genome.diet
        effs = [self._diet_eff(diet, t) for t in range(1, c.n_food_types + 1)]
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
                ft = food[cy, cx]
                if ft:
                    vec[i * 3] = effs[ft - 1] * (1.0 - k / rl)   # food, dim if indigestible
                    break
        vec[-2] = a.energy / c.max_energy               # interoception
        vec[-1] = 1.0                                    # bias
        return vec

    # -- one tick -----------------------------------------------------------

    def step(self) -> None:
        c = self.cfg
        plastic = c.brain == "nca" and c.nca_plastic   # within-life learning enabled
        self._regrow_food()
        dead: list[Agent] = []
        bred: set[int] = set()                     # at most one child per agent/tick
        for aid in sorted(self.agents):            # fixed order -> reproducible
            a = self.agents.get(aid)
            if a is None:
                continue
            if c.brain == "nca":                   # NCA may explore (seeded) to not freeze
                action = a.brain.act(self._sense(a), rng=self.rng,
                                     temp=c.nca_explore_temp)
            else:
                action = a.brain.act(self._sense(a))
            a.energy -= c.base_cost
            if c.brain == "nca":                   # a bigger brain costs more to run,
                a.energy -= c.nca_cell_cost * int(a.brain.genome.mask.sum())  # so size
                                                   # is under selection, not free drift
            action = int(action)
            reward = 0.0                           # energy eaten this tick (the learn signal)
            if action == FORWARD:
                a.energy -= c.move_cost
                dy, dx = _HEAD_VEC[a.heading]
                ny, nx = a.y + dy, a.x + dx
                if self._free(ny, nx):
                    self.occ[a.y, a.x] = -1
                    a.y, a.x = ny, nx
                    self.occ[ny, nx] = a.id
                    ft = self.food[ny, nx]
                    if ft:                              # step onto food = eat (as a plant)
                        self.food[ny, nx] = 0
                        g = a.brain.genome
                        gain = c.food_energy * self._diet_eff(g.diet, ft) * (1 - g.carnivore)
                        a.energy = min(c.max_energy, a.energy + gain)
                        reward = gain
                elif c.predation and self.occ[ny, nx] >= 0:
                    prey = self.agents.get(int(self.occ[ny, nx]))
                    if self._can_predate(a, prey):
                        reward = self._predate(a, ny, nx)   # eat the agent ahead
            elif action == TURN_LEFT:
                a.energy -= c.turn_cost
                a.heading = (a.heading - 1) % 4
            elif action == TURN_RIGHT:
                a.energy -= c.turn_cost
                a.heading = (a.heading + 1) % 4
            # action == STAY: stand still, paying only base_cost (already deducted) —
            # the cheapest tick, so a fed agent can park and conserve energy
            if plastic:                            # reinforce the decisions that fed it
                a.brain.learn(reward / c.food_energy)
            if a.energy >= c.repro_min_energy and a.id not in bred:
                self._try_reproduce(a, bred)           # innate drive: breed when able
            a.age += 1
            if a.energy <= 0:
                dead.append(a)

        for a in dead:
            self._kill(a)
        self.corpse[self.corpse > 0] -= 1              # corpses decay away
        self._drift_walls()                            # the living landscape shifts
        self._drift_halos()                            # the food patches wander
        self._immigrate()                              # competent newcomers, new lineage
        cutoff = self.tick - self.cfg.lifespan_window   # forget old death records
        while self.deaths_log and self.deaths_log[0][0] < cutoff:
            self.deaths_log.popleft()
        self.tick += 1
        if not self.agents:
            self._reseed()

    def _drift_walls(self) -> None:
        """The landscape lives: every `wall_drift_every` ticks each stroke may WALK
        one cell along its own axis (leading with either end, like a snake), keeping
        its shape. A stroke that would leave the interior reverses instead. A wall that
        walks onto an agent or food CRUSHES it (the agent dies, the food is buried) —
        nothing survives under a wall. Topology stays roughly constant; positions
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
            end = (ny + (cand["length"] - 1 if cand["axis"] == "v" else 0),
                   nx + (cand["length"] - 1 if cand["axis"] == "h" else 0))
            in_bounds = 1 <= ny and 1 <= nx and 0 < end[0] < self.h - 1 \
                and 0 < end[1] < self.w - 1
            if in_bounds:
                s["y"], s["x"] = ny, nx
                moved = True
            else:
                s["dir"] *= -1                          # would leave the map: turn back
        if moved:
            self._rasterize_walls()
            self._clear_under_walls()                  # crush whatever the walls buried

    def _clear_under_walls(self) -> None:
        """Nothing occupies a wall: remove any food under a wall cell and kill (crush)
        any agent a wall has walked onto. Keeps the invariant that food never sits on a
        wall, and makes walking walls a real hazard."""
        self.food[self.walls] = 0
        for a in [ag for ag in self.agents.values() if self.walls[ag.y, ag.x]]:
            self.occ[a.y, a.x] = -1
            self.agents.pop(a.id, None)
            self.stats["deaths"] += 1
            self._remember(a)                          # crushed: no corpse, just gone

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
        self._lamarck(child_g, parents)                    # inherit some learned skill
        if self._spawn_agent(child_g, cell=cell, energy=c.repro_cost * 0.8) is not None:
            self.stats["births"] += 1
            for p in parents:
                p.offspring += 1

    def _lamarck(self, child_g, parents) -> None:
        """Soft Lamarck: fold a fraction of the parents' LIFETIME-LEARNED readout delta
        (Phenotype.dW) into the child's genome P_out, so the child is born partly knowing
        what its parents learned and skill compounds across generations. No-op unless the
        NCA brain is plastic and nca_lamarck > 0. dW and P_out are both (C, n_out), so the
        fold is always shape-safe."""
        lam = self.cfg.nca_lamarck
        if self.cfg.brain != "nca" or lam <= 0.0:
            return
        dWs = [p.brain.dW for p in parents if getattr(p.brain, "dW", None) is not None]
        if dWs:
            child_g.P_out = child_g.P_out + lam * np.mean(dWs, axis=0)

    def _can_predate(self, pred: Agent, prey: Agent | None) -> bool:
        """A committed carnivore may eat the agent ahead only if it's a DIFFERENT
        colour-species — carnivores hunt other species, never their own kin/mates. So
        predation is a real cross-species trophic niche (a carnivore lineage can persist)
        instead of self-cannibalising into extinction."""
        c = self.cfg
        return (prey is not None
                and pred.brain.genome.carnivore >= c.carnivore_threshold
                and _hue_dist(pred.brain.genome.hue, prey.brain.genome.hue) > c.species_hue)

    def _predate(self, pred: Agent, ny: int, nx: int) -> float:
        """A carnivore steps onto the agent ahead, killing and eating it: the predator
        gains `meat_energy` (scaled by how committed a carnivore it is), the prey dies
        WITHOUT a corpse (consumed), and the predator takes the now-empty cell. Returns
        the energy gained (the plasticity reward), 0.0 if there was no prey."""
        prey = self.agents.get(int(self.occ[ny, nx]))
        if prey is None:
            return 0.0
        g = pred.brain.genome
        self.agents.pop(prey.id, None)                     # prey removed (no corpse)
        self.occ[prey.y, prey.x] = -1
        self.stats["deaths"] += 1
        self._remember(prey)
        gain = min(self.cfg.max_energy, pred.energy + self.cfg.meat_energy * g.carnivore) \
            - pred.energy
        pred.energy += gain
        self.occ[pred.y, pred.x] = -1
        pred.y, pred.x = ny, nx
        self.occ[ny, nx] = pred.id
        return gain

    def _kill(self, a: Agent) -> None:
        self.occ[a.y, a.x] = -1
        self.corpse[a.y, a.x] = self.cfg.corpse_decay      # body blocks, then decays
        self.agents.pop(a.id, None)
        self.stats["deaths"] += 1
        self._remember(a)

    def _remember(self, a: Agent) -> None:
        """Hall-of-fame of the most SUCCESSFUL genomes = long-lived AND fecund
        (age + a weight per child), so the saved pool selects for genomes that
        both survive and reproduce, not just old survivors. Also logs this death's
        lifespan for the rolling mean/median-age stat."""
        success = a.age + self.cfg.hall_offspring_weight * a.offspring
        self.hall.append((success, a.brain.genome.copy()))
        self.hall.sort(key=lambda t: t[0], reverse=True)
        del self.hall[max(self.cfg.reseed_count, 12):]
        # rolling record of ages-at-death (tick, age); old entries pruned in snapshot
        self.deaths_log.append((self.tick, a.age))

    def lifespan_stats(self) -> tuple[float, float]:
        """(mean, median) age-at-death over deaths in the last `lifespan_window`
        ticks — a rolling readout of how long agents are surviving. (0, 0) if none."""
        w = self.cfg.lifespan_window
        cutoff = self.tick - w
        ages = [age for t, age in self.deaths_log if t >= cutoff]
        if not ages:
            return 0.0, 0.0
        return float(np.mean(ages)), float(np.median(ages))

    def top_genomes(self, k: int) -> list[neat.Genome]:
        """The k most successful genomes recorded (for island migration)."""
        return [g.copy() for _, g in self.hall[:k]]

    def _immigrate(self) -> None:
        """Drop a few COMPETENT newcomers into a living world, drawn from the hall
        (so they already forage — "smart like the brothers") but re-coloured as a
        FRESH lineage (a new род). This seeds species diversity a homogeneous world
        won't grow on its own, and tests whether an outside lineage can invade."""
        c = self.cfg
        if (c.immigrate_every <= 0 or self.tick == 0 or self.tick % c.immigrate_every
                or not self.agents):
            return
        donors = [g for _, g in self.hall[:6]] or \
            [a.brain.genome for a in self.agents.values()]
        for _ in range(c.immigrate_count):
            leap = self.rng.random() < c.nca_leap_prob  # some newcomers are hopeful monsters
            g = self._mutate(donors[int(self.rng.integers(len(donors)))].copy(), leap=leap)
            g.hue = float(self.rng.random())           # a fresh lineage colour (new род)
            g.diet = float(self.rng.random())          # a fresh diet — may eat the OTHER
                                                       # food type the incumbents ignore
            self._spawn_agent(g)

    def _strong_mutate(self, g):
        """A big-JUMP variant: several compounded mutation passes, so an extinction
        reseed can escape the basin the last population died in instead of replaying
        the same genome. Some variants also take a STRUCTURAL LEAP (a hopeful monster —
        a bigger/deeper brain in one shot) so a collapse can be answered with new
        capacity, not just re-rolled weights."""
        leap_pass = int(self.rng.integers(self.cfg.reseed_mutate_passes)) \
            if self.rng.random() < self.cfg.nca_leap_prob else -1
        for i in range(self.cfg.reseed_mutate_passes):
            g = self._mutate(g, leap=(i == leap_pass))
        return g

    def _reseed(self) -> None:
        """Extinction: repopulate from the best-so-far genomes AND strongly-mutated
        variants of them. Re-dropping only the elites replays the population that just
        died (no new variation -> evolution stalls); the mutants inject exploration on
        every collapse. Falls back to the seed pool / random when the hall is empty."""
        self.epoch += 1                                     # a new epoch begins
        self.epoch_start = self.tick
        elites = [g for _, g in self.hall[:self.cfg.reseed_elites]]
        if not elites:
            elites = [self._seed_genome()
                      for _ in range(min(self.cfg.reseed_elites, self.cfg.reseed_count))]
        for i in range(self.cfg.reseed_count):
            if i < len(elites):
                self._spawn_agent(elites[i].copy())         # keep the best ideas intact
            else:                                           # + big-jump mutants of them
                parent = elites[int(self.rng.integers(len(elites)))]
                self._spawn_agent(self._strong_mutate(parent))

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

    def species_groups(self) -> list[dict]:
        """Group the living population into colour-species (same hue clustering as
        species_count) and return per-species stats — count, mean hue/diet/age and
        carnivore share — largest first. Lets the HUD report EACH population, not just
        the aggregate."""
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
        thr = self.cfg.carnivore_threshold
        out = []
        for grp in groups:
            gs = [a.brain.genome for a in grp]
            out.append({"n": len(grp),
                        "hue": float(np.mean([g.hue for g in gs])),
                        "diet": float(np.mean([g.diet for g in gs])),
                        "carn": sum(1 for g in gs if g.carnivore >= thr) / len(gs),
                        "age": float(np.mean([a.age for a in grp]))})
        out.sort(key=lambda s: s["n"], reverse=True)
        return out

    def population_genomes(self, per: int = 3) -> list:
        """Top-`per` genomes (oldest first) from EACH living colour-species — a DIVERSITY
        snapshot of the CURRENT populations (not the all-time hall champions), so a save
        keeps the range of coexisting lineages, not just the single most successful line."""
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
        out = []
        for grp in groups:
            for a in sorted(grp, key=lambda a: a.age, reverse=True)[:per]:
                out.append(a.brain.genome.copy())
        return out

    def carnivore_fraction(self) -> float:
        """Share of the living population that are committed carnivores."""
        n = len(self.agents)
        if not n:
            return 0.0
        thr = self.cfg.carnivore_threshold
        return sum(1 for a in self.agents.values()
                   if a.brain.genome.carnivore >= thr) / n

    def snapshot(self) -> dict:
        life_mean, life_median = self.lifespan_stats()
        energies = [a.energy for a in self.agents.values()]
        # world energy budget: energy banked in living bodies + the potential energy
        # sitting in uneaten food. Food is the ONLY inflow; metabolism + death are the
        # outflows — so a rising total means inflow > burn (the balance is positive).
        e_bodies = float(sum(energies))
        e_food = float((self.food > 0).sum()) * self.cfg.food_energy
        return {"tick": self.tick, "population": self.population(),
                "energy_bodies": e_bodies, "energy_food": e_food,
                "energy_mean": float(np.mean(energies)) if energies else 0.0,
                "food": int((self.food > 0).sum()), "corpses": int((self.corpse > 0).sum()),
                "births": self.stats["births"], "deaths": self.stats["deaths"],
                "species": self.species_count(), "carnivores": self.carnivore_fraction(),
                "life_mean": life_mean, "life_median": life_median,
                "epoch": self.epoch, "epoch_age": self.tick - self.epoch_start,
                "food_rate": self.food_rate,
                "food_cap": int(self.food_cap * int((~self.walls).sum()))}
