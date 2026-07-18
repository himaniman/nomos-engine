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

from vivarium import nca, neat
from vivarium.config import Config
from vivarium.world import input_size, _hue_dist

_BRAINS = {"neat": neat, "nca": nca}
_TWO_PI = 2.0 * math.pi


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
    __slots__ = ("id", "y", "x", "heading", "energy", "age", "offspring", "brain")

    def __init__(self, aid, y, x, heading, energy, brain):
        self.id = aid
        self.y = float(y)
        self.x = float(x)
        self.heading = float(heading)      # radians; 0 = +x, pi/2 = +y (down)
        self.energy = energy
        self.age = 0
        self.offspring = 0
        self.brain = brain


class World:
    """A continuous bounded world of evolving brains. Grid-backed food/walls, but
    float agents that move and see smoothly."""

    def __init__(self, config: Config | None = None, genome_pool=None,
                 max_speed: float = 1.0, max_turn: float = 0.45,
                 radius: float = 0.35, mate_radius: float = 1.8,
                 ray_step: float = 0.6):
        self.cfg = config or Config()
        c = self.cfg
        self.max_speed = max_speed          # cells per tick at full forward thrust
        self.max_turn = max_turn            # radians per tick at full turn
        self.radius = radius                # agent body radius (collision/eat)
        self.mate_radius = mate_radius      # how close two agents breed
        self.ray_step = ray_step            # vision ray-march increment (cells)

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
        self.n_in = input_size(c.n_rays)
        self.n_out = self.brain.N_ACTIONS
        self._next_node_id = self.n_in + self.n_out
        if self.genome_pool:
            self._next_node_id = max([self._next_node_id]
                                     + [g.max_node_id() + 1 for g in self.genome_pool])

        self.strokes = self._generate_strokes()
        self.walls = np.zeros((self.h, self.w), dtype=bool)
        self._rasterize_walls()
        self.food = np.zeros((self.h, self.w), dtype=np.int8)
        self.halos = np.array([[self.rng.integers(2, self.h - 2),
                                self.rng.integers(2, self.w - 2)]
                               for _ in range(c.n_food_types)], dtype=np.float64)
        self.corpse = np.zeros((self.h, self.w), dtype=np.int64)

        self.agents: dict[int, Agent] = {}
        self.hall: list[tuple[int, object]] = []
        self.stats = {"births": 0, "deaths": 0}
        self.deaths_log: deque = deque()
        # per-heading ray fan is now continuous; precompute the relative angles only
        self._ray_angles = np.linspace(-c.fov / 2, c.fov / 2, c.n_rays)

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

    def _cell_blocked(self, y: float, x: float) -> bool:
        """Is the grid cell under a float point a wall or a corpse?"""
        cy, cx = int(round(y)), int(round(x))
        if not (0 <= cy < self.h and 0 <= cx < self.w):
            return True
        return bool(self.walls[cy, cx]) or self.corpse[cy, cx] > 0

    # -- food (grid, copied from vivarium) ----------------------------------

    def _free_food_cell(self, y: int, x: int) -> bool:
        return (0 < y < self.h - 1 and 0 < x < self.w - 1 and not self.walls[y, x]
                and self.corpse[y, x] == 0 and self.food[y, x] == 0
                and self._agent_near(y, x, self.radius) is None)

    def _random_free_cell(self):
        free = np.argwhere((~self.walls) & (self.corpse == 0) & (self.food == 0))
        if len(free) == 0:
            return None
        return tuple(free[self.rng.integers(len(free))])

    def _effective_regrow(self) -> float:
        c = self.cfg
        if c.food_pop_target <= 0:
            return self.food_rate
        over = len(self.agents) - c.food_pop_target
        return int(round(min(c.food_regrow_boot,
                             max(0.0, c.food_regrow_boot - c.food_regrow_gain * over))))

    def _regrow_food(self) -> None:
        c = self.cfg
        open_cells = int((~self.walls).sum())
        target = int(self.food_cap * open_cells)
        deficit = target - int((self.food > 0).sum())
        sigma = c.food_halo_sigma * min(self.h, self.w)
        rate = self._effective_regrow()
        count = int(rate)
        frac = rate - count
        if frac and self.rng.random() < frac:
            count += 1
        for _ in range(min(count, max(0, deficit))):
            k = int(self.rng.integers(c.n_food_types))
            if self.rng.random() < c.food_scatter_prob:
                cell = self._random_free_cell()
                if cell is not None:
                    self.food[cell] = k + 1
                continue
            cy, cx = self.halos[k]
            for _try in range(6):
                y = int(round(cy + self.rng.normal(0.0, sigma)))
                x = int(round(cx + self.rng.normal(0.0, sigma)))
                if self._free_food_cell(y, x):
                    self.food[y, x] = k + 1
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

    def _agent_near(self, y: float, x: float, r: float, exclude: int = -1):
        """Any agent whose centre is within `r` of (y, x) (O(n); populations are
        small). Returns the closest such agent or None."""
        best, best_d = None, r
        for a in self.agents.values():
            if a.id == exclude:
                continue
            d = math.hypot(a.y - y, a.x - x)
            if d < best_d:
                best, best_d = a, d
        return best

    def _random_free_pos(self):
        for _ in range(40):
            y = float(self.rng.uniform(1.5, self.h - 1.5))
            x = float(self.rng.uniform(1.5, self.w - 1.5))
            if not self._cell_blocked(y, x) and self._agent_near(y, x, 2 * self.radius) is None:
                return y, x
        return None

    def _free_pos_near(self, y: float, x: float):
        """A free spot ~one body away from (y, x) to drop a child."""
        d = 2.2 * self.radius
        for k in range(8):
            ang = k / 8.0 * _TWO_PI
            ny, nx = y + math.sin(ang) * d, x + math.cos(ang) * d
            if (1.0 <= ny < self.h - 1.0 and 1.0 <= nx < self.w - 1.0
                    and not self._cell_blocked(ny, nx)
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
        self._next_id += 1
        return a

    # -- vision (continuous ray-march) --------------------------------------

    def _sense(self, a: Agent) -> np.ndarray:
        """Ray vision fanned around the heading + interoception. Each ray reports
        the first thing it hits — food / obstacle / agent — and how near, by
        marching in float steps and sampling the grid (walls, food) and nearby
        agents. Diet dims indigestible food, exactly as in the grid world."""
        c = self.cfg
        vec = np.zeros(self.n_in)
        rl = float(c.ray_length)
        diet = a.brain.genome.diet
        effs = [self._diet_eff(diet, t) for t in range(1, c.n_food_types + 1)]
        # candidate agents once (within ray reach), for the per-ray agent test
        near = [(b, math.hypot(b.y - a.y, b.x - a.x),
                 math.atan2(b.y - a.y, b.x - a.x))
                for b in self.agents.values()
                if b.id != a.id and math.hypot(b.y - a.y, b.x - a.x) <= rl + self.radius]
        half_ray = 0.5 * c.fov / max(1, c.n_rays - 1) + self.radius / max(rl, 1.0)
        for i, rel in enumerate(self._ray_angles):
            ang = a.heading + rel
            dx, dy = math.cos(ang), math.sin(ang)
            hit_d, hit_ch, hit_val = rl, -1, 0.0
            k = self.ray_step
            while k <= rl:                                   # march grid for wall/food
                py, px = a.y + dy * k, a.x + dx * k
                cy, cx = int(round(py)), int(round(px))
                if not (0 <= cy < self.h and 0 <= cx < self.w) or self.walls[cy, cx] \
                        or self.corpse[cy, cx] > 0:
                    hit_d, hit_ch, hit_val = k, 1, 1.0 - k / rl      # obstacle
                    break
                ft = self.food[cy, cx]
                if ft:
                    hit_d, hit_ch, hit_val = k, 0, effs[ft - 1] * (1.0 - k / rl)
                    break
                k += self.ray_step
            for b, d, bearing in near:                       # an agent nearer than that?
                if d < hit_d and abs(_wrap_angle(bearing - ang)) <= half_ray:
                    hit_d, hit_ch, hit_val = d, 2, 1.0 - d / rl
            if hit_ch >= 0:
                vec[i * 3 + hit_ch] = hit_val
        vec[-2] = a.energy / c.max_energy
        vec[-1] = 1.0
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
            if plastic:
                a.brain.learn(reward / c.food_energy)
            if a.energy >= c.repro_min_energy and a.id not in bred:
                self._try_reproduce(a, bred)
            a.age += 1
            if a.energy <= 0:
                dead.append(a)

        for a in dead:
            self._kill(a)
        self.corpse[self.corpse > 0] -= 1
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
        """Try to move agent `a` to (ny, nx). Blocked by the map border, walls,
        corpses, or another agent (opacity) — a blocked move just doesn't happen
        (heading already turned). Stepping onto food eats it; a carnivore stepping
        into another agent may eat it. Returns the energy gained (the learn signal)."""
        c = self.cfg
        if not (self.radius <= ny <= self.h - 1 - self.radius
                and self.radius <= nx <= self.w - 1 - self.radius):
            return 0.0
        if self._cell_blocked(ny, nx):
            return 0.0
        other = self._agent_near(ny, nx, 2 * self.radius, exclude=a.id)
        if other is not None:
            if c.predation and self._can_predate(a, other):
                return self._predate(a, other, ny, nx)
            return 0.0                                        # opacity: blocked
        a.y, a.x = ny, nx
        cy, cx = int(round(ny)), int(round(nx))
        ft = self.food[cy, cx]
        if ft:
            self.food[cy, cx] = 0
            g = a.brain.genome
            gain = c.food_energy * self._diet_eff(g.diet, ft) * (1 - g.carnivore)
            a.energy = min(c.max_energy, a.energy + gain)
            return gain
        return 0.0

    # -- reproduction -------------------------------------------------------

    def _try_reproduce(self, a: Agent, bred: set) -> None:
        """Sexual reproduction by PROXIMITY: a fed, same-species agent within
        `mate_radius` mates (crossover of the two, fitter leads); a very-full lone
        agent clones asexually. The child drops at a free spot beside the parent."""
        c = self.cfg
        mate = None
        for b in self.agents.values():
            if b.id == a.id or b.id in bred or b.energy < c.repro_min_energy:
                continue
            if math.hypot(b.y - a.y, b.x - a.x) > self.mate_radius:
                continue
            if _hue_dist(a.brain.genome.hue, b.brain.genome.hue) <= c.species_hue:
                mate = b
                break
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
        self.stats["deaths"] += 1
        self._remember(prey)
        gain = min(self.cfg.max_energy, pred.energy + self.cfg.meat_energy * g.carnivore) \
            - pred.energy
        pred.energy += gain
        pred.y, pred.x = ny, nx
        return gain

    # -- death / hall -------------------------------------------------------

    def _kill(self, a: Agent) -> None:
        self.corpse[int(round(a.y)), int(round(a.x))] = self.cfg.corpse_decay
        self.agents.pop(a.id, None)
        self.stats["deaths"] += 1
        self._remember(a)

    def _remember(self, a: Agent) -> None:
        success = a.age + self.cfg.hall_offspring_weight * a.offspring
        self.hall.append((success, a.brain.genome.copy()))
        self.hall.sort(key=lambda t: t[0], reverse=True)
        del self.hall[max(self.cfg.reseed_count, 12):]
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
            self._rasterize_walls()
            self._clear_under_walls()

    def _clear_under_walls(self) -> None:
        self.food[self.walls] = 0
        for a in [ag for ag in self.agents.values()
                  if self.walls[int(round(ag.y)), int(round(ag.x))]]:
            self.agents.pop(a.id, None)
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
                        "age": float(np.mean([a.age for a in grp]))})
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
                "energy_food": float((self.food > 0).sum()) * self.cfg.food_energy,
                "energy_mean": float(np.mean(energies)) if energies else 0.0,
                "food": int((self.food > 0).sum()), "corpses": int((self.corpse > 0).sum()),
                "births": self.stats["births"], "deaths": self.stats["deaths"],
                "species": self.species_count(), "carnivores": self.carnivore_fraction(),
                "life_mean": life_mean, "life_median": life_median,
                "epoch": self.epoch, "epoch_age": self.tick - self.epoch_start,
                "food_rate": self.food_rate,
                "food_cap": int(self.food_cap * int((~self.walls).sum()))}
