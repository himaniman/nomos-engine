"""The cortex world — a float agent on a tile-stored map, with food and one verb.

Stage 1-2 scope: geometry, movement, energy, FOOD as float entities, and a single
contextual INTERACT. No senses yet (that is the next chunk) — but the ACTION
INTERFACE is already the final one, because the keyboard and the brain must drive
the agent through exactly the same call or we are playtesting a different game than
the one the agent lives in.

Movement rules carried over as LESSONS (not code) from aqua:
  * the body is a CIRCLE and collision is tested against that circle, so what is
    drawn is what collides;
  * a move is MARCHED in sub-steps smaller than the body, so nothing tunnels a wall
    at any speed;
  * a blocked hop SLIDES along the free axis instead of dead-stopping, so the agent
    never wedges into a corner.

EATING IS A DECISION, not a collision: food is consumed by INTERACT, never by
walking over it. That is what makes poison a perception problem later — the agent
can walk across a poison berry harmlessly and simply decline to eat it, instead of
having to learn "never step there".

ONE VERB: `interact` does whatever the context affords — eat what is in reach, else
open/close the door in front, else chip the breakable tile in front, else shove the
pushable block in front. The brain only ever decides WHETHER to act, never which of
five verbs, which keeps the output space tiny and the credit assignment simple.

Determinism: a run is a pure function of (seed, size, the action sequence).
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import numpy as np

from cortex.tiles import (BREAKABLE, DOOR, DOOR_OPEN, FOLIAGE, GROUND, HP, PUSHABLE,
                          SOLID, TRUNK, TileMap, _mix)

# --- food ------------------------------------------------------------------------
# A food kind is a colour, an ODOUR MIXTURE and an energy value — never a "food" flag.
# The senses (next chunk) will report the colour and the odour; EDIBILITY is a
# conclusion the brain has to reach, which is the whole reason poison can be learned.
# Poison is defined here but not planted until its own stage (weight 0).
#              name       energy  rgb              odour (4 channels)
_FOODS = (("berry",     12.0,  (214, 66, 92),   (1.0, 0.15, 0.0, 0.0)),
          ("fruit",     22.0,  (235, 158, 52),  (0.2, 1.0, 0.0, 0.0)),
          ("mushroom",  32.0,  (196, 208, 236), (0.0, 0.1, 1.0, 0.0)),
          ("nightcap", -30.0,  (150, 60, 168),  (1.0, 0.15, 0.0, 0.0)))

BERRY, FRUIT, MUSHROOM, NIGHTCAP = range(4)
N_ODOUR = 4
FOOD_NAMES = tuple(f[0] for f in _FOODS)
FOOD_ENERGY = np.array([f[1] for f in _FOODS], dtype=np.float64)
FOOD_RGB = np.array([f[2] for f in _FOODS], dtype=np.uint8)
FOOD_HSV = np.array([colorsys.rgb_to_hsv(*(np.array(f[2]) / 255.0)) for f in _FOODS])
FOOD_ODOUR = np.array([f[3] for f in _FOODS], dtype=np.float64)


@dataclass
class FoodSpot:
    """A fixed PLACE that bears food. Renewable spots regrow on a timer — which is
    the only reason remembering a location can ever pay off (see the economy section
    of docs/cortex-world.md). One-shot spots reward exploration instead."""
    y: float
    x: float
    kind: int
    renewable: bool = True
    ready: bool = True
    regrow_at: int = 0


@dataclass
class Config:
    """World + body constants. The economy numbers are placeholders until they are
    tuned against a human baseline — see docs/cortex-world.md."""
    size: int | None = 512          # None = unbounded (the generator allows it)
    chunk: int = 32
    radius: float = 0.45            # body radius: about a wall's width, still under
                                    # 0.5 so it can thread a one-cell doorway
    substep: float = 0.4            # < 2*radius, so consecutive body circles overlap
    max_speed: float = 0.35         # cells per tick at full thrust
    max_turn: float = 0.12          # radians per tick at full turn
    start_energy: float = 100.0
    max_energy: float = 150.0       # a cap, so energy cannot be hoarded without limit
    move_cost: float = 0.02         # energy per tick at full speed
    stand_cost: float = 0.01        # ...and standing still: cheaper, never free, so
                                    # thinking beats wandering but waiting still costs
    max_age: int = 200_000          # a cap, NOT the cause of death — energy is life
    # --- interaction
    interact_cost: float = 0.05     # energy per attempt, whether or not it achieves
    interact_every: int = 6         # ticks between attempts (holding E chips, not melts)
    eat_reach: float = 1.1          # how far from the body centre food can be taken
    # --- food economy
    food_per_chunk: int = 9
    renewable_frac: float = 0.7     # the rest are one-shot: exploration vs memory
    regrow_ticks: int = 1200
    poison_weight: float = 0.0      # nightcaps are not planted until their own stage


@dataclass
class Agent:
    y: float
    x: float
    heading: float = 0.0            # radians; 0 = +x (east)
    speed: float = 0.0
    energy: float = 0.0
    age: int = 0
    alive: bool = True
    ate: list = field(default_factory=lambda: [0, 0, 0, 0])


class World:
    """The map, the food and the one agent. `step(thrust, turn, interact)` = a tick."""

    def __init__(self, seed: int = 0, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.seed = int(seed)
        self.tiles = TileMap(seed=seed, size=self.cfg.size, chunk=self.cfg.chunk)
        self.tick = 0
        self.food: dict[tuple[int, int], list[FoodSpot]] = {}
        self.damage: dict[tuple[int, int], int] = {}     # cell -> hits taken so far
        self.last_event = ""                             # HUD / debug feedback
        self._pending: list[FoodSpot] = []               # eaten spots awaiting regrowth
        self._next_interact = 0
        y, x = self._spawn_spot()
        self.agent = Agent(y=y, x=x, energy=self.cfg.start_energy)

    # --- spawn -------------------------------------------------------------------
    def _spawn_spot(self) -> tuple[float, float]:
        """The first free cell found spiralling out from the world's centre — a pure
        function of the map, so a seed always spawns in the same place."""
        c = self.cfg
        cy = cx = (c.size // 2) if c.size is not None else 0
        for r in range(0, 400):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dy), abs(dx)) != r:          # only the ring at radius r
                        continue
                    y, x = cy + dy, cx + dx
                    if not self.tiles.blocked_circle(float(y), float(x), c.radius):
                        return float(y), float(x)
        raise RuntimeError("no free spawn cell found near the world centre")

    # --- food ---------------------------------------------------------------------
    def _ensure_food(self, cy: int, cx: int) -> list[FoodSpot]:
        """This chunk's food spots, planted on first touch from a chunk-seeded rng —
        so food is deterministic, infinite-capable, and generated lazily like terrain.
        The KIND follows the surroundings: berries near foliage, fruit by trunks,
        mushrooms in enclosed (cave) cells."""
        spots = self.food.get((cy, cx))
        if spots is not None:
            return spots
        c, ch = self.cfg, self.cfg.chunk
        rng = np.random.default_rng(_mix(self.seed ^ 0x5EED, cy, cx))
        spots = []
        for _ in range(c.food_per_chunk):
            y = cy * ch + int(rng.integers(ch))
            x = cx * ch + int(rng.integers(ch))
            # jitter FIRST, then validate the position we actually plant at: a cell
            # whose centre is clear can still have its off-grid offset land a morsel
            # inside a wall, and unreachable food is worse than no food
            fy = float(y) + float(rng.uniform(-0.35, 0.35))
            fx = float(x) + float(rng.uniform(-0.35, 0.35))
            if self.tiles.blocked_circle(fy, fx, c.radius):
                continue
            spots.append(FoodSpot(y=fy, x=fx, kind=self._kind_for(y, x, rng),
                                  renewable=bool(rng.random() < c.renewable_frac)))
        self.food[(cy, cx)] = spots
        return spots

    def _kind_for(self, y: int, x: int, rng) -> int:
        """What grows here — read off the surroundings, so food sits where it looks
        like it should and the world's colours correlate with its rewards."""
        if self.cfg.poison_weight > 0 and rng.random() < self.cfg.poison_weight:
            return NIGHTCAP
        near = self.tiles.region(y - 2, x - 2, 5, 5)
        if (near == TRUNK).any() and rng.random() < 0.7:
            return FRUIT
        if (near == FOLIAGE).any() and rng.random() < 0.8:
            return BERRY
        if float(SOLID[near].mean()) > 0.45:                 # walled in: a cave nook
            return MUSHROOM
        return BERRY if rng.random() < 0.7 else FRUIT

    def nearby_food(self, y: float, x: float, r: float) -> list[FoodSpot]:
        """Every READY spot within `r` of (y, x), by true distance. Scans only the
        chunks the radius touches, so this stays local in an unbounded world."""
        ch = self.cfg.chunk
        out = []
        for cy in range(int(math.floor((y - r) / ch)), int(math.floor((y + r) / ch)) + 1):
            for cx in range(int(math.floor((x - r) / ch)),
                            int(math.floor((x + r) / ch)) + 1):
                if self.cfg.size is not None and not (
                        0 <= cy * ch < self.cfg.size and 0 <= cx * ch < self.cfg.size):
                    continue
                for s in self._ensure_food(cy, cx):
                    if s.ready and math.hypot(s.y - y, s.x - x) <= r:
                        out.append(s)
        return out

    def _regrow(self) -> None:
        """Renewable spots come back on their timer. Only spots actually WAITING are
        looked at (a pending list), so the per-tick cost is the number of eaten spots,
        not the number of spots the agent has ever walked past."""
        if not self._pending:
            return
        still = []
        for s in self._pending:
            if self.tick >= s.regrow_at:
                s.ready = True
            else:
                still.append(s)
        self._pending = still

    # --- the tick ----------------------------------------------------------------
    def step(self, thrust: float = 0.0, turn: float = 0.0, interact: float = 0.0) -> None:
        """One world tick. `thrust` in [0, 1], `turn` in [-1, 1], `interact` in [0, 1]
        (> 0.5 = pressed). This is the one entry point: the keyboard and the brain
        both come through here."""
        a = self.agent
        if not a.alive:
            return
        c = self.cfg
        a.heading = (a.heading + max(-1.0, min(1.0, turn)) * c.max_turn) % (2 * math.pi)
        a.speed = max(0.0, min(1.0, thrust)) * c.max_speed
        if a.speed > 0.0:
            self._advance(a.y + math.sin(a.heading) * a.speed,
                          a.x + math.cos(a.heading) * a.speed)
        if interact > 0.5 and self.tick >= self._next_interact:
            self._next_interact = self.tick + c.interact_every
            a.energy -= c.interact_cost
            self._interact()
        frac = a.speed / c.max_speed
        a.energy -= c.stand_cost + (c.move_cost - c.stand_cost) * frac
        a.energy = min(a.energy, c.max_energy)
        a.age += 1
        self.tick += 1
        self._regrow()
        if a.energy <= 0.0 or a.age >= c.max_age:
            a.alive = False

    def _advance(self, ny: float, nx: float) -> None:
        """March the body from where it is toward (ny, nx) in sub-steps, colliding as
        a circle the whole way. A blocked hop retries the two axes separately, which
        is what makes the agent slide along a wall instead of sticking to it."""
        a, c = self.agent, self.cfg
        dy, dx = ny - a.y, nx - a.x
        dist = math.hypot(dy, dx)
        if dist < 1e-12:
            return
        n = max(1, int(math.ceil(dist / c.substep)))
        sy, sx = dy / n, dx / n
        blocked = self.tiles.blocked_circle
        for _ in range(n):
            ty, tx = a.y + sy, a.x + sx
            if not blocked(ty, tx, c.radius):
                a.y, a.x = ty, tx
            elif abs(sx) > 1e-12 and not blocked(a.y, tx, c.radius):
                a.x = tx                                     # slide along x
            elif abs(sy) > 1e-12 and not blocked(ty, a.x, c.radius):
                a.y = ty                                     # slide along y
            else:
                break                                        # wedged in a corner

    # --- the one verb --------------------------------------------------------------
    def _interact(self) -> str:
        """Do whatever the context affords, in priority order: EAT what is in reach,
        else act on the tile in front — open/close a door, chip a breakable, shove a
        pushable. Sets and returns `last_event` (the HUD's feedback line)."""
        a = self.agent
        food = self.nearby_food(a.y, a.x, self.cfg.eat_reach)
        if food:
            s = min(food, key=lambda f: math.hypot(f.y - a.y, f.x - a.x))
            return self._eat(s)
        fy, fx = self.facing_cell()
        mat = self.tiles.material(fy, fx)
        if mat == DOOR:
            self.tiles.set(fy, fx, DOOR_OPEN)
            return self._event("opened the door")
        if mat == DOOR_OPEN:
            self.tiles.set(fy, fx, DOOR)                     # ...but never onto our own
            if self.tiles.blocked_circle(a.y, a.x, self.cfg.radius):   # body: a door in
                self.tiles.set(fy, fx, DOOR_OPEN)            # the doorway we are standing
                return self._event("standing in the doorway")   # in would trap us inside
            return self._event("closed the door")
        if BREAKABLE[mat]:
            return self._break(fy, fx, mat)
        if mat == PUSHABLE:
            return self._push(fy, fx)
        return self._event("nothing to do here")

    def _eat(self, s: FoodSpot) -> str:
        a = self.agent
        s.ready = False
        if s.renewable:
            s.regrow_at = self.tick + self.cfg.regrow_ticks
            self._pending.append(s)
        gain = float(FOOD_ENERGY[s.kind])
        a.energy = min(a.energy + gain, self.cfg.max_energy)
        a.ate[s.kind] += 1
        return self._event(f"ate {FOOD_NAMES[s.kind]} {gain:+.0f}")

    def _break(self, y: int, x: int, mat: int) -> str:
        """Chipping: a breakable tile takes several hits, so opening a shortcut is a
        real cost paid in energy and time, not a free click."""
        hits = self.damage.get((y, x), 0) + 1
        need = int(HP[mat])
        if hits >= need:
            self.damage.pop((y, x), None)
            self.tiles.set(y, x, GROUND)
            return self._event(f"broke the {self._name(mat)}")
        self.damage[(y, x)] = hits
        return self._event(f"chipping {self._name(mat)} {hits}/{need}")

    def _push(self, y: int, x: int) -> str:
        a = self.agent
        dy = int(round(math.sin(a.heading)))
        dx = int(round(math.cos(a.heading)))
        ty, tx = y + dy, x + dx
        if (dy or dx) and not SOLID[self.tiles.material(ty, tx)]:
            self.tiles.set(y, x, GROUND)
            self.tiles.set(ty, tx, PUSHABLE)
            return self._event("pushed the block")
        return self._event("the block will not budge")

    def _event(self, text: str) -> str:
        self.last_event = text
        return text

    @staticmethod
    def _name(mat: int) -> str:
        from cortex.tiles import NAMES
        return NAMES[mat]

    # --- helpers -------------------------------------------------------------------
    def facing_cell(self) -> tuple[int, int]:
        """The cell the agent is looking at, one body-length ahead — the target an
        INTERACT acts on when there is no food in reach."""
        a = self.agent
        d = self.cfg.radius + 0.6
        return (int(round(a.y + math.sin(a.heading) * d)),
                int(round(a.x + math.cos(a.heading) * d)))

    def afforded(self) -> str:
        """What pressing INTERACT would do right now — the HUD's prompt, and a handy
        one-line summary of the context rule above."""
        a = self.agent
        food = self.nearby_food(a.y, a.x, self.cfg.eat_reach)
        if food:
            s = min(food, key=lambda f: math.hypot(f.y - a.y, f.x - a.x))
            return f"eat {FOOD_NAMES[s.kind]}"
        fy, fx = self.facing_cell()
        mat = self.tiles.material(fy, fx)
        if mat == DOOR:
            return "open door"
        if mat == DOOR_OPEN:
            return "close door"
        if BREAKABLE[mat]:
            hits = self.damage.get((fy, fx), 0)
            return f"break {self._name(mat)} ({hits}/{int(HP[mat])})"
        if mat == PUSHABLE:
            return "push block"
        return ""
