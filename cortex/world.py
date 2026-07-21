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
from collections import deque
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
    assist: float = 0.06            # CORRIDOR CENTRING: when a hop is fully blocked,
                                    # drift this far toward the centre line of the cell
                                    # we occupy, then retry. At radius 0.45 a one-cell
                                    # gap leaves +-0.05 of clearance — passable, as the
                                    # player found, but only with sub-cell precision.
                                    # Measured: without this, threading works from an
                                    # offset of 0.00 and fails from 0.10 onward; with
                                    # it, anywhere in the right ROW works. That is the
                                    # difference between a body that fits on paper and
                                    # one a player (or a brain) can actually drive.
    substep: float = 0.4            # < 2*radius, so consecutive body circles overlap
    max_speed: float = 0.35         # cells per tick at full thrust
    max_turn: float = 0.12          # radians per tick at full turn
    start_energy: float = 100.0
    max_energy: float = 150.0       # a cap, so energy cannot be hoarded without limit
    move_cost: float = 0.02         # energy per tick at full speed
    stand_cost: float = 0.01        # ...and standing still: cheaper, never free, so
                                    # thinking beats wandering but waiting still costs
    max_age: int = 200_000          # a cap, NOT the cause of death — energy is life
    spawn_room: int = 600           # a spawn must reach at least this many cells (via
                                    # doors and breakables) or it counts as a coffin
    # --- senses (see cortex/senses.py)
    n_rays: int = 64                # a single agent can afford a real retina: at 64 over
                                    # a 120 deg fov a ray is 1.9 deg, so a 2x2 tree ten
                                    # cells away spans ~6 rays and its SHAPE is legible.
                                    # At aqua's 3-12 rays it would be one ray and a guess
    fov: float = math.radians(120)
    ray_reach: float = 22.0         # how far sight carries (cells)
    food_radius: float = 0.25       # a morsel's visual size
    smell_radius: int = 24          # how far the geodesic scent BFS is computed
    smell_sharp: float = 5.0        # half strength at this many cells of WALKED path
    # --- interaction
    interact_cost: float = 0.05     # energy per attempt, whether or not it achieves
    interact_every: int = 6         # ticks between attempts (holding E chips, not melts)
    eat_reach: float = 1.1          # how far from the body centre food can be taken
    # --- food economy
    food_per_chunk: int = 1         # scattered spots (cut hard after the first
                                    # playtest: the world was too generous to starve in)
    larder_doors: int = 3           # rooms per chunk that get stocked...
    larder_max: int = 2             # ...with up to this many berries each
    tree_spots: int = 2             # fruit planted right at a trunk (its only habitat)
    cave_spots: int = 4             # extra attempts that only take in walled-in cells,
                                    # so mushrooms are actually worth going caving for
    renewable_frac: float = 0.7     # the rest are one-shot: exploration vs memory
    regrow_ticks: int = 1200
    # POISON. A nightcap only ever replaces a BERRY — same habitat, identical odour,
    # different hue — so the only way to tell them apart is to LOOK, and the only way
    # to know which hue to avoid is to have eaten one. That is the experiment.
    poison_weight: float = 0.12     # of berries (0 = a world with no poison in it)
    poison_rate: float = 0.15       # energy drained per tick while poisoned: survivable
                                    # on purpose. A morsel that kills outright teaches
                                    # nothing, because the dead do not learn.


@dataclass
class Agent:
    y: float
    x: float
    heading: float = 0.0            # radians; 0 = +x (east)
    speed: float = 0.0
    last_turn: float = 0.0          # proprioception: what we just commanded
    energy: float = 0.0
    poison: float = 0.0             # energy still to be drained by something eaten
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

    # --- enclosure ------------------------------------------------------------------
    def enclosure(self, y: int, x: int, limit: int = 400,
                  openable: bool = False) -> tuple[list, bool, bool]:
        """Flood-fill the region containing cell (y, x), capped at `limit`. Returns
        (cells, has_door, bounded). Deterministic (fixed neighbour order).

        `openable=False` walks only what is walkable RIGHT NOW — that is a ROOM test
        (bounded + has_door = a room worth stocking with a larder).

        `openable=True` also walks what the agent can OPEN or BREAK — doors and
        breakables. That is the CAN I EVER GET OUT test, and it is the one the spawn
        uses. The difference is not academic: the player was shut in a room that had a
        perfectly good door with solid rock behind it (two rock masses overlapped), and
        a has_door check called that an exit. Only walking THROUGH the door finds out."""
        start = (int(y), int(x))
        if SOLID[self.tiles.material(*start)]:
            return [], False, True
        seen = {start}
        queue = deque([start])
        cells: list[tuple[int, int]] = []
        has_door = False
        while queue:
            qy, qx = queue.popleft()
            cells.append((qy, qx))
            if len(cells) > limit:
                return cells, has_door, False                # too big to be a pocket
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (qy + dy, qx + dx)
                if n in seen:
                    continue
                seen.add(n)
                m = self.tiles.material(*n)
                if m in (DOOR, DOOR_OPEN):
                    has_door = True                          # a door — not yet an exit
                if not SOLID[m] or (openable and (m == DOOR or BREAKABLE[m])):
                    queue.append(n)
        return cells, has_door, True

    # --- spawn -------------------------------------------------------------------
    def _spawn_spot(self) -> tuple[float, float]:
        """The first free cell spiralling out from the world's centre from which the
        agent can ACTUALLY REACH the wider world — opening doors and breaking rubble
        counts, a door with rock behind it does not. A pure function of the map, so a
        seed always spawns in the same place.

        This test earned two revisions. First it did not exist and seed 1 started the
        player inside a hollow rock. Then it accepted "the pocket has a door", and seed
        1 started him in a room whose door opened onto solid stone. It is not a world,
        it is a coffin, twice over."""
        c = self.cfg
        cy = cx = (c.size // 2) if c.size is not None else 0
        for r in range(0, 400):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dy), abs(dx)) != r:          # only the ring at radius r
                        continue
                    y, x = cy + dy, cx + dx
                    if self.tiles.blocked_circle(float(y), float(x), c.radius):
                        continue
                    _, _, bounded = self.enclosure(y, x, limit=c.spawn_room, openable=True)
                    if bounded:
                        continue                            # sealed even to a pickaxe
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

        def plant(y: int, x: int, kind: int | None = None):
            # jitter FIRST, then validate the position we actually plant at: a cell
            # whose centre is clear can still have its off-grid offset land a morsel
            # inside a wall, and unreachable food is worse than no food
            fy = float(y) + float(rng.uniform(-0.35, 0.35))
            fx = float(x) + float(rng.uniform(-0.35, 0.35))
            if self.tiles.blocked_circle(fy, fx, c.radius):
                return
            k = self._kind_for(y, x, rng) if kind is None else kind
            if k == BERRY and rng.random() < c.poison_weight:
                k = NIGHTCAP                                 # a berry that is not one
            spots.append(FoodSpot(y=fy, x=fx, kind=k,
                                  renewable=bool(rng.random() < c.renewable_frac)))

        for _ in range(c.food_per_chunk):                    # scattered over the chunk
            plant(cy * ch + int(rng.integers(ch)), cx * ch + int(rng.integers(ch)))

        # ORCHARDS: fruit hangs by a TRUNK and nowhere else, so a tree — the most
        # visible landmark the world has — is a place worth walking to. Anchored on
        # actual trunks, because random scatter finds them far too rarely.
        trunks = np.argwhere(self.tiles.region(cy * ch, cx * ch, ch, ch) == TRUNK)
        if len(trunks):
            for k in rng.choice(len(trunks), size=min(c.tree_spots, len(trunks)),
                                replace=False):
                ty, tx = trunks[int(k)]
                oy, ox = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
                plant(cy * ch + int(ty) + oy, cx * ch + int(tx) + ox, kind=FRUIT)

        # CAVE CACHES: extra attempts that only take in a walled-in cell. Random
        # scatter almost never lands in a tunnel, so without this the best food in the
        # world is too rare to be a reason to go anywhere.
        for _ in range(c.cave_spots):
            y = cy * ch + int(rng.integers(ch))
            x = cx * ch + int(rng.integers(ch))
            near = self.tiles.region(y - 2, x - 2, 5, 5)
            if not SOLID[self.tiles.material(y, x)] and float(SOLID[near].mean()) > 0.3:
                plant(y, x, kind=MUSHROOM)

        # LARDERS: a room behind a door gets berries, so a closed room is worth the
        # trouble of opening. Empty rooms were the user's complaint — a door that
        # leads to nothing teaches the agent that doors are not worth opening.
        block = self.tiles.region(cy * ch, cx * ch, ch, ch)
        doors = np.argwhere((block == DOOR) | (block == DOOR_OPEN))[:c.larder_doors]
        for dy, dx in doors:
            for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cells, _, bounded = self.enclosure(cy * ch + int(dy) + oy,
                                                   cx * ch + int(dx) + ox, limit=260)
                if not bounded or len(cells) < 4:
                    continue                                 # outside, or a crack
                inside = [p for p in cells                   # keep larders in the chunk
                          if cy * ch <= p[0] < (cy + 1) * ch
                          and cx * ch <= p[1] < (cx + 1) * ch]
                for _ in range(int(rng.integers(1, c.larder_max + 1))):
                    if inside:
                        py, px = inside[int(rng.integers(len(inside)))]
                        plant(py, px, kind=BERRY)
                break                                        # one side is the interior
        self.food[(cy, cx)] = spots
        return spots

    def _kind_for(self, y: int, x: int, rng) -> int:
        """What grows here — read off the surroundings, so food sits where it looks
        like it should and the world's colours correlate with its rewards. Fruit grows
        ONLY by a trunk (it is tree fruit); mushrooms want walls around them."""
        near = self.tiles.region(y - 2, x - 2, 5, 5)
        if (near == TRUNK).any():
            return FRUIT if rng.random() < 0.85 else BERRY
        if float(SOLID[near].mean()) > 0.3:                  # walled in: a cave nook
            return MUSHROOM
        return BERRY

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
        a.last_turn = max(-1.0, min(1.0, turn))
        a.heading = (a.heading + a.last_turn * c.max_turn) % (2 * math.pi)
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
        if a.poison > 0.0:                                   # the nightcap taking its
            drain = min(a.poison, c.poison_rate)             # time, tick by tick
            a.poison -= drain
            a.energy -= drain
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
            elif not self._centre_up(sy, sx):
                break                                        # wedged in a corner

    def _centre_up(self, sy: float, sx: float) -> bool:
        """A hop is fully blocked: drift toward the CENTRE LINE of the cell we are in,
        across the direction of travel, and retry. This is what makes a one-cell
        doorway or cave tunnel passable in practice — a body of diameter 0.6 fits a
        1.0 gap with 0.2 to spare, but only if it is roughly centred, and asking a
        player (or a brain) to hold a heading to the hundredth of a cell is not a
        game. Aim at the right ROW and the geometry does the last bit.

        It never moves onto a wall (every candidate is collision-tested), and it only
        fires when the move failed outright, so ordinary walking is untouched.
        Returns True if it managed to make progress."""
        a, c = self.agent, self.cfg
        blocked = self.tiles.blocked_circle
        across_y = abs(sx) >= abs(sy)                    # travelling in x -> centre y
        here = a.y if across_y else a.x
        want = round(here)
        if abs(want - here) < 1e-9:
            return False                                 # already centred: truly wedged
        step = math.copysign(min(c.assist, abs(want - here)), want - here)
        ny, nx = (a.y + step, a.x) if across_y else (a.y, a.x + step)
        if blocked(ny, nx, c.radius):
            return False
        a.y, a.x = ny, nx
        if not blocked(a.y + sy, a.x + sx, c.radius):    # the gap opened up: go through
            a.y, a.x = a.y + sy, a.x + sx
        return True

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
        a.ate[s.kind] += 1
        if gain < 0:                                         # POISON does not hit at
            a.poison += -gain                                # once: it drains, so the
            return self._event(f"ate {FOOD_NAMES[s.kind]} — it burns")   # agent lives
        a.energy = min(a.energy + gain, self.cfg.max_energy) # to connect cause to effect
        return self._event(f"ate {FOOD_NAMES[s.kind]} {gain:+.0f}")

    def _break(self, y: int, x: int, mat: int) -> str:
        """Chipping: a breakable tile takes several hits, so opening a shortcut is a
        real cost paid in energy and time, not a free click. Felling a TRUNK takes its
        crown with it — a floating canopy over a stump reads as a bug, and the foliage
        is part of the same object as far as the eye is concerned."""
        hits = self.damage.get((y, x), 0) + 1
        need = int(HP[mat])
        if hits < need:
            self.damage[(y, x)] = hits
            return self._event(f"chipping {self._name(mat)} {hits}/{need}")
        self.damage.pop((y, x), None)
        self.tiles.set(y, x, GROUND)
        if mat == TRUNK:
            self._drop_crown(y, x)
            return self._event("felled the tree")
        return self._event(f"broke the {self._name(mat)}")

    def _drop_crown(self, y: int, x: int, cap: int = 12) -> None:
        """Clear the FOLIAGE connected to a felled trunk (8-connected flood fill,
        capped). A canopy floating over a stump reads as a bug, and to an eye the
        leaves and the trunk are one object. Capped because two trees whose crowns
        touch should not chain into a clear-cut."""
        queue = deque([(y, x)])
        seen = {(y, x)}
        removed = 0
        while queue and removed < cap:
            qy, qx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (qy + dy, qx + dx)
                    if n in seen or self.tiles.material(*n) != FOLIAGE:
                        continue
                    seen.add(n)
                    self.tiles.set(n[0], n[1], GROUND)
                    removed += 1
                    queue.append(n)

    def damage_frac(self, y: int, x: int) -> float:
        """How chewed-up the tile at (y, x) is, 0..1. The RENDERER darkens by this and
        (next chunk) VISION will read the same call, so damage is visible to the player
        and to the agent through one shared fact — nobody sees a wall the other does
        not."""
        hits = self.damage.get((int(y), int(x)), 0)
        if not hits:
            return 0.0
        return min(1.0, hits / max(1, int(HP[self.tiles.material(y, x)])))

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
