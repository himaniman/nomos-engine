"""The cortex world — a float agent on a tile-stored map.

Stage 1 scope: geometry, movement and energy. No senses, no food, no interaction
effects yet (they are the next chunks) — but the ACTION INTERFACE is already the
final one, because the keyboard and the brain must drive the agent through exactly
the same call or we are playtesting a different game than the one the agent lives in.

Movement rules carried over as LESSONS (not code) from aqua:
  * the body is a CIRCLE and collision is tested against that circle, so what is
    drawn is what collides;
  * a move is MARCHED in sub-steps smaller than the body, so nothing tunnels a wall
    at any speed;
  * a blocked hop SLIDES along the free axis instead of dead-stopping, so the agent
    never wedges into a corner.

Determinism: a run is a pure function of (seed, size, the action sequence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cortex.tiles import TileMap


@dataclass
class Config:
    """World + body constants. The economy numbers are placeholders until stage 4
    tunes them against a human baseline — see docs/cortex-world.md."""
    size: int | None = 512          # None = unbounded (the generator allows it)
    chunk: int = 32
    radius: float = 0.45            # body radius: about a wall's width, still under
                                    # 0.5 so it can thread a one-cell doorway
    substep: float = 0.4            # < 2*radius, so consecutive body circles overlap
    max_speed: float = 0.35         # cells per tick at full thrust
    max_turn: float = 0.12          # radians per tick at full turn
    start_energy: float = 100.0
    move_cost: float = 0.02         # energy per tick at full speed
    stand_cost: float = 0.01        # ...and standing still: cheaper, never free, so
                                    # thinking beats wandering but waiting still costs
    max_age: int = 200_000          # a cap, NOT the cause of death — energy is life


@dataclass
class Agent:
    y: float
    x: float
    heading: float = 0.0            # radians; 0 = +x (east)
    speed: float = 0.0
    energy: float = 0.0
    age: int = 0
    alive: bool = True


class World:
    """The map plus the one agent. `step(thrust, turn, interact)` advances a tick."""

    def __init__(self, seed: int = 0, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.seed = int(seed)
        self.tiles = TileMap(seed=seed, size=self.cfg.size, chunk=self.cfg.chunk)
        self.tick = 0
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

    # --- the tick ----------------------------------------------------------------
    def step(self, thrust: float = 0.0, turn: float = 0.0, interact: float = 0.0) -> None:
        """One world tick. `thrust` in [0, 1], `turn` in [-1, 1], `interact` in [0, 1]
        (its effects — breaking, doors, pushing — land in the next chunk). This is the
        one entry point: the keyboard and the brain both come through here."""
        a = self.agent
        if not a.alive:
            return
        c = self.cfg
        a.heading = (a.heading + max(-1.0, min(1.0, turn)) * c.max_turn) % (2 * math.pi)
        a.speed = max(0.0, min(1.0, thrust)) * c.max_speed
        if a.speed > 0.0:
            self._advance(a.y + math.sin(a.heading) * a.speed,
                          a.x + math.cos(a.heading) * a.speed)
        frac = a.speed / c.max_speed
        a.energy -= c.stand_cost + (c.move_cost - c.stand_cost) * frac
        a.age += 1
        self.tick += 1
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

    # --- helpers -------------------------------------------------------------------
    def facing_cell(self) -> tuple[int, int]:
        """The cell the agent is looking at, one body-length ahead — the target an
        INTERACT will act on once interaction lands."""
        a = self.agent
        d = self.cfg.radius + 0.6
        return (int(round(a.y + math.sin(a.heading) * d)),
                int(round(a.x + math.cos(a.heading) * d)))
