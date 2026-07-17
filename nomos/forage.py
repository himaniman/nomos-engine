"""A walled foraging world — the harder external game the cell-grid brain drives.

The map is generated from the seed: border walls plus random wall strokes (each
with a doorway gap) carve the open space into rooms and nooks. The player is a
single dot that starts at the centre, moves in the 4 cardinal directions (a wall
blocks the move; no move = stand still), and must keep finding food to survive:
hunger grows every move and `starve_after` moves without a meal kill it. There
is no other death — the challenge is navigation, not collision. `food_count`
items live on the map at once and respawn when eaten, biased into nooks (cells
hugged by walls), so eating means steering around walls, not just surfing a
straight gradient.

Deterministic: the whole game — map, food placement, respawns — is a pure
function of (size, seed, the move sequence). NOT toroidal: walls bound the world.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

# move name -> (dy, dx)
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class Forage:
    """Point-agent foraging in a seeded walled map. Score = meals eaten.

    The map is meant to be WIDE (default 96x96) — the player starts at the
    centre and the render follows it egocentrically via `view()`; `food_count`
    defaults to one item per ~768 cells so the world stays spacious."""

    def __init__(self, size: int | tuple[int, int] = 96, seed: int = 0,
                 food_count: int | None = None, starve_after: int = 120,
                 idle_cost: float = 0.5):
        self.h, self.w = (size, size) if isinstance(size, int) else tuple(size)
        self.rng = random.Random(seed)
        self.pos: tuple[int, int] = (self.h // 2, self.w // 2)
        self.walls = self._generate_walls()
        self.reachable = self._flood_fill(self.pos)
        self._nook = self._nook_weights()
        self.alive = True
        self.starved = False
        self.score = 0
        self.steps = 0
        self.starve_after = starve_after
        self.idle_cost = idle_cost
        self.hunger = 0.0                                # grows until the next meal
        self.foods: set[tuple[int, int]] = set()
        self._scent_cache: dict[tuple[int, int], np.ndarray] = {}
        self._vis_cache: dict[tuple[tuple[int, int], int], np.ndarray] = {}
        if food_count is None:
            food_count = max(4, (self.h * self.w) // 768)
        for _ in range(food_count):
            self._place_food()

    # -- map generation -----------------------------------------------------

    def _generate_walls(self) -> np.ndarray:
        """Border walls + random strokes, each pierced by one doorway so strokes
        partition space into rooms without (usually) sealing it. Strokes are
        added until ~16% of the interior is wall, so wall density is the same
        whatever the map size. The centre 3x3 is always cleared so the player
        never spawns inside a wall."""
        walls = np.zeros((self.h, self.w), dtype=bool)
        walls[0, :] = walls[-1, :] = True
        walls[:, 0] = walls[:, -1] = True
        target = int(0.16 * (self.h - 2) * (self.w - 2))
        lo = max(3, min(self.h, self.w) // 8)
        hi = max(lo + 1, min(self.h, self.w) // 3)
        placed = 0
        for _ in range(10_000):                          # hard cap, never spins
            if placed >= target:
                break
            length = self.rng.randint(lo, hi)
            door = self.rng.randrange(length)            # leave one gap
            if self.rng.random() < 0.5:                  # horizontal stroke
                y = self.rng.randint(2, self.h - 3)
                x0 = self.rng.randint(1, max(1, self.w - 1 - length))
                cells = [(y, x0 + i) for i in range(length)]
            else:                                        # vertical stroke
                x = self.rng.randint(2, self.w - 3)
                y0 = self.rng.randint(1, max(1, self.h - 1 - length))
                cells = [(y0 + i, x) for i in range(length)]
            for i, (y, x) in enumerate(cells):
                if i != door and 0 < y < self.h - 1 and 0 < x < self.w - 1:
                    placed += int(not walls[y, x])
                    walls[y, x] = True
        py, px = self.pos
        walls[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = False
        return walls

    def _flood_fill(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        """Open cells reachable from `start` — food only ever spawns here, so a
        stroke that seals off a pocket just makes that pocket irrelevant."""
        seen = {start}
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            for dy, dx in DIRECTIONS.values():
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.h and 0 <= nx < self.w
                        and not self.walls[ny, nx] and (ny, nx) not in seen):
                    seen.add((ny, nx))
                    queue.append((ny, nx))
        return seen

    def _nook_weights(self) -> np.ndarray:
        """Per-cell food-placement weight, precomputed once (the walls are
        static): cells hugged by walls weigh more — food prefers nooks and dead
        ends, so reaching it takes actual navigation."""
        p = np.pad(self.walls, 1, constant_values=True).astype(np.float64)
        n = sum(p[1 + dy:self.h + 1 + dy, 1 + dx:self.w + 1 + dx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        return 1.0 + 2.0 * n

    def _place_food(self) -> None:
        free = sorted(self.reachable - self.foods - {self.pos})
        if not free:
            return
        weights = [self._nook[c] for c in free]
        self.foods.add(self.rng.choices(free, weights=weights, k=1)[0])

    # -- game dynamics ------------------------------------------------------

    def hunger_fraction(self) -> float:
        """0 just after eating, approaching 1 as starvation nears."""
        return min(1.0, self.hunger / max(1, self.starve_after))

    def scent(self, food: tuple[int, int]) -> np.ndarray:
        """GEODESIC distance map from a food item over open cells (-1 where
        unreachable), BFS-computed once per item and cached (walls are static).
        This is how smell physically works: it diffuses through the maze, so
        its gradient at the player's nose points along the PATH to the food,
        around walls — not through them. A food sealed off from the player has
        no scent at all."""
        if food not in self._scent_cache:
            dist = np.full((self.h, self.w), -1, dtype=np.int32)
            dist[food] = 0
            queue = deque([food])
            while queue:
                y, x = queue.popleft()
                for dy, dx in DIRECTIONS.values():
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.h and 0 <= nx < self.w
                            and not self.walls[ny, nx] and dist[ny, nx] < 0):
                        dist[ny, nx] = dist[y, x] + 1
                        queue.append((ny, nx))
            self._scent_cache[food] = dist
        return self._scent_cache[food]

    def blocked(self, move: str | None) -> bool:
        """Would this move run into a wall (or off the map)?"""
        if move not in DIRECTIONS:
            return False
        dy, dx = DIRECTIONS[move]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        return not (0 <= ny < self.h and 0 <= nx < self.w) or bool(self.walls[ny, nx])

    def step(self, move: str | None) -> None:
        """Advance one game tick. A wall blocks the move (the player stays);
        `move` of None means stand still. Hunger grows either way, but standing
        costs only `idle_cost` (resting metabolism) while a move — even a
        blocked one, pressing is effort — costs a full point: pausing to THINK
        is cheaper than wandering wrongly, yet never free, so the immortal
        stand-forever loophole stays closed. No-op once dead."""
        if not self.alive:
            return
        self.steps += 1
        attempted = move in DIRECTIONS
        if attempted:
            dy, dx = DIRECTIONS[move]
            ny, nx = self.pos[0] + dy, self.pos[1] + dx
            if 0 <= ny < self.h and 0 <= nx < self.w and not self.walls[ny, nx]:
                self.pos = (ny, nx)
        if self.pos in self.foods:
            self.foods.discard(self.pos)
            self.score += 1
            self.hunger = 0.0
            self._place_food()
        else:
            self.hunger += 1.0 if attempted else self.idle_cost
            if self.hunger >= self.starve_after:
                self.alive = False
                self.starved = True

    def field(self) -> np.ndarray:
        """Board as a 0..1 intensity map for rendering: walls 0.3 (dim red in
        the fire palette), player 0.85 (orange), food 1.0 (white). Starvation
        flashes the player at full brightness so the death is visible."""
        f = np.zeros((self.h, self.w), dtype=np.float64)
        f[self.walls] = 0.3
        for fy, fx in self.foods:
            f[fy, fx] = 1.0
        f[self.pos] = 1.0 if self.starved else 0.85
        return f

    def _los(self, ty: int, tx: int) -> bool:
        """Bresenham line of sight from the player to (ty, tx): true when no
        wall stands strictly between (the target itself may be a wall — that
        is the visible surface)."""
        y0, x0 = self.pos
        dy, dx = abs(ty - y0), abs(tx - x0)
        sy = 1 if ty > y0 else -1
        sx = 1 if tx > x0 else -1
        err = dy - dx
        y, x = y0, x0
        while (y, x) != (ty, tx):
            if (y, x) != (y0, x0) and self.walls[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dx:
                err -= dx
                y += sy
            if e2 < dy:
                err += dy
                x += sx
        return True

    def visibility(self, radius: int) -> np.ndarray:
        """Boolean line-of-sight mask over the egocentric window, cached per
        (position, radius) — walls are static, so a revisited cell is free."""
        key = (self.pos, radius)
        if key not in self._vis_cache:
            side = 2 * radius + 1
            mask = np.zeros((side, side), dtype=bool)
            py, px = self.pos
            for wy in range(side):
                for wx in range(side):
                    ty, tx = py + wy - radius, px + wx - radius
                    if 0 <= ty < self.h and 0 <= tx < self.w:
                        mask[wy, wx] = self._los(ty, tx)
            self._vis_cache[key] = mask
        return self._vis_cache[key]

    def view(self, radius: int = 32, fade: float | None = None,
             fog: bool = False) -> np.ndarray:
        """Egocentric crop of `field()`: a (2*radius+1)^2 window with the player
        always at the CENTRE — the camera follows the player through the wide
        world. Cells beyond the map read as wall. With `fade` (a fraction of
        the radius, e.g. 0.7) the surroundings dim under a Gaussian vignette —
        the render matches how perception attenuates with distance — with a
        small floor so the layout stays legible; the centre cell is untouched.
        With `fog` the window is masked by LINE OF SIGHT: anything behind a
        wall is black — the player does not see through walls (the blocking
        wall itself stays lit; it is the visible surface). Food behind a wall
        is smelled, never seen."""
        f = self.field()
        side = 2 * radius + 1
        out = np.full((side, side), 0.3, dtype=np.float64)
        py, px = self.pos
        y0, y1 = max(0, py - radius), min(self.h, py + radius + 1)
        x0, x1 = max(0, px - radius), min(self.w, px + radius + 1)
        out[y0 - py + radius:y1 - py + radius,
            x0 - px + radius:x1 - px + radius] = f[y0:y1, x0:x1]
        if fog:
            mask = self.visibility(radius)
            out *= mask
            out[mask & (out < 0.08)] = 0.08   # visible floor glows against the fog
        if fade is not None:
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            sigma = fade * radius
            vignette = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
            out *= np.maximum(vignette, 0.12)
            out[radius, radius] = f[py, px]      # the player never fades
        return out
