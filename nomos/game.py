"""A deterministic Snake — the external world the cell-grid 'brain' drives.

Kept fully separate from the engine: the game is a pure function of (game size,
seed, the sequence of moves fed to it). The board wraps (toroidal, like the
world), food is placed by a seeded RNG, and the snake dies only by running into
its own body. That determinism is what keeps the coupled brain+game system
reproducible — the whole thing is still a function of (law, seed, cadence).
"""

from __future__ import annotations

import random

import numpy as np

# move name -> (dy, dx)
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class Snake:
    """Toroidal Snake. Head is body[0]; eating grows it and scores a point.

    Hunger: every step without food the snake gets hungrier, and after
    `starve_every` such steps it SHRINKS by one segment (loses its tail). Shrink
    a length-1 snake and it starves to death. So a snake that never eats cannot
    live forever — survival now requires foraging, which balances the reward
    against the trivial "flee food and circle forever" strategy.
    """

    def __init__(self, size: int | tuple[int, int], seed: int = 0,
                 starve_every: int = 50):
        self.h, self.w = (size, size) if isinstance(size, int) else tuple(size)
        self.rng = random.Random(seed)
        cy, cx = self.h // 2, self.w // 2
        self.body: list[tuple[int, int]] = [(cy, cx), (cy, (cx - 1) % self.w),
                                            (cy, (cx - 2) % self.w)]
        self.direction = "right"
        self.alive = True
        self.score = 0
        self.steps = 0
        self.starve_every = starve_every
        self.hunger = 0                                  # steps since last meal
        self.starved = False
        self.death_cell: tuple[int, int] | None = None   # where the head crashed
        self.food: tuple[int, int] | None = None
        self._place_food()

    def hunger_fraction(self) -> float:
        """0 just after eating, approaching 1 as the next shrink nears."""
        return min(1.0, self.hunger / max(1, self.starve_every))

    def head(self) -> tuple[int, int]:
        return self.body[0]

    def _place_food(self) -> None:
        occupied = set(self.body)
        free = [(y, x) for y in range(self.h) for x in range(self.w)
                if (y, x) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def step(self, move: str | None) -> None:
        """Advance one game tick. `move` turns the snake unless it reverses into
        itself or is None (keep going straight). No-op once dead."""
        if not self.alive:
            return
        if move in DIRECTIONS and move != _OPPOSITE[self.direction]:
            self.direction = move
        dy, dx = DIRECTIONS[self.direction]
        hy, hx = self.body[0]
        new_head = ((hy + dy) % self.h, (hx + dx) % self.w)
        ate = new_head == self.food
        # the tail cell frees up this tick unless we grow into it
        blocking = set(self.body if ate else self.body[:-1])
        if new_head in blocking:
            self.alive = False
            self.death_cell = new_head          # the body cell the head ran into
            self.steps += 1
            return
        self.body.insert(0, new_head)
        if ate:
            self.score += 1
            self.hunger = 0                     # a meal resets hunger
            self._place_food()
        else:
            self.body.pop()                     # normal move: length unchanged
            self.hunger += 1
            if self.hunger >= self.starve_every:
                self.hunger = 0
                if len(self.body) <= 1:
                    self.alive = False          # starved: nothing left to shrink
                    self.starved = True
                    self.steps += 1
                    return
                self.body.pop()                 # shrink: lose another tail segment
        self.steps += 1

    def field(self) -> np.ndarray:
        """Board as a 0..1 intensity map for rendering: body 0.5, head 0.85,
        food 1.0, empty 0 — so it reads naturally through the fire colormap.
        On death the crash cell is flashed at full brightness so the collision
        is visible (the step that kills never advances the head, so without this
        the death frame would look identical to the one before it)."""
        f = np.zeros((self.h, self.w), dtype=np.float64)
        for y, x in self.body:
            f[y, x] = 0.5
        hy, hx = self.body[0]
        f[hy, hx] = 0.85
        if self.food is not None:
            f[self.food] = 1.0
        if not self.alive:                      # flash the death for a visible beat
            if self.death_cell is not None:
                f[self.death_cell] = 1.0        # self-collision
            elif self.starved:
                f[hy, hx] = 1.0                 # starvation (no crash cell)
        return f
