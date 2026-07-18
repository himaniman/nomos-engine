"""A tiny deterministic uniform-BUCKET spatial index for the continuous aqua world.

This is the grid's afterlife: the old cell array's real job was "what is near (y, x)?",
and that's all this keeps — an O(1)-ish neighbourhood lookup so the agent-agent
proximity queries (collision every sub-step, the mate search, vision) stop being
O(n^2). Buckets hold ITEM IDS only; the caller reads live positions from its own store,
so an id sitting in the right cell is the single source of truth (no stale coordinates
cached here). It's rebuilt cheaply each tick and kept live as agents move / spawn / die.

Determinism: cells are visited in a fixed (row-major) order and ids within a cell in
insertion order, and every caller SELECTS BY NEAREST DISTANCE, so the result never
depends on iteration order — runs stay reproducible.
"""

from __future__ import annotations

import math


class Buckets:
    """Maps a coarse integer cell -> the ids whose point falls in it."""

    def __init__(self, cell: float = 2.0):
        self.cell = cell
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.where: dict[int, tuple[int, int]] = {}          # id -> its current cell

    def clear(self) -> None:
        self.cells.clear()
        self.where.clear()

    def _key(self, y: float, x: float) -> tuple[int, int]:
        return (int(math.floor(y / self.cell)), int(math.floor(x / self.cell)))

    def insert(self, iid: int, y: float, x: float) -> None:
        k = self._key(y, x)
        self.cells.setdefault(k, []).append(iid)
        self.where[iid] = k

    def remove(self, iid: int) -> None:
        k = self.where.pop(iid, None)
        if k is None:
            return
        lst = self.cells.get(k)
        if lst is not None:
            try:
                lst.remove(iid)
            except ValueError:
                pass
            if not lst:
                del self.cells[k]

    def move(self, iid: int, y: float, x: float) -> None:
        """Reindex an id to its new position (a no-op if it stayed in the same cell)."""
        if self.where.get(iid) == self._key(y, x):
            return
        self.remove(iid)
        self.insert(iid, y, x)

    def neighbors(self, y: float, x: float, r: float):
        """Yield the ids in every cell overlapping the (y, x)+r box — a SUPERSET of
        those within r; the caller filters by exact distance."""
        rc = int(math.ceil(r / self.cell))
        by, bx = self._key(y, x)
        for cy in range(by - rc, by + rc + 1):
            for cx in range(bx - rc, bx + rc + 1):
                lst = self.cells.get((cy, cx))
                if lst is not None:
                    yield from lst
