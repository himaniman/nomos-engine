"""Tile STORAGE for the cortex world — the map is a tile array, the physics is float.

The world is generated from a seed in CHUNKS, on demand, so the generator is
infinite-capable; `size` bounds the playable region anyway (outside is solid rock).
Bounded is the default on purpose: a geodesic scent needs a boundary, and — the real
reason — in an endless world MEMORY never pays, because "there is always more ahead"
beats "go back to the patch I remember". See docs/cortex-world.md.

Chunk generation is deterministic and SEAM-FREE: a chunk rasterises the features of
the 3x3 block of chunks around it, each feature drawn from its own chunk's rng, so a
rock mass that starts in one chunk continues correctly into its neighbour no matter
which chunk is visited first.

Coordinate convention (same as the rest of the project): an INTEGER coordinate is a
cell CENTRE, so cell k owns the world interval [k - 0.5, k + 0.5).
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# --- materials -------------------------------------------------------------------
# Eight of them, as agreed: enough for colour and odour to carry information, few
# enough that we can tell what the brain learned. Food is NOT a material — it is a
# float entity (next stage).
GROUND, STONE, RUBBLE, TRUNK, FOLIAGE, DOOR, DOOR_OPEN, PUSHABLE = range(8)

#            name        solid  breakable  hp   rgb
_MATERIALS = (("ground",   False, False,     0,  (54, 48, 40)),
              ("stone",    True,  False,     0,  (118, 118, 128)),
              ("rubble",   True,  True,      3,  (150, 126, 92)),
              ("trunk",    True,  True,      5,  (104, 70, 40)),
              ("foliage",  False, False,     0,  (58, 148, 62)),
              ("door",     True,  False,     0,  (196, 152, 68)),
              ("door_open", False, False,    0,  (110, 86, 40)),
              ("pushable", True,  False,     0,  (168, 108, 172)))

N_MATERIALS = len(_MATERIALS)
NAMES = tuple(m[0] for m in _MATERIALS)
SOLID = np.array([m[1] for m in _MATERIALS], dtype=bool)
BREAKABLE = np.array([m[2] for m in _MATERIALS], dtype=bool)
HP = np.array([m[3] for m in _MATERIALS], dtype=np.int64)
RGB = np.array([m[4] for m in _MATERIALS], dtype=np.uint8)
# HSV per material — ONE source of truth shared by the renderer and (next stage) the
# vision sensor, so what the user sees is literally what the agent senses.
HSV = np.array([colorsys.rgb_to_hsv(*(np.array(m[4]) / 255.0)) for m in _MATERIALS],
               dtype=np.float64)


def _mix(seed: int, cy: int, cx: int) -> int:
    """A deterministic 31-bit seed for chunk (cy, cx) — pure function of the world
    seed and the chunk coordinates, so chunks generate identically in any order."""
    h = (int(seed) & 0xFFFFFFFF) * 0x9E3779B1
    h ^= (int(cy) & 0xFFFFFFFF) * 0x85EBCA77
    h ^= (int(cx) & 0xFFFFFFFF) * 0xC2B2AE3D
    h ^= (h >> 15)
    return h & 0x7FFFFFFF


class TileMap:
    """The map. `material(y, x)` generates the containing chunk on first touch and
    caches it; `set(y, x, m)` edits it (destruction, doors). Out of bounds is STONE."""

    def __init__(self, seed: int = 0, size: int | None = 512, chunk: int = 32):
        self.seed = int(seed)
        self.size = size                      # None = unbounded (the generator allows it)
        self.chunk = int(chunk)
        self._chunks: dict[tuple[int, int], np.ndarray] = {}

    # --- access ------------------------------------------------------------------
    def in_bounds(self, y: int, x: int) -> bool:
        return self.size is None or (0 <= y < self.size and 0 <= x < self.size)

    def material(self, y: int, x: int) -> int:
        y, x = int(y), int(x)
        if not self.in_bounds(y, x):
            return STONE                      # the world is walled by rock
        ch = self.chunk
        cy, cx = y // ch, x // ch
        arr = self._chunks.get((cy, cx))
        if arr is None:
            arr = self._generate(cy, cx)
            self._chunks[(cy, cx)] = arr
        return int(arr[y - cy * ch, x - cx * ch])

    def set(self, y: int, x: int, mat: int) -> None:
        y, x = int(y), int(x)
        if not self.in_bounds(y, x):
            return
        self.material(y, x)                   # ensure the chunk exists
        ch = self.chunk
        self._chunks[(y // ch, x // ch)][y - (y // ch) * ch, x - (x // ch) * ch] = mat

    def solid(self, y: int, x: int) -> bool:
        return bool(SOLID[self.material(y, x)])

    def region(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        """An (h, w) block of material ids with its top-left at (y0, x0), copied CHUNK
        BY CHUNK (a per-cell `material()` loop would be far too slow for the renderer,
        which reads a whole camera window every frame). Out of bounds reads as STONE."""
        ch = self.chunk
        out = np.full((h, w), STONE, dtype=np.int8)
        for cy in range(math.floor(y0 / ch), math.floor((y0 + h - 1) / ch) + 1):
            for cx in range(math.floor(x0 / ch), math.floor((x0 + w - 1) / ch) + 1):
                # the overlap of this chunk with the requested window, in world coords
                wy0, wx0 = max(y0, cy * ch), max(x0, cx * ch)
                wy1, wx1 = min(y0 + h, (cy + 1) * ch), min(x0 + w, (cx + 1) * ch)
                if self.size is not None:                    # clip to the walled world
                    wy0, wx0 = max(wy0, 0), max(wx0, 0)
                    wy1, wx1 = min(wy1, self.size), min(wx1, self.size)
                if wy1 <= wy0 or wx1 <= wx0:
                    continue
                arr = self._chunks.get((cy, cx))
                if arr is None:
                    arr = self._generate(cy, cx)
                    self._chunks[(cy, cx)] = arr
                out[wy0 - y0:wy1 - y0, wx0 - x0:wx1 - x0] = \
                    arr[wy0 - cy * ch:wy1 - cy * ch, wx0 - cx * ch:wx1 - cx * ch]
        return out

    # --- collision ---------------------------------------------------------------
    def blocked_circle(self, y: float, x: float, r: float) -> bool:
        """Would a BODY circle of radius `r` centred at (y, x) overlap a solid tile?
        Each solid cell k is the unit square [k-0.5, k+0.5]; the test is the exact
        circle-vs-AABB one (distance to the square's closest point < r). What you see
        is what collides — the lesson aqua paid for with bodies poking through walls."""
        k0y, k1y = int(math.floor(y - r - 0.5)), int(math.floor(y + r + 0.5)) + 1
        k0x, k1x = int(math.floor(x - r - 0.5)), int(math.floor(x + r + 0.5)) + 1
        r2 = r * r
        for ky in range(k0y, k1y + 1):
            for kx in range(k0x, k1x + 1):
                if not SOLID[self.material(ky, kx)]:
                    continue
                ny = min(max(y, ky - 0.5), ky + 0.5)      # closest point on the square
                nx = min(max(x, kx - 0.5), kx + 0.5)
                dy, dx = y - ny, x - nx
                if dy * dy + dx * dx < r2:
                    return True
        return False

    # --- generation --------------------------------------------------------------
    def _generate(self, cy: int, cx: int) -> np.ndarray:
        """Rasterise chunk (cy, cx). Features are drawn from the 3x3 neighbourhood of
        chunks (each from its OWN rng) and clipped into this one, so a feature that
        straddles a chunk edge is identical from both sides — no seams, no dependence
        on visit order."""
        ch = self.chunk
        arr = np.full((ch, ch), GROUND, dtype=np.int8)
        oy, ox = cy * ch, cx * ch
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for f in self._features(cy + dy, cx + dx):
                    self._draw(arr, oy, ox, f)
        return arr

    def _features(self, cy: int, cx: int) -> list[tuple]:
        """This chunk's feature list — a pure function of (seed, cy, cx). Features are
        anchored inside the chunk and never larger than it, so the 3x3 sweep above is
        enough to catch every feature that can reach into a neighbour."""
        rng = np.random.default_rng(_mix(self.seed, cy, cx))
        ch = self.chunk
        oy, ox = cy * ch, cx * ch
        def spot():
            return oy + int(rng.integers(ch)), ox + int(rng.integers(ch))

        feats: list[tuple] = []
        for _ in range(int(rng.integers(2, 5))):              # rock masses, some hollow
            h, w = int(rng.integers(5, 16)), int(rng.integers(5, 16))
            y, x = spot()
            hollow = bool(rng.random() < 0.7) and h >= 7 and w >= 7
            feats.append(("mass", y, x, h, w, hollow, int(rng.integers(4))))
        if rng.random() < 0.45:                               # a CAVE: a rock block with
            h, w = int(rng.integers(12, 26)), int(rng.integers(12, 26))   # tunnels bored
            y, x = spot()                                     # through it (the maze the
            feats.append(("cave", y, x, h, w, int(rng.integers(1 << 30))))  # user wanted)
        for _ in range(int(rng.integers(1, 4))):              # long thin veins with a gap
            ln = int(rng.integers(10, ch))
            y, x = spot()
            feats.append(("vein", y, x, ln, bool(rng.random() < 0.5),
                          int(rng.integers(2, ln - 2))))
        for _ in range(int(rng.integers(1, 5))):              # breakable rubble blobs
            y, x = spot()
            feats.append(("blob", y, x, int(rng.integers(1, 4)), RUBBLE))
        for _ in range(int(rng.integers(4, 12))):             # trees: a trunk in a leafy
            y, x = spot()                                     # clump, 2x2..3x3, shapes
            feats.append(("tree", y, x, int(rng.integers(1 << 30))))   # vary a little
        for _ in range(int(rng.integers(0, 3))):              # pushable blocks
            y, x = spot()
            feats.append(("push", y, x))
        return feats

    def _draw(self, arr: np.ndarray, oy: int, ox: int, f: tuple) -> None:
        kind = f[0]
        if kind == "mass":
            _, y, x, h, w, hollow, side = f
            _rect(arr, oy, ox, y, x, h, w, STONE)
            if hollow:                                        # a room / cave inside...
                _rect(arr, oy, ox, y + 1, x + 1, h - 2, w - 2, GROUND)
                dy, dx = (y + h // 2, x) if side == 0 else \
                         (y + h // 2, x + w - 1) if side == 1 else \
                         (y, x + w // 2) if side == 2 else (y + h - 1, x + w // 2)
                _put(arr, oy, ox, dy, dx, DOOR)               # ...reached through a door
        elif kind == "vein":
            _, y, x, ln, vert, gap = f
            for i in range(ln):
                if i == gap:                                  # doorways, so veins divide
                    continue                                  # without sealing anything
                _put(arr, oy, ox, y + i if vert else y, x if vert else x + i, STONE)
        elif kind == "cave":
            # a solid rock block with tunnels BORED through it by random walks that
            # start on the rim, so a cave always has mouths and is never a sealed
            # pocket. The carve pattern is a pure function of the feature's own seed,
            # so a cave straddling a chunk edge is drawn identically from both sides.
            _, y, x, h, w, cseed = f
            _rect(arr, oy, ox, y, x, h, w, STONE)
            rng = np.random.default_rng(cseed)
            for _ in range(int(rng.integers(3, 6))):
                side = int(rng.integers(4))
                cy = y if side == 0 else (y + h - 1 if side == 1 else
                                          y + int(rng.integers(h)))
                cx = x + int(rng.integers(w)) if side < 2 else \
                    (x if side == 2 else x + w - 1)
                for _ in range(int(rng.integers(h + w, 2 * (h + w)))):
                    _put(arr, oy, ox, cy, cx, GROUND)
                    if rng.random() < 0.5:
                        cy += int(rng.integers(-1, 2))
                    else:
                        cx += int(rng.integers(-1, 2))
                    cy = min(max(cy, y), y + h - 1)           # stay inside the block
                    cx = min(max(cx, x), x + w - 1)
        elif kind == "blob":
            _, y, x, r, mat = f
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    if i * i + j * j <= r * r:
                        _put(arr, oy, ox, y + i, x + j, mat, only_ground=True)
        elif kind == "tree":
            # a compact CLUMP: a solid trunk with foliage filling most of a 2x2..3x3
            # box around it. Mostly-filled (not a sparse bitmask) so it reads as one
            # object — which is the point, since the eye is meant to resolve its shape.
            _, y, x, tseed = f
            rng = np.random.default_rng(tseed)
            big = rng.random() < 0.45
            r0, r1 = (-1, 2) if big else (0, 2)
            _put(arr, oy, ox, y, x, TRUNK, only_ground=True)
            for i in range(r0, r1):
                for j in range(r0, r1):
                    if (i or j) and rng.random() < 0.8:
                        _put(arr, oy, ox, y + i, x + j, FOLIAGE, only_ground=True)
        elif kind == "push":
            _, y, x = f
            _put(arr, oy, ox, y, x, PUSHABLE, only_ground=True)


def _put(arr: np.ndarray, oy: int, ox: int, y: int, x: int, mat: int,
         only_ground: bool = False) -> None:
    """Write world cell (y, x) into the chunk array, clipped. `only_ground` keeps
    decoration (trees, rubble) from eating structural rock."""
    i, j = y - oy, x - ox
    if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
        if not only_ground or arr[i, j] == GROUND:
            arr[i, j] = mat


def _rect(arr: np.ndarray, oy: int, ox: int, y: int, x: int, h: int, w: int,
          mat: int) -> None:
    i0, j0 = max(0, y - oy), max(0, x - ox)
    i1, j1 = min(arr.shape[0], y - oy + h), min(arr.shape[1], x - ox + w)
    if i1 > i0 and j1 > j0:
        arr[i0:i1, j0:j1] = mat
