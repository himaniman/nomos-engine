"""The atlas: the sheet as a MAP — a cellular automaton that plans by wave.

    .\\.venv\\Scripts\\python.exe -m cortex.atlas --probes            # the ladder
    .\\.venv\\Scripts\\python.exe -m cortex.atlas --seeds 0-7          # open-world lives
    .\\.venv\\Scripts\\python.exe -m cortex.atlas --gif out/cortex/atlas/seed0.gif

Why this exists (docs/lab-journal.md, 2026-09-14). The cortex brain used its 2D sheet
as a slow, noisy MLP: a 1D retina painted in by column, a command read as a mean over
effectors, and a random per-cell perceptron in between that had no reason to transmit
anything, let alone compute. The one thing a cellular automaton is actually GOOD at —
local rules producing global SPATIAL structure: wavefronts, distance fields, gradients —
was never asked of it. In that arrangement a perceptron beats a CA, which is the
opposite of the thesis.

So here the lattice IS a map. The agent sits at the centre; the sheet scrolls under it
as it walks (the sense of one's own motion is proprioception, and it is the only extra
sense this brain gets). Every sense paints WHERE IT REFERS TO: a ray that stops paints
"solid" at the cell it hit and "free" along the cells it crossed; a seen morsel paints a
bright thing at its cell; a smell paints a source out along the walkable direction it
reports. Nothing is labelled — a cell holds solidity, freeness, hue, brightness — and
the map remembers: painted evidence fades slowly, so the brain carries a decaying
picture of where it has been and what was there.

The RULE is one local update, run a few times per thought, and it is the whole
computation: a value wave spreads from every source through cells that are not solid,
losing a fixed share per step. That is Bellman / brushfire / BFS, done by cells that
each look only at their eight neighbours — exactly what `Pathfinding Neural Cellular
Automata` (Earle et al. 2023) shows an NCA can learn to do. The command is read at ONE
cell, the centre: which neighbour holds the highest value is which way to step. The
wave routes around every wall the map knows about; a wall the map has not seen yet is
optimistically passable, and the moment a ray sees it the wave re-routes.

Two behaviours fall out for free that the old brain never had. EXPLORATION: unknown
cells bordering known-free ones are faint sources (frontier exploration), so with
nothing to smell the agent walks toward what it has not seen instead of circling.
PATIENCE: no source anywhere means no wave, and no wave means the read-out says stand
still, which is what a brain that knows its energy is for should say.

The rule is hand-written here, and that is the point of stage 1: a CA that can play at
all, to grow and to learn on top of. Stage 2 replaces the hand rule with a per-cell
perceptron and asks whether evolution or gradient can FIND this rule — which is a small
search over a few hundred weights against a working target, not a needle hunt over
thirty thousand weights of a random medium.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from cortex.senses import (INTERO, N_TOUCH, ODOUR_CH, RAY_CH, TOUCH_CH, ScentField,
                           layout, sense)
from cortex.world import N_ODOUR, World
from cortex.world import Config as WorldConfig

# --- the channels a map cell holds ---------------------------------------------------
SOLID, FREE, THING, HUE_C, HUE_S, BRIGHT, SCENT, VALUE = range(8)
N_CH = 8
CH_NAMES = ("solid", "free", "thing", "hue_c", "hue_s", "bright", "scent", "value")

# the eight neighbours, and how far each is (a diagonal step costs sqrt 2)
_NBRS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
_NBR_DIST = np.array([math.hypot(dy, dx) for dy, dx in _NBRS])


class Config:
    """The atlas's own constants. Every one of these is a candidate gene — the map's
    radius above all, since it is the horizon of both planning and memory and it costs
    energy to keep."""

    def __init__(self, **kw):
        self.radius = 12            # map half-width in cells: horizon of memory AND
                                    # of planning, and what the map costs to keep.
                                    # Measured over 16 worlds under upkeep (journal,
                                    # 2026-09-14): r6 12.6 meals, r9 20.1, r12 28.1,
                                    # r16 20.2, r24 17.9 — an optimum, because a map
                                    # bigger than the walk between meals is memory
                                    # nobody reads, paid for every tick
        self.subticks = 6           # wave passes per thought (state persists between)
        self.think_every = 3        # world ticks per thought
        self.decay = 0.06           # value lost per cell of travel: V ~ (1-decay)^d
        self.forget = 0.0015        # evidence lost per world tick: the map's memory
        self.frontier = 0.12        # source strength of an unseen cell next to a seen
                                    # free one — exploration. Small, so anything real
                                    # (a smell, a seen morsel) always wins
        self.smell_gain = 1.0       # source strength of a painted smell
        self.thing_gain = 1.0       # ...and of a seen bright non-solid thing
        self.bright_min = 0.6       # what counts as a THING: brighter than any floor
                                    # or leaf (0.58), i.e. the innate appetite is "go
                                    # for the bright stuff" — hue is left for learning
        self.leak = 0.12            # how much of the wave gets THROUGH a solid cell.
                                    # Zero and a room with a door is a coffin; this is
                                    # the innate guess that a wall might open, and the
                                    # read-out presses INTERACT on a solid it is led
                                    # into. Which hues actually open is the door
                                    # experiment, and a per-hue leak is where it lives
        self.eat_dist = 1.2         # map distance at which to try INTERACT
        self.smell_here = 0.85      # ...or a smell this strong (same bar as the reflex)
        self.turn_gain = 3.0        # how hard a heading error is steered against
        self.slow_angle = 1.0       # radians of error above which the body slows
        self.cell_cost = 0.000015   # energy per map cell per tick — same tariff as a
                                    # cortex brain cell, so a bigger map has to pay
        self.__dict__.update(kw)

    def to_dict(self) -> dict:
        return dict(vars(self))


class Atlas:
    """A map brain. `act(vec, pose)` paints the senses, runs the wave, reads the
    command at the centre. Pure function of its state and inputs; determinism as
    everywhere else here."""

    def __init__(self, cfg: Config | None = None, wcfg: WorldConfig | None = None):
        self.cfg = cfg or Config()
        self.wcfg = wcfg or WorldConfig()
        self.layout = layout(self.wcfg.n_rays)
        R = self.cfg.radius
        self.R = R
        self.side = 2 * R + 1
        self.state = np.zeros((N_CH, self.side, self.side))
        self.cell = None            # the agent's integer cell the map is centred on
        self.thoughts = 0
        self.last_command = (0.0, 0.0, 0.0)
        self.last_target = None     # the neighbour the wave chose, for the HUD
        n = self.wcfg.n_rays
        step = self.wcfg.fov / n
        self._ray_rel = -self.wcfg.fov / 2 + (np.arange(n) + 0.5) * step
        self._t = np.arange(0.5, self.wcfg.ray_reach, 0.5)      # samples along a ray
        # source hues, remembered so a strip can be drawn
        self.sources = np.zeros((self.side, self.side))

    def cells(self) -> int:
        return self.side * self.side

    # --- the pose: proprioception ----------------------------------------------------
    def _scroll(self, y: float, x: float) -> None:
        """Keep the agent at the centre: when it crosses into a new cell the whole map
        shifts the other way and the strip that came into view is blank (unknown)."""
        cy, cx = int(round(y)), int(round(x))
        if self.cell is None:
            self.cell = (cy, cx)
            return
        dy, dx = cy - self.cell[0], cx - self.cell[1]
        if dy == 0 and dx == 0:
            return
        st = self.state
        st[:] = np.roll(st, (-dy, -dx), axis=(1, 2))
        if dy > 0:
            st[:, -dy:, :] = 0.0
        elif dy < 0:
            st[:, :-dy, :] = 0.0
        if dx > 0:
            st[:, :, -dx:] = 0.0
        elif dx < 0:
            st[:, :, :-dx] = 0.0
        self.cell = (cy, cx)

    # --- painting: every sense lands where it refers to -----------------------------
    def _paint(self, vec: np.ndarray, y: float, x: float, heading: float) -> None:
        st, R, S = self.state, self.R, self.side
        cy, cx = self.cell
        oy, ox = y - cy, x - cx                       # sub-cell offset of the body
        n = self.wcfg.n_rays
        rays = vec[:n * RAY_CH].reshape(n, RAY_CH)
        reach = self.wcfg.ray_reach
        dist = (1.0 - rays[:, 4]) * reach
        ang = heading + self._ray_rel
        sa, ca = np.sin(ang), np.cos(ang)
        # FREE along the ray, up to just short of what it hit
        t = self._t
        py = oy + sa[:, None] * t[None, :]
        px = ox + ca[:, None] * t[None, :]
        my = np.rint(py).astype(int) + R
        mx = np.rint(px).astype(int) + R
        ok = (t[None, :] < dist[:, None] - 0.45) & (my >= 0) & (my < S) & (mx >= 0) & (mx < S)
        fy, fx = my[ok], mx[ok]
        st[FREE, fy, fx] = 1.0
        st[SOLID, fy, fx] *= 0.5                      # a ray got through: not solid
        # the HIT, for rays that stopped inside their reach
        hit = dist < reach - 1e-6
        hy = np.rint(oy + sa * (dist + 0.01)).astype(int) + R
        hx = np.rint(ox + ca * (dist + 0.01)).astype(int) + R
        inb = hit & (hy >= 0) & (hy < S) & (hx >= 0) & (hx < S)
        sol = inb & (rays[:, 5] > 0.5)
        st[SOLID, hy[sol], hx[sol]] = 1.0
        st[FREE, hy[sol], hx[sol]] = 0.0
        # a thing you no longer see where you saw one is gone: a ray crossing a cell
        # that used to hold a thing clears it. BEFORE the hits are painted — a morsel is
        # a quarter of a cell wide and the rays beside the ones it claims pass through
        # the rest of its cell, so clearing afterwards would erase every morsel seen
        st[THING, fy, fx] = 0.0
        thing = inb & (rays[:, 5] <= 0.5)
        st[THING, hy[thing], hx[thing]] = np.where(rays[thing, 3] >= self.cfg.bright_min,
                                                   1.0, 0.0)
        st[FREE, hy[thing], hx[thing]] = 1.0          # a morsel or a leaf: walkable
        seen = inb
        st[HUE_C, hy[seen], hx[seen]] = rays[seen, 0]
        st[HUE_S, hy[seen], hx[seen]] = rays[seen, 1]
        st[BRIGHT, hy[seen], hx[seen]] = rays[seen, 3]
        # where the body is, is free
        st[FREE, R, R] = 1.0
        st[SOLID, R, R] = 0.0
        # TOUCH: what presses against the body is solid, one cell out
        tlo, _ = self.layout["slices"]["touch"]
        for k in range(N_TOUCH):
            if vec[tlo + k * TOUCH_CH] > 0.0:
                a = heading + k * math.pi / 2
                ty = int(round(oy + math.sin(a))) + R
                tx = int(round(ox + math.cos(a))) + R
                if 0 <= ty < S and 0 <= tx < S and (ty, tx) != (R, R):
                    st[SOLID, ty, tx] = max(st[SOLID, ty, tx], 0.8)
        # SMELL: not a memory, a current reading — re-painted from scratch each thought
        # at the distance the intensity implies, out along the walkable direction the
        # world reports. Cardinal far away, exact up close (senses._smell), so the
        # painted source is wrong in detail and right in direction, and it corrects as
        # the agent moves.
        st[SCENT] = 0.0
        olo, _ = self.layout["slices"]["odour"]
        sh, ch = math.sin(heading), math.cos(heading)
        for j in range(N_ODOUR):
            amp = float(vec[olo + j * ODOUR_CH])
            if amp < 0.005:
                continue
            fwd, lat = float(vec[olo + j * ODOUR_CH + 1]), float(vec[olo + j * ODOUR_CH + 2])
            dy, dx = fwd * sh + lat * ch, fwd * ch - lat * sh
            norm = math.hypot(dy, dx)
            if norm < 1e-6:
                continue
            d = self.wcfg.smell_sharp * math.sqrt(max(0.0, 1.0 / amp - 1.0))
            d = max(1.0, min(float(R - 1), d))
            sy = int(round(oy + dy / norm * d)) + R
            sx = int(round(ox + dx / norm * d)) + R
            sy, sx = max(0, min(S - 1, sy)), max(0, min(S - 1, sx))
            st[SCENT, sy, sx] = max(st[SCENT, sy, sx], amp * self.cfg.smell_gain)

    # --- the rule: a wave through what is not solid ---------------------------------
    def _sources(self) -> np.ndarray:
        st, c = self.state, self.cfg
        src = st[SCENT] + c.thing_gain * st[THING]
        if c.frontier > 0.0:
            # FRONTIER: unknown cells that touch a known-free one. Walking to them is
            # how the map grows; it is exploration without a random number.
            known_free = st[FREE] > 0.5
            unknown = (st[FREE] < 0.2) & (st[SOLID] < 0.5)
            nb = np.zeros_like(known_free)
            for dy, dx in _NBRS:
                nb |= np.roll(known_free, (dy, dx), axis=(0, 1))
            src = src + c.frontier * (unknown & nb)
        return src

    def _passable(self) -> np.ndarray:
        """How much of a wave a cell lets through: all of it when free, `leak` of it
        when solid — a wall is a cost, not an absolute, so a door can be worth a try
        and a sealed room is not a death sentence."""
        sol = np.clip(self.state[SOLID], 0.0, 1.0)
        return 1.0 - sol * (1.0 - self.cfg.leak)

    def _wave(self, src: np.ndarray, passes: int) -> None:
        """Bellman on a lattice: V = max(source, best neighbour * decay) * passable.

        Computed on a copy PADDED by one ring of free cells that wrap nothing but touch
        each other, so 'there is always a way round, out beyond the map' is true: a
        source at the map's edge with a wall in the way can be reached by walking along
        the rim rather than only through the wall. The ring is the map's horizon."""
        st, c = self.state, self.cfg
        S = self.side
        passable = np.ones((S + 2, S + 2))
        passable[1:-1, 1:-1] = self._passable()
        source = np.zeros((S + 2, S + 2))
        source[1:-1, 1:-1] = src
        V = np.zeros((S + 2, S + 2))
        V[1:-1, 1:-1] = st[VALUE]
        for _ in range(passes):
            best = np.zeros_like(V)
            for (dy, dx), d in zip(_NBRS, _NBR_DIST):
                shifted = np.roll(V, (dy, dx), axis=(0, 1)) * (1.0 - c.decay) ** d
                if dy > 0:                       # the roll wraps: cut the wrap
                    shifted[:dy, :] = 0.0
                elif dy < 0:
                    shifted[dy:, :] = 0.0
                if dx > 0:
                    shifted[:, :dx] = 0.0
                elif dx < 0:
                    shifted[:, dx:] = 0.0
                np.maximum(best, shifted, out=best)
            V = np.maximum(source, best) * passable
        st[VALUE] = V[1:-1, 1:-1]

    # --- the read-out at one cell: the centre -----------------------------------------
    def _read(self, heading: float, vec: np.ndarray) -> tuple[float, float, float]:
        st, c, R = self.state, self.cfg, self.R
        V = st[VALUE]
        passable = self._passable()
        # is there something to eat right here? A seen thing within reach, or a smell
        # that says it is under the nose.
        olo, ohi = self.layout["slices"]["odour"]
        amps = vec[olo:ohi:ODOUR_CH]
        near_thing = None
        win = st[THING, R - 2:R + 3, R - 2:R + 3]
        if win.max() > 0.5:
            ys, xs = np.nonzero(win > 0.5)
            d = np.hypot(ys - 2, xs - 2)
            k = int(np.argmin(d))
            if d[k] <= c.eat_dist + 0.5:
                near_thing = (ys[k] - 2, xs[k] - 2, d[k])
        # which neighbour does the wave say to step to?
        vals = np.array([V[R + dy, R + dx] * passable[R + dy, R + dx] for dy, dx in _NBRS])
        k = int(np.argmax(vals))
        self.last_target = _NBRS[k] if vals[k] > 1e-4 else None
        if near_thing is not None or (amps.max() > c.smell_here and self.last_target):
            # EAT: face it, creep, press. Same "approach slowly" the reflex found. Face
            # the thing if one is seen, else the way the wave points — a smell this
            # strong is a source painted a cell or two out, at its true bearing. (The
            # first version crept along its old heading here and spent a life pressing
            # E into a rubble pile with the berry behind its back.)
            if near_thing is not None and near_thing[2] > 1e-6:
                want = math.atan2(near_thing[0], near_thing[1])
            else:
                want = math.atan2(*self.last_target)
            err = (want - heading + math.pi) % (2 * math.pi) - math.pi
            turn = max(-1.0, min(1.0, err * c.turn_gain))
            return (0.25 if abs(err) < c.slow_angle else 0.0), turn, 1.0
        if self.last_target is None:
            return 0.0, 0.0, 0.0                    # nothing anywhere: stand still
        dy, dx = self.last_target
        want = math.atan2(dy, dx)
        err = (want - heading + math.pi) % (2 * math.pi) - math.pi
        turn = max(-1.0, min(1.0, err * c.turn_gain))
        thrust = 1.0 if abs(err) < c.slow_angle else 0.3
        if st[SOLID, R + dy, R + dx] > 0.5:
            # led INTO a solid: the wave says the way is through. Face it, lean on it,
            # and use the one verb — which opens a door, chips rubble, shoves a block,
            # and does nothing at all to stone (that is what the hue is for, later)
            return (0.5 if abs(err) < c.slow_angle else 0.0), turn, 1.0
        return thrust, turn, 0.0

    # --- one thought -----------------------------------------------------------------
    def act(self, vec: np.ndarray, y: float, x: float, heading: float) -> tuple:
        c = self.cfg
        self._scroll(y, x)
        # the map forgets, slowly: evidence is a memory, not a fact
        f = 1.0 - c.forget * c.think_every
        self.state[SOLID] *= f
        self.state[FREE] *= f
        self.state[THING] *= f
        self._paint(vec, y, x, heading)
        src = self._sources()
        self.sources = src
        self._wave(src, c.subticks)
        self.thoughts += 1
        self.last_command = self._read(heading, vec)
        return self.last_command


class AtlasMind:
    """The atlas wired into a life, on the same terms as `agent.Mind`: it thinks every
    `think_every` ticks, holds the command in between, and pays for its cells."""

    def __init__(self, cfg: Config | None = None, wcfg: WorldConfig | None = None):
        self.atlas = Atlas(cfg, wcfg)
        self.cfg = self.atlas.cfg
        self.field: ScentField | None = None
        self.command = (0.0, 0.0, 0.0)
        self.sensors = None

    def attach(self, w: World) -> None:
        self.field = ScentField(w.cfg.smell_radius)

    def step(self, w: World) -> None:
        if self.field is None:
            self.attach(w)
        a = w.agent
        if w.tick % self.cfg.think_every == 0:
            self.sensors = sense(w, self.field)
            self.command = self.atlas.act(self.sensors, a.y, a.x, a.heading)
        w.step(*self.command)
        a.energy -= self.cfg.cell_cost * self.atlas.cells()


class Driver:
    """The atlas at a probe (`probes._run` calls `driver(vec, w)`). It reads the pose
    off the world — that is its proprioception, the one sense the vector lacks."""

    def __init__(self, cfg: Config | None = None):
        self.atlas = None
        self.cfg = cfg

    def __call__(self, vec, w):
        if self.atlas is None:
            self.atlas = Atlas(self.cfg, w.cfg)
        a = w.agent
        return self.atlas.act(vec, a.y, a.x, a.heading)


def live(seed: int = 0, max_ticks: int = 20000, cfg: Config | None = None,
         wcfg: WorldConfig | None = None, frames: list | None = None,
         every: int = 12) -> tuple[World, AtlasMind]:
    """One life, headless. A pure function of (cfg, seed, limits). With `frames` a
    list, an image of world + map is appended every `every` ticks."""
    w = World(seed=seed, cfg=wcfg or WorldConfig())
    mind = AtlasMind(cfg, w.cfg)
    mind.attach(w)
    while w.agent.alive and w.tick < max_ticks:
        mind.step(w)
        if frames is not None and w.tick % every == 0:
            frames.append(frame(w, mind.atlas))
    return w, mind


# --- watchable -------------------------------------------------------------------------

def map_image(atlas: Atlas, scale: int = 4) -> np.ndarray:
    """The map as RGB: value wave in blue, solid in grey, seen things in their own
    hue, scent sources in yellow, the agent a white dot at the centre."""
    from cortex.app import _hsv_rgb
    st, S = atlas.state, atlas.side
    img = np.zeros((S, S, 3))
    V = st[VALUE]
    vmax = max(1e-6, float(V.max()))
    img[:, :, 2] = np.clip(V / vmax, 0, 1) * 0.9
    img[:, :, 1] += np.clip(st[FREE], 0, 1) * 0.18
    g = np.clip(st[SOLID], 0, 1)
    img = img * (1 - g[:, :, None]) + g[:, :, None] * np.array([0.55, 0.55, 0.6])
    ys, xs = np.nonzero(st[THING] > 0.3)
    for y, x in zip(ys, xs):
        img[y, x] = np.array(_hsv_rgb(st[HUE_C, y, x], st[HUE_S, y, x], 0.9, 1.0)) / 255.0
    ys, xs = np.nonzero(st[SCENT] > 0.01)
    for y, x in zip(ys, xs):
        img[y, x] = np.array([1.0, 0.9, 0.2]) * min(1.0, 0.4 + st[SCENT, y, x])
    if atlas.last_target is not None:
        dy, dx = atlas.last_target
        img[atlas.R + dy, atlas.R + dx] = [0.2, 1.0, 0.3]
    img[atlas.R, atlas.R] = 1.0
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def world_image(w: World, radius: int, scale: int = 4) -> np.ndarray:
    """The same window of the world, so a map can be read against the truth."""
    from cortex.tiles import RGB
    from cortex.world import FOOD_RGB
    a = w.agent
    cy, cx = int(round(a.y)), int(round(a.x))
    S = 2 * radius + 1
    mats = w.tiles.region(cy - radius, cx - radius, S, S)
    img = RGB[mats].astype(np.uint8).copy()
    for s in w.nearby_food(a.y, a.x, radius * 1.5):
        yy, xx = int(round(s.y)) - cy + radius, int(round(s.x)) - cx + radius
        if 0 <= yy < S and 0 <= xx < S:
            img[yy, xx] = FOOD_RGB[s.kind]
    img[radius, radius] = (255, 255, 255)
    hy = int(round(radius + math.sin(a.heading) * 1.5))
    hx = int(round(radius + math.cos(a.heading) * 1.5))
    if 0 <= hy < S and 0 <= hx < S:
        img[hy, hx] = (255, 80, 80)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def frame(w: World, atlas: Atlas, scale: int = 4) -> np.ndarray:
    left = world_image(w, atlas.R, scale)
    right = map_image(atlas, scale)
    gap = np.full((left.shape[0], 2 * scale, 3), 30, dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def save_gif(frames: list, path: str, fps: int = 12) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps),
                loop=0)


# --- the command line ------------------------------------------------------------------

def _seeds(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(q) for q in s.split(",")]


def _one_life(args) -> dict:
    seed, ticks, cfg_d = args
    from cortex.world import FOOD_ENERGY
    w, mind = live(seed, ticks, Config(**cfg_d))
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"seed": seed, "ticks": w.tick, "meals": int(sum(w.agent.ate)),
            "gained": gained, "walked": float(w.distance), "ate": list(w.agent.ate),
            "alive": bool(w.agent.alive)}


def lives(seeds, ticks: int = 20000, cfg: Config | None = None, jobs: int = 1) -> list[dict]:
    """Whole lives over many worlds, in parallel when asked. Each is a pure function of
    (cfg, seed, ticks), so the order they run in cannot matter."""
    cfg_d = (cfg or Config()).to_dict()
    args = [(int(s), int(ticks), cfg_d) for s in seeds]
    if jobs > 1 and len(args) > 1:
        from multiprocessing import Pool
        with Pool(min(jobs, len(args))) as pool:
            return pool.map(_one_life, args)
    return [_one_life(a) for a in args]


def table(rows: list[dict], who: str = "atlas") -> str:
    lines = [f"{who}  seed {r['seed']:>3}  ticks {r['ticks']:>6}  meals {r['meals']:>3}"
             f"  gained {r['gained']:>7.1f}  walked {r['walked']:>6.0f}  {r['ate']}"
             for r in rows]
    n = len(rows)
    lines.append(f"{who}  MEAN over {n}: ticks {np.mean([r['ticks'] for r in rows]):>7.0f}"
                 f"  meals {np.mean([r['meals'] for r in rows]):>5.1f}"
                 f"  gained {np.mean([r['gained'] for r in rows]):>7.1f}"
                 f"  immortal {sum(r['alive'] for r in rows)}/{n}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="The atlas: a map brain that plans by wave.")
    p.add_argument("--probes", action="store_true", help="sit the probe ladder")
    p.add_argument("--probe-ticks", type=int, default=900)
    p.add_argument("--seeds", default="", help="open-world lives, e.g. 0-7")
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--radius", type=int, default=12)
    p.add_argument("--free", action="store_true",
                   help="charge nothing for the map (like-for-like with the reflex)")
    p.add_argument("--jobs", type=int, default=1, help="lives in parallel")
    p.add_argument("--gif", default="", help="record one life (first seed) here")
    p.add_argument("--strip", default="out/cortex/atlas/probes.png")
    args = p.parse_args(argv)
    cfg = Config(radius=args.radius)
    if args.free:
        cfg.cell_cost = 0.0
    if args.probes:
        from cortex import probes
        out = probes._suite(lambda: Driver(cfg), args.probe_ticks)
        print("atlas   " + probes.summarise(out))
        if args.strip:
            probes.render(out, args.strip)
            print(f"strip -> {args.strip}")
    if args.seeds:
        from cortex.session import record
        seeds = _seeds(args.seeds)
        if args.gif:
            frames = []
            w, mind = live(seeds[0], args.ticks, cfg, frames=frames)
            save_gif(frames, args.gif)
            print(f"gif -> {args.gif}")
        rows = lives(seeds, args.ticks, cfg, jobs=args.jobs)
        print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
