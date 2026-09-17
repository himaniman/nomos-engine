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
SOLID, FREE, THING, HUE_C, HUE_S, BRIGHT, SCENT, VALUE, BAD, GHOST = range(10)
N_CH = 10
CH_NAMES = ("solid", "free", "thing", "hue_c", "hue_s", "bright", "scent", "value", "bad",
            "ghost")

# the hue population code: N_HUE preferred hues round the circle, tuning width kappa
N_HUE = 16
HUE_KAPPA = 12.0          # ~20 degrees half-width: berry (349) and nightcap (288) share
                          # almost nothing; at 8 bumps of 30 degrees they did, and one
                          # berry after a burn undid the lesson
_HUE_MU = np.arange(N_HUE) * 2 * math.pi / N_HUE
_HUE_COS, _HUE_SIN = np.cos(_HUE_MU), np.sin(_HUE_MU)

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
        self.smell_gain = 1.0       # source strength of a painted smell...
        self.smell_reach = 99.0     # ...and the furthest out it is painted (the map's
                                    # edge if larger). The world's nose points at a
                                    # WAYPOINT at most 6 cells out when the walk
                                    # bends; painting the source at the walk's full
                                    # length along that bearing puts it past the
                                    # turn — beside a door it landed 11 cells along
                                    # the wall and the agent ran past the door both
                                    # ways for ever (the door probe, 2026-09-17)
        self.smell_floor = 0.35     # ...and the least a faint one is painted at.
                                    # Swept over 16 worlds: 0 -> 27.4 meals, 0.35 ->
                                    # 32.2, 1.0 -> 18.8 (chases every far smell to the
                                    # map edge instead of exploring what is near)
        self.thing_gain = 1.0       # ...and of a seen bright non-solid thing
        self.bright_min = 0.6       # what counts as a THING: brighter than any floor
                                    # or leaf (0.58), i.e. the innate appetite is "go
                                    # for the bright stuff" — hue is left for learning
        self.leak = 0.5             # how much of the wave gets THROUGH a solid cell —
                                    # the INNATE guess that a wall might open; the
                                    # read-out presses INTERACT on a solid it is led
                                    # into. 0.12 for months, and at 0.12 the horizon
                                    # ring round the map (a walk of ~2R cells,
                                    # 0.94^26 = 0.2) beats leaking through ANY wall,
                                    # so no door was ever pressed; 0.5 is CURIOUS —
                                    # the door probe 0.08 -> 1.00, doors opened in
                                    # the wild 0 -> ~1.5 a life — and the per-hue
                                    # learning below is what pays for the curiosity
        self.linger = 0.99          # COMMITMENT: a painted source keeps this much of
                                    # itself per wave pass after the paint moves on
                                    # (0 = the plain wave). Found as a LAW first
                                    # (`laws/atlas_ghost.nomos`, 2026-09-17): the
                                    # smell of food behind a wall is painted where the
                                    # geodesic bearing points, and that bearing flips
                                    # between the two ways round as the agent steps
                                    # each way — a reactive limit cycle at the sensor.
                                    # A lingering source holds the first target.
                                    # 0.99: 16 worlds 14.6 -> 20.2 meals. Where the
                                    # agent stands the ghost is erased (nothing there)
        self.leak_lr = 1.0          # THE DOOR EXPERIMENT: with this > 0 the leak is
                                    # PER HUE — a perceptron over the hue population
                                    # code, bias = logit(leak) (the innate guess that
                                    # any wall might open), taught by the body: a press
                                    # that opens a door or chips rubble is +1 for that
                                    # hue, a press that does nothing ("nothing to do
                                    # here") is -1. Then a door is worth going through
                                    # and stone is not, without either ever being named
        self.leak_clip = 6.0
        self.press_angle = 0.35     # heading error below which a solid is pressed on
        self.eat_dist = 1.2         # map distance at which to try INTERACT
        self.smell_here = 0.85      # ...or a smell this strong (same bar as the reflex)
        self.turn_gain = 3.0        # how hard a heading error is steered against
        self.slow_angle = 1.0       # radians of error above which the body slows
        self.cell_cost = 0.000015   # energy per map cell per tick — same tariff as a
                                    # cortex brain cell, so a bigger map has to pay
        # APPETITE: the one thing this brain LEARNS within a life. A seen thing's pull,
        # and whether it is eaten when reached, is a tiny perceptron over its HUE —
        # innately "anything bright is dinner", corrected by what eating it did to the
        # body (interoception: the poison channel rising after a meal). Nothing else
        # can tell a nightcap from a berry: they smell identical by design.
        self.appetite_lr = 3.0      # how hard one meal's verdict moves the weights:
                                    # one burn is enough to refuse that hue next time
        self.appetite_clip = 6.0    # ...and how far they can go, so a brain stays
                                    # teachable after twenty meals of the same kind
        self.appetite_delay = 60    # ticks after a meal to wait for the verdict
        self.bad_radius = 2         # a refused or regretted thing suppresses every
                                    # source this close to it (including the smell that
                                    # led there), so the agent LEAVES instead of standing
                                    # over the one thing it will not eat
        self.__dict__.update(kw)

    def to_dict(self) -> dict:
        return dict(vars(self))


class Atlas:
    """A map brain. `act(vec, pose)` paints the senses, runs the wave, reads the
    command at the centre. Pure function of its state and inputs; determinism as
    everywhere else here."""

    def __init__(self, cfg: Config | None = None, wcfg: WorldConfig | None = None,
                 rule=None):
        self.cfg = cfg or Config()
        self.wcfg = wcfg or WorldConfig()
        self.layout = layout(self.wcfg.n_rays)
        R = self.cfg.radius
        self.R = R
        self.side = 2 * R + 1
        # THE RULE. None is the hand-written wave (`_wave`); anything else is an object
        # with `hidden` extra channels of its own and `step(state, src, passable,
        # passes)` that writes VALUE — cortex/wave.py grows one out of a per-cell
        # perceptron and asks whether evolution can find the wave on its own
        self.rule = rule
        hidden = int(getattr(rule, "hidden", 0))
        self.state = np.zeros((N_CH + hidden, self.side, self.side))
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
        # the appetite perceptron over a POPULATION CODE of hue (eight bumps round the
        # colour circle, like colour-tuned cells) plus a bias: born liking everything.
        # A perceptron over plain (cos, sin) cannot do the job — the nightcap's hue sits
        # BETWEEN the berry's and the mushroom's on the circle, so no half-plane keeps
        # the two foods and drops the poison; measured, it learned to hate mushrooms.
        self.appetite_w = np.zeros(N_HUE + 1)
        self.appetite_w[-1] = 2.0
        self.last_thing = None      # (dy, dx, features) of the thing within reach
        self.meals = 0              # verdicts learned from, for the HUD
        # the leak perceptron: born believing every solid leaks `leak`
        self.leak_w = np.zeros(N_HUE + 1)
        self.leak_w[-1] = math.log(self.cfg.leak / (1.0 - self.cfg.leak))
        self.last_press = None      # (dy, dx, features) of the solid being pressed on
        self.presses = 0            # verdicts on presses learned from

    def cells(self) -> int:
        return self.side * self.side

    # --- appetite: a perceptron over hue, taught by the body -------------------------
    @staticmethod
    def _feat(hue_c, hue_s) -> np.ndarray:
        """The population code: bump k fires as exp(kappa (cos(hue - mu_k) - 1)), and
        cos(hue - mu) = cos hue cos mu + sin hue sin mu, so it needs no angle. Last
        entry is the bias. Works on arrays (trailing axis = features)."""
        hc, hs = np.asarray(hue_c, dtype=np.float64), np.asarray(hue_s, dtype=np.float64)
        cosd = hc[..., None] * _HUE_COS + hs[..., None] * _HUE_SIN
        bumps = np.exp(HUE_KAPPA * (cosd - 1.0))
        return np.concatenate([bumps, np.ones(bumps.shape[:-1] + (1,))], axis=-1)

    def appetite(self, hue_c, hue_s):
        """0..1: how much a thing of this hue is wanted. Works on arrays too."""
        z = self._feat(hue_c, hue_s) @ self.appetite_w
        return 1.0 / (1.0 + np.exp(-z))

    def learn(self, feat: np.ndarray, reward: float) -> None:
        """The verdict on a meal: +1 it was food, -1 it burned. One Hebbian step, the
        weights clipped so the brain stays teachable."""
        c = self.cfg
        step = c.appetite_lr * float(reward) * feat
        step[-1] = 0.0                  # the bias is the innate prior; it is not taught
        self.appetite_w += step
        np.clip(self.appetite_w, -c.appetite_clip, c.appetite_clip, out=self.appetite_w)
        self.meals += 1

    def leak_map(self) -> np.ndarray:
        """0..1 per cell: how much wave the solid there would let through, by its hue.
        Cells never seen (no hue) get the innate leak: the bumps all fire ~0."""
        z = self._feat(self.state[HUE_C], self.state[HUE_S]) @ self.leak_w
        return 1.0 / (1.0 + np.exp(-z))

    def learn_leak(self, feat: np.ndarray, reward: float) -> None:
        """The verdict on a press: +1 something gave, -1 nothing did."""
        c = self.cfg
        step = c.leak_lr * float(reward) * feat
        step[-1] = 0.0
        self.leak_w += step
        np.clip(self.leak_w, -c.leak_clip, c.leak_clip, out=self.leak_w)
        self.presses += 1

    def gave(self, dy: int, dx: int, still: float) -> None:
        """A pressed cell yielded: it is `still` as solid as it was (0 for a door
        that opened, a fraction for rubble that was only chipped)."""
        y, x = self.R + dy, self.R + dx
        if 0 <= y < self.side and 0 <= x < self.side:
            self.state[SOLID, y, x] *= still
            if still == 0.0:
                self.state[FREE, y, x] = 1.0

    def regret(self, dy: int, dx: int) -> None:
        """Remember a bad thing WHERE it is, so the sources around it go quiet."""
        y, x = self.R + dy, self.R + dx
        if 0 <= y < self.side and 0 <= x < self.side:
            self.state[BAD, y, x] = 1.0

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
            d = max(1.0, min(float(R - 1), self.cfg.smell_reach, d))
            sy = int(round(oy + dy / norm * d)) + R
            sx = int(round(ox + dx / norm * d)) + R
            sy, sx = max(0, min(S - 1, sy)), max(0, min(S - 1, sx))
            # painted no fainter than `smell_floor`: how far it is has already been
            # said by WHERE it is painted, and the wave discounts that distance itself.
            # (Painted at `amp` alone, a smell sixteen cells off was worth 0.09 and
            # lost to the 0.12 of an unexplored cell beside the body.)
            st[SCENT, sy, sx] = max(st[SCENT, sy, sx],
                                    self.cfg.smell_gain * max(amp, self.cfg.smell_floor))

    # --- the rule: a wave through what is not solid ---------------------------------
    def _sources(self) -> np.ndarray:
        st, c = self.state, self.cfg
        want = self.appetite(st[HUE_C], st[HUE_S])
        src = st[SCENT] + c.thing_gain * st[THING] * want
        if c.frontier > 0.0:
            # FRONTIER: unknown cells that touch a known-free one. Walking to them is
            # how the map grows; it is exploration without a random number.
            known_free = st[FREE] > 0.5
            unknown = (st[FREE] < 0.2) & (st[SOLID] < 0.5)
            nb = np.zeros_like(known_free)
            for dy, dx in _NBRS:
                nb |= np.roll(known_free, (dy, dx), axis=(0, 1))
            src = src + c.frontier * (unknown & nb)
        if st[BAD].max() > 0.05:
            # a remembered bad thing silences every source near it: the smell that led
            # here is a lie this brain has learned not to follow, for a while
            hush = st[BAD].copy()
            for _ in range(int(c.bad_radius)):
                grown = hush.copy()
                for dy, dx in _NBRS:
                    np.maximum(grown, np.roll(hush, (dy, dx), axis=(0, 1)), out=grown)
                hush = grown
            hush = np.clip(hush, 0.0, 1.0)
            src = src * (1.0 - hush)
            st[VALUE] *= 1.0 - hush         # and the wave already there, at once
            st[GHOST] *= 1.0 - hush
        return src

    def _linger(self, src: np.ndarray) -> np.ndarray:
        """One pass of commitment: the wave's source is the ghost of every source
        painted lately, erased under the agent and its ring (it is standing there:
        there is nothing). Returns the ghost as it was — snapshot semantics, the same
        pass order as `laws/atlas_ghost.nomos`, so the two are one rule."""
        R, c = self.R, self.cfg
        old = self.state[GHOST].copy()
        g = np.maximum(src, c.linger * old)
        g[R - 1:R + 2, R - 1:R + 2] = 0.0
        self.state[GHOST] = g
        return old

    def _passable(self) -> np.ndarray:
        """How much of a wave a cell lets through: all of it when free, `leak` of it
        when solid — a wall is a cost, not an absolute, so a door can be worth a try
        and a sealed room is not a death sentence."""
        sol = np.clip(self.state[SOLID], 0.0, 1.0)
        if self.cfg.leak_lr > 0.0:
            return 1.0 - sol * (1.0 - self.leak_map())
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
            if c.linger > 0.0:
                source[1:-1, 1:-1] = self._linger(src)
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
        # a remembered bad thing within reach counts as refused even when it is out of
        # view: the press eats whatever is nearest, sight or no sight
        refused = bool(st[BAD, R - 2:R + 3, R - 2:R + 3].max() > 0.3)
        self.last_thing = None
        self.last_press = None
        win = st[THING, R - 2:R + 3, R - 2:R + 3]
        if win.max() > 0.5:
            ys, xs = np.nonzero(win > 0.5)
            d = np.hypot(ys - 2, xs - 2)
            k = int(np.argmin(d))
            if d[k] <= c.eat_dist + 0.5:
                dy, dx = int(ys[k] - 2), int(xs[k] - 2)
                hc, hs = st[HUE_C, R + dy, R + dx], st[HUE_S, R + dy, R + dx]
                if self.appetite(hc, hs) >= 0.5:
                    near_thing = (dy, dx, d[k])
                    self.last_thing = (dy, dx, self._feat(hc, hs))
                else:
                    refused = True
                    self.regret(dy, dx)             # not eating THAT: note it and go
        # which neighbour does the wave say to step to? Ties — and a wave with no
        # diagonal discount has whole rows of them — go to the neighbour that needs
        # the least turning, so a tie is never an excuse to veer
        # the wave has already charged a solid neighbour its leak (V there is leak x
        # the best of ITS neighbours); charging it again here made "led into a solid"
        # a branch that never ran — 1.4% of a value loses to any free cell — so no door
        # was ever pressed and the leak could not be taught. Measured 2026-09-17: 0
        # door presses in 24000 ticks over three worlds
        vals = np.array([V[R + dy, R + dx] for dy, dx in _NBRS])
        top = float(vals.max())
        if top > 1e-4:
            tied = [i for i in range(len(_NBRS)) if vals[i] >= top * (1.0 - 1e-3)]
            k = min(tied, key=lambda i: abs(
                (math.atan2(*_NBRS[i]) - heading + math.pi) % (2 * math.pi) - math.pi))
            self.last_target = _NBRS[k]
        else:
            self.last_target = None
        if near_thing is not None or (amps.max() > c.smell_here and self.last_target
                                      and not refused):
            # (never press on smell alone with a REFUSED thing in reach: the press
            # eats whatever is nearest, and that would be the thing just refused)
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
            # and does nothing at all to stone — which is what the hue is for: the
            # body's verdict on this press teaches the leak of this hue
            # ...and only pressed when FACING it: the verb acts on the cell ahead of
            # the nose, and a press thrown while still turning lands on the stone
            # beside the door and blames the door's hue for it (measured: learning
            # drove the door's leak DOWN, 0.5 -> 0.36, and cost 2.4 meals a life)
            facing = abs(err) < c.press_angle
            if facing:
                self.last_press = (dy, dx, self._feat(st[HUE_C, R + dy, R + dx],
                                                      st[HUE_S, R + dy, R + dx]))
            return (0.5 if abs(err) < c.slow_angle else 0.0), turn, (1.0 if facing else 0.0)
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
        self.state[BAD] *= f
        self._paint(vec, y, x, heading)
        src = self._sources()
        self.sources = src
        if self.rule is None:
            self._wave(src, c.subticks)
        else:
            self.rule.step(self.state, src, self._passable(), c.subticks)
        self.thoughts += 1
        self.last_command = self._read(heading, vec)
        return self.last_command


class AtlasMind:
    """The atlas wired into a life, on the same terms as `agent.Mind`: it thinks every
    `think_every` ticks, holds the command in between, and pays for its cells."""

    def __init__(self, cfg: Config | None = None, wcfg: WorldConfig | None = None,
                 rule=None):
        self.atlas = Atlas(cfg, wcfg, rule)
        self.cfg = self.atlas.cfg
        self.field: ScentField | None = None
        self.command = (0.0, 0.0, 0.0)
        self.sensors = None
        self._eaten = 0
        self._pending: list = []    # [tick eaten, features, (dy, dx) then, y, x then]

    def attach(self, w: World) -> None:
        self.field = ScentField(w.cfg.smell_radius)

    def panel(self, app):
        return Panel(app)

    def step(self, w: World) -> None:
        if self.field is None:
            self.attach(w)
        a = w.agent
        if w.tick % self.cfg.think_every == 0:
            self.sensors = sense(w, self.field)
            self.command = self.atlas.act(self.sensors, a.y, a.x, a.heading)
        poison_before = a.poison
        gave = w.deeds["door"] + w.deeds["break"] + w.deeds["push"]
        doors, idle = w.deeds["door"], w.deeds["idle"]
        w.step(*self.command)
        a.energy -= self.cfg.cell_cost * self.atlas.cells()
        self._digest(w, poison_before)
        press = self.atlas.last_press
        if press is not None and self.cfg.leak_lr > 0.0 and self.command[2] > 0.0:
            if w.deeds["door"] + w.deeds["break"] + w.deeds["push"] > gave:
                self.atlas.learn_leak(press[2], +1.0)
                # and the map learns the cell gave: a door is open, rubble is less.
                # Left solid until a ray happened to look, the next press landed on
                # bare ground, "nothing to do here", and the -1 went to the hue of
                # the thing that had just yielded (rubble and door share a bump)
                self.atlas.gave(press[0], press[1], 0.0 if w.deeds["door"] > doors else 0.6)
            elif w.deeds["idle"] > idle:
                self.atlas.learn_leak(press[2], -1.0)

    def _digest(self, w: World, poison_before: float) -> None:
        """Learning from what a meal did. The body knows it swallowed something (the
        deed) and, a little later, whether it burns (the poison channel of
        interoception rising). Those two facts, and the hue of what was within reach
        when it was eaten, are all the appetite perceptron is ever told."""
        a = w.agent
        if w.deeds["eat"] > self._eaten:
            self._eaten = w.deeds["eat"]
            lt = self.atlas.last_thing
            if lt is not None:
                self._pending.append([w.tick, lt[2], (lt[0], lt[1]), a.y, a.x])
        if not self._pending:
            return
        burned = a.poison > poison_before + 1e-9
        keep = []
        for m in self._pending:
            if burned:
                self.atlas.learn(m[1], -1.0)
                # and remember WHERE: the thing is gone, but its patch and its smell
                # are not, and the same kind may regrow there
                self.atlas.regret(int(round(m[3] - a.y)) + m[2][0],
                                  int(round(m[4] - a.x)) + m[2][1])
            elif w.tick - m[0] >= self.cfg.appetite_delay:
                self.atlas.learn(m[1], +1.0)
            else:
                keep.append(m)
        self._pending = keep


class Driver:
    """The atlas at a probe (`probes._run` calls `driver(vec, w)`). It reads the pose
    off the world — that is its proprioception, the one sense the vector lacks."""

    def __init__(self, cfg: Config | None = None, rule=None):
        self.atlas = None
        self.cfg = cfg
        self.rule = rule

    def __call__(self, vec, w):
        if self.atlas is None:
            self.atlas = Atlas(self.cfg, w.cfg, self.rule)
        a = w.agent
        return self.atlas.act(vec, a.y, a.x, a.heading)


def live(seed: int = 0, max_ticks: int = 20000, cfg: Config | None = None,
         wcfg: WorldConfig | None = None, frames: list | None = None,
         every: int = 12, rule=None) -> tuple[World, AtlasMind]:
    """One life, headless. A pure function of (cfg, seed, limits). With `frames` a
    list, an image of world + map is appended every `every` ticks."""
    w = World(seed=seed, cfg=wcfg or WorldConfig())
    mind = AtlasMind(cfg, w.cfg, rule)
    mind.attach(w)
    while w.agent.alive and w.tick < max_ticks:
        mind.step(w)
        if frames is not None and w.tick % every == 0:
            frames.append(frame(w, mind.atlas))
    return w, mind


# --- the lesson: the one exam that needs a life before it ---------------------------------

def poison_lesson(cfg: Config | None = None, ticks: int = 900, rule=None) -> list[dict]:
    """The poison arena TWICE with the same brain — a nightcap dead ahead, a berry off to
    the side, identical smell, different hue. The first time it has never been burned.
    The second time the map is wiped and the arena fresh, and all it carries over is
    its appetite. `bad` is whether it ate the nightcap; the lesson is the second row."""
    from cortex import probes
    at = None
    out = []
    for _ in range(2):
        w, sp = probes.build("poison")
        if at is None:
            at = Atlas(cfg, w.cfg, rule)
        else:
            at.state[:] = 0.0                   # a fresh arena: forget the map...
            at.cell = None                      # ...keep what the body taught
        mind = AtlasMind(cfg, w.cfg, rule)
        mind.atlas = at
        mind.attach(w)
        while w.agent.alive and w.tick < ticks:
            mind.step(w)
        out.append({"good": int(not sp["food"].ready), "bad": int(not sp["poison"].ready),
                    "verdicts": at.meals})
    return out


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
    ys, xs = np.nonzero(st[BAD] > 0.3)
    for y, x in zip(ys, xs):
        img[y, x] = [0.9, 0.1, 0.6]
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


class Panel:
    """The atlas's column in the live HUD (`python -m cortex.app --atlas`): the map,
    drawn as the agent holds it, plus the numbers that say what it is doing. Same slot
    the cortex brain's panel takes; the keys that mean nothing here are no-ops."""

    trace_on = False
    trace_i = 0
    picked = None

    def __init__(self, app):
        self.app = app

    def cycle_view(self) -> str:
        return "map"

    def slower(self) -> None:
        pass

    def faster(self) -> None:
        pass

    def advance_trace(self) -> None:
        pass

    def click(self, pos) -> bool:
        return False

    def draw(self, surf, x: int) -> None:
        import pygame
        from cortex.brainpanel import BRAIN_W
        at = self.app.mind.atlas
        font = self.app.font
        pw = BRAIN_W - 28
        scale = max(1, pw // at.side)
        img = map_image(at, scale)
        pic = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        y = 14
        surf.blit(font.render(f"ATLAS  r{at.R}  {at.cells()} cells  thought {at.thoughts}",
                              True, (210, 210, 220)), (x + 14, y))
        y += 22
        surf.blit(pic, (x + 14, y))
        pygame.draw.rect(surf, (40, 42, 52), pygame.Rect(x + 14, y, img.shape[1], img.shape[0]), 1)
        y += img.shape[0] + 10
        t, tr, act = at.last_command
        lines = [f"command  thrust {t:.2f}  turn {tr:+.2f}  {'E' if act else '-'}",
                 f"step     {at.last_target}",
                 f"value at centre {at.state[VALUE][at.R, at.R]:.3f}   max {at.state[VALUE].max():.3f}",
                 f"known free {int((at.state[FREE] > 0.5).sum())}   solid "
                 f"{int((at.state[SOLID] > 0.5).sum())}   things {int((at.state[THING] > 0.5).sum())}",
                 f"scent sources {int((at.state[SCENT] > 0.01).sum())}   bad "
                 f"{int((at.state[BAD] > 0.3).sum())}",
                 f"appetite w {np.round(at.appetite_w, 2)}  verdicts {at.meals}",
                 "",
                 "blue = the wave, grey = solid, yellow = smell,",
                 "coloured = a seen thing, green = the chosen step"]
        for ln in lines:
            surf.blit(font.render(ln, True, (170, 170, 185)), (x + 14, y))
            y += 18


# --- the command line ------------------------------------------------------------------

def _seeds(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(q) for q in s.split(",")]


def _one_life(args) -> dict:
    seed, ticks, cfg_d, rule = args
    from cortex.world import FOOD_ENERGY
    w, mind = live(seed, ticks, Config(**cfg_d), rule=rule)
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"seed": seed, "ticks": w.tick, "meals": int(sum(w.agent.ate)),
            "gained": gained, "walked": float(w.distance), "ate": list(w.agent.ate),
            "alive": bool(w.agent.alive)}


def lives(seeds, ticks: int = 20000, cfg: Config | None = None, jobs: int = 1,
          rule=None) -> list[dict]:
    """Whole lives over many worlds, in parallel when asked. Each is a pure function of
    (cfg, seed, ticks, rule), so the order they run in cannot matter."""
    cfg_d = (cfg or Config()).to_dict()
    args = [(int(s), int(ticks), cfg_d, rule) for s in seeds]
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
    p.add_argument("--poison", action="store_true",
                   help="the lesson: the poison arena twice with the same brain")
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
    if args.poison:
        first, second = poison_lesson(cfg)
        print(f"poison lesson  first visit: berry {first['good']} nightcap {first['bad']}"
              f"   second visit: berry {second['good']} nightcap {second['bad']}"
              f"   (verdicts {second['verdicts']})")
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
