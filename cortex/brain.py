"""The brain: a sheet of cells that thinks by relaxing.

Forked from `vivarium/nca.py` — the ideas are the same and were earned there — but
this is one BIG brain for ONE agent, not a population of small ones, and it plugs
into the cortex sensorium and its three continuous commands. See
docs/cortex-design.md; this file is stage 0, the running baseline everything later
is measured against.

What a brain is here:

  * a bounded G x G lattice with a heritable ALIVE MASK, so the tissue has a grown
    SHAPE rather than a rectangle;
  * every cell has a TYPE pointing at one of a small set of update RULES, so cells
    DIFFERENTIATE; each rule is a stack of layers of evolvable depth and width;
  * ORGANS: sensory cells hear one MODALITY, and which ray they hear is decided by
    their POSITION on the lattice (retinotopy) — so geometry does work and an "eye"
    region can form. Effector cells are where the action is read from. Signal has to
    travel organ -> tissue -> effector, which is the point of having a shape at all;
  * a HORMONE channel gates every cell's rule (neuromodulation), so the same tissue
    computes differently in different internal states;
  * within-life PLASTICITY: reward-modulated Hebbian deltas on the motor readout and
    on each rule's output layer. Reward is the change in energy, which is dense,
    honest and needs no shaping.

Thinking is `subticks` rounds of relaxation per world tick, so a signal can cross the
sheet before the action is read. Cell state PERSISTS between ticks: the medium is the
memory.

The genome is deliberately small and structural (shape, types, rule stacks) rather
than a wall of weights — docs/cortex-design.md wants an LLM to be able to read and
write it, and later for it to become a program for GROWING the sheet rather than the
sheet itself.
"""

from __future__ import annotations

import math

import numpy as np

# the three commands the world takes, exactly as a human's keyboard produces them
THRUST, TURN, INTERACT = 0, 1, 2
N_OUT = 3

MOD_ALL = 0                      # an organ that hears the whole sensor vector
_MIN_DEPTH = 2                   # a rule always has an input and an output layer
_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

try:                             # numba is an optional accelerator; numpy is the
    from numba import njit as _njit        # reference path and the only one without it
    from numba.typed import List as _NbList
    _NUMBA = True
except ImportError:                                  # pragma: no cover - env dependent
    _NUMBA = False


class Config:
    """Brain-shape constants. Deliberately separate from the world's Config: the world
    does not care how the agent thinks, and the brain does not care where food grows."""

    def __init__(self, **kw):
        self.grid = 64               # lattice side. 64 so an eye region CAN resolve
                                     # all 64 rays — retinotopy hands out ray slots by
                                     # column, so a narrower sheet would see double
        self.channels = 12           # state channels per cell (last one = the hormone)
        self.hidden = 24             # hidden width of a fresh rule
        self.start = 9               # side of the seed block of living cells
        self.subticks = 24           # relaxation rounds per world tick
        self.max_cells = 4096
        self.max_types = 6
        self.max_layers = 5
        self.max_hidden = 48
        self.min_hidden = 4
        self.organ_mods = 9          # 1..6 per-ray channels, 7 odour, 8 touch, 9 intero
        self.neuromod = True
        # within-life learning (see Phenotype.learn)
        self.plastic = True
        self.plastic_lr = 0.03
        self.plastic_forget = 0.002
        self.plastic_elig = 0.5
        self.plastic_clip = 1.0
        self.rule_plastic = True
        self.rule_lr = 0.006
        self.rule_clip = 0.3
        # mutation
        self.weight_mut = 0.08
        self.weight_mut_prob = 0.10
        self.grow_prob = 0.30
        self.shrink_prob = 0.06
        self.add_in_prob = 0.12
        self.del_in_prob = 0.03
        self.add_out_prob = 0.06
        self.del_out_prob = 0.02
        self.organ_mod_prob = 0.08
        self.retype_prob = 0.05
        self.split_prob = 0.04
        self.add_layer_prob = 0.05
        self.del_layer_prob = 0.015
        self.widen_prob = 0.06
        self.narrow_prob = 0.02
        self.subtick_mut_prob = 0.10
        self.subticks_min, self.subticks_max = 8, 48
        self.__dict__.update(kw)


def _fresh_rule(in_dim, H, C, depth, rng, scale=0.5) -> list[dict]:
    dims = [in_dim] + [H] * (depth - 1) + [C]
    return [{"W": rng.normal(0.0, scale / math.sqrt(a), (b, a)), "b": np.zeros(b)}
            for a, b in zip(dims[:-1], dims[1:])]


def _copy_rule(rule) -> list[dict]:
    return [{"W": L["W"].copy(), "b": L["b"].copy()} for L in rule]


class Genome:
    """Shape + cell types + rule stacks + the sensor/motor projections."""

    def __init__(self, n_in, mask, type_map, rules, P_in, P_out, b_out,
                 channels, hidden, subticks, in_mask, out_mask, organ_mod):
        self.n_in = int(n_in)
        self.n_out = N_OUT
        self.mask = mask.astype(bool)
        self.type_map = type_map.astype(np.int64)
        self.in_mask = in_mask.astype(bool)
        self.out_mask = out_mask.astype(bool)
        self.organ_mod = organ_mod.astype(np.int64)
        self.C = int(channels)
        self.H = int(hidden)
        self.rules = rules
        self.P_in = np.asarray(P_in, dtype=np.float64)      # (n_in, C) sensor -> drive
        self.P_out = np.asarray(P_out, dtype=np.float64)    # (C, N_OUT) pooled -> command
        self.b_out = np.asarray(b_out, dtype=np.float64)
        self.subticks = int(subticks)

    @property
    def G(self) -> int:
        return self.mask.shape[0]

    @property
    def n_types(self) -> int:
        return len(self.rules)

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.mask.copy(), self.type_map.copy(),
                      [_copy_rule(r) for r in self.rules], self.P_in.copy(),
                      self.P_out.copy(), self.b_out.copy(), self.C, self.H,
                      self.subticks, self.in_mask.copy(), self.out_mask.copy(),
                      self.organ_mod.copy())

    # --- the numbers the HUD and the LLM read -------------------------------------
    def cells(self) -> int:
        return int(self.mask.sum())

    def organs(self) -> tuple[int, int]:
        return int((self.in_mask & self.mask).sum()), int((self.out_mask & self.mask).sum())

    def types_used(self) -> int:
        return int(len(np.unique(self.type_map[self.mask]))) if self.mask.any() else 0

    def max_depth(self) -> int:
        return max((len(r) for r in self.rules), default=_MIN_DEPTH)

    def max_width(self) -> int:
        return max((int(r[0]["W"].shape[0]) for r in self.rules), default=self.H)

    def params(self) -> int:
        n = self.P_in.size + self.P_out.size + self.b_out.size
        for r in self.rules:
            for L in r:
                n += L["W"].size + L["b"].size
        return int(n)

    def to_dict(self) -> dict:
        return {"n_in": self.n_in, "C": self.C, "H": self.H, "subticks": self.subticks,
                "mask": self.mask.astype(int).tolist(),
                "type_map": self.type_map.tolist(),
                "in_mask": self.in_mask.astype(int).tolist(),
                "out_mask": self.out_mask.astype(int).tolist(),
                "organ_mod": self.organ_mod.tolist(),
                "rules": [[[L["W"].tolist(), L["b"].tolist()] for L in r]
                          for r in self.rules],
                "P_in": self.P_in.tolist(), "P_out": self.P_out.tolist(),
                "b_out": self.b_out.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        rules = [[{"W": np.array(W, dtype=np.float64), "b": np.array(b, dtype=np.float64)}
                  for W, b in r] for r in d["rules"]]
        return cls(int(d["n_in"]), np.array(d["mask"], dtype=bool),
                   np.array(d["type_map"]), rules, d["P_in"], d["P_out"], d["b_out"],
                   int(d["C"]), int(d["H"]), int(d["subticks"]),
                   np.array(d["in_mask"], dtype=bool), np.array(d["out_mask"], dtype=bool),
                   np.array(d["organ_mod"], dtype=np.int64))


def _start_mask(G: int, start: int) -> np.ndarray:
    m = np.zeros((G, G), dtype=bool)
    off = (G - start) // 2
    m[off:off + start, off:off + start] = True
    return m


def random_genome(n_in: int, rng, cfg: Config | None = None) -> Genome:
    """A seed brain wired as a small ORGANISM: sensory organs on the PERIMETER (each
    given a random modality, so retinotopy is engaged from birth), effectors at the
    CENTRE, interneurons in between. Sense at the edge, act from the core — so the
    shape has a job on the very first tick."""
    cfg = cfg or Config()
    G, C, H = cfg.grid, cfg.channels, cfg.hidden
    mask = _start_mask(G, cfg.start)
    type_map = np.zeros((G, G), dtype=np.int64)
    in_mask = np.zeros((G, G), dtype=bool)
    out_mask = np.zeros((G, G), dtype=bool)
    organ_mod = np.zeros((G, G), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    for y, x in zip(ys, xs):
        if y in (y0, y1) or x in (x0, x1):
            in_mask[y, x] = True
            organ_mod[y, x] = int(rng.integers(cfg.organ_mods + 1))
    out_mask[(y0 + y1) // 2, (x0 + x1) // 2] = True
    rules = [_fresh_rule(9 * C, H, C, _MIN_DEPTH, rng)]
    # Xavier on the sensor projection: 412 inputs summed into C channels would
    # saturate tanh instantly at N(0, 0.5) — the same regression that once broke NEAT
    # in the vivarium when the sensor vector got wide (see docs/lab-journal.md).
    P_in = rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, C))
    P_out = rng.normal(0.0, 0.5, (C, N_OUT))
    b_out = np.zeros(N_OUT)
    b_out[THRUST] += 0.5                                   # a nudge to get moving
    return Genome(n_in, mask, type_map, rules, P_in, P_out, b_out, C, H,
                  cfg.subticks, in_mask, out_mask, organ_mod)


# --- running a brain ----------------------------------------------------------------

def _apply_rule(x, rule) -> np.ndarray:
    for L in rule[:-1]:
        x = np.tanh(x @ L["W"].T + L["b"])
    return x @ rule[-1]["W"].T + rule[-1]["b"]


def _apply_rule_plastic(x, rule, dW_last, elig):
    for L in rule[:-1]:
        x = np.tanh(x @ L["W"].T + L["b"])
    y = x @ (rule[-1]["W"].T + dW_last) + rule[-1]["b"]
    elig += x.T @ y
    return y


if _NUMBA:
    @_njit(cache=True, nogil=True)
    def _relax_nb(st, drive, padF, centre, nbr, type_rows, Wts, bs, layer_cnt,
                  mod_ch, dW_last, elig_out, plastic, subticks):   # pragma: no cover
        """All `subticks` rounds as one kernel: scatter the alive rows into the padded
        buffer, gather each cell's 9-neighbour perception, push each TYPE's rows
        through its layer stack, then the residual tanh update with the hormone gate."""
        n, C = st.shape
        for _ in range(subticks):
            for i in range(n):
                for c in range(C):
                    padF[centre[i], c] = st[i, c]
            perc = np.empty((n, 9 * C))
            for i in range(n):
                for j in range(9):
                    r = nbr[i, j]
                    for c in range(C):
                        perc[i, j * C + c] = padF[r, c]
            delta = np.empty((n, C))
            li = 0
            for k in range(len(layer_cnt)):
                rows = type_rows[k]
                nl = layer_cnt[k]
                if rows.size > 0:
                    x = perc[rows]
                    for L in range(nl - 1):
                        x = np.tanh(x @ Wts[li + L] + bs[li + L])
                    last = li + nl - 1
                    if plastic == 1:
                        y = x @ (Wts[last] + dW_last[k]) + bs[last]
                        elig_out[k] += np.dot(x.T, y)
                    else:
                        y = x @ Wts[last] + bs[last]
                    for ii in range(rows.size):
                        for c in range(C):
                            delta[rows[ii], c] = y[ii, c]
                li += nl
            for i in range(n):
                gate = (1.0 + math.tanh(st[i, mod_ch])) if mod_ch >= 0 else 1.0
                for c in range(C):
                    st[i, c] = np.tanh(st[i, c] + gate * delta[i, c] + drive[i, c])
        return st


def build_listen(g: Genome, ys, xs, layout) -> np.ndarray:
    """Row i = which sensor indices alive-cell i hears. MOD_ALL hears everything; a
    per-ray modality hears the ONE ray its COLUMN selects (retinotopy); a static
    modality hears its index block. Non-organs hear nothing. No organs at all -> all
    cells hear all, so a brain is never struck deaf by mutation."""
    n = len(ys)
    M = np.zeros((n, g.n_in))
    organs = g.in_mask[ys, xs]
    if not organs.any() or layout is None:
        M[organs if organs.any() else slice(None)] = 1.0
        return M
    ray_ch, n_rays = layout["ray_ch"], layout["n_rays"]
    ray_mod, static_mod = layout["ray_mod_channel"], layout["static_mod_idx"]
    xmin, span = int(xs.min()), max(1, int(xs.max()) - int(xs.min()))
    for i in range(n):
        if not organs[i]:
            continue
        m = int(g.organ_mod[ys[i], xs[i]])
        if m in ray_mod:                                    # POSITION picks the ray
            r = int(round((int(xs[i]) - xmin) / span * (n_rays - 1)))
            idx = r * ray_ch + ray_mod[m]
            M[i, idx] = 1.0
        elif m in static_mod:
            M[i, static_mod[m]] = 1.0
        else:
            M[i, :] = 1.0
    return M


class Phenotype:
    """A runnable brain. `act(sensors)` injects the senses at the organs, relaxes the
    sheet, and reads the three commands from the effectors."""

    def __init__(self, genome: Genome, cfg: Config | None = None, layout=None):
        self.genome = genome
        self.cfg = cfg = cfg or Config()
        self.layout = layout
        G, C = genome.G, genome.C
        self._mod_ch = (C - 1) if cfg.neuromod else -1
        self.state = np.zeros((G, G, C))
        ys, xs = np.nonzero(genome.mask)
        self._ys, self._xs = ys, xs
        self._padF = np.zeros(((G + 2) * (G + 2), C))
        self._centre = (ys + 1) * (G + 2) + (xs + 1)
        self._nbr = np.stack([(ys + 1 + dy) * (G + 2) + (xs + 1 + dx)
                              for dy, dx in _OFFSETS], axis=1)
        types = genome.type_map[ys, xs]
        self._type_rows = [np.nonzero(types == k)[0] for k in range(genome.n_types)]
        self._M = build_listen(genome, ys, xs, layout)
        self._out_rows = np.nonzero(genome.out_mask[ys, xs])[0]
        if len(self._out_rows) == 0:
            self._out_rows = np.arange(len(ys))
        if _NUMBA:
            self._nb_rows = _NbList(self._type_rows)
            Wts, bs = _NbList(), _NbList()
            for r in genome.rules:
                for L in r:
                    Wts.append(np.ascontiguousarray(L["W"].T))
                    bs.append(np.ascontiguousarray(L["b"]))
            self._nb_W, self._nb_b = Wts, bs
            self._nb_cnt = np.array([len(r) for r in genome.rules], dtype=np.int64)
        # learning within one life
        self.dW = np.zeros((C, N_OUT))
        self._E = np.zeros((C, N_OUT))
        outs = [int(r[-1]["W"].shape[1]) for r in genome.rules]
        self._dW_rule = [np.zeros((o, C)) for o in outs]
        self._E_rule = [np.zeros((o, C)) for o in outs]
        self._elig = [np.zeros((o, C)) for o in outs]
        if _NUMBA:
            self._nb_dW = _NbList(self._dW_rule)
            self._nb_elig = _NbList(self._elig)
        self.last_input = None
        self.last_logits = np.zeros(N_OUT)
        self.last_command = (0.0, 0.0, 0.0)
        self.record = False                # inspector only: keep every sub-tick frame
        self.trace: list = []

    # --- the tick ------------------------------------------------------------------
    def act(self, sensors: np.ndarray) -> tuple[float, float, float]:
        g = self.genome
        drive = np.ascontiguousarray((self._M * sensors) @ g.P_in)
        n = len(self._ys)
        st = self.state[self._ys, self._xs]
        rp = 1 if self.cfg.rule_plastic else 0
        if rp:
            for e in self._elig:
                e[:] = 0.0
        if self.record:
            self.trace = []
        if _NUMBA and n:
            # when recording the "watch it think" trace the kernel is called ONE
            # sub-tick at a time, so the frames come from the same arithmetic —
            # watching an agent think must never change what it thinks
            for _ in range(g.subticks if self.record else 1):
                st = _relax_nb(st, drive, self._padF, self._centre, self._nbr,
                               self._nb_rows, self._nb_W, self._nb_b, self._nb_cnt,
                               self._mod_ch, self._nb_dW, self._nb_elig, rp,
                               1 if self.record else g.subticks)
                if self.record:
                    self._snap(st)
        else:
            for _ in range(g.subticks):
                self._padF[self._centre] = st
                perc = self._padF[self._nbr].reshape(n, 9 * g.C)
                delta = np.empty_like(st)
                for k, rows in enumerate(self._type_rows):
                    if len(rows):
                        delta[rows] = (_apply_rule_plastic(perc[rows], g.rules[k],
                                                           self._dW_rule[k], self._elig[k])
                                       if rp else _apply_rule(perc[rows], g.rules[k]))
                if self._mod_ch >= 0:
                    delta = delta * (1.0 + np.tanh(st[:, self._mod_ch]))[:, None]
                st = np.tanh(st + delta + drive)
                if self.record:
                    self._snap(st)
        if rp:
            for k in range(len(self._E_rule)):
                nrm = g.subticks * max(1, len(self._type_rows[k]))
                self._E_rule[k] *= self.cfg.plastic_elig
                self._E_rule[k] += self._elig[k] / nrm
        self.state[self._ys, self._xs] = st
        pooled = st[self._out_rows].mean(axis=0)
        eff = g.P_out + self.dW if self.cfg.plastic else g.P_out
        logits = pooled @ eff + g.b_out
        self.last_input, self.last_logits = sensors, logits
        # the three commands, in exactly the range World.step expects
        thrust = 1.0 / (1.0 + math.exp(-float(logits[THRUST])))
        turn = math.tanh(float(logits[TURN]))
        interact = 1.0 if logits[INTERACT] > 0.0 else 0.0
        if self.cfg.plastic:                               # credit what we just did
            self._E *= self.cfg.plastic_elig
            self._E[:, THRUST] += pooled * (thrust - 0.5) * 2.0
            self._E[:, TURN] += pooled * turn
            self._E[:, INTERACT] += pooled * (1.0 if interact else -1.0)
        self.last_command = (thrust, turn, interact)
        return thrust, turn, interact

    def _snap(self, st) -> None:
        full = np.zeros_like(self.state)
        full[self._ys, self._xs] = st
        self.trace.append(full)

    def learn(self, reward: float) -> None:
        """Three-factor update: the deltas always forget slowly, and whatever is still
        in the eligibility trace is reinforced when a reward lands. Reward here is the
        change in ENERGY, which is dense and needs no shaping — eating is positive,
        walking and poison are negative, so the sheet is pushed toward what fed it."""
        c = self.cfg
        if c.plastic:
            self.dW *= (1.0 - c.plastic_forget)
            if reward:
                self.dW += c.plastic_lr * reward * self._E
                np.clip(self.dW, -c.plastic_clip, c.plastic_clip, out=self.dW)
        if c.rule_plastic:
            for k in range(len(self._dW_rule)):
                self._dW_rule[k] *= (1.0 - c.plastic_forget)
                if reward:
                    self._dW_rule[k] += c.rule_lr * reward * self._E_rule[k]
                    np.clip(self._dW_rule[k], -c.rule_clip, c.rule_clip,
                            out=self._dW_rule[k])

    def plastic_norm(self) -> float:
        return float(np.abs(self.dW).sum())


# --- mutation -----------------------------------------------------------------------
# The genome is STRUCTURAL, so mutation is mostly structural too: grow or lose a cell,
# turn a cell into an organ or give it another modality, re-type it, split a type into
# a new rule, deepen or widen a rule. Weight jitter rides along. Appearance is biased
# over disappearance so the sheet trends toward more tissue and richer senses; the
# energy cost of carrying cells is what argues the other way.

def _jitter(arr, rng, prob, std) -> None:
    arr += rng.normal(0.0, std, arr.shape) * (rng.random(arr.shape) < prob)


def _grow_candidates(mask) -> list:
    G = mask.shape[0]
    cand = set()
    for y, x in np.argwhere(mask):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < G and 0 <= nx < G and not mask[ny, nx]:
                cand.add((ny, nx))
    return sorted(cand)


def _neighbour_type(mask, type_map, y, x, rng) -> int:
    G = mask.shape[0]
    types = [int(type_map[ny, nx])
             for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
             for ny, nx in [(y + dy, x + dx)]
             if 0 <= ny < G and 0 <= nx < G and mask[ny, nx]]
    return int(rng.choice(types)) if types else 0


def _identity_layer(H, rng, jitter=0.05) -> dict:
    """A near-IDENTITY layer, so DEEPENING a rule barely changes it at first
    (tanh(Ix) ~ x for small x) — depth grows without wrecking a working brain."""
    return {"W": np.eye(H) + rng.normal(0.0, jitter, (H, H)), "b": np.zeros(H)}


def _resize_width(rule, grow, rng, cfg) -> None:
    """Change a rule's hidden width by one unit. A GROWN unit gets ZERO outgoing
    weights, so extra capacity changes nothing until something learns to use it."""
    w, n = int(rule[0]["W"].shape[0]), len(rule)
    if grow and w >= cfg.max_hidden:
        return
    if not grow and w <= cfg.min_hidden:
        return
    j = -1 if grow else int(rng.integers(w))
    for i, L in enumerate(rule):
        if grow:
            if i < n - 1:
                L["W"] = np.vstack([L["W"], rng.normal(0.0, 0.1, (1, L["W"].shape[1]))])
                L["b"] = np.append(L["b"], 0.0)
            if i > 0:
                L["W"] = np.hstack([L["W"], np.zeros((L["W"].shape[0], 1))])
        else:
            if i < n - 1:
                L["W"] = np.delete(L["W"], j, axis=0)
                L["b"] = np.delete(L["b"], j)
            if i > 0:
                L["W"] = np.delete(L["W"], j, axis=1)


def _split_type(g, rng, cfg) -> None:
    """Duplicate a type's rule with a jitter and move half its cells to it —
    differentiation from within, rather than a type appearing out of nowhere."""
    src = int(rng.integers(g.n_types))
    new = _copy_rule(g.rules[src])
    for L in new:
        _jitter(L["W"], rng, 1.0, cfg.weight_mut)
    g.rules.append(new)
    idx = g.n_types - 1
    for y, x in np.argwhere(g.mask & (g.type_map == src)):
        if rng.random() < 0.5:
            g.type_map[y, x] = idx


def mutate(genome, rng, cfg=None):
    """A mutated copy. Every step is bounded and mostly silent on arrival, so it
    rarely destroys a working brain outright; selection between lives decides whether
    the new tissue earns its keep."""
    cfg = cfg or Config()
    g = genome.copy()
    for r in g.rules:
        for L in r:
            _jitter(L["W"], rng, cfg.weight_mut_prob, cfg.weight_mut)
            _jitter(L["b"], rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.P_in, rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.P_out, rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.b_out, rng, cfg.weight_mut_prob, cfg.weight_mut)

    if rng.random() < cfg.subtick_mut_prob:                 # heritable THINK TIME
        g.subticks = int(min(cfg.subticks_max, max(cfg.subticks_min,
                             g.subticks + int(rng.integers(-3, 4)))))

    n_alive = g.cells()
    if n_alive < cfg.max_cells and rng.random() < cfg.grow_prob:
        cand = _grow_candidates(g.mask)
        if cand:
            y, x = cand[int(rng.integers(len(cand)))]
            g.mask[y, x] = True                             # born an INTERNEURON
            g.type_map[y, x] = _neighbour_type(g.mask, g.type_map, y, x, rng)
            g.in_mask[y, x] = g.out_mask[y, x] = False
    elif n_alive > 1 and rng.random() < cfg.shrink_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.mask[y, x] = False
        g.in_mask[y, x] = g.out_mask[y, x] = False
        g.organ_mod[y, x] = 0

    if rng.random() < cfg.add_in_prob:                      # a cell becomes an ORGAN
        cand = np.argwhere(g.mask & ~g.in_mask)
        if len(cand):
            y, x = cand[int(rng.integers(len(cand)))]
            g.in_mask[y, x] = True
            g.organ_mod[y, x] = int(rng.integers(cfg.organ_mods + 1))
    elif rng.random() < cfg.del_in_prob:
        organs = np.argwhere(g.mask & g.in_mask)
        if len(organs):
            y, x = organs[int(rng.integers(len(organs)))]
            g.in_mask[y, x] = False
            g.organ_mod[y, x] = 0
    if rng.random() < cfg.add_out_prob:                     # ...or an EFFECTOR
        cand = np.argwhere(g.mask & ~g.out_mask)
        if len(cand):
            y, x = cand[int(rng.integers(len(cand)))]
            g.out_mask[y, x] = True
    elif rng.random() < cfg.del_out_prob:
        effs = np.argwhere(g.mask & g.out_mask)
        if len(effs) > 1:
            y, x = effs[int(rng.integers(len(effs)))]
            g.out_mask[y, x] = False
    if rng.random() < cfg.organ_mod_prob:                   # an organ SPECIALISES
        organs = np.argwhere(g.mask & g.in_mask)
        if len(organs):
            y, x = organs[int(rng.integers(len(organs)))]
            g.organ_mod[y, x] = int(rng.integers(cfg.organ_mods + 1))

    if g.n_types > 1 and rng.random() < cfg.retype_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.type_map[y, x] = int(rng.integers(g.n_types))
    if g.n_types < cfg.max_types and rng.random() < cfg.split_prob:
        _split_type(g, rng, cfg)

    if rng.random() < cfg.add_layer_prob:                   # DEPTH
        rule = g.rules[int(rng.integers(g.n_types))]
        if len(rule) < cfg.max_layers:
            rule.insert(len(rule) - 1, _identity_layer(int(rule[-1]["W"].shape[1]), rng))
    if rng.random() < cfg.del_layer_prob:
        k = int(rng.integers(g.n_types))
        if len(g.rules[k]) > _MIN_DEPTH:
            del g.rules[k][int(rng.integers(1, len(g.rules[k]) - 1))]
    if rng.random() < cfg.widen_prob:                       # WIDTH
        _resize_width(g.rules[int(rng.integers(g.n_types))], True, rng, cfg)
    if rng.random() < cfg.narrow_prob:
        _resize_width(g.rules[int(rng.integers(g.n_types))], False, rng, cfg)
    return g
