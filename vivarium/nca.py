"""NCA brain: an agent's brain is itself a small CELLULAR AUTOMATON.

Where `neat.py` grows a graph of scalar neurons, here every agent's brain is a
tiny multi-channel grid (a Neural Cellular Automaton). Cells sit on a bounded
G x G lattice; a heritable MASK says which cells are ALIVE, so the brain has an
arbitrary GROWN SHAPE (start 2x2, mutation adds/removes a cell). Signals injected
at the senses relax across the grid over several sub-ticks before a motor decision
is read out; cell state persists between game ticks, so the medium has memory.

v1 — cell TYPES: the genome carries a SET of update rules (small MLPs over the 3x3
neighbourhood), and every cell has a TYPE index pointing at one, so cells
DIFFERENTIATE. A cell's type is heritable (grown cells copy a neighbour's type);
mutation can re-type a cell or SPLIT a type into a new one (clone+jitter a rule,
hand it half the cells) — that is how a diversity of nets-in-cells emerges. v0 is
the single-type (K=1) special case. The sensor/motor projections are shared across
types (global wiring). Still ahead (v2): NEAT-style topology growth inside a rule,
sensory organs on the shape's edge.

Sensor/motor wiring is size-invariant (survives growth): the sensor vector is
projected to a per-cell DRIVE broadcast into every alive cell each sub-tick, and
the action is read from the mean-POOLED cell state through a motor projection. At
one cell the whole thing degrades to a plain recurrent net.

The module mirrors neat.py's public surface (Genome / Phenotype / random_genome /
mutate / crossover + action constants) so the World can use either brain.
"""

from __future__ import annotations

import numpy as np

# actions (outputs) — identical ids to neat.py so the World's step logic is shared
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2
N_ACTIONS = 3

# the 3x3 neighbourhood order (self included), used to build the perception vector
_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class Genome:
    """A CA brain: a heritable alive-MASK and per-cell TYPE-map over a G x G lattice,
    a SET of update rules (one per type), and the shared sensor/motor projections.
    Fixed (G, C, H) for a run; the number of types varies. Rule arrays carry a
    leading type axis: W1 (K,H,9C), b1 (K,H), W2 (K,C,H), b2 (K,C)."""

    def __init__(self, n_in, n_out, mask, type_map, W1, b1, W2, b2,
                 P_in, P_out, b_out, channels, hidden, hue=0.0):
        self.kind = "nca"
        self.n_in = n_in
        self.n_out = n_out
        self.mask = mask.astype(bool)               # (G, G) which cells are alive
        self.type_map = type_map.astype(np.int64)   # (G, G) each cell's rule index
        self.C = channels
        self.H = hidden
        self.W1 = np.asarray(W1, dtype=np.float64)  # (K, H, 9C)
        self.b1 = np.asarray(b1, dtype=np.float64)  # (K, H)
        self.W2 = np.asarray(W2, dtype=np.float64)  # (K, C, H)
        self.b2 = np.asarray(b2, dtype=np.float64)  # (K, C)
        self.P_in = np.asarray(P_in, dtype=np.float64)    # (n_in, C)  sensor -> drive
        self.P_out = np.asarray(P_out, dtype=np.float64)  # (C, n_out) pooled -> logits
        self.b_out = np.asarray(b_out, dtype=np.float64)  # (n_out,)
        self.hue = hue                              # heritable lineage colour

    @property
    def G(self) -> int:
        return self.mask.shape[0]

    @property
    def n_types(self) -> int:
        return self.W1.shape[0]

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.n_out, self.mask.copy(), self.type_map.copy(),
                      self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(),
                      self.P_in.copy(), self.P_out.copy(), self.b_out.copy(),
                      self.C, self.H, self.hue)

    def max_node_id(self) -> int:
        # NCA has no node-id bookkeeping; a harmless value for the World's shared
        # _next_node_id logic (only NEAT actually uses it).
        return self.n_in + self.n_out

    def complexity(self) -> tuple[int, int]:
        """(alive cells, distinct types IN USE) — the HUD's two brain numbers."""
        used = len(np.unique(self.type_map[self.mask])) if self.mask.any() else 0
        return int(self.mask.sum()), int(used)

    def to_dict(self) -> dict:
        return {"brain": "nca", "n_in": self.n_in, "n_out": self.n_out,
                "C": self.C, "H": self.H, "hue": self.hue,
                "mask": self.mask.astype(int).tolist(),
                "type_map": self.type_map.tolist(),
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                "P_in": self.P_in.tolist(), "P_out": self.P_out.tolist(),
                "b_out": self.b_out.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(int(d["n_in"]), int(d["n_out"]), np.array(d["mask"], dtype=bool),
                   np.array(d["type_map"]), d["W1"], d["b1"], d["W2"], d["b2"],
                   d["P_in"], d["P_out"], d["b_out"], int(d["C"]), int(d["H"]),
                   float(d.get("hue", 0.0)))

    def key(self) -> bytes:
        """Canonical fingerprint of shape + types + rules (rounded)."""
        parts = [self.mask.astype(np.int8).tobytes(),
                 (self.type_map * self.mask).astype(np.int16).tobytes()]
        for a in (self.W1, self.b1, self.W2, self.b2, self.P_in, self.P_out, self.b_out):
            parts.append(np.round(a, 1).tobytes())
        return b"|".join(parts)


def _start_mask(G: int, start: int) -> np.ndarray:
    """A `start` x `start` alive block at the grid centre — the seed shape."""
    mask = np.zeros((G, G), dtype=bool)
    off = (G - start) // 2
    mask[off:off + start, off:off + start] = True
    return mask


def random_genome(n_in, n_out, rng, forward_bias=0.0, cfg=None) -> Genome:
    """A minimal starting CA brain: a `nca_start`-square alive block, a SINGLE cell
    type (K=1) with small random rule weights, and a FORWARD motor bias so fresh
    brains tend to move (the bootstrap prior)."""
    G = getattr(cfg, "nca_grid", 12)
    C = getattr(cfg, "nca_channels", 8)
    H = getattr(cfg, "nca_hidden", 16)
    start = getattr(cfg, "nca_start", 2)
    mask = _start_mask(G, start)
    type_map = np.zeros((G, G), dtype=np.int64)
    s = 0.5
    W1 = rng.normal(0.0, s, (1, H, 9 * C))
    b1 = np.zeros((1, H))
    W2 = rng.normal(0.0, s, (1, C, H))
    b2 = np.zeros((1, C))
    P_in = rng.normal(0.0, s, (n_in, C))
    P_out = rng.normal(0.0, s, (C, n_out))
    b_out = np.zeros(n_out)
    b_out[FORWARD] += forward_bias
    return Genome(n_in, n_out, mask, type_map, W1, b1, W2, b2, P_in, P_out, b_out,
                  C, H, hue=float(rng.random()))


class Phenotype:
    """Runnable CA brain. Holds the persistent grid state; `act` injects the senses
    as a drive, relaxes the per-type rules a few sub-ticks so signals cross the grid,
    then reads the action from the pooled cell state."""

    def __init__(self, genome: Genome, subticks: int = 32):
        self.genome = genome
        self.subticks = subticks
        G, C = genome.G, genome.C
        self.state = np.zeros((G, G, C), dtype=np.float64)
        self._m = genome.mask[:, :, None]           # (G, G, 1) broadcast mask
        # per-type boolean cell masks, so each rule is applied to its own cells
        self._type_sel = [genome.mask & (genome.type_map == k)
                          for k in range(genome.n_types)]
        self.last_input = None
        self.last_logits = None
        # when `record` is on (set only for the inspected agent), act() keeps a per
        # sub-tick TRACE of the grid state so the viewer can animate the brain
        # thinking; `show_state`, if set, is what the inspector draws instead of the
        # live final state (one trace frame). Neither affects the computation.
        self.record = False
        self.trace: list = []
        self.show_state = None

    def _perceive(self, s: np.ndarray) -> np.ndarray:
        """3x3 neighbourhood of every cell, BOUNDED (zero-padded, no torus wrap),
        with dead cells contributing nothing -> (G, G, 9C)."""
        G = self.genome.G
        sm = s * self._m                                     # dead cells = 0
        pad = np.pad(sm, ((1, 1), (1, 1), (0, 0)))
        blocks = [pad[1 + dy:1 + dy + G, 1 + dx:1 + dx + G, :] for dy, dx in _OFFSETS]
        return np.concatenate(blocks, axis=-1)

    def act(self, inputs: np.ndarray) -> int:
        g = self.genome
        drive = inputs @ g.P_in                              # (C,) sensory current
        st = self.state
        if self.record:
            self.trace = []
        single = g.n_types == 1                              # common case: fast path
        for _ in range(self.subticks):
            perc = self._perceive(st)                        # (G, G, 9C)
            if single:                                       # one rule over the whole grid
                h = np.tanh(perc @ g.W1[0].T + g.b1[0])      # (G, G, H)
                delta = h @ g.W2[0].T + g.b2[0]              # (G, G, C)
            else:                                            # each type runs its own rule
                delta = np.zeros_like(st)
                for k, sel in enumerate(self._type_sel):
                    if not sel.any():
                        continue
                    hk = np.tanh(perc[sel] @ g.W1[k].T + g.b1[k])   # (n_k, H)
                    delta[sel] = hk @ g.W2[k].T + g.b2[k]           # (n_k, C)
            st = np.tanh(st + delta + drive)                 # residual + clamped
            st = st * self._m                                # dead cells stay 0
            if self.record:
                self.trace.append(st.copy())                 # one frame per sub-tick
        self.state = st
        pooled = st[g.mask].mean(axis=0)                     # (C,) mean over alive
        logits = pooled @ g.P_out + g.b_out                  # (n_out,)
        self.last_input = inputs
        self.last_logits = logits
        return int(np.argmax(logits))


def _jitter(arr: np.ndarray, rng, prob: float, std: float) -> None:
    """In-place per-element gaussian jitter, each element perturbed with `prob`."""
    arr += rng.normal(0.0, std, arr.shape) * (rng.random(arr.shape) < prob)


def _grow_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    """Dead cells 4-adjacent to an alive cell — where the blob can grow next."""
    G = mask.shape[0]
    cand = set()
    for y, x in np.argwhere(mask):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < G and 0 <= nx < G and not mask[ny, nx]:
                cand.add((ny, nx))
    return sorted(cand)


def _neighbour_type(mask, type_map, y, x, rng) -> int:
    """The type of a random alive 4-neighbour of (y, x) — so a grown cell inherits a
    LOCAL type and types form spatial regions."""
    G = mask.shape[0]
    types = [int(type_map[ny, nx])
             for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
             for ny, nx in [(y + dy, x + dx)]
             if 0 <= ny < G and 0 <= nx < G and mask[ny, nx]]
    return int(rng.choice(types)) if types else 0


def mutate(genome: Genome, rng, cfg, next_node_id: int):
    """Return (mutated copy, next_node_id). Weight jitter always; structural mutation
    grows/shrinks the SHAPE, re-types a cell, or SPLITS a type into a new rule — so
    both the brain's size and its cell-type composition evolve. `next_node_id` is
    passed through untouched (NCA keeps no node ids)."""
    g = genome.copy()
    if rng.random() < cfg.hue_drift_prob:
        g.hue = (g.hue + float(rng.normal(0.0, cfg.hue_drift))) % 1.0
    for arr in (g.W1, g.b1, g.W2, g.b2, g.P_in, g.P_out, g.b_out):
        _jitter(arr, rng, cfg.weight_mut_prob, cfg.weight_mut)

    n_alive = int(g.mask.sum())
    if n_alive < cfg.nca_max_cells and rng.random() < cfg.nca_grow_prob:
        cand = _grow_candidates(g.mask)
        if cand:
            y, x = cand[int(rng.integers(len(cand)))]
            g.mask[y, x] = True
            g.type_map[y, x] = _neighbour_type(g.mask, g.type_map, y, x, rng)
    elif n_alive > 1 and rng.random() < cfg.nca_shrink_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.mask[y, x] = False

    # re-type a cell: hand one alive cell to a different existing type
    if g.n_types > 1 and rng.random() < cfg.nca_retype_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.type_map[y, x] = int(rng.integers(g.n_types))

    # split a type: clone a rule (jittered) into a new type, reassign ~half its cells
    if g.n_types < cfg.nca_max_types and rng.random() < cfg.nca_split_prob:
        g = _split_type(g, rng, cfg)
    return g, next_node_id


def _split_type(g: Genome, rng, cfg) -> Genome:
    """Duplicate an existing type's rule (with a jitter) as a NEW type, and move a
    random half of that type's cells to it — differentiation from within."""
    src = int(rng.integers(g.n_types))
    new_rule = (g.W1[src:src + 1].copy(), g.b1[src:src + 1].copy(),
                g.W2[src:src + 1].copy(), g.b2[src:src + 1].copy())
    for arr in new_rule:
        _jitter(arr, rng, 1.0, cfg.weight_mut)
    g.W1 = np.concatenate([g.W1, new_rule[0]], axis=0)
    g.b1 = np.concatenate([g.b1, new_rule[1]], axis=0)
    g.W2 = np.concatenate([g.W2, new_rule[2]], axis=0)
    g.b2 = np.concatenate([g.b2, new_rule[3]], axis=0)
    new_idx = g.n_types - 1
    cells = np.argwhere(g.mask & (g.type_map == src))
    for y, x in cells:
        if rng.random() < 0.5:
            g.type_map[y, x] = new_idx
    return g


def crossover(g_fit: Genome, g_other: Genome, rng) -> Genome:
    """Child genome: the SHAPE (mask), TYPE-map and type count come from the fitter
    parent; rules shared by index with the other parent are an ELEMENTWISE random mix
    (types the fitter alone has are inherited whole), as are the projections. Colour
    from a random parent."""
    def mix(a, b):
        return np.where(rng.random(a.shape) < 0.5, a, b)

    kf, ko = g_fit.n_types, g_other.n_types
    k = min(kf, ko)
    W1, b1, W2, b2 = (g_fit.W1.copy(), g_fit.b1.copy(),
                      g_fit.W2.copy(), g_fit.b2.copy())
    W1[:k] = mix(g_fit.W1[:k], g_other.W1[:k])
    b1[:k] = mix(g_fit.b1[:k], g_other.b1[:k])
    W2[:k] = mix(g_fit.W2[:k], g_other.W2[:k])
    b2[:k] = mix(g_fit.b2[:k], g_other.b2[:k])
    P_in = mix(g_fit.P_in, g_other.P_in)
    P_out = mix(g_fit.P_out, g_other.P_out)
    b_out = mix(g_fit.b_out, g_other.b_out)
    hue = g_fit.hue if rng.random() < 0.5 else g_other.hue
    return Genome(g_fit.n_in, g_fit.n_out, g_fit.mask.copy(), g_fit.type_map.copy(),
                  W1, b1, W2, b2, P_in, P_out, b_out, g_fit.C, g_fit.H, hue=hue)
