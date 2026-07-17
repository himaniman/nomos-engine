"""NCA-lite: a brain that is itself a small CELLULAR AUTOMATON.

Where `neat.py` grows a graph of scalar neurons, here every agent's brain is a
tiny multi-channel grid (a Neural Cellular Automaton). Cells sit on a bounded
G x G lattice; a heritable MASK says which cells are ALIVE, so the brain has an
arbitrary GROWN SHAPE (start 2x2, mutation adds/removes a cell) rather than a
fixed square. All alive cells run ONE shared update rule — a small MLP over the
3x3 neighbourhood — so a signal injected at the senses relaxes across the grid
over several sub-ticks before a motor decision is read out. State persists
between game ticks, so the medium has memory for free.

Sensor/motor wiring is size-invariant (survives growth): the sensor vector is
projected to a per-cell DRIVE broadcast into every alive cell each sub-tick, and
the action is read from the mean-POOLED cell state through a motor projection.
At one cell the whole thing degrades to a plain recurrent net; with many cells it
is a spatial computer.

This is v0 of DESIGN.md's "grid + alive-mask" plan: single rule, evolvable
weights + shape. Cell TYPES (a SET of rules) and NEAT-inside-the-rule come later.
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
    """A CA brain: a heritable alive-MASK over a G x G lattice plus the weights of
    one shared update rule and the sensor/motor projections. Fixed (G, C, H) for a
    run, so every genome has the same array shapes — crossover is elementwise and
    the whole thing serializes cleanly."""

    def __init__(self, n_in: int, n_out: int, mask: np.ndarray, W1, b1, W2, b2,
                 P_in, P_out, b_out, channels: int, hidden: int, hue: float = 0.0):
        self.kind = "nca"
        self.n_in = n_in
        self.n_out = n_out
        self.mask = mask.astype(bool)            # (G, G) which cells are alive
        self.C = channels
        self.H = hidden
        # rule: perception (9*C) -> hidden (H) -> channel delta (C)
        self.W1 = np.asarray(W1, dtype=np.float64)     # (H, 9C)
        self.b1 = np.asarray(b1, dtype=np.float64)     # (H,)
        self.W2 = np.asarray(W2, dtype=np.float64)     # (C, H)
        self.b2 = np.asarray(b2, dtype=np.float64)     # (C,)
        # sensor -> per-cell drive, and pooled state -> action logits
        self.P_in = np.asarray(P_in, dtype=np.float64)   # (n_in, C)
        self.P_out = np.asarray(P_out, dtype=np.float64)  # (C, n_out)
        self.b_out = np.asarray(b_out, dtype=np.float64)  # (n_out,)
        # a HERITABLE lineage hue (0..1), like neat.Genome — colour = lineage
        self.hue = hue

    @property
    def G(self) -> int:
        return self.mask.shape[0]

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.n_out, self.mask.copy(), self.W1.copy(),
                      self.b1.copy(), self.W2.copy(), self.b2.copy(),
                      self.P_in.copy(), self.P_out.copy(), self.b_out.copy(),
                      self.C, self.H, self.hue)

    def max_node_id(self) -> int:
        # NCA has no node-id bookkeeping; return a harmless value for the World's
        # shared _next_node_id logic (which only NEAT actually uses).
        return self.n_in + self.n_out

    def complexity(self) -> tuple[int, int]:
        """(alive cells, channels) — the HUD prints these where NEAT shows
        (hidden, connections)."""
        return int(self.mask.sum()), self.C

    def to_dict(self) -> dict:
        return {"brain": "nca", "n_in": self.n_in, "n_out": self.n_out,
                "C": self.C, "H": self.H, "hue": self.hue,
                "mask": self.mask.astype(int).tolist(),
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                "P_in": self.P_in.tolist(), "P_out": self.P_out.tolist(),
                "b_out": self.b_out.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(int(d["n_in"]), int(d["n_out"]), np.array(d["mask"], dtype=bool),
                   d["W1"], d["b1"], d["W2"], d["b2"], d["P_in"], d["P_out"],
                   d["b_out"], int(d["C"]), int(d["H"]), float(d.get("hue", 0.0)))

    def key(self) -> bytes:
        """Canonical fingerprint of shape + rule (rounded), for naming/colour."""
        parts = [self.mask.astype(np.int8).tobytes()]
        for a in (self.W1, self.b1, self.W2, self.b2, self.P_in, self.P_out, self.b_out):
            parts.append(np.round(a, 1).tobytes())
        return b"|".join(parts)


def _start_mask(G: int, start: int, rng: np.random.Generator) -> np.ndarray:
    """A `start` x `start` alive block placed at the grid centre — the seed shape
    every fresh brain grows from."""
    mask = np.zeros((G, G), dtype=bool)
    off = (G - start) // 2
    mask[off:off + start, off:off + start] = True
    return mask


def random_genome(n_in: int, n_out: int, rng: np.random.Generator,
                  forward_bias: float = 0.0, cfg=None) -> Genome:
    """A minimal starting CA brain: a `nca_start`-square alive block, small random
    rule weights, and a FORWARD motor bias so fresh brains tend to move (the
    bootstrap prior, mirroring neat.random_genome's forward_bias)."""
    G = getattr(cfg, "nca_grid", 12)
    C = getattr(cfg, "nca_channels", 8)
    H = getattr(cfg, "nca_hidden", 16)
    start = getattr(cfg, "nca_start", 2)
    mask = _start_mask(G, start, rng)
    s = 0.5
    W1 = rng.normal(0.0, s, (H, 9 * C))
    b1 = np.zeros(H)
    W2 = rng.normal(0.0, s, (C, H))
    b2 = np.zeros(C)
    P_in = rng.normal(0.0, s, (n_in, C))
    P_out = rng.normal(0.0, s, (C, n_out))
    b_out = np.zeros(n_out)
    b_out[FORWARD] += forward_bias
    return Genome(n_in, n_out, mask, W1, b1, W2, b2, P_in, P_out, b_out, C, H,
                  hue=float(rng.random()))


class Phenotype:
    """Runnable CA brain. Holds the persistent grid state; `act` injects the senses
    as a drive, relaxes the rule a few sub-ticks so signals cross the grid, then
    reads the action from the pooled cell state."""

    def __init__(self, genome: Genome, subticks: int = 32):
        self.genome = genome
        self.subticks = subticks
        G, C = genome.G, genome.C
        self.state = np.zeros((G, G, C), dtype=np.float64)
        self._m = genome.mask[:, :, None]        # (G, G, 1) broadcast mask

    def _perceive(self, s: np.ndarray) -> np.ndarray:
        """3x3 neighbourhood of every cell, BOUNDED (zero-padded, no torus wrap),
        with dead cells contributing nothing -> (G, G, 9C)."""
        G, C = self.genome.G, self.genome.C
        sm = s * self._m                                     # dead cells = 0
        pad = np.pad(sm, ((1, 1), (1, 1), (0, 0)))
        blocks = [pad[1 + dy:1 + dy + G, 1 + dx:1 + dx + G, :] for dy, dx in _OFFSETS]
        return np.concatenate(blocks, axis=-1)

    def act(self, inputs: np.ndarray) -> int:
        g = self.genome
        drive = inputs @ g.P_in                              # (C,) sensory current
        st = self.state
        for _ in range(self.subticks):
            perc = self._perceive(st)                        # (G, G, 9C)
            h = np.tanh(perc @ g.W1.T + g.b1)                # (G, G, H)
            delta = h @ g.W2.T + g.b2                        # (G, G, C)
            st = np.tanh(st + delta + drive)                 # residual + clamped
            st = st * self._m                                # dead cells stay 0
        self.state = st
        pooled = st[g.mask].mean(axis=0)                     # (C,) mean over alive
        logits = pooled @ g.P_out + g.b_out                  # (n_out,)
        return int(np.argmax(logits))


def _jitter(arr: np.ndarray, rng, prob: float, std: float) -> None:
    """In-place per-element gaussian jitter, each element perturbed with `prob`."""
    noise = rng.normal(0.0, std, arr.shape) * (rng.random(arr.shape) < prob)
    arr += noise


def mutate(genome: Genome, rng: np.random.Generator, cfg, next_node_id: int):
    """Return (mutated copy, next_node_id). Weight jitter always; structural
    mutation GROWS the shape (activate a dead cell adjacent to the alive blob) or
    SHRINKS it (deactivate one) at low rates, so the brain's size/shape evolves.
    `next_node_id` is passed through untouched (NCA keeps no node ids)."""
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
    elif n_alive > 1 and rng.random() < cfg.nca_shrink_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.mask[y, x] = False
    return g, next_node_id


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


def crossover(g_fit: Genome, g_other: Genome, rng: np.random.Generator) -> Genome:
    """Child genome: rule/projection weights are an ELEMENTWISE random mix of the
    two parents (shapes match within a run); the SHAPE (alive mask) is inherited
    from the fitter parent. Colour inherited from a random parent."""
    def mix(a, b):
        pick = rng.random(a.shape) < 0.5
        return np.where(pick, a, b)

    W1 = mix(g_fit.W1, g_other.W1)
    b1 = mix(g_fit.b1, g_other.b1)
    W2 = mix(g_fit.W2, g_other.W2)
    b2 = mix(g_fit.b2, g_other.b2)
    P_in = mix(g_fit.P_in, g_other.P_in)
    P_out = mix(g_fit.P_out, g_other.P_out)
    b_out = mix(g_fit.b_out, g_other.b_out)
    hue = g_fit.hue if rng.random() < 0.5 else g_other.hue
    return Genome(g_fit.n_in, g_fit.n_out, g_fit.mask.copy(), W1, b1, W2, b2,
                  P_in, P_out, b_out, g_fit.C, g_fit.H, hue=hue)
