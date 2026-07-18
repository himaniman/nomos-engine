"""NCA brain: an agent's brain is itself a small CELLULAR AUTOMATON.

Where `neat.py` grows a graph of scalar neurons, here every agent's brain is a
tiny multi-channel grid (a Neural Cellular Automaton). Cells sit on a bounded
G x G lattice; a heritable MASK says which cells are ALIVE, so the brain has an
arbitrary GROWN SHAPE (start 2x2, mutation adds/removes a cell). Signals injected
at the senses relax across the grid over several sub-ticks before a motor decision
is read out; cell state persists between game ticks, so the medium has memory.

v1 — cell TYPES: the genome carries a SET of update rules, and every cell has a
TYPE index pointing at one, so cells DIFFERENTIATE (a grown cell copies a
neighbour's type; mutation re-types a cell or SPLITs a type into a new rule).

v2 — evolvable DEPTH: each type's rule is a STACK of layers of variable length,
not a fixed 2-layer MLP. Mutation can ADD a layer (a near-identity hidden layer,
inserted so it barely changes behaviour at first) or REMOVE one, capped at
`nca_max_layers`, so the DEPTH of a cell's little network evolves. Hidden width is
fixed at H; only the number of layers grows. v0 is the single-type / depth-2 case.
The sensor/motor projections are shared across types (global wiring). Still ahead:
sensory ORGANS on the shape's edge instead of a uniform broadcast.

Sensor/motor wiring is size-invariant (survives growth): the sensor vector is
projected to a per-cell DRIVE broadcast into every alive cell each sub-tick, and
the action is read from the mean-POOLED cell state through a motor projection. At
one cell the whole thing degrades to a plain recurrent net.

The module mirrors neat.py's public surface (Genome / Phenotype / random_genome /
mutate / crossover + action constants) so the World can use either brain.

A rule is a list of LAYERS; each layer is {"W": (out, in), "b": (out,)}. The chain
is 9C -> H -> ... -> H -> C (all but the last layer use tanh; the last is linear and
produces the channel delta). Depth = number of layers (>= 2: an input and an output).
"""

from __future__ import annotations

import numpy as np

# actions (outputs) — identical ids to neat.py so the World's step logic is shared
FORWARD, TURN_LEFT, TURN_RIGHT, STAY = 0, 1, 2, 3
N_ACTIONS = 4

# the 3x3 neighbourhood order (self included), used to build the perception vector
_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

_MIN_DEPTH = 2          # a rule always has at least an input and an output layer


def _fresh_rule(in_dim, H, C, depth, rng, scale=0.5) -> list[dict]:
    """A random rule of `depth` layers chaining in_dim -> H -> ... -> H -> C."""
    dims = [in_dim] + [H] * (depth - 1) + [C]
    return [{"W": rng.normal(0.0, scale, (b, a)), "b": np.zeros(b)}
            for a, b in zip(dims[:-1], dims[1:])]


def _copy_rule(rule: list[dict]) -> list[dict]:
    return [{"W": L["W"].copy(), "b": L["b"].copy()} for L in rule]


class Genome:
    """A CA brain: a heritable alive-MASK and per-cell TYPE-map over a G x G lattice,
    a SET of update rules (one variable-depth layer stack per type), and the shared
    sensor/motor projections. Fixed (G, C, H) for a run; the number of types and the
    depth of each rule vary."""

    def __init__(self, n_in, n_out, mask, type_map, rules, P_in, P_out, b_out,
                 channels, hidden, hue=0.0, diet=0.5, carnivore=0.0):
        self.kind = "nca"
        self.n_in = n_in
        self.n_out = n_out
        self.mask = mask.astype(bool)               # (G, G) which cells are alive
        self.type_map = type_map.astype(np.int64)   # (G, G) each cell's rule index
        self.C = channels
        self.H = hidden
        self.rules = rules                          # list (per type) of layer lists
        self.P_in = np.asarray(P_in, dtype=np.float64)    # (n_in, C)  sensor -> drive
        self.P_out = np.asarray(P_out, dtype=np.float64)  # (C, n_out) pooled -> logits
        self.b_out = np.asarray(b_out, dtype=np.float64)  # (n_out,)
        self.hue = hue                              # heritable lineage colour
        self.diet = diet                            # heritable niche traits (see neat.py):
        self.carnivore = carnivore                  # diet type + herbivore/carnivore mix

    @property
    def G(self) -> int:
        return self.mask.shape[0]

    @property
    def n_types(self) -> int:
        return len(self.rules)

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.n_out, self.mask.copy(), self.type_map.copy(),
                      [_copy_rule(r) for r in self.rules], self.P_in.copy(),
                      self.P_out.copy(), self.b_out.copy(), self.C, self.H,
                      self.hue, self.diet, self.carnivore)

    def max_node_id(self) -> int:
        return self.n_in + self.n_out

    def complexity(self) -> tuple[int, int]:
        """(alive cells, distinct types IN USE) — the HUD's two brain numbers."""
        used = len(np.unique(self.type_map[self.mask])) if self.mask.any() else 0
        return int(self.mask.sum()), int(used)

    def max_depth(self) -> int:
        return max((len(r) for r in self.rules), default=_MIN_DEPTH)

    def max_width(self) -> int:
        """Widest hidden layer across the type rules (a rule's hidden width = its input
        layer's output dim). The HUD's 'w' number."""
        return max((int(r[0]["W"].shape[0]) for r in self.rules), default=self.H)

    def to_dict(self) -> dict:
        return {"brain": "nca", "n_in": self.n_in, "n_out": self.n_out,
                "C": self.C, "H": self.H, "hue": self.hue,
                "diet": self.diet, "carnivore": self.carnivore,
                "mask": self.mask.astype(int).tolist(),
                "type_map": self.type_map.tolist(),
                "rules": [[[L["W"].tolist(), L["b"].tolist()] for L in r]
                          for r in self.rules],
                "P_in": self.P_in.tolist(), "P_out": self.P_out.tolist(),
                "b_out": self.b_out.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        rules = [[{"W": np.array(W, dtype=np.float64), "b": np.array(b, dtype=np.float64)}
                  for W, b in r] for r in d["rules"]]
        return cls(int(d["n_in"]), int(d["n_out"]), np.array(d["mask"], dtype=bool),
                   np.array(d["type_map"]), rules, d["P_in"], d["P_out"], d["b_out"],
                   int(d["C"]), int(d["H"]), float(d.get("hue", 0.0)),
                   float(d.get("diet", 0.5)), float(d.get("carnivore", 0.0)))

    def key(self) -> bytes:
        """Canonical fingerprint of shape + types + rule stacks (rounded)."""
        parts = [self.mask.astype(np.int8).tobytes(),
                 (self.type_map * self.mask).astype(np.int16).tobytes()]
        for r in self.rules:
            for L in r:
                parts.append(np.round(L["W"], 1).tobytes())
        for a in (self.P_in, self.P_out, self.b_out):
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
    type with a depth-2 rule, and a FORWARD motor bias so fresh brains tend to move."""
    G = getattr(cfg, "nca_grid", 12)
    C = getattr(cfg, "nca_channels", 8)
    H = getattr(cfg, "nca_hidden", 16)
    start = getattr(cfg, "nca_start", 2)
    mask = _start_mask(G, start)
    type_map = np.zeros((G, G), dtype=np.int64)
    rules = [_fresh_rule(9 * C, H, C, _MIN_DEPTH, rng)]
    P_in = rng.normal(0.0, 0.5, (n_in, C))
    P_out = rng.normal(0.0, 0.5, (C, n_out))
    b_out = np.zeros(n_out)
    b_out[FORWARD] += forward_bias
    return Genome(n_in, n_out, mask, type_map, rules, P_in, P_out, b_out, C, H,
                  hue=float(rng.random()),
                  diet=float(rng.random()), carnivore=float(rng.random()) * 0.6)


def _apply_rule(x: np.ndarray, rule: list[dict]) -> np.ndarray:
    """Forward a (n, 9C) perception batch through a rule's layer stack -> (n, C)
    delta. Hidden layers use tanh; the last layer is linear."""
    for L in rule[:-1]:
        x = np.tanh(x @ L["W"].T + L["b"])
    last = rule[-1]
    return x @ last["W"].T + last["b"]


class Phenotype:
    """Runnable CA brain. Holds the persistent grid state; `act` injects the senses
    as a drive, relaxes the per-type rules a few sub-ticks so signals cross the grid,
    then reads the action from the pooled cell state."""

    def __init__(self, genome: Genome, subticks: int = 32, plastic: bool = False,
                 lr: float = 0.03, forget: float = 0.002, elig_decay: float = 0.5,
                 clip: float = 1.0):
        self.genome = genome
        self.subticks = subticks
        G, C = genome.G, genome.C
        self.state = np.zeros((G, G, C), dtype=np.float64)
        self._m = genome.mask[:, :, None]           # (G, G, 1) broadcast mask
        self._type_sel = [genome.mask & (genome.type_map == k)
                          for k in range(genome.n_types)]
        self.last_input = None
        self.last_logits = None
        # LIFETIME LEARNING (meta-plasticity): a plastic delta on the motor readout
        # learned within one life. `dW` is added to P_out; `E` is the eligibility trace
        # (recent pooled-state x chosen-action), decayed each act and reinforced by
        # reward in `learn`. Reset per phenotype, so it is NOT inherited.
        self.plastic = plastic
        self.lr, self.forget, self.elig_decay, self.clip = lr, forget, elig_decay, clip
        self.dW = np.zeros((C, genome.n_out), dtype=np.float64)
        self._E = np.zeros((C, genome.n_out), dtype=np.float64)
        # inspector-only (never used by the computation): a per-sub-tick TRACE for the
        # "watch it think" animation, and a state override for drawing one trace frame.
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

    def act(self, inputs: np.ndarray, rng=None, temp: float = 0.0) -> int:
        g = self.genome
        drive = inputs @ g.P_in                              # (C,) sensory current
        st = self.state
        if self.record:
            self.trace = []
        single = g.n_types == 1                              # common case: fast path
        for _ in range(self.subticks):
            perc = self._perceive(st)                        # (G, G, 9C)
            if single:                                       # one rule over the whole grid
                flat = perc.reshape(-1, perc.shape[-1])
                delta = _apply_rule(flat, g.rules[0]).reshape(st.shape)
            else:                                            # each type runs its own rule
                delta = np.zeros_like(st)
                for k, sel in enumerate(self._type_sel):
                    if sel.any():
                        delta[sel] = _apply_rule(perc[sel], g.rules[k])
            st = np.tanh(st + delta + drive)                 # residual + clamped
            st = st * self._m                                # dead cells stay 0
            if self.record:
                self.trace.append(st.copy())                 # one frame per sub-tick
        self.state = st
        pooled = st[g.mask].mean(axis=0)                     # (C,) mean over alive
        eff_out = g.P_out + self.dW if self.plastic else g.P_out  # readout + learned delta
        logits = pooled @ eff_out + g.b_out                  # (n_out,)
        self.last_input = inputs
        self.last_logits = logits
        if rng is not None and temp > 0:                     # explore: sample vs freeze
            z = logits / temp
            z -= z.max()
            p = np.exp(z)
            a = int(rng.choice(len(p), p=p / p.sum()))
        else:
            a = int(np.argmax(logits))
        if self.plastic:                                     # credit the ACTION TAKEN:
            self._E *= self.elig_decay                       # decay the trace, then credit
            self._E[:, a] += pooled                          # outer(pooled, onehot(a))
        return a

    def learn(self, reward: float) -> None:
        """Reward-modulated (three-factor) update of the plastic readout delta, called
        once per game tick. The plastic delta always forgets slowly; when a REWARD lands
        (the agent ate), the decisions still in the eligibility trace are reinforced —
        so actions that led to food get more likely. No-op unless plastic."""
        if not self.plastic:
            return
        self.dW *= (1.0 - self.forget)
        if reward:
            self.dW += self.lr * reward * self._E
            np.clip(self.dW, -self.clip, self.clip, out=self.dW)


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
    grows/shrinks the SHAPE, re-types a cell, SPLITs a type, or grows/shrinks a rule's
    DEPTH (add/remove a layer) — so shape, cell-type composition AND network depth all
    evolve. `next_node_id` is passed through untouched."""
    g = genome.copy()
    if rng.random() < cfg.hue_drift_prob:
        g.hue = (g.hue + float(rng.normal(0.0, cfg.hue_drift))) % 1.0
    if rng.random() < cfg.trait_drift_prob:
        g.diet = min(1.0, max(0.0, g.diet + float(rng.normal(0.0, cfg.trait_drift))))
    if rng.random() < cfg.trait_drift_prob:
        g.carnivore = min(1.0, max(0.0,
                          g.carnivore + float(rng.normal(0.0, cfg.trait_drift))))
    if rng.random() < cfg.macro_mut_prob:                # rare SPECIATION jolt: a big
        g.hue = float(rng.random())                      # colour jump (new lineage) +
        g.diet = float(rng.random())                     # a new diet + maybe a predator
        g.carnivore = float(rng.random())                # + strong rewiring
        for r in g.rules:
            for L in r:
                _jitter(L["W"], rng, 1.0, cfg.weight_mut * cfg.macro_mut_scale)
    for r in g.rules:                                    # ordinary weight jitter
        for L in r:
            _jitter(L["W"], rng, cfg.weight_mut_prob, cfg.weight_mut)
            _jitter(L["b"], rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.P_in, rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.P_out, rng, cfg.weight_mut_prob, cfg.weight_mut)
    _jitter(g.b_out, rng, cfg.weight_mut_prob, cfg.weight_mut)

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

    if g.n_types > 1 and rng.random() < cfg.nca_retype_prob:
        alive = np.argwhere(g.mask)
        y, x = alive[int(rng.integers(len(alive)))]
        g.type_map[y, x] = int(rng.integers(g.n_types))

    if g.n_types < cfg.nca_max_types and rng.random() < cfg.nca_split_prob:
        g = _split_type(g, rng, cfg)

    _mutate_depth(g, rng, cfg)                           # add / remove a rule layer
    _mutate_width(g, rng, cfg)                            # widen / narrow the hidden layer
    return g, next_node_id


def _resize_rule_width(rule: list[dict], grow: bool, rng, cfg) -> None:
    """Change a rule's HIDDEN WIDTH by one unit, uniformly across all its hidden layers
    (so every middle layer stays square and depth mutations remain valid). GROW: append
    a hidden unit whose OUTGOING weights are zero, so the new channel is disconnected
    downstream and adding it doesn't change behaviour at first (evolution/learning can
    wake it). NARROW: drop one hidden unit everywhere. The input width (9C) and output
    width (C) are untouched."""
    w = int(rule[0]["W"].shape[0])
    n = len(rule)
    if grow:
        if w >= cfg.nca_max_hidden:
            return
        for i, L in enumerate(rule):
            if i < n - 1:                                # this layer emits a hidden unit
                L["W"] = np.vstack([L["W"], rng.normal(0.0, 0.1, (1, L["W"].shape[1]))])
                L["b"] = np.append(L["b"], 0.0)
            if i > 0:                                    # this layer reads a hidden unit
                L["W"] = np.hstack([L["W"], np.zeros((L["W"].shape[0], 1))])
    else:
        if w <= cfg.nca_min_hidden:
            return
        j = int(rng.integers(w))
        for i, L in enumerate(rule):
            if i < n - 1:
                L["W"] = np.delete(L["W"], j, axis=0)
                L["b"] = np.delete(L["b"], j)
            if i > 0:
                L["W"] = np.delete(L["W"], j, axis=1)


def _mutate_width(g: Genome, rng, cfg) -> None:
    """Grow or shrink one type-rule's hidden width (the number of nodes in its hidden
    layers), so the perceptron's WIDTH evolves alongside its depth and count."""
    if rng.random() < cfg.nca_widen_prob:
        _resize_rule_width(g.rules[int(rng.integers(g.n_types))], True, rng, cfg)
    if rng.random() < cfg.nca_narrow_prob:
        _resize_rule_width(g.rules[int(rng.integers(g.n_types))], False, rng, cfg)


def _identity_layer(H: int, rng, jitter=0.05) -> dict:
    """A near-IDENTITY H->H layer, so inserting it barely changes the rule at first
    (tanh(I x) ~ x for small x) — depth grows without wrecking a working brain."""
    W = np.eye(H) + rng.normal(0.0, jitter, (H, H))
    return {"W": W, "b": np.zeros(H)}


def _mutate_depth(g: Genome, rng, cfg) -> None:
    """Grow or shrink one rule's DEPTH: ADD a near-identity hidden layer (up to
    nca_max_layers) or REMOVE a middle layer (down to _MIN_DEPTH). Only middle layers
    are touched, so the input (9C->H) and output (H->C) layers always frame the stack
    and the chain stays valid."""
    if rng.random() < cfg.nca_add_layer_prob:
        k = int(rng.integers(g.n_types))
        rule = g.rules[k]
        if len(rule) < cfg.nca_max_layers:
            w = int(rule[-1]["W"].shape[1])              # width feeding the output layer
            rule.insert(len(rule) - 1, _identity_layer(w, rng))
    if rng.random() < cfg.nca_del_layer_prob:
        k = int(rng.integers(g.n_types))
        if len(g.rules[k]) > _MIN_DEPTH:
            mid = int(rng.integers(1, len(g.rules[k]) - 1))   # a middle layer
            del g.rules[k][mid]


def _split_type(g: Genome, rng, cfg) -> Genome:
    """Duplicate an existing type's rule (with a jitter) as a NEW type, and move a
    random half of that type's cells to it — differentiation from within."""
    src = int(rng.integers(g.n_types))
    new_rule = _copy_rule(g.rules[src])
    for L in new_rule:
        _jitter(L["W"], rng, 1.0, cfg.weight_mut)
    g.rules.append(new_rule)
    new_idx = g.n_types - 1
    for y, x in np.argwhere(g.mask & (g.type_map == src)):
        if rng.random() < 0.5:
            g.type_map[y, x] = new_idx
    return g


def _mix_rule(rf: list[dict], ro: list[dict], rng) -> list[dict]:
    """Crossover two rules: layers shared by index (same shapes, since widths are
    fixed) are an elementwise mix; layers only the fitter rule has are taken whole."""
    out = []
    for i, Lf in enumerate(rf):
        if i < len(ro) and ro[i]["W"].shape == Lf["W"].shape:
            Lo = ro[i]
            pw = rng.random(Lf["W"].shape) < 0.5
            pb = rng.random(Lf["b"].shape) < 0.5
            out.append({"W": np.where(pw, Lf["W"], Lo["W"]),
                        "b": np.where(pb, Lf["b"], Lo["b"])})
        else:
            out.append({"W": Lf["W"].copy(), "b": Lf["b"].copy()})
    return out


def crossover(g_fit: Genome, g_other: Genome, rng) -> Genome:
    """Child genome: the SHAPE (mask), TYPE-map and type set come from the fitter
    parent; each rule is crossed with the other parent's same-index rule (extra rules
    inherited whole), as are the projections. Colour/diet/carnivore from a parent."""
    def mix(a, b):
        return np.where(rng.random(a.shape) < 0.5, a, b)

    rules = []
    for k, rf in enumerate(g_fit.rules):
        ro = g_other.rules[k] if k < g_other.n_types else rf
        rules.append(_mix_rule(rf, ro, rng))
    P_in = mix(g_fit.P_in, g_other.P_in)
    P_out = mix(g_fit.P_out, g_other.P_out)
    b_out = mix(g_fit.b_out, g_other.b_out)
    hue = g_fit.hue if rng.random() < 0.5 else g_other.hue
    diet = g_fit.diet if rng.random() < 0.5 else g_other.diet
    carn = g_fit.carnivore if rng.random() < 0.5 else g_other.carnivore
    return Genome(g_fit.n_in, g_fit.n_out, g_fit.mask.copy(), g_fit.type_map.copy(),
                  rules, P_in, P_out, b_out, g_fit.C, g_fit.H, hue=hue,
                  diet=diet, carnivore=carn)
