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

# The BLOODSTREAM: what the body tells the whole sheet at once, no organ involved.
# Energy first, because it is the one that has to swing the whole brain's mode.
N_HORMONE = 3                    # energy, poison load, recent reward

# MORPHOGENESIS. A rule's output is not only a state delta: the last N_MORPH values are
# what the cell wants to DO WITH ITSELF. They are computed by the very same little
# network the cell thinks with, from the very same 3x3 perception — there is no separate
# "growth brain", which is the whole point (docs/cortex-design.md).
GROW, DIR_Y, DIR_X, DIE, BE_SENSE, BE_ACT, SPROUT = range(7)
N_MORPH = 7
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
        self.channels = 14           # state channels per cell (see the channel map in
                                     # `Genome.channel_map` — the tail is special)
        self.hidden = 28             # hidden width of a fresh rule
        # MORPHOLOGY. The seed is not a rectangle: it is a blob GROWN by accretion,
        # with lacunae punched through it, so the tissue has an irregular border and
        # interior gaps from birth. A square has no morphology to select on, and the
        # organs would inherit its symmetry — which is the opposite of the point.
        self.start_cells = 340       # living cells in the seed (many senses, big sheet)
        self.hole_frac = 0.06        # of interior cells punched out as lacunae
        self.organ_frac = 0.55       # of BORDER cells that become sensory organs
        self.effector_frac = 0.04    # of DEEP cells that become effectors
        self.subticks = 20           # relaxation rounds per world tick
        self.max_cells = 4096
        self.max_types = 6
        self.max_layers = 5
        self.max_hidden = 48
        self.min_hidden = 4
        self.organ_mods = 9          # 1..6 per-ray channels, 7 odour, 8 touch, 9 intero
        # THREE SIGNAL LAYERS, at three speeds (docs/cortex-design.md).
        # 1 ELECTRIC: the 3x3 neighbourhood, every sub-tick. Fast, local, precise.
        # 2 CHEMICAL: a few channels that DIFFUSE and DECAY. Slow, blurry, and the
        #   only carrier with a spatial GRADIENT — which is what a growing tissue can
        #   read its own position from, and what an axon would later grow along.
        # 3 HORMONAL: a global vector from the BODY, reaching every cell at once
        #   regardless of organs (a bloodstream, not a wire), and gating every rule —
        #   so the same tissue computes differently when starving or poisoned.
        self.chem_channels = 3       # how many of the tail channels diffuse
        self.chem_mix = 0.35         # share of the neighbourhood average taken per pass
        self.chem_decay = 0.04       # per pass, so a plume fades instead of piling up
        self.chem_every = 2          # sub-ticks per diffusion pass (slower than thought)
        self.hormone = True          # the global gate + bloodstream drive
        self.hormone_gain = 0.6      # how hard it swings each cell's rule
        self.neuromod = True         # the LOCAL hormone channel (per-cell gate)
        # DEVELOPMENT. The genome carries a SEED and a program; the body is grown.
        # Growth fires on a rare DEVELOPMENT TICK, never per sub-tick: partly because
        # tissue is slower than signal, and partly because a shape change rebuilds the
        # phenotype's index arrays, which is what would make a per-tick version
        # unaffordable. Rare development is not a compromise, it is the enabling trick.
        self.develop = True
        self.seed_cells = 24         # the SEED: a genome is a program, not a body
        self.seed_types = 3          # cell TYPES the seed already has. One meant every
                                     # cell ran the same network until a 4%-per-mutation
                                     # split happened to fire, so a fresh brain was
                                     # uniform tissue. Laid out as REGIONS, not
                                     # salt-and-pepper: a daughter inherits her mother's
                                     # type, so regions grow outward with the body.
        self.embryo_ticks = 900      # a head start in which growth is fast and FREE...
        self.dev_every_embryo = 8    # ...development ticks this often, then:
        self.dev_every = 120         # slow structural plasticity for the rest of life
        self.grow_bias = 0.55        # the seed's INNATE urge to divide, written into
                                     # the rule's output bias. An embryo grows unless
                                     # told not to — without it a fresh genome's morph
                                     # outputs sit near zero, no cell ever asks for
                                     # anything, and "development" is a no-op that
                                     # merely made the brain smaller. Mutation shapes
                                     # this from the first generation.
        self.sense_bias = 0.10       # ...and to become a SENSE. Deliberately BELOW the
                                     # threshold, so only cells whose perception pushes
                                     # them over it become organs. Too low and the sheet
                                     # is deaf (18 organs in 700 cells, nothing to
                                     # select on); too high and it is all ears (424 in
                                     # 471) with no interneurons left to think with.
        self.act_bias = 0.12         # below threshold: only strongly-driven cells
                                     # become effectors, so a body has a few, not a sea
        self.max_born_per_dev = 5    # an ABSOLUTE ceiling on births per development
                                     # tick. The percentage alone compounds: at 5% over
                                     # a 112-tick embryo a seed of 24 reached 2344 cells
                                     # and the tick cost 7 ms. With a ceiling, growth is
                                     # linear in developmental time and a body lands in
                                     # the hundreds, which is the size that thinks fast
                                     # enough to live a long life.
        self.max_change_frac = 0.05  # of the body may be BORN in one dev tick...
        self.die_ratio = 0.4         # ...and this fraction of that may die. Appearance
                                     # is biased over disappearance on purpose: with
                                     # equal budgets the sheet sat at its seed size
                                     # forever, churning and never becoming anything.
        self.grow_thresh = 0.25      # how loud a cell's GROW output must be to divide
        self.die_thresh = 0.25       # ...and its DIE output to apoptose. Set where the
                                     # natural spread of a fresh rule can actually
                                     # REACH it: at 0.55 no cell ever died, so
                                     # "structural plasticity" was growth only and the
                                     # apoptosis half of it was dead code
        self.organ_thresh = 0.30
        self.divide_cost = 0.35      # RESOURCE spent by a division, taken from the cell
        self.resource_spread = 1     # seepage passes per development tick
        self.resource_mix = 0.25     # how much of a neighbour's supply seeps across.
                                     # Kept LOW on purpose: at 3 passes x 0.5 the fresh
                                     # supply was carried off the border and pooled in
                                     # the interior — exactly where there is no room to
                                     # divide — so the cells that could grow never had
                                     # the means and the cells with the means could not
                                     # grow. Supply must stay where the growing is.
        self.resource_decay = 0.10   # what is not used is lost, so supply is a FLOW
        self.resource_cap = 3.0
        self.resource_gain = 0.55    # resource fed in at the BORDER each dev tick,
                                     # scaled by how full the body is — so the tissue
                                     # can only grow on what the agent has actually
                                     # eaten, and the deep interior grows last
        # AXONS: long-range directed links, the one thing the 3x3 neighbourhood cannot
        # express. A sheet without them is a RELAY — every message crosses cell by cell,
        # one cell per sub-tick, so a sense on one rim and an effector on the other are
        # further apart in time than a thought is long. An axon is a shortcut with a
        # weight, and a set of shortcuts is a TOPOLOGY: which regions can talk at all.
        # It grows like a growth cone, up the guidance chemical, and it is pruned by
        # decay, so the wiring is grown and lost rather than designed.
        self.axons = True
        self.sprout_bias = 0.26      # the seed's innate urge to send one out. Near the
                                     # threshold on purpose, like the other seed urges:
                                     # at 0.10 no cell ever reached it and the whole
                                     # mechanism was dead code mutation could not even
                                     # find, because a wire that is never grown is never
                                     # selected for or against.
        self.sprout_thresh = 0.30
        self.axon_min_len = 3        # shorter than this and the neighbourhood already
        self.axon_max_len = 22       # covers it; longer and it leaves the body
        self.axon_cost = 0.5         # RESOURCE to grow one — dearer than a cell, since
                                     # a wire that costs nothing would fill the sheet
        self.axon_w0 = 0.35
        self.axon_decay = 0.02       # per development tick: use it or lose it
        self.axon_min_w = 0.05       # below this it is pruned
        self.axon_lr = 0.02          # reward-gated Hebbian on the weight
        self.axon_clip = 1.2
        self.max_axons = 400
        self.cell_cost = 0.000015    # energy per cell per world tick, charged to the
                                     # AGENT. This is the counterweight to growth: a
                                     # bigger brain has to earn its keep by eating more
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
    """The output layer is C + N_MORPH wide: the state delta AND what the cell wants to
    do with itself, from one network and one perception."""
    dims = [in_dim] + [H] * (depth - 1) + [C + N_MORPH]
    return [{"W": rng.normal(0.0, scale / math.sqrt(a), (b, a)), "b": np.zeros(b)}
            for a, b in zip(dims[:-1], dims[1:])]


def _copy_rule(rule) -> list[dict]:
    return [{"W": L["W"].copy(), "b": L["b"].copy()} for L in rule]


class Genome:
    """Shape + cell types + rule stacks + the sensor/motor projections."""

    def __init__(self, n_in, mask, type_map, rules, P_in, P_out, b_out,
                 channels, hidden, subticks, in_mask, out_mask, organ_mod,
                 P_horm=None, organ_map=None):
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
        # (N_HORMONE, C): the bloodstream's drive into every cell, organs or not
        self.P_horm = (np.zeros((N_HORMONE, self.C)) if P_horm is None
                       else np.asarray(P_horm, dtype=np.float64))
        # ORGAN FATE: which sensory modality a cell that decides to become an organ
        # takes on, chosen by WHICH CHEMICAL is strongest where it stands. That is the
        # French-flag model — a concentration decides a fate — and it is what makes an
        # "eye region" a thing that can form rather than a thing we place.
        self.organ_map = (np.array([1, 7, 8], dtype=np.int64) if organ_map is None
                          else np.asarray(organ_map, dtype=np.int64))
        self.subticks = int(subticks)

    @property
    def G(self) -> int:
        return self.mask.shape[0]

    @property
    def n_types(self) -> int:
        return len(self.rules)

    def channel_map(self, cfg=None) -> dict:
        """Which channels are what. The tail is special: the last is the LOCAL hormone
        that gates a cell's own rule, and the few before it are CHEMICAL — they leak
        into their neighbours and fade, so they carry gradients instead of values."""
        cfg = cfg or Config()
        chem = int(min(cfg.chem_channels, self.C - 2))
        return {"electric": (0, self.C - 1 - chem),
                "chemical": (self.C - 1 - chem, self.C - 1),
                "hormone": (self.C - 1, self.C)}

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.mask.copy(), self.type_map.copy(),
                      [_copy_rule(r) for r in self.rules], self.P_in.copy(),
                      self.P_out.copy(), self.b_out.copy(), self.C, self.H,
                      self.subticks, self.in_mask.copy(), self.out_mask.copy(),
                      self.organ_mod.copy(), self.P_horm.copy(), self.organ_map.copy())

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
        n = self.P_in.size + self.P_out.size + self.b_out.size + self.P_horm.size
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
                "b_out": self.b_out.tolist(), "P_horm": self.P_horm.tolist(),
                "organ_map": self.organ_map.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        rules = [[{"W": np.array(W, dtype=np.float64), "b": np.array(b, dtype=np.float64)}
                  for W, b in r] for r in d["rules"]]
        return cls(int(d["n_in"]), np.array(d["mask"], dtype=bool),
                   np.array(d["type_map"]), rules, d["P_in"], d["P_out"], d["b_out"],
                   int(d["C"]), int(d["H"]), int(d["subticks"]),
                   np.array(d["in_mask"], dtype=bool), np.array(d["out_mask"], dtype=bool),
                   np.array(d["organ_mod"], dtype=np.int64),
                   np.array(d["P_horm"]) if "P_horm" in d else None,
                   np.array(d["organ_map"]) if "organ_map" in d else None)


def _grow_blob(G: int, n_cells: int, rng) -> np.ndarray:
    """Grow an irregular blob by ACCRETION from one cell: repeatedly pick a living
    cell and light up one of its free neighbours. The border comes out ragged and
    every seed is a different shape, which is what gives morphology something to be
    selected on — a rectangle has none, and its organs would inherit its symmetry."""
    m = np.zeros((G, G), dtype=bool)
    c = G // 2
    m[c, c] = True
    frontier = [(c, c)]
    n = 1
    while n < n_cells and frontier:
        i = int(rng.integers(len(frontier)))
        y, x = frontier[i]
        free = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= y + dy < G and 0 <= x + dx < G and not m[y + dy, x + dx]]
        if not free:
            frontier.pop(i)                      # fully surrounded: stop offering it
            continue
        ny, nx = free[int(rng.integers(len(free)))]
        m[ny, nx] = True
        frontier.append((ny, nx))
        n += 1
    return m


def _punch_holes(mask: np.ndarray, frac: float, rng) -> None:
    """Kill some INTERIOR cells, so the sheet has gaps signal must route around. Only
    cells whose four neighbours are all alive are eligible, so a hole never bites a
    piece off the border."""
    interior = [(int(y), int(x)) for y, x in np.argwhere(mask)
                if all(mask[y + dy, x + dx]
                       for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                       if 0 <= y + dy < mask.shape[0] and 0 <= x + dx < mask.shape[1])]
    for k in rng.permutation(len(interior))[:int(len(interior) * frac)]:
        y, x = interior[int(k)]
        mask[y, x] = False


def _border_and_deep(mask: np.ndarray):
    """Split the living cells into those touching the outside (where senses belong)
    and those buried at least one layer in (where a command can be read from)."""
    G = mask.shape[0]
    border, deep = [], []
    for y, x in np.argwhere(mask):
        exposed = any(not (0 <= y + dy < G and 0 <= x + dx < G and mask[y + dy, x + dx])
                      for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        (border if exposed else deep).append((int(y), int(x)))
    return border, deep


def random_genome(n_in: int, rng, cfg: Config | None = None) -> Genome:
    """A seed brain wired as a small ORGANISM: sensory organs on the PERIMETER (each
    given a random modality, so retinotopy is engaged from birth), effectors at the
    CENTRE, interneurons in between. Sense at the edge, act from the core — so the
    shape has a job on the very first tick."""
    cfg = cfg or Config()
    G, C, H = cfg.grid, cfg.channels, cfg.hidden
    # With development ON this is a SEED, not a body: a couple of dozen cells that the
    # tissue grows itself from. With it off it is the whole brain, as before.
    n0 = cfg.seed_cells if cfg.develop else cfg.start_cells
    mask = _grow_blob(G, n0, rng)
    if not cfg.develop:
        _punch_holes(mask, cfg.hole_frac, rng)
    type_map = np.zeros((G, G), dtype=np.int64)
    in_mask = np.zeros((G, G), dtype=bool)
    out_mask = np.zeros((G, G), dtype=bool)
    organ_mod = np.zeros((G, G), dtype=np.int64)
    border, deep = _border_and_deep(mask)
    # ORGANS ARE PLACED MORPHOLOGICALLY, not by a rule about a rectangle: only SOME
    # border cells become senses (the rest are skin), and effectors sit in the DEEP
    # tissue, so a command is read from cells no sense can reach directly.
    for y, x in border:
        if rng.random() < cfg.organ_frac:
            in_mask[y, x] = True
            organ_mod[y, x] = int(rng.integers(cfg.organ_mods + 1))
    for y, x in deep:
        if rng.random() < cfg.effector_frac:
            out_mask[y, x] = True
    if not out_mask.any():                       # a brain always has a voice
        y, x = (deep or border)[int(rng.integers(len(deep or border)))]
        out_mask[y, x] = True
    # TYPES AS REGIONS. Each type claims a random seed cell and every cell takes the
    # nearest claim (a small Voronoi). Because a daughter inherits her mother's type,
    # these patches then grow outward with the tissue instead of staying a speckle.
    n_types = max(1, int(cfg.seed_types))
    rules = [_fresh_rule(9 * C, H, C, _MIN_DEPTH, rng) for _ in range(n_types)]
    if n_types > 1:
        alive = np.argwhere(mask)
        claims = alive[rng.choice(len(alive), size=min(n_types, len(alive)),
                                  replace=False)]
        for y, x in alive:
            d = ((claims[:, 0] - y) ** 2 + (claims[:, 1] - x) ** 2)
            type_map[y, x] = int(np.argmin(d))
    # Xavier on the sensor projection: 412 inputs summed into C channels would
    # saturate tanh instantly at N(0, 0.5) — the same regression that once broke NEAT
    # in the vivarium when the sensor vector got wide (see docs/lab-journal.md).
    P_in = rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, C))
    P_out = rng.normal(0.0, 0.5, (C, N_OUT))
    b_out = np.zeros(N_OUT)
    b_out[THRUST] += 0.5                                   # a nudge to get moving
    for r in rules:                                        # the SEED PROGRAM's innate
        r[-1]["b"][C + GROW] += cfg.grow_bias              # urges: grow, sense, act,
        r[-1]["b"][C + BE_SENSE] += cfg.sense_bias         # and send a wire out.
        r[-1]["b"][C + BE_ACT] += cfg.act_bias             # Mutation edits these from
        r[-1]["b"][C + SPROUT] += cfg.sprout_bias          # the first generation on
    P_horm = rng.normal(0.0, 0.25, (N_HORMONE, C))
    n_chem = max(1, int(min(cfg.chem_channels, C - 2)))
    organ_map = rng.integers(1, cfg.organ_mods + 1, n_chem)
    return Genome(n_in, mask, type_map, rules, P_in, P_out, b_out, C, H,
                  cfg.subticks, in_mask, out_mask, organ_mod, P_horm, organ_map)


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
                  mod_ch, dW_last, elig_out, plastic, subticks,
                  chem0, chem1, chem_mix, chem_decay, chem_every, hgate,
                  tick0, morph, ax_src, ax_dst, ax_w, ax_buf, ax_elig):  # pragma: no cover
        """All `subticks` rounds as one kernel: scatter the alive rows into the padded
        buffer, gather each cell's 9-neighbour perception, push each TYPE's rows
        through its layer stack, then the residual tanh update with the hormone gate."""
        n, C = st.shape
        n_ax = ax_w.shape[0]
        for _ in range(subticks):
            # AXONS: what arrives down the wires, gathered from the state as it stands
            # and delivered as an additive drive, exactly like a sense.
            if n_ax > 0:
                for i in range(n):
                    for c in range(C):
                        ax_buf[i, c] = 0.0
                for k in range(n_ax):
                    a = ax_src[k]
                    b = ax_dst[k]
                    sa = 0.0
                    sb = 0.0
                    for c in range(C):
                        ax_buf[b, c] += ax_w[k] * st[a, c]
                        sa += abs(st[a, c])
                        sb += abs(st[b, c])
                    ax_elig[k] += (sa / C) * (sb / C)     # Hebbian: fired together
            for i in range(n):
                for c in range(C):
                    padF[centre[i], c] = st[i, c]
            perc = np.empty((n, 9 * C))
            for i in range(n):
                for j in range(9):
                    r = nbr[i, j]
                    for c in range(C):
                        perc[i, j * C + c] = padF[r, c]
            M = morph.shape[1]
            delta = np.empty((n, C + M))
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
                        for c in range(C + M):
                            delta[rows[ii], c] = y[ii, c]
                li += nl
            for i in range(n):
                gate = (1.0 + math.tanh(st[i, mod_ch])) if mod_ch >= 0 else 1.0
                gate *= hgate                              # the whole-sheet mode switch
                if n_ax > 0:
                    for c in range(C):
                        st[i, c] = np.tanh(st[i, c] + gate * delta[i, c]
                                           + drive[i, c] + ax_buf[i, c])
                else:
                    for c in range(C):
                        st[i, c] = np.tanh(st[i, c] + gate * delta[i, c] + drive[i, c])
                for m in range(M):                         # what the cell wants to DO
                    morph[i, m] += delta[i, C + m]         # with itself, accumulated
            # CHEMISTRY: a slow pass in which the chem channels leak into their
            # neighbours and fade. This is the only layer with a spatial gradient.
            tick0 += 1
            if chem1 > chem0 and (tick0 % chem_every) == 0:
                for i in range(n):
                    for c in range(chem0, chem1):
                        padF[centre[i], c] = st[i, c]
                for i in range(n):
                    for c in range(chem0, chem1):
                        acc = 0.0
                        for j in range(9):
                            acc += padF[nbr[i, j], c]
                        mixed = st[i, c] * (1.0 - chem_mix) + (acc / 9.0) * chem_mix
                        st[i, c] = mixed * (1.0 - chem_decay)
        return st


def build_listen(g: Genome, ys, xs, layout, in_mask=None, organ_mod=None) -> np.ndarray:
    """Row i = which sensor indices alive-cell i hears. MOD_ALL hears everything; a
    per-ray modality hears the ONE ray its COLUMN selects (retinotopy); a static
    modality hears its index block. Non-organs hear nothing. No organs at all -> all
    cells hear all, so a brain is never struck deaf.

    `in_mask`/`organ_mod` default to the genome's, but development passes the LIVING
    ones — under morphogenesis the genome describes a seed, not a body."""
    in_mask = g.in_mask if in_mask is None else in_mask
    organ_mod = g.organ_mod if organ_mod is None else organ_mod
    n = len(ys)
    M = np.zeros((n, g.n_in))
    organs = in_mask[ys, xs]
    if not organs.any() or layout is None:
        M[organs if organs.any() else slice(None)] = 1.0
        return M
    ray_ch, n_rays = layout["ray_ch"], layout["n_rays"]
    ray_mod, static_mod = layout["ray_mod_channel"], layout["static_mod_idx"]
    xmin, span = int(xs.min()), max(1, int(xs.max()) - int(xs.min()))
    for i in range(n):
        if not organs[i]:
            continue
        m = int(organ_mod[ys[i], xs[i]])
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
        cmap = genome.channel_map(cfg)
        self._chem0, self._chem1 = cmap["chemical"]
        if not cfg.chem_channels:
            self._chem1 = self._chem0
        self._subtick = 0
        self.hormone = np.zeros(N_HORMONE)
        self.state = np.zeros((G, G, C))
        # THE LIVING BODY. These start as copies of the genome's seed and are then
        # changed by DEVELOPMENT. The genome itself is never touched during a life —
        # that is what makes "only the genome crosses the boundary" true rather than
        # aspirational, and what stops a phenotype quietly becoming the heredity.
        self.mask = genome.mask.copy()
        self.type_map = genome.type_map.copy()
        self.in_mask = genome.in_mask.copy()
        self.out_mask = genome.out_mask.copy()
        self.organ_mod = genome.organ_mod.copy()
        self._padF = np.zeros(((G + 2) * (G + 2), C))
        # RESOURCE is metabolism, not signal, so it is NOT one of the cell's channels:
        # an early version used a chemical channel and the rule simply flooded it —
        # every cell wanted to divide and two had the means. A neuron does not write
        # its own blood supply. Fed at the border, diffused, spent by division.
        self.resource = np.zeros((G, G))
        # AXONS live as CELL COORDINATES, not row indices: rows are rebuilt every time
        # the body changes, and a wire that silently re-pointed at whatever cell
        # inherited its row number would be worse than no wire at all.
        self.axon_cells: list = []          # [sy, sx, dy, dx, weight]
        self._ax_src = np.zeros(0, dtype=np.int64)
        self._ax_dst = np.zeros(0, dtype=np.int64)
        self._ax_w = np.zeros(0)
        self._ax_elig = np.zeros(0)
        self.births = 0
        self.deaths = 0
        self.dev_ticks = 0
        self.morph = np.zeros((1, N_MORPH))
        if _NUMBA:
            Wts, bs = _NbList(), _NbList()
            for r in genome.rules:
                for L in r:
                    Wts.append(np.ascontiguousarray(L["W"].T))
                    bs.append(np.ascontiguousarray(L["b"]))
            self._nb_W, self._nb_b = Wts, bs
            self._nb_cnt = np.array([len(r) for r in genome.rules], dtype=np.int64)
        self._reindex()
        # learning within one life
        self.dW = np.zeros((C, N_OUT))
        self._E = np.zeros((C, N_OUT))
        outs = [int(r[-1]["W"].shape[1]) for r in genome.rules]
        self._dW_rule = [np.zeros((o, C + N_MORPH)) for o in outs]
        self._E_rule = [np.zeros((o, C + N_MORPH)) for o in outs]
        self._elig = [np.zeros((o, C + N_MORPH)) for o in outs]
        if _NUMBA:
            self._nb_dW = _NbList(self._dW_rule)
            self._nb_elig = _NbList(self._elig)
        self.last_input = None
        self.last_logits = np.zeros(N_OUT)
        self.last_command = (0.0, 0.0, 0.0)
        self.record = False                # inspector only: keep every sub-tick frame
        self.trace: list = []

    def _reindex(self) -> None:
        """Rebuild everything that depends on WHICH cells are alive and where. Called
        once at birth and once per development tick — O(alive), which is why growth is
        rare: at one rebuild per world tick this would cost more than the thinking."""
        G, C = self.genome.G, self.genome.C
        ys, xs = np.nonzero(self.mask)
        if len(ys) == 0:                                    # never let a body vanish
            cy = G // 2
            self.mask[cy, cy] = True
            ys, xs = np.nonzero(self.mask)
        self._ys, self._xs = ys, xs
        self._centre = (ys + 1) * (G + 2) + (xs + 1)
        self._nbr = np.stack([(ys + 1 + dy) * (G + 2) + (xs + 1 + dx)
                              for dy, dx in _OFFSETS], axis=1)
        types = self.type_map[ys, xs]
        self._type_rows = [np.nonzero(types == k)[0]
                           for k in range(self.genome.n_types)]
        self._M = build_listen(self.genome, ys, xs, self.layout,
                               self.in_mask, self.organ_mod)
        self._out_rows = np.nonzero(self.out_mask[ys, xs])[0]
        if len(self._out_rows) == 0:
            self._out_rows = np.arange(len(ys))
        self.morph = np.zeros((len(ys), N_MORPH))
        self._ax_buf = np.zeros((len(ys), C))
        self._rewire()
        if _NUMBA:
            self._nb_rows = _NbList(self._type_rows)

    def _rewire(self) -> None:
        """Re-point the axons at rows after the body changed, dropping any whose end
        died. Wiring outlives cells only as long as both ends are alive."""
        row = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(self._ys, self._xs))}
        kept, src, dst, w = [], [], [], []
        for ax in self.axon_cells:
            a, b = row.get((ax[0], ax[1])), row.get((ax[2], ax[3]))
            if a is None or b is None:
                continue
            kept.append(ax)
            src.append(a)
            dst.append(b)
            w.append(ax[4])
        self.axon_cells = kept
        self._ax_src = np.array(src, dtype=np.int64)
        self._ax_dst = np.array(dst, dtype=np.int64)
        self._ax_w = np.array(w, dtype=np.float64)
        self._ax_elig = np.zeros(len(kept))

    def cells(self) -> int:
        return int(self.mask.sum())

    def organs(self) -> tuple[int, int]:
        return (int((self.in_mask & self.mask).sum()),
                int((self.out_mask & self.mask).sum()))

    def types_used(self) -> int:
        return int(len(np.unique(self.type_map[self.mask]))) if self.mask.any() else 0

    # --- the tick ------------------------------------------------------------------
    def act(self, sensors: np.ndarray, hormone=None) -> tuple[float, float, float]:
        """`hormone` is the BODY talking to the whole sheet at once — energy, poison
        and the last reward. It reaches every cell regardless of organs (a bloodstream,
        not a wire) and it swings every rule's gain, so a starving brain is not the
        same computer as a fed one."""
        g = self.genome
        if hormone is not None:
            self.hormone = np.clip(np.asarray(hormone, dtype=np.float64), -1.0, 1.0)
        drive = np.ascontiguousarray((self._M * sensors) @ g.P_in)
        if self.cfg.hormone:
            drive = np.ascontiguousarray(drive + (self.hormone @ g.P_horm))
        hgate = 1.0
        if self.cfg.hormone:
            hgate += self.cfg.hormone_gain * float(self.hormone[0])
        n = len(self._ys)
        st = self.state[self._ys, self._xs]
        rp = 1 if self.cfg.rule_plastic else 0
        self.morph[:] = 0.0                    # fresh per thought: what the tissue
        if rp:                                 # wants NOW, not what it once wanted
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
                               1 if self.record else g.subticks,
                               self._chem0, self._chem1, self.cfg.chem_mix,
                               self.cfg.chem_decay, max(1, self.cfg.chem_every),
                               hgate, self._subtick, self.morph,
                               self._ax_src, self._ax_dst, self._ax_w,
                               self._ax_buf, self._ax_elig)
                self._subtick += 1 if self.record else g.subticks
                if self.record:
                    self._snap(st)
        else:
            for _ in range(g.subticks):
                ax = self._axon_drive(st)
                self._padF[self._centre] = st
                perc = self._padF[self._nbr].reshape(n, 9 * g.C)
                delta = np.empty((n, g.C + N_MORPH))
                for k, rows in enumerate(self._type_rows):
                    if len(rows):
                        delta[rows] = (_apply_rule_plastic(perc[rows], g.rules[k],
                                                           self._dW_rule[k], self._elig[k])
                                       if rp else _apply_rule(perc[rows], g.rules[k]))
                self.morph += delta[:, g.C:]                # the morphogenetic tail
                delta = delta[:, :g.C]
                gate = np.ones((n, 1))
                if self._mod_ch >= 0:
                    gate = (1.0 + np.tanh(st[:, self._mod_ch]))[:, None]
                st = np.tanh(st + gate * hgate * delta + drive + ax)
                self._subtick += 1
                if self._chem1 > self._chem0 and self._subtick % max(1, self.cfg.chem_every) == 0:
                    self._padF[self._centre] = st           # diffuse + decay the
                    hood = self._padF[self._nbr]            # chemical channels
                    avg = hood[:, :, self._chem0:self._chem1].mean(axis=1)
                    mix = self.cfg.chem_mix
                    st[:, self._chem0:self._chem1] = (
                        st[:, self._chem0:self._chem1] * (1.0 - mix) + avg * mix
                    ) * (1.0 - self.cfg.chem_decay)
                if self.record:
                    self._snap(st)
        if rp:
            for k in range(len(self._E_rule)):
                nrm = g.subticks * max(1, len(self._type_rows[k]))
                self._E_rule[k] *= self.cfg.plastic_elig
                self._E_rule[k] += self._elig[k] / nrm
        self.morph /= max(1, g.subticks)       # a mean, so think-time does not decide
        self.state[self._ys, self._xs] = st    # how loudly a cell asks to divide
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

    # --- development ------------------------------------------------------------------
    def develop(self, energy_frac: float = 1.0, embryo: bool = False) -> dict:
        """A DEVELOPMENT TICK: read what each cell has been asking for and let the
        tissue act on it. This is where the genome stops being a body and starts being
        a program — nothing here places a cell, it only decides whether a cell's own
        request is granted.

        Three things do the deciding, and all three are needed:

        RESOURCE, a chemical, fed in at the BORDER in proportion to how full the body
        is and diffusing inward by the ordinary chemistry. A division spends it. So
        growth is paid for out of what the agent actually ate, and the deep interior —
        farthest from the supply — grows last. This is what gives the sheet a shape
        instead of a disc.

        POSITION, read from the same chemistry. A cell that decides to become a sensory
        organ takes the modality its strongest local chemical maps to (`organ_map`).
        Concentration decides fate: the French-flag model, and the reason an eye region
        is something that can FORM rather than something we place.

        ORDER, fixed and row-major, so a run reproduces exactly.

        Returns what happened, for the HUD and the journal."""
        g, cfg = self.genome, self.cfg
        if not cfg.develop:
            return {"born": 0, "died": 0}
        ys, xs, morph = self._ys, self._xs, self.morph
        chem0, chem1 = self._chem0, self._chem1
        n_chem = max(0, chem1 - chem0)
        self._feed_resource(energy_frac)
        born = died = 0
        alive = int(self.mask.sum())
        # RATE LIMIT. At most this many cells may be born or die in one development
        # tick, strongest request first. Without it the sheet is not a body but a
        # boiling one: an early run turned over 13611 births and 11572 deaths in a
        # single life and settled nothing. Tissue changes slowly — that is not a
        # concession to stability, it is what makes a shape mean anything at all.
        budget = max(1, min(int(alive * cfg.max_change_frac), cfg.max_born_per_dev))
        die_budget = int(budget * cfg.die_ratio)
        order_die = [i for i in range(len(ys)) if morph[i, DIE] > cfg.die_thresh]
        order_die.sort(key=lambda i: (-float(morph[i, DIE]), i))
        for i in order_die[:die_budget]:
            if alive <= 4:
                break
            y, x = int(ys[i]), int(xs[i])
            self.mask[y, x] = False
            self.in_mask[y, x] = self.out_mask[y, x] = False
            self.organ_mod[y, x] = 0
            self.state[y, x, :] = 0.0
            alive -= 1
            died += 1
        # FEASIBILITY FIRST, then rank. Ranking by urge alone and checking afterwards
        # is why an early version grew nothing at all: the loudest cell was usually one
        # buried in the middle with nowhere to put a daughter, it burned the whole
        # budget failing, and the sheet never moved.
        grow_plan = {}
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            if not self.mask[y, x] or morph[i, GROW] <= cfg.grow_thresh:
                continue
            if self.resource[y, x] < cfg.divide_cost:
                continue
            spot = self._free_spot(y, x, float(morph[i, DIR_Y]), float(morph[i, DIR_X]))
            if spot is not None:
                grow_plan[i] = spot
        ranked = sorted(grow_plan, key=lambda i: (-float(morph[i, GROW]), i))[:budget]
        grow_allow = set(ranked)
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            if not self.mask[y, x]:                       # died a moment ago
                continue
            want = morph[i]
            # ORGAN FATE, re-read every development tick, so a cell can also stop being
            # an organ when the chemistry around it changes
            if want[BE_SENSE] > cfg.organ_thresh and not self.out_mask[y, x]:
                self.in_mask[y, x] = True
                if n_chem:
                    j = int(np.argmax(np.abs(self.state[y, x, chem0:chem1])))
                    self.organ_mod[y, x] = int(g.organ_map[j % len(g.organ_map)])
            elif want[BE_SENSE] < -cfg.organ_thresh:
                self.in_mask[y, x] = False
                self.organ_mod[y, x] = 0
            if want[BE_ACT] > cfg.organ_thresh and not self.in_mask[y, x]:
                self.out_mask[y, x] = True
            elif want[BE_ACT] < -cfg.organ_thresh:
                self.out_mask[y, x] = False
            # DIVISION, paid for in resource, rate-limited, bounded by the lattice
            if i not in grow_allow or alive >= cfg.max_cells:
                continue
            spot = self._free_spot(y, x, float(want[DIR_Y]), float(want[DIR_X]))
            if spot is None or self.resource[y, x] < cfg.divide_cost:
                continue                                   # the plan went stale
            ny, nx = spot
            self.mask[ny, nx] = True
            self.type_map[ny, nx] = self.type_map[y, x]     # a daughter of its mother
            self.in_mask[ny, nx] = self.out_mask[ny, nx] = False
            self.organ_mod[ny, nx] = 0
            # the daughter inherits HALF the mother's state. That is also what makes
            # MIGRATION free: divide toward, then apoptose behind, and the cell has
            # moved with what it was holding.
            self.state[ny, nx, :] = self.state[y, x, :] * 0.5
            self.state[y, x, :] *= 0.5
            self.resource[y, x] -= cfg.divide_cost
            self.resource[ny, nx] = self.resource[y, x] * 0.5
            self.resource[y, x] *= 0.5
            alive += 1
            born += 1
        grown_ax = self._sprout(ys, xs, morph)
        if born or died:
            self._reindex()
        elif grown_ax:
            self._rewire()
        self._prune_axons()
        self.births += born
        self.deaths += died
        self.dev_ticks += 1
        return {"born": born, "died": died, "cells": alive, "embryo": embryo,
                "axons": len(self.axon_cells)}

    def _sprout(self, ys, xs, morph) -> int:
        """Grow axons. A GROWTH CONE: from the cell, walk up the guidance chemical one
        neighbour at a time, and wherever it ends up — if that is far enough away and
        alive — becomes the far end. So wiring follows the same gradient development
        reads its position from, which is the point of having the chemical layer at all,
        and the topology is GROWN rather than designed."""
        cfg = self.cfg
        if not cfg.axons or self._chem1 <= self._chem0:
            return 0
        guide = self._chem1 - 1                          # the last chemical guides
        made = 0
        order = [i for i in range(len(ys)) if morph[i, SPROUT] > cfg.sprout_thresh]
        order.sort(key=lambda i: (-float(morph[i, SPROUT]), i))
        for i in order:
            if len(self.axon_cells) >= cfg.max_axons or made >= cfg.max_born_per_dev:
                break
            y, x = int(ys[i]), int(xs[i])
            if not self.mask[y, x] or self.resource[y, x] < cfg.axon_cost:
                continue
            cy, cx = y, x
            seen = {(y, x)}
            # A cone that stopped at the first local peak walked two cells and gave up:
            # the guidance chemical is written by the cells themselves and is bumpy. So
            # it always MOVES, to the best neighbour it has not already visited — uphill
            # where there is a hill, onward where there is not.
            for _ in range(cfg.axon_max_len):
                best, by, bx = -1e18, cy, cx
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < self.genome.G and 0 <= nx < self.genome.G
                            and self.mask[ny, nx] and (ny, nx) not in seen
                            and self.state[ny, nx, guide] > best):
                        best, by, bx = self.state[ny, nx, guide], ny, nx
                if (by, bx) == (cy, cx):                 # nowhere left to go
                    break
                cy, cx = by, bx
                seen.add((cy, cx))
            if math.hypot(cy - y, cx - x) < cfg.axon_min_len:
                continue                                 # the neighbourhood covers this
            if any(a[0] == y and a[1] == x and a[2] == cy and a[3] == cx
                   for a in self.axon_cells):
                continue
            self.axon_cells.append([y, x, cy, cx, cfg.axon_w0])
            self.resource[y, x] -= cfg.axon_cost
            made += 1
        return made

    def _prune_axons(self) -> None:
        """Use it or lose it. Weight decays every development tick and a wire too weak
        to matter is removed, so a topology is lost as well as grown — otherwise the
        sheet only ever accumulates wiring and nothing is ever a decision."""
        if not self.axon_cells:
            return
        keep = []
        for ax in self.axon_cells:
            ax[4] *= (1.0 - self.cfg.axon_decay)
            if abs(ax[4]) >= self.cfg.axon_min_w:
                keep.append(ax)
        if len(keep) != len(self.axon_cells):
            self.axon_cells = keep
            self._rewire()
        else:
            self._ax_w = np.array([a[4] for a in keep])

    def _free_spot(self, y: int, x: int, wy: float, wx: float):
        """Where a daughter can go: the four neighbours in the order the cell ASKED
        for, first free one wins. A strict single direction meant a cell that wanted
        to grow into an occupied neighbour simply did not divide, which made the whole
        interior sterile; a preference with fallbacks keeps the request meaningful
        while letting a crowded cell still find room."""
        G = self.genome.G
        dirs = sorted(((-1, 0), (1, 0), (0, -1), (0, 1)),
                      key=lambda d: (-(d[0] * wy + d[1] * wx), d))
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < G and 0 <= nx < G and not self.mask[ny, nx]:
                return ny, nx
        return None

    def _feed_resource(self, energy_frac: float) -> None:
        """Supply reaches the tissue from OUTSIDE — cells with a free neighbour take it
        in, in proportion to how full the body is — and then seeps inward and decays.
        So the deep interior is always the last to be fed and the last to grow, which
        is what stops a sheet from being a disc and gives development a geometry."""
        cfg = self.cfg
        ys, xs = self._ys, self._xs
        G = self.genome.G
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            if any(not self.mask[y + dy, x + dx]
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= y + dy < G and 0 <= x + dx < G):
                self.resource[y, x] += cfg.resource_gain * max(0.0, energy_frac)
        for _ in range(cfg.resource_spread):               # seep inward
            src = self.resource
            acc = np.zeros(len(ys))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                acc += src[np.clip(ys + dy, 0, G - 1), np.clip(xs + dx, 0, G - 1)]                     * self.mask[np.clip(ys + dy, 0, G - 1), np.clip(xs + dx, 0, G - 1)]
            mix = cfg.resource_mix
            self.resource[ys, xs] = (src[ys, xs] * (1 - mix) + acc / 4.0 * mix)
        self.resource[ys, xs] *= (1.0 - cfg.resource_decay)
        np.clip(self.resource, 0.0, cfg.resource_cap, out=self.resource)

    def _axon_drive(self, st) -> np.ndarray:
        """What arrives down the wires this sub-tick. A shortcut, not a new kind of
        signal: it lands as an additive DRIVE exactly like a sense does, so nothing
        about the cell's own rule has to change to receive one."""
        if len(self._ax_w) == 0:
            return 0.0
        self._ax_buf[:] = 0.0
        np.add.at(self._ax_buf, self._ax_dst, self._ax_w[:, None] * st[self._ax_src])
        # HEBBIAN eligibility: how much this wire's two ends fired together, which is
        # what `learn` reinforces when a reward lands. A topology that never changed
        # would be a wiring diagram, not a brain.
        self._ax_elig += np.abs(st[self._ax_src]).mean(axis=1) *             np.abs(st[self._ax_dst]).mean(axis=1)
        return self._ax_buf

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
        self.axon_learn(reward)
        if c.rule_plastic:
            for k in range(len(self._dW_rule)):
                self._dW_rule[k] *= (1.0 - c.plastic_forget)
                if reward:
                    self._dW_rule[k] += c.rule_lr * reward * self._E_rule[k]
                    np.clip(self._dW_rule[k], -c.rule_clip, c.rule_clip,
                            out=self._dW_rule[k])

    def axon_learn(self, reward: float) -> None:
        """Reward-gated Hebb on the wires, then forget a little. Same three-factor idea
        as the readout, applied to the topology instead of the weights."""
        if len(self._ax_w) == 0 or not self.cfg.axons:
            return
        c = self.cfg
        if reward:
            n = max(1e-9, float(self._ax_elig.max()))
            self._ax_w += c.axon_lr * reward * (self._ax_elig / n)
            np.clip(self._ax_w, -c.axon_clip, c.axon_clip, out=self._ax_w)
            for i, ax in enumerate(self.axon_cells):
                ax[4] = float(self._ax_w[i])
        self._ax_elig *= 0.5

    def axon_stats(self) -> dict:
        if not len(self._ax_w):
            return {"n": 0, "mean_len": 0.0, "mean_w": 0.0}
        lens = [math.hypot(a[2] - a[0], a[3] - a[1]) for a in self.axon_cells]
        return {"n": len(self._ax_w), "mean_len": round(float(np.mean(lens)), 1),
                "mean_w": round(float(np.abs(self._ax_w).mean()), 3)}

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
    _jitter(g.P_horm, rng, cfg.weight_mut_prob, cfg.weight_mut)
    if rng.random() < cfg.organ_mod_prob:                   # re-read one chemical as a
        j = int(rng.integers(len(g.organ_map)))             # different organ FATE
        g.organ_map[j] = int(rng.integers(1, cfg.organ_mods + 1))

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
