"""NEAT-lite: brains whose TOPOLOGY evolves, not just their weights.

A genome is a graph — input/output/hidden nodes and weighted connections — that
grows by mutation (add a connection, split a connection into a new node, perturb
weights, toggle links). Two genomes reproduce by CROSSOVER: connections are
aligned by their (src, dst) node ids (nodes carry globally-unique ids, so the
same historical innovation keeps its id across a lineage), matching genes are
inherited at random, and disjoint genes come from the fitter parent.

The phenotype is evaluated as a small recurrent net: node activations persist and
relax a few synchronous steps per decision, so signals cross hidden layers within
a tick and the net gets memory for free. Inputs = the agent's senses, outputs =
the action logits (argmax picks the move). Reproduction is NOT an output — it's an
innate proximity drive in the world.

Kept simpler than canonical NEAT: no global innovation registry (endpoint ids do
that job) and no speciation yet (DESIGN.md notes when to add it).
"""

from __future__ import annotations

import math

import numpy as np

from vivarium.vision import (DEFAULT_N_RAYS, DEFAULT_RAY_LEN, cross_vision,
                             drift_vision, random_vision)

# actions (outputs), matching the world's movement model. STAY = stand still and
# burn only the base metabolic cost (no move/turn) — an idle agent conserves energy,
# so parking to digest is a real strategy evolution can find.
FORWARD, TURN_LEFT, TURN_RIGHT, STAY = 0, 1, 2, 3
N_ACTIONS = 4

_ACT_RELAX = 3          # synchronous relaxation steps per decision


class Genome:
    """Graph brain. nodes: {id: kind in 'in'|'out'|'hidden'}; conns: list of
    {'in','out','w','enabled'}. Input ids are 0..n_in-1 (the last input is the
    constant bias), output ids are n_in..n_in+n_out-1; hidden ids are unique and
    larger."""

    def __init__(self, n_in: int, n_out: int, nodes: dict, conns: list[dict],
                 hue: float = 0.0, diet: float = 0.5, carnivore: float = 0.0,
                 n_rays: int = DEFAULT_N_RAYS, ray_len: float = DEFAULT_RAY_LEN):
        self.kind = "neat"
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = nodes
        self.conns = conns
        # heritable NICHE traits (inherited from a parent, drift slowly):
        #   diet      — which food TYPE this creature digests best (0..1)
        #   carnivore — plant-vs-meat allocation (0..1): 0 herbivore, 1 predator
        self.diet = diet
        self.carnivore = carnivore
        # heritable VISION (see vivarium.vision): how many rays this creature fans and
        # how far each reaches. Rides along on the grid world; steers the aqua world.
        self.n_rays = int(n_rays)
        self.ray_len = float(ray_len)
        # A HERITABLE lineage hue (0..1): children inherit a parent's hue and it
        # drifts only slowly, so a family shares a colour and related lineages sit
        # near each other in hue-space — unlike the brain fingerprint (`key`),
        # which avalanche-changes on any mutation. Colour = lineage; name = brain.
        self.hue = hue

    def copy(self) -> "Genome":
        return Genome(self.n_in, self.n_out, dict(self.nodes),
                      [dict(c) for c in self.conns], self.hue,
                      self.diet, self.carnivore, self.n_rays, self.ray_len)

    def max_node_id(self) -> int:
        return max(self.nodes) if self.nodes else self.n_in + self.n_out - 1

    def complexity(self) -> tuple[int, int]:
        n_hidden = sum(1 for k in self.nodes.values() if k == "hidden")
        n_enabled = sum(1 for c in self.conns if c["enabled"])
        return n_hidden, n_enabled

    def to_dict(self) -> dict:
        return {"brain": "neat", "n_in": self.n_in, "n_out": self.n_out,
                "hue": self.hue, "diet": self.diet, "carnivore": self.carnivore,
                "n_rays": self.n_rays, "ray_len": self.ray_len,
                "nodes": [[i, k] for i, k in self.nodes.items()],
                "conns": [[c["in"], c["out"], c["w"], c["enabled"]] for c in self.conns]}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        nodes = {int(i): k for i, k in d["nodes"]}
        conns = [{"in": int(a), "out": int(b), "w": float(w), "enabled": bool(e)}
                 for a, b, w, e in d["conns"]]
        return cls(int(d["n_in"]), int(d["n_out"]), nodes, conns,
                   float(d.get("hue", 0.0)), float(d.get("diet", 0.5)),
                   float(d.get("carnivore", 0.0)),
                   int(d.get("n_rays", DEFAULT_N_RAYS)),
                   float(d.get("ray_len", DEFAULT_RAY_LEN)))

    def key(self) -> bytes:
        """A canonical fingerprint of the topology + weights, for naming/colour."""
        parts = sorted((c["in"], c["out"], round(c["w"], 1), c["enabled"])
                       for c in self.conns)
        return repr(parts).encode()


def random_genome(n_in: int, n_out: int, rng: np.random.Generator,
                  forward_bias: float = 0.0, cfg=None) -> Genome:
    """A minimal starting brain: inputs fully connected to outputs with small
    random weights (a reactive single layer). The bias input (last input) gets a
    `forward_bias` weight to the FORWARD output so fresh brains tend to move.

    Weights are scaled by 1/sqrt(n_in) (Xavier-style): the drive into each output is
    a SUM over all inputs, so with a wide sensor (the heritable-vision world sizes it
    for n_rays_max rays) fixed-variance weights would saturate tanh and a fresh brain
    would twitch itself to death. Scaling keeps the drive ~O(1) at any input width, so
    a wider eye doesn't destabilize the population — and the forward_bias still leads."""
    nodes = {i: "in" for i in range(n_in)}
    nodes.update({n_in + j: "out" for j in range(n_out)})
    scale = 1.0 / math.sqrt(max(1, n_in))
    conns = []
    for i in range(n_in):
        for j in range(n_out):
            w = float(rng.normal(0.0, scale))
            if i == n_in - 1 and j == FORWARD:
                w += forward_bias
            conns.append({"in": i, "out": n_in + j, "w": w, "enabled": True})
    n_rays, ray_len = random_vision(rng, cfg)
    return Genome(n_in, n_out, nodes, conns, hue=float(rng.random()),
                  diet=float(rng.random()), carnivore=float(rng.random()) * 0.6,
                  n_rays=n_rays, ray_len=ray_len)


class Phenotype:
    """Runnable net built from a genome: a dense weight matrix over the small node
    set, plus persistent activations relaxed a few steps per decision."""

    def __init__(self, genome: Genome):
        self.genome = genome
        ids = list(genome.nodes)
        self.idx = {nid: k for k, nid in enumerate(ids)}
        n = len(ids)
        self.W = np.zeros((n, n), dtype=np.float64)
        for c in genome.conns:
            if c["enabled"] and c["in"] in self.idx and c["out"] in self.idx:
                self.W[self.idx[c["out"]], self.idx[c["in"]]] += c["w"]
        self.in_idx = np.array([self.idx[i] for i in range(genome.n_in)])
        self.out_idx = np.array([self.idx[genome.n_in + j]
                                 for j in range(genome.n_out)])
        self.is_input = np.zeros(n, dtype=bool)
        self.is_input[self.in_idx] = True
        self.state = np.zeros(n, dtype=np.float64)

    def act(self, inputs: np.ndarray) -> int:
        st = self.state
        st[self.in_idx] = inputs
        for _ in range(_ACT_RELAX):
            z = np.tanh(self.W @ st)
            st = np.where(self.is_input, st, z)
        self.state = st
        return int(np.argmax(st[self.out_idx]))


def mutate(genome: Genome, rng: np.random.Generator, cfg, next_node_id: int,
           leap: bool = False):
    """Return (mutated copy, next_node_id). Weight jitter always; structural
    mutations (add connection, split a connection into a node, toggle) at low
    rates so complexity grows gradually. `leap=True` forces the macro jolt (the
    NCA brain grows real structure on a leap; here it's a strong rewiring — the
    parameter exists so the World can call either brain's mutate uniformly)."""
    g = genome.copy()
    if rng.random() < cfg.hue_drift_prob:                # the lineage colour drifts slowly
        g.hue = (g.hue + float(rng.normal(0.0, cfg.hue_drift))) % 1.0
    if rng.random() < cfg.trait_drift_prob:              # niche traits drift slowly
        g.diet = min(1.0, max(0.0, g.diet + float(rng.normal(0.0, cfg.trait_drift))))
    if rng.random() < cfg.trait_drift_prob:
        g.carnivore = min(1.0, max(0.0,
                          g.carnivore + float(rng.normal(0.0, cfg.trait_drift))))
    drift_vision(g, rng, cfg)                            # heritable ray count / reach
    if leap or rng.random() < cfg.macro_mut_prob:        # rare SPECIATION jolt: a big
        g.hue = float(rng.random())                      # colour jump (new lineage) +
        g.diet = float(rng.random())                     # a new diet + maybe a predator
        g.carnivore = float(rng.random())                # + strong rewiring
        g.n_rays, g.ray_len = random_vision(rng, cfg)    # + re-rolled eyes
        for c in g.conns:
            c["w"] += float(rng.normal(0.0, cfg.weight_mut * cfg.macro_mut_scale))
    for c in g.conns:
        if rng.random() < cfg.weight_mut_prob:
            c["w"] += float(rng.normal(0.0, cfg.weight_mut))

    if rng.random() < cfg.add_conn_prob:
        srcs = [i for i, k in g.nodes.items() if k != "out"]
        dsts = [i for i, k in g.nodes.items() if k != "in"]
        if srcs and dsts:
            a = int(rng.choice(srcs))
            b = int(rng.choice(dsts))
            if a != b and not any(c["in"] == a and c["out"] == b for c in g.conns):
                g.conns.append({"in": a, "out": b,
                                "w": float(rng.normal(0.0, 1.0)), "enabled": True})

    enabled = [c for c in g.conns if c["enabled"]]
    if enabled and rng.random() < cfg.add_node_prob:
        c = enabled[int(rng.integers(len(enabled)))]
        c["enabled"] = False
        h = next_node_id
        next_node_id += 1
        g.nodes[h] = "hidden"
        g.conns.append({"in": c["in"], "out": h, "w": 1.0, "enabled": True})
        g.conns.append({"in": h, "out": c["out"], "w": c["w"], "enabled": True})

    if g.conns and rng.random() < cfg.toggle_prob:
        c = g.conns[int(rng.integers(len(g.conns)))]
        c["enabled"] = not c["enabled"]
    return g, next_node_id


def crossover(g_fit: Genome, g_other: Genome, rng: np.random.Generator) -> Genome:
    """Child genome: matching connections (same in,out) inherited at random from
    either parent; connections only the FITTER parent has are inherited too;
    those only the other parent has are dropped. Nodes are the union referenced by
    the child's connections plus all inputs/outputs."""
    other = {(c["in"], c["out"]): c for c in g_other.conns}
    child_conns = []
    for c in g_fit.conns:
        key = (c["in"], c["out"])
        pick = c if (key not in other or rng.random() < 0.5) else other[key]
        nc = dict(pick)
        # a disabled gene in either parent has a chance to re-enable in the child
        if not nc["enabled"] and rng.random() < 0.25:
            nc["enabled"] = True
        child_conns.append(nc)

    nodes = {i: "in" for i in range(g_fit.n_in)}
    nodes.update({g_fit.n_in + j: "out" for j in range(g_fit.n_out)})
    for c in child_conns:
        for nid in (c["in"], c["out"]):
            if nid not in nodes:
                nodes[nid] = "hidden"
    # each heritable trait inherited independently from a random parent
    hue = g_fit.hue if rng.random() < 0.5 else g_other.hue
    diet = g_fit.diet if rng.random() < 0.5 else g_other.diet
    carn = g_fit.carnivore if rng.random() < 0.5 else g_other.carnivore
    n_rays, ray_len = cross_vision(g_fit, g_other, rng)
    return Genome(g_fit.n_in, g_fit.n_out, nodes, child_conns, hue=hue,
                  diet=diet, carnivore=carn, n_rays=n_rays, ray_len=ray_len)
