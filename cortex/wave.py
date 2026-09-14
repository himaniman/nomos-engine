"""The rule that can be found: a per-cell perceptron in place of the hand-written wave.

    .\\.venv\\Scripts\\python.exe -m cortex.wave --evolve --generations 60 --jobs 16
    .\\.venv\\Scripts\\python.exe -m cortex.wave --exact --probes      # the built target
    .\\.venv\\Scripts\\python.exe -m cortex.wave --load out/cortex/wave/best.npy --seeds 0-15

`cortex/atlas.py` plays the game with a rule somebody WROTE: value = max(source, best
neighbour x decay) x passable. That settles that a cellular automaton with the right
local rule forages better than the reflex. It does not settle the question this
playground exists for — whether such a rule can be GROWN: found by a search over the
parameters of a small per-cell network, with nothing but the game to say which rule is
better. That is the question `Pathfinding Neural Cellular Automata` (Earle et al. 2023)
answers with gradient descent against a hand-coded BFS; here the teacher is the ladder
and the optimiser is evolution, because that is what a lineage has.

The cell. Each map cell holds the painted evidence (source, solid, value) plus a few
HIDDEN channels of its own, and every sub-tick the same small network reads its 3x3
neighbourhood and writes its value and hidden channels. Perception is fixed, the way
Mordvintsev's NCA fixes Sobel filters: for every channel the cell sees itself, the eight
neighbours, and the neighbourhood MAX and MEAN. Everything after that is weights. A
two-layer ReLU net over that perception can express the wave exactly — `exact()` builds
the weights that do, so the target is known to be in the space and a search that fails
has only itself to blame.

The perceptron is a PART, not the structure: the structure is the lattice, the locality,
the persistence of state between thoughts and the scroll under the body. What is being
searched for is the one local rule, a few hundred numbers.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from cortex import atlas as A

# what the rule sees per cell, in this order: the painted source, how solid the cell
# is, the value it holds, then its hidden channels
N_SEEN = 3
_NB = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class NeuralRule:
    """A per-cell perceptron: 3x3 perception (raw + max + mean per channel) -> ReLU
    hidden layer -> new value and hidden channels. Pure numpy, whole-lattice."""

    def __init__(self, theta: np.ndarray | None = None, width: int = 12,
                 hidden: int = 2, seed: int = 0, scale: float = 0.1):
        self.width, self.hidden = int(width), int(hidden)
        self.n_in = N_SEEN + self.hidden
        self.n_feat = self.n_in * 11                  # 9 raw + max + mean
        self.n_out = 1 + self.hidden
        self.n_params = self.n_feat * self.width + self.width + self.width * self.n_out + self.n_out
        if theta is None:
            theta = np.random.default_rng(seed).normal(0.0, scale, self.n_params)
        self.theta = np.asarray(theta, dtype=np.float64)
        assert self.theta.shape == (self.n_params,), (self.theta.shape, self.n_params)
        self._unpack()

    def _unpack(self) -> None:
        t, i = self.theta, 0
        self.W1 = t[i:i + self.n_feat * self.width].reshape(self.n_feat, self.width)
        i += self.n_feat * self.width
        self.b1 = t[i:i + self.width]
        i += self.width
        self.W2 = t[i:i + self.width * self.n_out].reshape(self.width, self.n_out)
        i += self.width * self.n_out
        self.b2 = t[i:i + self.n_out]

    def copy(self) -> "NeuralRule":
        return NeuralRule(self.theta.copy(), self.width, self.hidden)

    # --- one pass over the lattice ---------------------------------------------------
    def perceive(self, chans: np.ndarray) -> np.ndarray:
        """(n_in, S, S) -> (S, S, n_feat): every channel's 3x3 raw values, then its
        neighbourhood max and mean. Zero beyond the map's edge."""
        n, S, _ = chans.shape
        pad = np.zeros((n, S + 2, S + 2))
        pad[:, 1:-1, 1:-1] = chans
        raw = np.stack([pad[:, 1 + dy:1 + dy + S, 1 + dx:1 + dx + S] for dy, dx in _NB],
                       axis=0)                                   # (9, n, S, S)
        nb = raw[[k for k, o in enumerate(_NB) if o != (0, 0)]]  # the eight others
        feats = np.concatenate([raw.transpose(2, 3, 1, 0).reshape(S, S, n * 9),
                                nb.max(axis=0).transpose(1, 2, 0),
                                nb.mean(axis=0).transpose(1, 2, 0)], axis=2)
        return feats

    def step(self, state: np.ndarray, src: np.ndarray, passable: np.ndarray,
             passes: int) -> None:
        S = state.shape[1]
        solid = 1.0 - passable
        for _ in range(passes):
            chans = np.concatenate([src[None], solid[None], state[A.VALUE][None],
                                    state[A.N_CH:A.N_CH + self.hidden]], axis=0)
            feats = self.perceive(chans)
            h = np.maximum(0.0, feats @ self.W1 + self.b1)
            out = h @ self.W2 + self.b2
            state[A.VALUE] = np.clip(out[:, :, 0], 0.0, 2.0)
            if self.hidden:
                state[A.N_CH:A.N_CH + self.hidden] = np.tanh(out[:, :, 1:]).transpose(2, 0, 1)


def exact(decay: float = 0.06, width: int = 12, hidden: int = 2) -> NeuralRule:
    """The weights that ARE the hand-written wave, in this architecture:
        h0 = relu(src - k * max8(V)),  h1 = relu(k * max8(V)),  V = relu(h0 + h1 - 2 solid)
    so V = max(src, k * max8(V)) - 2 solid, clipped at zero: through a free cell, cut
    at a solid one. (The diagonal discount and the leak are dropped; they are details
    of the hand rule, not of the claim.)"""
    r = NeuralRule(np.zeros(NeuralRule(width=width, hidden=hidden).n_params), width, hidden)
    n_in = r.n_in
    i_src_c, i_sol_c = 0 * 9 + 4, 1 * 9 + 4            # centre of the raw block
    i_v_max = 9 * n_in + 2                             # max block, value channel
    k = 1.0 - decay
    r.W1[i_src_c, 0] = 1.0
    r.W1[i_v_max, 0] = -k
    r.W1[i_v_max, 1] = k
    r.W2[0, 0] = 1.0
    r.W2[1, 0] = 1.0
    r.W1[i_sol_c, 2] = 1.0                             # h2 = relu(solid)
    r.W2[2, 0] = -2.0
    return r


# --- fitness --------------------------------------------------------------------------

def ladder(theta, width: int, hidden: int, ticks: int = 600, cfg_d: dict | None = None) -> dict:
    """The probe ladder with this rule in the map. Deterministic, so a candidate has ONE
    score and a (1+lambda) strategy needs no re-measuring."""
    from cortex import probes
    cfg = A.Config(**(cfg_d or {}))
    rule = NeuralRule(np.asarray(theta), width, hidden)
    out = probes._suite(lambda: A.Driver(cfg, rule=rule), ticks)
    return out["profile"]


def _fit(args):
    theta, width, hidden, ticks, cfg_d = args
    return ladder(theta, width, hidden, ticks, cfg_d)


def evolve(generations: int = 60, lam: int = 31, sigma: float = 0.05, seed: int = 0,
           width: int = 12, hidden: int = 2, ticks: int = 600, jobs: int = 16,
           start: np.ndarray | None = None, out_dir: str = "out/cortex/wave",
           log=print, cfg_d: dict | None = None) -> dict:
    """(1 + lambda) over the rule's weights, scored on the ladder. Rechenberg's 1/5 rule
    on sigma. The parent is never re-measured because the fitness is exact."""
    from multiprocessing import Pool
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    parent = NeuralRule(start, width, hidden, seed=seed) if start is not None else \
        NeuralRule(None, width, hidden, seed=seed)
    best = ladder(parent.theta, width, hidden, ticks, cfg_d)
    history = [best["mean"]]
    log(f"gen   0  fit {best['mean']:.3f}  {_row(best)}  sigma {sigma:.4f}")
    pool = Pool(jobs) if jobs > 1 else None
    try:
        for gen in range(1, generations + 1):
            t0 = time.time()
            kids = [parent.theta + rng.normal(0.0, sigma, parent.n_params) for _ in range(lam)]
            args = [(k, width, hidden, ticks, cfg_d) for k in kids]
            scores = pool.map(_fit, args) if pool else [_fit(a) for a in args]
            i = int(np.argmax([s["mean"] for s in scores]))
            kept = scores[i]["mean"] > best["mean"]
            if kept:
                parent, best = NeuralRule(kids[i], width, hidden), scores[i]
                np.save(os.path.join(out_dir, "best.npy"), parent.theta)
            sigma = min(0.5, max(0.002, sigma * (1.3 if kept else 0.93)))
            history.append(best["mean"])
            log(f"gen {gen:>3}  fit {best['mean']:.3f}  {_row(best)}  sigma {sigma:.4f}"
                f"  {'KEPT' if kept else '    '}  kids max {scores[i]['mean']:.3f}"
                f" mean {np.mean([s['mean'] for s in scores]):.3f}  {time.time() - t0:.0f}s")
    finally:
        if pool:
            pool.close()
            pool.join()
    return {"rule": parent, "profile": best, "history": history}


def _row(p: dict) -> str:
    return " ".join(f"{k[:4]} {v:.2f}" for k, v in p.items() if k != "mean")


# --- the command line ------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Find the wave: evolve a per-cell perceptron.")
    p.add_argument("--evolve", action="store_true")
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--lam", type=int, default=31)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--hidden", type=int, default=2)
    p.add_argument("--ticks", type=int, default=600, help="per probe, while evolving")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--from-exact", action="store_true", help="start the search at the target")
    p.add_argument("--exact", action="store_true", help="use the built target rule")
    p.add_argument("--load", default="", help="a saved theta (.npy)")
    p.add_argument("--probes", action="store_true")
    p.add_argument("--seeds", default="", help="open-world lives with the rule")
    p.add_argument("--life-ticks", type=int, default=20000)
    p.add_argument("--out", default="out/cortex/wave")
    args = p.parse_args(argv)

    rule = None
    if args.exact:
        rule = exact(width=args.width, hidden=args.hidden)
    elif args.load:
        rule = NeuralRule(np.load(args.load), args.width, args.hidden)
    if args.evolve:
        os.makedirs(args.out, exist_ok=True)
        logf = open(os.path.join(args.out, "evolve.log"), "a", encoding="utf-8")

        def log(s):
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()
        start = exact(width=args.width, hidden=args.hidden).theta if args.from_exact else \
            (rule.theta if rule is not None else None)
        res = evolve(args.generations, args.lam, args.sigma, args.seed, args.width,
                     args.hidden, args.ticks, args.jobs, start, args.out, log)
        rule = res["rule"]
        logf.close()
    if rule is None:
        rule = NeuralRule(None, args.width, args.hidden, seed=args.seed)
    if args.probes:
        from cortex import probes
        out = probes._suite(lambda: A.Driver(A.Config(), rule=rule), 900)
        print("wave    " + probes.summarise(out))
        probes.render(out, os.path.join(args.out, "probes.png"))
    if args.seeds:
        rows = A.lives(A._seeds(args.seeds), args.life_ticks, A.Config(), jobs=args.jobs,
                       rule=rule)
        print(A.table(rows, "wave "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
