"""Logs for the prediction bench: what an agent senses, what it does, what is really around.

    python -m cortex.pred.log --jobs 18

An agent walks on `laws/atlas_course.nomos`, but keeps stopping to look around at random:
a forward walker hardly ever looks back at what it has seen, and then memory could not
pay. One frame = 2 thoughts = 6 ticks. `act[t]` is what was done between `sens[t]` and
`sens[t+1]`; `ego[t]` is the true solid map around the body in the body's frame
(forward, side), and `pose[t]` the true (y, x, heading): both kept only to examine what a
brain knows and what a perfect memory could do, never as inputs.
"""

from __future__ import annotations

import argparse
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cortex import atlas as A
from cortex import atlas_law as LAW
from cortex import atlas_lineage as L
from cortex.senses import RAY_CH
from cortex.tiles import SOLID

OUT = Path("out/cortex/pred/data")
LAW_FILE = "laws/atlas_course.nomos"
SPLITS = {"train": range(2000, 2060), "val": range(2060, 2076), "test": range(2080, 2112)}
EGO_R = 8                 # the true map kept: (2R+1)^2 cells around the body
THOUGHTS_PER_FRAME = 2


def reduce_senses(v: np.ndarray, n_rays: int) -> np.ndarray:
    """Every second ray (64 -> 32); smell, touch and the inner senses as they are."""
    rays = v[:n_rays * RAY_CH].reshape(n_rays, RAY_CH)[::2].ravel()
    return np.concatenate([rays, v[n_rays * RAY_CH:]]).astype(np.float32)


def ego_map(w, r: int = EGO_R) -> np.ndarray:
    a = w.agent
    pad = int(r * 1.5) + 2
    y0, x0 = int(a.y) - pad, int(a.x) - pad
    side = 2 * pad + 1
    solid = SOLID[w.tiles.region(y0, x0, side, side)]
    f, s = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    sn, cs = math.sin(a.heading), math.cos(a.heading)
    yy = np.floor(a.y + sn * f + cs * s).astype(int) - y0
    xx = np.floor(a.x + cs * f - sn * s).astype(int) - x0
    return solid[yy, xx].astype(np.uint8)


def life(args) -> dict:
    seed, ticks, look_p = args
    text = Path(LAW_FILE).read_text(encoding="utf-8")
    from cortex.world import World
    rule = LAW.LawRule(text, seed)
    w = World(seed=int(seed), cfg=L.WorldConfig(**L.WORLDS["default"]))
    mind = A.AtlasMind(L.Program({}).config(), w.cfg, rule)
    mind.attach(w)
    rng = np.random.default_rng(int(seed) + 77)
    look = {"left": 0, "cmd": None}
    plan = mind.atlas.act

    def act(vec, y, x, heading):
        cmd = plan(vec, y, x, heading)         # the map goes on being kept either way
        if look["left"] <= 0 and rng.random() < look_p:
            look["left"] = int(rng.integers(4, 17))
            look["cmd"] = (float(rng.choice([0.0, 0.0, 0.5])), float(rng.choice([-1.0, 1.0])), 0.0)
        if look["left"] > 0:
            look["left"] -= 1
            return look["cmd"]
        return cmd
    mind.atlas.act = act
    n = w.cfg.n_rays
    sens, acts, egos, poses, cur = [], [], [], [], []
    think = mind.cfg.think_every
    thought = 0
    while w.agent.alive and w.tick < ticks:
        thinks = w.tick % think == 0
        record = thinks and thought % THOUGHTS_PER_FRAME == 0
        ego = ego_map(w) if record else None
        pose = (w.agent.y, w.agent.x, w.agent.heading)
        mind.step(w)
        if thinks:
            if record:
                if cur:
                    acts.append(np.mean(cur, axis=0))
                    cur = []
                sens.append(reduce_senses(mind.sensors, n))
                egos.append(ego)
                poses.append(pose)
            cur.append(mind.command)
            thought += 1
    k = len(acts)
    return {"sens": np.array(sens[:k], np.float32), "act": np.array(acts[:k], np.float32),
            "ego": np.array(egos[:k], np.uint8),
            "pose": np.array(poses[:k], np.float32), "seed": int(seed), "meals": int(sum(w.agent.ate))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jobs", type=int, default=18)
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--look", type=float, default=1 / 40, help="chance per thought to look round")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.jobs) as pool:
        for name, seeds in SPLITS.items():
            rows = pool.map(life, [(s, args.ticks, args.look) for s in seeds], chunksize=1)
            lens = [len(r["sens"]) for r in rows]
            np.savez_compressed(OUT / f"{name}.npz",
                                sens=np.concatenate([r["sens"] for r in rows]),
                                act=np.concatenate([r["act"] for r in rows]),
                                ego=np.concatenate([r["ego"] for r in rows]),
                                pose=np.concatenate([r["pose"] for r in rows]),
                                lens=np.array(lens), seeds=np.array([r["seed"] for r in rows]))
            print(f"{name}: {len(rows)} lives, {sum(lens)} frames, median {int(np.median(lens))}, "
                  f"meals {np.mean([r['meals'] for r in rows]):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
