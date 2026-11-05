"""The simplest memory there is: stand still, turn round twice, the second lap with closed eyes.

    python -m cortex.pred.spin --make --jobs 8               # the episodes
    python -m cortex.pred.spin --arm gru16 --steps 3000      # train one brain
    python -m cortex.pred.spin --table                       # compare

The body does not move, the heading is felt all along, and the second lap shows exactly what
the first one did. Nothing has to be shifted or aligned: a brain only has to keep the view by
direction and look it up, the job a 1-D strip could do. If training cannot find even that,
the walking exam says nothing about any architecture. A hand-made PANORAMA (the seen rays,
looked up by their absolute angle) is the ceiling; MEAN is knowing nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cortex import atlas_lineage as L
from cortex.pred.bench import COS_H, FOV, GEOM, SIN_H
from cortex.pred.log import reduce_senses
from cortex.pred.model import ARMS, FLOW, PLAIN, RETINA, build
from cortex.senses import ScentField, sense
from cortex.world import World

OUT = Path("out/cortex/pred/spin")
SPLITS = {"train": (range(3000, 3040), 75), "val": (range(3060, 3072), 34),
          "test": (range(3080, 3092), 34)}
FRAMES = 64
TICKS = 6
TURNS = (0.3, 0.4, 0.5, 0.7, 1.0)
SPREAD = 100.0
N_LOGGED_RAYS = 64
STEP = FOV / N_LOGGED_RAYS


def world_episodes(args) -> dict:
    seed, n = args
    w = World(seed=int(seed), cfg=L.WorldConfig(**L.WORLDS["default"]))
    rng = np.random.default_rng(int(seed) + 5)
    field = ScentField(w.cfg.smell_radius)
    a, c = w.agent, w.cfg
    y0, x0 = a.y, a.x
    sens, act, lap = [], [], []
    for _ in range(n):
        for _ in range(200):                     # somewhere free enough to stand
            y = float(y0 + rng.uniform(-SPREAD, SPREAD))
            x = float(x0 + rng.uniform(-SPREAD, SPREAD))
            if not any(w.tiles.solid(int(y) + i, int(x) + j) for i in (-1, 0, 1) for j in (-1, 0, 1)):
                break
        a.y, a.x = y, x
        a.heading = float(rng.uniform(0, 2 * math.pi))
        a.energy, a.poison, a.alive = c.max_energy, 0.0, True
        turn = float(rng.choice(TURNS)) * float(rng.choice([-1.0, 1.0]))
        s = []
        for _ in range(FRAMES):
            s.append(reduce_senses(sense(w, field), c.n_rays))
            a.energy = c.max_energy
            for _ in range(TICKS):
                w.step(0.0, turn, 0.0)
        sens.append(s)
        act.append(np.tile([0.0, turn, 0.0], (FRAMES, 1)))
        lap.append(math.ceil(2 * math.pi / (TICKS * c.max_turn * abs(turn))))
    return {"sens": np.array(sens, np.float32), "act": np.array(act, np.float32), "lap": np.array(lap)}


def make(jobs: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(jobs) as pool:
        for name, (seeds, n) in SPLITS.items():
            t = time.time()
            rows = pool.map(world_episodes, [(s, n) for s in seeds], chunksize=1)
            sens = np.concatenate([r["sens"] for r in rows])
            np.savez_compressed(OUT / f"{name}.npz", sens=sens,
                                act=np.concatenate([r["act"] for r in rows]),
                                lap=np.concatenate([r["lap"] for r in rows]))
            print(f"{name}: {len(sens)} episodes in {time.time() - t:.0f} s", flush=True)


class Eps:
    def __init__(self, split, mean=None, std=None):
        d = np.load(OUT / f"{split}.npz")
        self.raw, self.act, self.lap = d["sens"], d["act"], d["lap"]
        self.mean = self.raw.mean((0, 1)) if mean is None else mean
        self.std = np.maximum(self.raw.std((0, 1)), 0.05) if std is None else std
        self.sens = (self.raw - self.mean) / self.std

    def eyes(self, idx):
        e = np.ones((len(idx), FRAMES), np.float32)
        for b, i in enumerate(idx):
            e[b, self.lap[i]:] = 0.0
        return torch.tensor(e)


def scored(lap: int) -> np.ndarray:
    """Frames of the second lap, skipping the first two after the eyes close."""
    return np.arange(lap + 2, min(2 * lap, FRAMES))


def ray_angles(heading: np.ndarray) -> np.ndarray:
    return heading[..., None] - FOV / 2 + STEP / 2 + np.arange(0, N_LOGGED_RAYS, 2) * STEP


def rivals(d: Eps) -> dict[str, np.ndarray]:
    """Per episode error on the geometry for each simple prediction, over the scored frames."""
    out = {"mean": [], "hold": [], "panorama": []}
    near_solid = GEOM.reshape(32, 2)
    for i in range(len(d.sens)):
        lap, f = int(d.lap[i]), scored(int(d.lap[i]))
        tgt = d.sens[i][f][:, GEOM]
        head = np.arctan2(d.raw[i, :, SIN_H], d.raw[i, :, COS_H])
        out["mean"].append((tgt ** 2).mean())
        out["hold"].append(((d.sens[i, lap - 1][GEOM] - tgt) ** 2).mean())
        seen_ang = ray_angles(head[:lap]).ravel()
        seen = d.sens[i, :lap][:, near_solid].reshape(-1, 2)        # (rays seen, 2)
        want = ray_angles(head[f])                                  # (frames, 32)
        diff = (want[..., None] - seen_ang[None, None] + math.pi) % (2 * math.pi) - math.pi
        pick = np.abs(diff).argmin(-1)
        out["panorama"].append(((seen[pick].reshape(len(f), -1) - tgt) ** 2).mean())
    return {k: np.array(v) for k, v in out.items()}


@torch.no_grad()
def brain_errors(brain, d: Eps, batch: int = 64) -> np.ndarray:
    out = []
    for at in range(0, len(d.sens), batch):
        idx = np.arange(at, min(at + batch, len(d.sens)))
        s, a = torch.tensor(d.sens[idx]), torch.tensor(d.act[idx])
        pred = brain.run(s[:, :-1], a[:, :-1], d.eyes(idx)[:, :-1])[0].numpy()
        for b, i in enumerate(idx):
            f = scored(int(d.lap[i]))
            out.append(((pred[b, f - 1][:, GEOM] - d.sens[i][f][:, GEOM]) ** 2).mean())
    return np.array(out)


def share(err: np.ndarray, riv: dict) -> float:
    """0 = knows nothing, 1 = as good as the panorama."""
    m, p = riv["mean"].mean(), riv["panorama"].mean()
    return float((m - err.mean()) / (m - p))


def train(args) -> int:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr = Eps("train")
    del tr.raw
    val = Eps("val", tr.mean, tr.std)
    riv = rivals(val)
    brain = build(args.arm, args.seed, args.side)
    plain = args.arm in PLAIN
    name = f"{args.arm}-s{args.seed}" if plain else f"{args.arm}-{args.side}-s{args.seed}"
    sides = [int(x) for x in args.sides.split(",")] if args.sides else None
    if sides:                      # every step on a random size, the same weights for all
        name = f"{args.arm}-grow{'_'.join(map(str, sides))}-s{args.seed}"
    run = OUT / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    print(f"{name}: weights {sum(p.numel() for p in brain.parameters())}", flush=True)
    opt = torch.optim.Adam(brain.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    losses, curve = [], []
    first = 1
    ckpt = run / "ckpt.pt"
    if ckpt.exists():              # a stopped run goes on where it stopped
        c = torch.load(ckpt, weights_only=False)
        brain.load_state_dict(c["brain"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        torch.set_rng_state(c["torch"])
        curve, first = c["curve"], c["step"] + 1
        print(f"{name}: resumed at step {first}", flush=True)
    for step in range(first, args.steps + 1):
        idx = rng.integers(len(tr.sens), size=args.batch)
        if sides:
            brain.set_side(int(rng.choice(sides)))
        s, a = torch.tensor(tr.sens[idx]), torch.tensor(tr.act[idx])
        pred = brain.run(s[:, :-1], a[:, :-1], tr.eyes(idx)[:, :-1])[0]
        loss = ((pred - s[:, 1:]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.every == 0 or step == args.steps:
            if sides:
                by_size = {}
                for side in sides:
                    brain.set_side(side)
                    by_size[side] = round(share(brain_errors(brain, val), riv), 3)
            e = brain_errors(brain, val)
            curve.append({"step": step, "s": round(time.time() - t0), "err": round(float(e.mean()), 4),
                          "share": round(share(e, riv), 3), "loss": round(float(np.mean(losses[-100:])), 4)})
            if sides:
                curve[-1]["by_size"] = by_size
            print(name, curve[-1], flush=True)
            torch.save({"brain": brain.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch": torch.get_rng_state(), "step": step, "curve": curve}, ckpt)
    torch.save(brain.state_dict(), run / "brain.pt")
    e = brain_errors(brain, val)
    (run / "result.json").write_text(json.dumps(
        {"args": vars(args), "weights": sum(p.numel() for p in brain.parameters()),
         "curve": curve, "val_err": e.tolist()}), encoding="utf-8")
    return 0


def table() -> int:
    tr = Eps("train")
    val = Eps("val", tr.mean, tr.std)
    riv = rivals(val)
    print(f"{len(val.sens)} episodes, second lap with closed eyes; error on the geometry of the view\n")
    for k, v in riv.items():
        print(f"{k:<22}{'':>9}  {v.mean():.3f}")
    print(f"\n{'brain':<22}{'weights':>9}  error  share of the road mean -> panorama")
    rows = []
    for f in sorted(glob.glob(str(OUT / "runs" / "*" / "result.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        e = np.array(r["val_err"])
        rows.append((Path(f).parent.name, r["weights"], e))
    for name, w, e in sorted(rows, key=lambda x: x[2].mean()):
        se = e.std(ddof=1) / math.sqrt(len(e))
        print(f"{name:<22}{w:>9}  {e.mean():.3f}  {share(e, riv):+.2f}  (+- {se:.3f})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--make", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--arm", choices=sorted(ARMS) + sorted(RETINA) + sorted(FLOW))
    p.add_argument("--side", type=int, default=16)
    p.add_argument("--sides", default="", help="train across sizes, e.g. 8,16,32 (flow arms only)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--every", type=int, default=500)
    args = p.parse_args(argv)
    if args.make:
        make(args.jobs)
        return 0
    if args.table:
        return table()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
