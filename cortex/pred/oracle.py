"""How much is there to remember? A hand-made memory on the same exam as the brains.

    python -m cortex.pred.oracle            # validation worlds

During the 32 frames with open eyes it paints every sensed ray onto a map using the TRUE
pose; with closed eyes it casts the 32 rays from the true future pose through that map,
stepping over gaps of one cell (32 rays leave holes far out). A ray that reaches a known
wall predicts that wall; one that runs out into the unseen predicts the mean.

The exam is also scored on ANSWERABLE rays only: those whose true end is a solid cell that
was seen as a wall during the open-eyes frames. There a good memory has the answer, while
on the other rays it cannot have one however good it is, and an agent walking into new
land makes most rays of that other kind. This split is the point of the module.

Also re-marks the saved brains on denser windows with a paired bootstrap, so that small
differences between arms can be told from noise.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from cortex.pred.bench import COS_H, FOV, GEOM, HORIZONS, SIN_H, Data, rotated
from cortex.pred.model import build

REACH = 22.0
N_LOGGED_RAYS = 64
STEP = FOV / N_LOGGED_RAYS
FREE, WALL = 1, 2
STRIDE = 50
BURN, CLOSED = 32, 32
NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def ray_angles(heading: float) -> np.ndarray:
    return heading - FOV / 2 + STEP / 2 + np.arange(0, N_LOGGED_RAYS, 2) * STEP


def hit_cell(y, x, ang, d):
    return int(math.floor(y + math.sin(ang) * (d + 0.01))), int(math.floor(x + math.cos(ang) * (d + 0.01)))


def paint(mem: dict, raw: np.ndarray, pose: np.ndarray) -> None:
    y, x, h = pose
    for j, ang in enumerate(ray_angles(h)):
        d = (1.0 - raw[6 * j + 4]) * REACH
        sn, cs = math.sin(ang), math.cos(ang)
        for t in np.arange(0.5, d, 0.5):
            mem.setdefault((int(math.floor(y + sn * t)), int(math.floor(x + cs * t))), FREE)
        if raw[6 * j + 5] > 0.5:
            mem[hit_cell(y, x, ang, d)] = WALL


def cast(mem: dict, pose: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalised (near, solid) per ray; 0 = the mean, for what memory cannot say."""
    y, x, h = pose
    out = np.zeros(64, np.float32)
    for j, ang in enumerate(ray_angles(h)):
        sn, cs = math.sin(ang), math.cos(ang)
        gap, last = 0.0, None
        near, solid = 0.0, 0.0
        known_end = True
        for t in np.arange(0.5, REACH, 0.5):
            cell = (int(math.floor(y + sn * t)), int(math.floor(x + cs * t)))
            if cell == last:
                continue
            last = cell
            m = mem.get(cell)
            if m == WALL:
                near, solid = 1.0 - t / REACH, 1.0
                break
            if m is None:
                gap += 1
                if gap > 1:
                    known_end = False
                    break
            else:
                gap = 0
        if known_end:
            out[2 * j] = (near - mean[2 * j]) / std[2 * j]
            out[2 * j + 1] = (solid - mean[2 * j + 1]) / std[2 * j + 1]
    return out


def windows(data: Data) -> list[int]:
    n = BURN + CLOSED
    return [s + o for s, l in zip(data.starts, data.lens) for o in range(0, l - n - 2, STRIDE)]


def answerable(mem: dict, raw: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Per ray of a future frame: does it really end on a wall the memory holds?"""
    y, x, h = pose
    out = np.zeros(32, bool)
    for j, ang in enumerate(ray_angles(h)):
        if raw[6 * j + 5] > 0.5:
            cy, cx = hit_cell(y, x, ang, (1.0 - raw[6 * j + 4]) * REACH)
            out[j] = any(mem.get((cy + a, cx + b)) == WALL for a, b in NEIGHBOURS)
    return out


def oracle_errors(data: Data, pose: np.ndarray):
    mean, std = data.mean[GEOM], data.std[GEOM]
    err: dict[int, list] = {h: [] for h in HORIZONS}
    ans: dict[int, list] = {h: [] for h in HORIZONS}
    for o in windows(data):
        mem: dict = {}
        for t in range(BURN):
            paint(mem, data.raw[o + t], pose[o + t])
        for h in HORIZONS:
            f = o + BURN - 1 + h
            err[h].append((cast(mem, pose[f], mean, std) - data.sens[f][GEOM]) ** 2)
            ans[h].append(answerable(mem, data.raw[f], pose[f]))
    return {h: np.array(v) for h, v in err.items()}, {h: np.array(v) for h, v in ans.items()}


def rivals(data: Data):
    r: dict = {"mean": {}, "hold": {}, "rot": {}}
    w = np.array(windows(data))
    last = data.sens[w + BURN - 1][:, GEOM]
    h0 = np.arctan2(data.raw[w + BURN - 1, SIN_H], data.raw[w + BURN - 1, COS_H])
    for h in HORIZONS:
        f = w + BURN - 1 + h
        tgt = data.sens[f][:, GEOM]
        dh = (np.arctan2(data.raw[f, SIN_H], data.raw[f, COS_H]) - h0 + math.pi) % (2 * math.pi) - math.pi
        r["mean"][h] = tgt ** 2
        r["hold"][h] = (last - tgt) ** 2
        r["rot"][h] = (rotated(last, dh) - tgt) ** 2
    return r


@torch.no_grad()
def brain_errors(brain, data: Data, batch: int = 64) -> dict[int, np.ndarray]:
    w = windows(data)
    n = BURN + CLOSED
    out: dict[int, list] = {h: [] for h in HORIZONS}
    for at in range(0, len(w), batch):
        ss = w[at:at + batch]
        s = torch.tensor(np.array([data.sens[o:o + n + 1] for o in ss]))
        a = torch.tensor(np.array([data.act[o:o + n + 1] for o in ss]))
        e = torch.ones(len(ss), n)
        e[:, BURN:] = 0.0
        pred = brain.run(s[:, :n], a[:, :n], e)[0].numpy()
        sn = s.numpy()
        for h in HORIZONS:
            out[h].append((pred[:, BURN - 2 + h][:, GEOM] - sn[:, BURN - 1 + h][:, GEOM]) ** 2)
    return {h: np.concatenate(v) for h, v in out.items()}


def on_rays(err: np.ndarray, mask: np.ndarray) -> float:
    """Mean error over the selected rays; err is (windows, 64), mask (windows, 32)."""
    return float(err[np.repeat(mask, 2, axis=1)].mean())


def boot(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """Mean of a - b and its bootstrap SE: paired over the same windows."""
    rng = np.random.default_rng(0)
    d = a - b
    idx = rng.integers(len(d), size=(n, len(d)))
    return float(d.mean()), float(d[idx].mean(1).std())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="val")
    p.add_argument("--match", default="", help="only runs whose name contains this")
    args = p.parse_args(argv)
    train = Data("train")
    data = Data(args.split, train.mean, train.std)
    pose = np.load(f"out/cortex/pred/data/{args.split}.npz")["pose"]
    riv = rivals(data)
    orc, ans = oracle_errors(data, pose)
    every = {h: np.ones_like(ans[h]) for h in HORIZONS}
    errs = {}
    for f in sorted(glob.glob("out/cortex/pred/runs/*/result.json")):
        run = Path(f).parent
        if args.match and args.match not in run.name:
            continue
        a = json.loads(Path(f).read_text(encoding="utf-8"))["args"]
        brain = build(a["arm"], a["seed"], a["side"])
        brain.load_state_dict(torch.load(run / "brain.pt"))
        errs[run.name] = brain_errors(brain, data)
    head = f"{'':<22}" + "".join(f"  h{h:<5}" for h in HORIZONS)
    print(f"{len(windows(data))} windows; error on the geometry of the view, lower is better")
    print("\nshare of rays that end on a wall already seen: "
          + "  ".join(f"h{h} {ans[h].mean():.0%}" for h in HORIZONS))
    for title, mask in (("ALL RAYS", every), ("ANSWERABLE RAYS", ans)):
        print(f"\n{title}\n{head}")
        for k in ("mean", "hold", "rot"):
            print(f"{k:<22}" + "".join(f"{on_rays(riv[k][h], mask[h]):7.3f} " for h in HORIZONS))
        print(f"{'HAND MEMORY':<22}" + "".join(f"{on_rays(orc[h], mask[h]):7.3f} " for h in HORIZONS))
        for name, e in errs.items():
            print(f"{name:<22}" + "".join(f"{on_rays(e[h], mask[h]):7.3f} " for h in HORIZONS))
    ref = "gru16-s0"
    if ref in errs:
        print("\nanswerable rays, h8: error minus gru16 per window (+- bootstrap SE, negative = better)")
        m = ans[8].astype(float)
        keep = m.sum(1) > 0

        def per_window(e):
            per_ray = e[:, ::2] + e[:, 1::2]
            return ((per_ray * m)[keep].sum(1) / m[keep].sum(1)) / 2
        for name, e in errs.items():
            if name != ref:
                d, se = boot(per_window(e[8]), per_window(errs[ref][8]))
                print(f"{name:<22}{d:+.3f} +- {se:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
