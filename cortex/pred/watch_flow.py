"""Watch a flow ring keep the world while the body turns, at several sizes on the same weights.

    python -m cortex.pred.watch_flow --run flowc-grow8_16_32-s0 --sides 8,16,32,64

Validation worlds only; the same episode choice and replay as `watch`, so what is drawn is what
the brain was fed. The space-time picture is the point: x = direction relative to the body,
y = time. A wall that stays put in the world while the body turns is a slanted stripe, so memory
that really flows keeps slanting after the eyes close.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from cortex import atlas_lineage as L
from cortex.pred.bench import COS_H, FOV, GEOM, SIN_H
from cortex.pred.model import FlowBrain, build
from cortex.pred.spin import OUT, Eps, ray_angles
from cortex.pred.watch import (BG, BRAIN_C, CLOSED_C, DIM, INK, NEAR, OPEN_C, big, label, pick_episode,
                               replay, world_panel)

NEAR_CH = 4                    # nearness among a ray's six channels
TIME_PX, DIR_PX = 7, 2
DIRS = 180                     # columns of every picture, whatever the size of the ring


def belief_by_direction(brain: FlowBrain, state: torch.Tensor, d: Eps) -> np.ndarray:
    """(t, DIRS): nearness each direction around the body is believed to have, resampled."""
    with torch.no_grad():
        z = brain.belief(state)[:, :, NEAR_CH].numpy()
    near = z * d.std[NEAR_CH] + d.mean[NEAR_CH]
    w = near.shape[1]
    at = np.arange(DIRS) / DIRS * w
    lo = np.floor(at).astype(int)
    f = at - lo
    return near[:, lo % w] * (1 - f) + near[:, (lo + 1) % w] * f


def truth_by_direction(raw: np.ndarray, head: np.ndarray) -> np.ndarray:
    """(t, DIRS): the real nearness by relative direction, from everything seen in the episode.
    The body does not move, so what a ray saw at some world angle is there all along."""
    world = np.full(DIRS, np.nan)
    for t in range(len(raw)):
        idx = ((ray_angles(head[t]) % (2 * math.pi)) / (2 * math.pi) * DIRS).astype(int) % DIRS
        world[idx] = raw[t, NEAR]
    known = np.flatnonzero(~np.isnan(world))
    world = np.interp(np.arange(DIRS), known, world[known], period=DIRS)
    out = np.zeros((len(raw), DIRS))
    for t in range(len(raw)):
        at = ((head[t] % (2 * math.pi)) / (2 * math.pi) * DIRS + np.arange(DIRS)).astype(int) % DIRS
        out[t] = world[at]
    return out


def centred(a: np.ndarray) -> np.ndarray:
    """Straight ahead in the middle of the picture."""
    return np.roll(a, DIRS // 2, axis=-1)


def picture(a: np.ndarray, t: int | None = None) -> Image.Image:
    g = (28 + np.clip(centred(a), 0, 1) * 215).astype(np.uint8)
    if t is not None:
        g = g.copy()
        g[t + 1:] = 24
    rgb = np.stack([g, g, g], -1)
    return Image.fromarray(rgb).resize((DIRS * DIR_PX, len(a) * TIME_PX), Image.NEAREST)


def run_brain(brain: FlowBrain, d: Eps, i: int) -> torch.Tensor:
    keep: list[torch.Tensor] = []
    idx = np.array([i])
    s, a = torch.tensor(d.sens[idx]), torch.tensor(d.act[idx])
    with torch.no_grad():
        brain.run(s[:, :-1], a[:, :-1], d.eyes(idx)[:, :-1], keep=keep)
    return torch.cat(keep)


def belief_polygon(img: Image.Image, belief: np.ndarray, reach: float, head: float) -> None:
    """The belief all round the body, drawn on the world panel, in that panel's coordinates."""
    from cortex.pred.watch import CELL, WORLD_R
    d = ImageDraw.Draw(img, "RGBA")
    cx = (WORLD_R + 0.5) * CELL
    pts = []
    for k in range(DIRS + 1):
        a = head + k % DIRS / DIRS * 2 * math.pi
        dist = (1 - float(np.clip(belief[k % DIRS], 0, 1))) * reach
        pts.append((cx + math.cos(a) * dist * CELL, cx + math.sin(a) * dist * CELL))
    d.line(pts, fill=(*BRAIN_C, 255), width=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="flowc-grow8_16_32-s0")
    p.add_argument("--arm", default="flowc")
    p.add_argument("--sides", default="8,16,32,64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    tr = Eps("train")
    val = Eps("val", tr.mean, tr.std)
    del tr
    i = pick_episode(val) if args.episode is None else args.episode
    lap = int(val.lap[i])
    frames = replay(i, val.raw[i])
    raw = val.raw[i]
    head = np.arctan2(raw[:, SIN_H], raw[:, COS_H])
    state = torch.load(OUT / "runs" / args.run / "brain.pt", weights_only=True)
    sides = [int(x) for x in args.sides.split(",")]
    beliefs = {}
    for side in sides:
        b = build(args.arm, args.seed, side).eval()
        b.load_state_dict(state)
        beliefs[side] = belief_by_direction(b, run_brain(b, val, i), val)
    truth = truth_by_direction(raw[:-1], head[:-1])
    reach = L.WorldConfig(**L.WORLDS["default"]).ray_reach
    n = len(truth)
    biggest = beliefs[sides[-1]]
    pan_w, pan_h = DIRS * DIR_PX, n * TIME_PX
    cols = 1 + len(sides)
    w = 14 + 410 + cols * (pan_w + 12) + 10
    h = max(pan_h + 90, 470)
    imgs = []
    for t in range(n):
        im = Image.new("RGB", (w, h), BG)
        dr = ImageDraw.Draw(im)
        open_now = t < lap
        col = OPEN_C if open_now else CLOSED_C
        label(dr, (14, 10), "EYES OPEN - lap 1" if open_now else "EYES CLOSED - lap 2", 20, col)
        label(dr, (250, 15), f"frame {t + 1}/{n}", 13, DIM)
        pred = None
        panel = world_panel(frames, t, raw[t], pred, open_now, reach)
        belief_polygon(panel, biggest[t], reach, head[t])
        im.paste(panel, (14, 42))
        label(dr, (14, 438), f"blue: what the {sides[-1]}-column ring believes, all round", 13, BRAIN_C)
        x = 438
        for name, a in [("truth", truth)] + [(f"{s} columns", beliefs[s]) for s in sides]:
            label(dr, (x, 15), name, 14, INK if name == "truth" else BRAIN_C)
            im.paste(picture(a, t), (x, 42))
            y_eye = 42 + min(lap, n) * TIME_PX
            dr.line([x, y_eye, x + pan_w, y_eye], fill=CLOSED_C, width=1)
            x += pan_w + 12
        label(dr, (438, 42 + pan_h + 6), "x: direction from the body, behind - ahead - behind.  y: time, red line = eyes close.  bright = near", 13, DIM)
        label(dr, (438, 42 + pan_h + 24), f"the same 6 381 weights at every size; {args.run}, validation world", 13, DIM)
        imgs.append(im)
    out = OUT / "watch"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}-ep{i}"
    imgs[0].save(out / f"{stem}.gif", save_all=True, append_images=imgs[1:] + [imgs[-1]] * 8,
                 duration=int(1000 / args.fps), loop=0, optimize=True)
    imgs[-1].save(out / f"{stem}.png")
    print(f"wrote {out / stem}.gif, episode {i}, lap {lap}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
