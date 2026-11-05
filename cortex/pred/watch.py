"""Watch a sheet sit the spin exam: the world, the tissue and what it believes it would see.

    python -m cortex.pred.watch --arm sheet --side 16
    python -m cortex.pred.watch --arm sheet --episode 136 --rival wide512

Validation worlds only. The episode is re-simulated to get the map and checked against the
logged senses, so what is drawn is what the brain was fed. Episodes are picked by how much
there is to remember in the view, never by how the brain did on them. Writes a GIF, two key
frames and numbers about the tissue next to the run's checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cortex import atlas_lineage as L
from cortex.pred.bench import COS_H, FOV, GEOM, SIN_H
from cortex.pred.log import reduce_senses
from cortex.pred.model import PLAIN, Sheet, build
from cortex.pred.spin import (FRAMES, OUT, SPLITS, SPREAD, TICKS, TURNS, Eps, brain_errors,
                              ray_angles, rivals, scored, share)
from cortex.senses import ScentField, sense
from cortex.tiles import _MATERIALS
from cortex.world import World

NEAR = GEOM[::2]                      # the nearness of each of the 32 rays
WORLD_R, CELL = 24, 8
BINS = 96
EPISODES_PER_WORLD = SPLITS["val"][1]
BG, INK, DIM = (17, 19, 24), (232, 232, 226), (140, 144, 152)
OPEN_C, CLOSED_C, BRAIN_C, RIVAL_C = (112, 200, 120), (226, 120, 90), (90, 220, 255), (255, 190, 90)
MAT_RGB = np.array([m[4] for m in _MATERIALS], np.uint8)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def run_name(arm: str, side: int, seed: int) -> str:
    return f"{arm}-s{seed}" if arm in PLAIN else f"{arm}-{side}-s{seed}"


def load(arm: str, side: int, seed: int) -> tuple[torch.nn.Module, str]:
    brain = build(arm, seed, side)
    run = OUT / "runs" / run_name(arm, side, seed)
    if (run / "brain.pt").exists():
        brain.load_state_dict(torch.load(run / "brain.pt", weights_only=True))
        return brain.eval(), "final"
    c = torch.load(run / "ckpt.pt", weights_only=False)
    brain.load_state_dict(c["brain"])
    return brain.eval(), f"step {c['step']}"


def pick_episode(d: Eps) -> int:
    """The most to remember in a view that closes two whole laps, on a slow turn."""
    best, best_v = 0, -1.0
    for i in range(len(d.raw)):
        lap = int(d.lap[i])
        if not 16 <= lap <= 32:
            continue
        v = float(d.raw[i, :lap][:, NEAR].mean(1).std() + d.raw[i, :lap][:, NEAR].std())
        if v > best_v:
            best, best_v = i, v
    return best


def replay(i: int, logged: np.ndarray) -> list[dict]:
    """Episode `i` of the validation split again, as `spin.world_episodes` lived it."""
    seeds = SPLITS["val"][0]
    seed, want = seeds[i // EPISODES_PER_WORLD], i % EPISODES_PER_WORLD
    w = World(seed=int(seed), cfg=L.WorldConfig(**L.WORLDS["default"]))
    rng = np.random.default_rng(int(seed) + 5)
    field = ScentField(w.cfg.smell_radius)
    a, c = w.agent, w.cfg
    y0, x0 = a.y, a.x
    frames: list[dict] = []
    for ep in range(want + 1):
        for _ in range(200):
            y = float(y0 + rng.uniform(-SPREAD, SPREAD))
            x = float(x0 + rng.uniform(-SPREAD, SPREAD))
            if not any(w.tiles.solid(int(y) + p, int(x) + q) for p in (-1, 0, 1) for q in (-1, 0, 1)):
                break
        a.y, a.x = y, x
        a.heading = float(rng.uniform(0, 2 * math.pi))
        a.energy, a.poison, a.alive = c.max_energy, 0.0, True
        turn = float(rng.choice(TURNS)) * float(rng.choice([-1.0, 1.0]))
        for t in range(FRAMES):
            s = reduce_senses(sense(w, field), c.n_rays)
            if ep == want:
                assert np.allclose(s, logged[t], atol=1e-4), f"replay left the log at frame {t}"
                frames.append({"y": a.y, "x": a.x, "heading": a.heading})
            a.energy = c.max_energy
            for _ in range(TICKS):
                w.step(0.0, turn, 0.0)
    r = WORLD_R + 1
    y0, x0 = int(a.y) - r, int(a.x) - r
    frames[0]["tiles"] = w.tiles.region(y0, x0, 2 * r + 1, 2 * r + 1)
    frames[0]["origin"] = (y0, x0)
    frames[0]["food"] = [(s.y, s.x) for s in w.nearby_food(a.y, a.x, r * 1.5)]
    return frames


@torch.no_grad()
def trace(brain, sens: torch.Tensor, act: torch.Tensor, eyes: torch.Tensor) -> dict:
    """`Brain.run` with the tissue kept: state (b, t, ch, side, side), read-out (b, t, cells)."""
    b, t = sens.shape[:2]
    core = brain.core
    s = core.init_state(b)
    preds, states, gates = [], [], []
    for i in range(t):
        e = eyes[:, i, None]
        seen = torch.cat([sens[:, i, :212] * e, sens[:, i, 212:]], 1)
        out, s = core(brain.enc(torch.cat([seen, act[:, i], e], 1)), s)
        preds.append(brain.dec(out))
        states.append(s)
        gates.append(torch.softmax(core.gate(s).flatten(1), 1))
    return {"pred": torch.stack(preds, 1), "state": torch.stack(states, 1),
            "gate": torch.stack(gates, 1)}


@torch.no_grad()
def predict(brain, d: Eps, idx: np.ndarray) -> np.ndarray:
    s, a = torch.tensor(d.sens[idx]), torch.tensor(d.act[idx])
    return brain.run(s[:, :-1], a[:, :-1], d.eyes(idx)[:, :-1])[0].numpy()


def liveliness(pred: np.ndarray, d: Eps, idx: np.ndarray) -> dict:
    """Over the closed lap: does the belief about a ray move at all, and does it move right?"""
    ratios, corrs = [], []
    for b, i in enumerate(idx):
        f = scored(int(d.lap[i]))
        if len(f) < 4:
            continue
        p, q = pred[b, f - 1][:, NEAR], d.sens[i][f][:, NEAR]
        ps, qs = p.std(0), q.std(0)
        ratios.append(ps.mean() / max(qs.mean(), 1e-6))
        ok = (ps > 1e-6) & (qs > 1e-6)
        if ok.any():
            pc, qc = p[:, ok] - p[:, ok].mean(0), q[:, ok] - q[:, ok].mean(0)
            corrs.append(((pc * qc).mean(0) / (ps[ok] * qs[ok])).mean())
    return {"belief_moves": float(np.mean(ratios)), "belief_corr": float(np.mean(corrs))}


def tissue_numbers(brain, d: Eps, batch: int = 32, every: int = 3) -> dict:
    """What 2048 numbers actually do, over validation episodes."""
    idx = np.arange(0, len(d.sens), every)
    states, gates, closed = [], [], []
    for at in range(0, len(idx), batch):
        k = idx[at:at + batch]
        s, a = torch.tensor(d.sens[k]), torch.tensor(d.act[k])
        e = d.eyes(k)[:, :-1]
        tr = trace(brain, s[:, :-1], a[:, :-1], e)
        states.append(tr["state"].numpy())
        gates.append(tr["gate"].numpy())
        closed.append(e.numpy() == 0)
    st = np.concatenate(states)                       # (ep, t, ch, y, x)
    gate = np.concatenate(gates)
    shut = np.concatenate(closed)
    ep, t = st.shape[:2]
    flat = st.reshape(ep * t, -1)
    centred = flat - flat.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False) ** 2
    cum = np.cumsum(sv) / sv.sum()
    total = centred.var(0).sum()
    uniform = st.mean((3, 4)).reshape(ep * t, -1)     # each channel averaged over the sheet
    uniform_var = uniform.var(0).sum() * st.shape[3] * st.shape[4]
    by_episode = st.mean(1, keepdims=True)
    within = float(((st - by_episode) ** 2).mean((0, 1)).sum() / max(total, 1e-9))
    ent = -(gate * np.log(gate + 1e-12)).sum(-1)
    moves = np.abs(np.diff(st, axis=1)).mean((2, 3, 4))
    return {"numbers": int(flat.shape[1]),
            "dims_90": int(np.searchsorted(cum, 0.90) + 1),
            "dims_99": int(np.searchsorted(cum, 0.99) + 1),
            "top5_variance": [round(float(x), 3) for x in (sv[:5] / sv.sum())],
            "same_everywhere": round(float(uniform_var / max(total, 1e-9)), 3),
            "over_time_share": round(within, 3),
            "cells_read": round(float(np.exp(ent).mean()), 1),
            "move_eyes_open": round(float(moves[shut[:, 1:] == 0].mean()), 4),
            "move_eyes_closed": round(float(moves[shut[:, 1:]].mean()), 4)}


def diverging(v: np.ndarray, scale: float) -> np.ndarray:
    x = np.clip(v / max(scale, 1e-6), -1, 1)
    neg, pos = np.array([64, 120, 240.0]), np.array([240, 90, 70.0])
    base = np.array([24, 26, 32.0])
    out = base + np.clip(x, 0, 1)[..., None] * (pos - base) + np.clip(-x, 0, 1)[..., None] * (neg - base)
    return out.astype(np.uint8)


def big(rgb: np.ndarray, px: int) -> Image.Image:
    return Image.fromarray(rgb).resize((rgb.shape[1] * px, rgb.shape[0] * px), Image.NEAREST)


def pca_rgb(state: np.ndarray) -> np.ndarray:
    """The tissue of one episode (t, ch, y, x) -> (t, y, x, 3): three main ways cells differ."""
    t, ch, h, w = state.shape
    cells = state.transpose(0, 2, 3, 1).reshape(-1, ch)
    centred = cells - cells.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = centred @ vt[:3].T
    lo, hi = np.percentile(pcs, 1, axis=0), np.percentile(pcs, 99, axis=0)
    scaled = np.clip((pcs - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return (scaled.reshape(t, h, w, 3) * 255).astype(np.uint8)


class Panorama:
    """Nearness by absolute direction, filled in as the head turns."""

    def __init__(self):
        self.v = np.full(BINS, np.nan)

    def write(self, angles: np.ndarray, near: np.ndarray) -> None:
        idx = ((angles % (2 * math.pi)) / (2 * math.pi) * BINS).astype(int) % BINS
        self.v[idx] = near

    def image(self, width: int, height: int) -> Image.Image:
        g = np.nan_to_num(np.clip(self.v, 0, 1), nan=0.0)
        rgb = np.stack([g, g, g], -1)
        rgb = 28 + rgb * 215
        rgb[np.isnan(self.v)] = (34, 36, 44)
        return Image.fromarray(rgb[None].astype(np.uint8)).resize((width, height), Image.NEAREST)


def world_panel(frames, t, raw, pred_near, eyes_open, reach) -> Image.Image:
    info = frames[0]
    f = frames[t]
    tiles, (oy, ox) = info["tiles"], info["origin"]
    cy, cx = int(f["y"]), int(f["x"])
    r = WORLD_R
    crop = tiles[cy - oy - r:cy - oy + r + 1, cx - ox - r:cx - ox + r + 1]
    img = big(MAT_RGB[crop], CELL).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    left, top = cx - r, cy - r

    def xy(py: float, px: float) -> tuple[float, float]:
        return (px - left) * CELL, (py - top) * CELL

    for fy, fx in info["food"]:
        p = xy(fy, fx)
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(240, 90, 170, 255))
    ang = ray_angles(np.array(f["heading"]))
    ray_c = (250, 240, 150, 110 if eyes_open else 30)
    me = xy(f["y"], f["x"])
    true_pts = []
    for a, n in zip(ang, raw[NEAR]):
        dist = (1 - float(n)) * reach
        p = xy(f["y"] + math.sin(a) * dist, f["x"] + math.cos(a) * dist)
        d.line([me, p], fill=ray_c, width=1)
        true_pts.append(p)
    d.line(true_pts, fill=(250, 240, 150, 255), width=2)
    if pred_near is not None:
        pts = []
        for a, n in zip(ang, pred_near):
            dist = (1 - float(np.clip(n, 0, 1))) * reach
            pts.append(xy(f["y"] + math.sin(a) * dist, f["x"] + math.cos(a) * dist))
        d.line(pts, fill=(*BRAIN_C, 255), width=3)
    d.ellipse([me[0] - 5, me[1] - 5, me[0] + 5, me[1] + 5], fill=(255, 255, 255, 255))
    return img


def label(d: ImageDraw.ImageDraw, xy, text, size=14, fill=INK) -> None:
    d.text(xy, text, font=font(size), fill=fill)


def render(frames, d: Eps, i: int, tr: dict, rival: np.ndarray | None, rival_name: str, tag: str,
           stats: dict, reach: float) -> list[Image.Image]:
    lap = int(d.lap[i])
    raw, std, mean = d.raw[i], d.std, d.mean
    state = tr["state"][0].numpy()                     # (t, ch, y, x)
    gate = tr["gate"][0].numpy()
    pred = tr["pred"][0].numpy()
    t_len = len(pred)
    side = state.shape[-1]
    near_of = lambda z: z[NEAR] * std[NEAR] + mean[NEAR]
    head = np.arctan2(raw[:, SIN_H], raw[:, COS_H])
    pca = pca_rgb(state)
    scales = np.percentile(np.abs(state), 99, axis=(0, 2, 3))
    gate_scale = np.percentile(gate, 99.5)
    delta = np.abs(np.diff(state, axis=0, prepend=state[:1])).mean(1)
    delta_scale = max(float(np.percentile(delta, 99)), 1e-6)
    err = ((pred[:, GEOM] - d.sens[i, 1:][:, GEOM]) ** 2).mean(1)
    err_r = None if rival is None else ((rival[:, GEOM] - d.sens[i, 1:][:, GEOM]) ** 2).mean(1)
    err_mean = (d.sens[i, 1:][:, GEOM] ** 2).mean(1)
    w, h = 1190, 800
    x_w, x_s, x_c = 14, 432, 722
    pano = {k: Panorama() for k in ("seen", "truth", "brain", "rival")}
    images = []
    for t in range(t_len):
        im = Image.new("RGB", (w, h), BG)
        dr = ImageDraw.Draw(im)
        eyes_open = t < lap
        ang = ray_angles(head[t])
        p_near = near_of(pred[t - 1]) if t else None
        # panoramas fill up with the frame being shown
        if eyes_open:
            pano["seen"].write(ang, raw[t, NEAR])
        else:
            pano["truth"].write(ang, raw[t, NEAR])
            pano["brain"].write(ang, p_near)
            if rival is not None:
                pano["rival"].write(ang, near_of(rival[t - 1]))
        # header
        c = OPEN_C if eyes_open else CLOSED_C
        label(dr, (x_w, 10), "EYES OPEN  - lap 1" if eyes_open else "EYES CLOSED  - lap 2", 22, c)
        label(dr, (x_w + 300, 15), f"frame {t + 1}/{t_len}", 14, DIM)
        label(dr, (x_s, 15), f"sheet {side}x{side}, 8 channels per cell  ({tag}), validation world", 14, DIM)
        # world
        im.paste(world_panel(frames, t, raw[t], p_near, eyes_open, reach), (x_w, 42))
        dr.rectangle([x_w, 42, x_w + (2 * WORLD_R + 1) * CELL, 42 + (2 * WORLD_R + 1) * CELL], outline=(60, 62, 70))
        label(dr, (x_w, 438), "yellow: what the rays really hit", 13, (250, 240, 150))
        label(dr, (x_w + 220, 438), "blue: what the brain expects", 13, BRAIN_C)
        # the tissue
        cell_px = 256 // side
        label(dr, (x_s, 42), "cells: 3 main patterns as RGB", 13, INK)
        im.paste(big(pca[t], cell_px), (x_s, 62))
        small = 128 // side
        label(dr, (x_s, 328), "read-out listens", 13, INK)
        im.paste(big(diverging(gate[t].reshape(side, side), gate_scale), small), (x_s, 348))
        label(dr, (x_s + 136, 328), "change this frame", 13, INK)
        im.paste(big(diverging(delta[t], delta_scale), small), (x_s + 136, 348))
        label(dr, (x_c, 42), "the 8 channels (blue -, red +)", 13, INK)
        tile = 112
        for ch in range(state.shape[1]):
            cx, cy = x_c + (ch % 4) * (tile + 4), 62 + (ch // 4) * (tile + 4)
            im.paste(big(diverging(state[t, ch], scales[ch]), tile // side), (cx, cy))
        for n, line in enumerate((f"state numbers: {stats['numbers']}",
                                  f"directions holding 90% of the change: {stats['dims_90']}, 99%: {stats['dims_99']}",
                                  f"variance same across the whole sheet: {stats['same_everywhere']:.1%}",
                                  f"cells the read-out listens to: {stats['cells_read']:.0f} of {side * side}",
                                  f"change per frame, eyes open {stats['move_eyes_open']:.3f} / closed {stats['move_eyes_closed']:.3f}")):
            label(dr, (x_c, 304 + n * 20), line, 13, DIM)
        # panorama by absolute direction
        y = 500
        px, pw, ph = 150, w - 170, 34
        label(dr, (x_w, y - 24), "the view by absolute direction, -180..180 degrees (bright = near)", 14, INK)
        rows = [("lap 1, seen", "seen", OPEN_C), ("lap 2, truth", "truth", INK), ("lap 2, sheet", "brain", BRAIN_C)]
        if rival is not None:
            rows.append((f"lap 2, {rival_name}", "rival", RIVAL_C))
        for n, (text, key, col) in enumerate(rows):
            ry = y + n * (ph + 6)
            label(dr, (x_w, ry + 9), text, 13, col)
            im.paste(pano[key].image(pw, ph), (px, ry))
        # current field of view on the panorama
        lo = ((head[t] - FOV / 2) % (2 * math.pi)) / (2 * math.pi) * pw + px
        hi = ((head[t] + FOV / 2) % (2 * math.pi)) / (2 * math.pi) * pw + px
        by = y + len(rows) * (ph + 6)
        if lo <= hi:
            dr.line([lo, by, hi, by], fill=c, width=3)
        else:
            dr.line([px, by, hi, by], fill=c, width=3)
            dr.line([lo, by, px + pw, by], fill=c, width=3)
        # timeline
        ty, th = by + 18, 800 - by - 18 - 22
        label(dr, (x_w, ty + th // 2 - 8), "error per frame", 13, DIM)
        dr.rectangle([px, ty, px + pw, ty + th], outline=(50, 52, 60))
        fx = lambda k: px + k / max(t_len - 1, 1) * pw
        top_err = 2.0
        fy = lambda v: ty + th - min(v / top_err, 1.0) * th
        dr.rectangle([px, ty + th + 3, fx(min(lap, t_len)), ty + th + 8], fill=OPEN_C)
        dr.rectangle([fx(min(lap, t_len)), ty + th + 3, px + pw, ty + th + 8], fill=CLOSED_C)
        for series, col in ((err_mean, (110, 110, 118)), (err_r, RIVAL_C), (err, BRAIN_C)):
            if series is not None:
                pts = [(fx(k), fy(v)) for k, v in enumerate(series[:t + 1])]
                if len(pts) > 1:
                    dr.line(pts, fill=col, width=2)
        dr.line([fx(t), ty, fx(t), ty + th], fill=(200, 200, 200), width=1)
        images.append(im)
    return images


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="sheet")
    p.add_argument("--side", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--rival", default="wide512", help="a trained plain arm to show next to it, or none")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    tr = Eps("train")
    val = Eps("val", tr.mean, tr.std)
    del tr
    brain, tag = load(args.arm, args.side, args.seed)
    assert isinstance(brain.core, Sheet), "only a sheet in a Brain has a tissue to draw here"
    i = pick_episode(val) if args.episode is None else args.episode
    lap = int(val.lap[i])
    print(f"{run_name(args.arm, args.side, args.seed)} ({tag}), episode {i}, lap {lap} frames", flush=True)
    frames = replay(i, val.raw[i])
    idx = np.array([i])
    s, a = torch.tensor(val.sens[idx]), torch.tensor(val.act[idx])
    e = val.eyes(idx)[:, :-1]
    t = trace(brain, s[:, :-1], a[:, :-1], e)
    ran = predict(brain, val, idx)
    assert np.allclose(t["pred"].numpy(), ran, atol=1e-5), "trace() drifted from Brain.run"
    rival, rival_name = None, ""
    numbers: dict = {"run": run_name(args.arm, args.side, args.seed), "tag": tag, "episode": i, "lap": lap}
    riv = rivals(val)
    err = brain_errors(brain, val)
    numbers["share_val"] = round(share(err, riv), 3)
    numbers["err_val"] = round(float(err.mean()), 4)
    all_idx = np.arange(len(val.sens))
    numbers["sheet"] = liveliness(np.concatenate([predict(brain, val, all_idx[k:k + 64])
                                                  for k in range(0, len(all_idx), 64)]), val, all_idx)
    if args.rival != "none":
        other, _ = load(args.rival, args.side, args.seed)
        rival, rival_name = predict(other, val, idx)[0], args.rival
        numbers[args.rival] = liveliness(np.concatenate([predict(other, val, all_idx[k:k + 64])
                                                         for k in range(0, len(all_idx), 64)]), val, all_idx)
    tissue = tissue_numbers(brain, val)
    numbers["tissue"] = tissue
    print(json.dumps(numbers, indent=1), flush=True)
    images = render(frames, val, i, t, rival, rival_name, tag, tissue, L.WorldConfig(**L.WORLDS["default"]).ray_reach)
    out = OUT / "watch"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name(args.arm, args.side, args.seed)}-ep{i}"
    images[0].save(out / f"{stem}.gif", save_all=True, append_images=images[1:] + [images[-1]] * 8,
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    images[min(lap - 1, len(images) - 1)].save(out / f"{stem}-lap1.png")
    images[min(2 * lap - 2, len(images) - 1)].save(out / f"{stem}-lap2.png")
    (out / f"{stem}.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(f"wrote {out / stem}.gif", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
