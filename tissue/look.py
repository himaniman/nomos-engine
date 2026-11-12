"""A picture of one life: which cells speak, tick by tick, and what the brain says in the dark.

    python -m tissue.look runs/tissue/gate/gate-s0.pt --fan 32 --field 24 [--loops 24]
                          [--small 256] [--big 2048] [--zoom 8] [--out out/tissue/look.png]

The life is the one of exam 3 with `--alone` (`grow.alone`, the arm that grows): a brain
that is told nothing, neither where a thing lies nor when to grow. One life is recorded
and drawn in two parts:

  A  the whole life. Across: ticks. Up: the cells of the ring, cell 0 at the bottom, new
     cells born above the last one. Brightness: how loud a cell is. Above it, whether the
     eyes are open; below it, how many bits the brain had right in each closing.
  B  the first closings of the test, tick by tick: the cells again, then the twelve bits
     the world showed on each tick and the twelve the brain said about that tick. With the
     eyes closed nobody shows it anything: what it says there it says by itself.

Nothing is measured here that the exams do not measure better; it is a thing to look at.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from tissue import grow
from tissue.train import load

PLANE, SURFACE = (13, 13, 13), (26, 26, 25)             # nothing there; a silent cell
INK, INK2, MUTED, AXIS = (240, 239, 236), (195, 194, 183), (137, 135, 129), (56, 56, 53)
RAMP = ((26, 26, 25), (13, 54, 107), (28, 92, 171), (57, 135, 229), (134, 182, 239), (205, 226, 251))
PLUS, MINUS, WRONG, DOTS = (134, 182, 239), (24, 79, 149), (230, 103, 103), (217, 89, 38)
LENGTH, LAPS, OPEN = 8, 2, 3                            # the shape of a life in `grow.alone`


def font(size: int, bold: bool = False):
    for name in (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold
                 else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def record(path: str, loops: int, batches: int, small: int, big: int, closed: int = 8,
           seed: int = 60_000, **switches) -> tuple[list, dict]:
    """One life of the brain that grows by itself: (a row per tick, the exam's result)."""
    tissue, task, _ = load(path, **switches)
    if not tissue.plan.field:
        raise ValueError(f"{path} has no sense of place")
    rows = []

    def watch(**tick):
        rows.append({k: v[0].clone().numpy() if torch.is_tensor(v) else v for k, v in tick.items()})

    res = grow.alone(tissue, task, loops, batches, small, big, "grows", lives=1, closed=closed,
                     seed=seed, watch=watch)
    return rows, res


def ramp(v: np.ndarray) -> np.ndarray:
    """Loudness in [0, 1] (NaN: a cell not born yet) -> RGB."""
    stops = np.array(RAMP, dtype=float)
    x = np.clip(np.nan_to_num(v), 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    rgb[np.isnan(v)] = PLANE
    return rgb.astype(np.uint8)


def pooled(a: np.ndarray, by: int) -> np.ndarray:
    """(ticks, cells) -> (cells // by, ticks): the loudest of each `by` cells, cell 0 at the bottom."""
    t, n = a.shape
    a = np.pad(a, ((0, 0), (0, (-n) % by)), constant_values=np.nan).reshape(t, -1, by)
    top = np.where(np.isnan(a), -1.0, a).max(-1)
    return np.where(top < 0, np.nan, top).T[::-1]


def draw(rows: list, name: str, loops: int, batches: int, closed: int = 8, zoom: int = 8) -> Image.Image:
    ticks = len(rows) - 1                       # the last tick only closes the score
    cells = np.array([len(r["act"]) for r in rows[:ticks]])
    most = int(cells.max())
    act = np.full((ticks, most), np.nan)
    for t, r in enumerate(rows[:ticks]):
        act[t, :len(r["act"])] = np.abs(r["act"])
    act = act / max(float(np.nanpercentile(act[act > 0], 99)) if (act > 0).any() else 1.0, 1e-9)
    seen = np.array([r["seen"] for r in rows[:ticks]])
    study = batches * LAPS * (loops // batches) * (LENGTH + 1)
    blink = OPEN + closed
    right = np.array([float(((r["said"] > 0) == (r["about"] > 0)).mean()) for r in rows[1:]])
    dark = np.array([r["about_closed"] for r in rows[1:]])      # index t: was tick t a closed one

    px = max(1, round(1900 / ticks))
    by = max(1, -(-most // 420))
    x0, xr, wide = 260, 50, ticks * px
    f_title, f_head, f_text, f_small = font(46, True), font(32, True), font(27), font(23)
    height_a = -(-most // by)
    zoom = min(zoom, (ticks - study) // blink)
    cw = max(4, wide // (zoom * blink))
    by_b = max(1, -(-most // 200))
    height_b = -(-most // by_b)
    bit = 15
    d = len(rows[1]["about"])
    total = 210 + 90 + 40 + height_a + 80 + 120 + 110 + 80 + 44 + height_b + 36 + 2 * (d * bit + 36) + 120
    img = Image.new("RGB", (x0 + wide + xr, total), SURFACE)
    g = ImageDraw.Draw(img)

    def label(y, text, f=f_text, fill=INK2):    # in the left margin, beside a row
        g.text((x0 - 16, y), text, font=f, fill=fill, anchor="rm")

    g.text((32, 30), "One life of a brain that is told nothing", font=f_title, fill=INK)
    g.text((32, 96), f"{name}: born with {int(cells[0])} cells, it ends with {most}. The world is"
                     f" {loops * LENGTH} items in {loops} loops of {LENGTH}.", font=f_text, fill=INK2)
    g.text((32, 132), "Where on its ring a loop lives, and when to grow, the brain decides.",
           font=f_text, fill=INK2)

    # --- A: the whole life
    y = 210
    g.text((32, y), "A   The whole life", font=f_head, fill=INK)
    g.text((340, y + 5), f"brightness: how loud a cell is (the loudest of every {by})" if by > 1
           else "brightness: how loud a cell is", font=f_text, fill=MUTED)
    y += 90
    xs = x0 + study * px
    g.text((x0, y - 34), f"study: eyes open, {loops} loops walked round twice, in {batches} batches",
           font=f_small, fill=INK2)
    g.text((xs + 8, y - 34), f"test: the eyes see {OPEN} items, then close for {closed} ticks;"
                             f" {2 * loops} times", font=f_small, fill=INK2)
    strip = np.where(seen[None, :, None], np.array(INK2), np.array(AXIS)).astype(np.uint8)
    img.paste(Image.fromarray(np.repeat(np.repeat(strip, px, 1), 18, 0)), (x0, y))
    label(y + 9, "eyes (light: open)", f_small)
    y += 40
    kymo = ramp(pooled(act, by))
    img.paste(Image.fromarray(np.repeat(kymo, px, 1)), (x0, y))
    base = y + height_a
    for c in range(0, most + 1, 256):           # the ring, in cells
        yy = base - c // by
        g.line([(x0 - 8, yy), (x0 - 1, yy)], fill=MUTED)
        label(yy, str(c), f_small, MUTED)
    g.text((32, y + height_a // 2), "cell on\nthe ring", font=f_text, fill=INK2, anchor="lm")
    grown = [t for t in range(1, ticks) if cells[t] != cells[t - 1]]
    for t in grown:                             # the moment the brain gives itself cells
        x = x0 + t * px
        g.line([(x, base - int(cells[t]) // by), (x, base - int(cells[t - 1]) // by)], fill=INK, width=2)
        g.text((x - 12, base - int(cells[t - 1]) // by - 10), f"+{int(cells[t] - cells[t - 1])} cells",
               font=f_small, fill=INK, anchor="rb")
    for t in [0] + grown:                       # the edge of the ring as it is
        end = next((u for u in grown if u > t), ticks)
        yy = base - int(cells[t]) // by
        g.line([(x0 + t * px, yy), (x0 + end * px, yy)], fill=INK, width=2)
    g.line([(xs, y - 40), (xs, base + 8)], fill=MUTED)
    for t in range(0, ticks + 1, 100):
        g.line([(x0 + t * px, base), (x0 + t * px, base + 7)], fill=MUTED)
        g.text((x0 + t * px, base + 12), str(t), font=f_small, fill=MUTED, anchor="mt")
    g.text((x0 + wide, base + 44), "tick", font=f_small, fill=MUTED, anchor="rt")
    y = base + 80

    # bits right in each closing: its own small chart, the same ticks across
    top, bottom = y + 10, y + 110
    for value, text in ((1.0, f"all {d} bits"), (0.5, "chance")):
        yy = bottom - (value - 0.5) * 2 * (bottom - top)
        g.line([(x0, yy), (x0 + wide, yy)], fill=AXIS)
        label(yy, text, f_small, MUTED)
    g.text((32, (top + bottom) // 2), "right,\neyes closed", font=f_text, fill=INK2, anchor="lm")
    for k in range((ticks - study) // blink):
        t = study + k * blink
        hops = right[t + OPEN:t + blink][dark[t + OPEN:t + blink]]
        if len(hops):
            cx = x0 + (t + OPEN + closed / 2) * px
            cy = bottom - (float(hops.mean()) - 0.5) * 2 * (bottom - top)
            g.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=DOTS, outline=SURFACE, width=2)
    y = bottom + 110

    # --- B: the first closings, tick by tick
    g.text((32, y), f"B   The first {zoom} closings of the test, tick by tick", font=f_head, fill=INK)
    y += 80
    span = range(study, study + zoom * blink)
    for k in range(zoom):
        xa = x0 + k * blink * cw
        g.rectangle([xa + 1, y, xa + OPEN * cw - 2, y + 28], fill=INK2)
        g.rectangle([xa + OPEN * cw + 1, y, xa + blink * cw - 2, y + 28], fill=AXIS)
        g.text((xa + OPEN * cw / 2, y + 14), "sees", font=f_small, fill=PLANE, anchor="mm")
        g.text((xa + OPEN * cw + closed * cw / 2, y + 14), "eyes closed", font=f_small, fill=INK2, anchor="mm")
    y += 44
    img.paste(Image.fromarray(np.repeat(ramp(pooled(act[span.start:span.stop], by_b)), cw, 1)), (x0, y))
    g.text((32, y + height_b // 2), "cells on\nthe ring", font=f_text, fill=INK2, anchor="lm")
    for c in (0, most):
        label(y + height_b - c // by_b, str(c), f_small, MUTED)
    y += height_b + 36
    for title, key in (("the world\nshowed", "about"), ("the brain\nsaid", "said")):
        g.text((32, y + d * bit // 2), title, font=f_text, fill=INK2, anchor="lm")
        for j, t in enumerate(span):
            row = rows[t + 1]                   # what is said on the next tick is about this one
            for i in range(d):
                good = (row["said"][i] > 0) == (row["about"][i] > 0)
                fill = (PLUS if row[key][i] > 0 else MINUS) if (key == "about" or good) else WRONG
                g.rectangle([x0 + j * cw + 1, y + i * bit + 1, x0 + (j + 1) * cw - 2, y + (i + 1) * bit - 2],
                            fill=fill)
        y += d * bit + 36
    x = x0
    for fill, text in ((PLUS, "a bit that is 1"), (MINUS, "a bit that is 0"), (WRONG, "a bit the brain got wrong")):
        g.rectangle([x, y + 2, x + 26, y + 28], fill=fill)
        g.text((x + 38, y + 15), text, font=f_small, fill=INK2, anchor="lm")
        x += 80 + int(g.textlength(text, font=f_small))
    g.text((x0, y + 62), "With the eyes closed the world goes on round its loop and shows nothing:"
                         " what the brain says there it says by itself.", font=f_small, fill=MUTED, anchor="lm")
    return img


def draw_thought(t: dict, name: str, cells: int) -> Image.Image:
    """One thought of exam 5: every string of ways walked in the dark by a copy of the brain,
    and how near what the copy says comes to the wanted room."""
    strings, says, true, items = t["strings"], t["says"], t["true"], t["items"]
    n, depth, bits = says.shape
    want = items[t["want"]]
    agree = ((says > 0) == (want > 0)).sum(-1)                      # bits of the wanted room said
    knows = ((says > 0) == (items[true] > 0)).sum(-1) >= bits - 1   # the copy says the room really there
    f_title, f_head, f_text, f_small = font(46, True), font(32, True), font(27), font(23)
    x0, y0, cw, ch = 330, 300, 190, 46
    img = Image.new("RGB", (x0 + depth * cw + 700, y0 + n * ch + 150), SURFACE)
    g = ImageDraw.Draw(img)
    g.text((32, 30), "One thought: walks in the dark toward a wanted room", font=f_title, fill=INK)
    g.text((32, 96), f"{name}, {cells} cells; a world of {len(items)} rooms with {int(strings.max()) + 1} ways out"
                     f" of each, learned in this life. The wanted room is {depth} steps away.", font=f_text, fill=INK2)
    g.text((32, 132), "A copy of the brain walks every string of ways with its eyes closed. A number: how"
                      f" many of the wanted room's {bits} bits the copy says after that hop.", font=f_text, fill=INK2)
    g.text((32, 216), "the ways tried", font=f_head, fill=INK)
    for h in range(depth):
        g.text((x0 + h * cw + cw // 2, 236), f"hop {h + 1}", font=f_text, fill=INK2, anchor="mm")
    g.text((x0 + depth * cw + 30, 236), "where that string really leads", font=f_text, fill=INK2, anchor="lm")
    plans = [(int((agree[s] >= bits - 1).float().argmax()) if bool((agree[s] >= bits - 1).any()) else depth + 1)
             for s in range(n)]
    best = min(range(n), key=lambda s: plans[s])
    for s in range(n):
        y = y0 + s * ch
        chosen = plans[s] <= depth and s == best
        g.text((x0 - 24, y + ch // 2), "  ".join("ABCDEFGH"[w] for w in strings[s].tolist()), font=f_text,
               fill=INK if chosen else INK2, anchor="rm")
        for h in range(depth):
            v = float(agree[s, h])
            box = [x0 + h * cw + 2, y + 2, x0 + (h + 1) * cw - 3, y + ch - 3]
            fill = tuple(ramp(np.array([[(v - bits / 2) / (bits / 2)]]))[0, 0].tolist())
            g.rectangle(box, fill=fill)
            g.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), f"{int(v)}", font=f_small,
                   fill=PLANE if v >= bits - 2 else INK2, anchor="mm")
            if not bool(knows[s, h]):               # the copy is lost: it says a room that is not there
                g.rectangle([box[2] - 14, box[1], box[2], box[1] + 14], fill=WRONG)
            if v >= bits - 1:
                g.rectangle(box, outline=INK, width=3)
        reach = [h + 1 for h in range(depth) if int(true[s, h]) == t["want"]]
        note = f"the wanted room at hop {reach[0]}" if reach else "elsewhere"
        if chosen:
            note += "   <- the plan: its first way is taken"
        g.text((x0 + depth * cw + 30, y + ch // 2), note, font=f_small,
               fill=INK if reach else MUTED, anchor="lm")
    y = y0 + n * ch + 30
    g.rectangle([x0, y, x0 + 26, y + 26], outline=INK, width=3)
    g.text((x0 + 40, y + 13), "the copy says the wanted room (all bits but one at most)", font=f_small,
           fill=INK2, anchor="lm")
    g.rectangle([x0, y + 44, x0 + 14, y + 58], fill=WRONG)
    g.text((x0 + 40, y + 51), "the copy is lost: what it says is not the room that string leads to",
           font=f_small, fill=INK2, anchor="lm")
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genome")
    p.add_argument("--thought", action="store_true",
                   help="draw one thought of exam 5 instead: every string of ways walked in the dark")
    p.add_argument("--rooms", type=int, default=32, help="--thought: rooms in the world")
    p.add_argument("--depth", type=int, default=4, help="--thought: how far the wanted room lies")
    p.add_argument("--cells", type=int, default=256, help="--thought: the size of the brain")
    p.add_argument("--loops", type=int, default=24, help="loops in the world, eight items each")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--small", type=int, default=256, help="cells the brain is born with")
    p.add_argument("--big", type=int, default=2048, help="the most it may grow to")
    p.add_argument("--closed", type=int, default=8)
    p.add_argument("--fan", type=int, default=None)
    p.add_argument("--field", type=int, default=None)
    p.add_argument("--lean", type=float, default=None)
    p.add_argument("--seed", type=int, default=60_000)
    p.add_argument("--zoom", type=int, default=8, help="closings drawn tick by tick")
    p.add_argument("--out", default="out/tissue/look.png")
    a = p.parse_args()
    torch.set_num_threads(4)
    switches = {k: v for k, v in (("fan", a.fan), ("field", a.field), ("lean", a.lean)) if v is not None}
    name = os.path.splitext(os.path.basename(a.genome))[0]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.thought:
        from tissue import fork, seek
        tissue, task, _ = load(a.genome, **switches)
        gen = torch.Generator().manual_seed(a.seed)
        walk = fork.Walk(rooms=a.rooms, ways=2)
        items, nxt = fork.world(task.d_out, walk, 1, gen)
        with torch.no_grad():
            t = seek.thought(tissue.grow(a.cells), items, nxt, walk, gen, a.depth)
        draw_thought(t, name, a.cells).save(a.out)
        print(f"{a.genome}: one thought, {a.rooms} rooms, {a.depth} deep -> {a.out}")
        return
    rows, res = record(a.genome, a.loops, a.batches, a.small, a.big, a.closed, a.seed, **switches)
    img = draw(rows, name, a.loops, a.batches, a.closed, a.zoom)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    img.save(a.out)
    print(f"{a.genome}: {res['cells']} cells at the end; right with the eyes closed, hop 1"
          f" {res['hop'][0]:.3f}, hop {a.closed} {res['hop'][-1]:.3f}; {len(rows)} ticks -> {a.out}")


if __name__ == "__main__":
    main()
