"""A picture of one life in a land (exam 6): what the brain comes to know and where it goes.

    python -m tissue.view runs/tissue/hold/hold-s0.pt [--side 6] [--cells 256] [--life 0]
                          [--out out/tissue/world.png] [--film out/tissue/world.gif]

Top row, the map in its head at three ticks of the life and one thought:
  the land with its walls and trees, and out of every room a mark for each way. Blue: asked
  by the examiner (a copy carried there, one hop in the dark) the brain says the room that
  way really leads to, or, where a wall stands, that it would stay. Red: it says something
  else. Grey: it has never gone that way.
  The thought: the body (a ring), the room it wants (a dot), every hop its copies made
  between rooms that are there, and the road it chose.
Bottom row: where the body went in the first third of the life and in the last, and the
  meals eaten by then against the yardsticks in the same land.

`--film`: the same life as a film, a frame a tick: its first ticks, then its last. The body,
the fruit, where it has just been, and in orange the road it has thought of on that tick
(no orange: the thought found none, and the step is a way at random).

Nothing is measured here that the exam does not measure better; it is a thing to look at.
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image, ImageDraw

from tissue import world
from tissue.look import AXIS, DOTS, INK, INK2, MUTED, PLANE, PLUS, SURFACE, WRONG, font
from tissue.train import load

NAMES = {"oracle": "born knowing every way", "thinks": "the brain", "perfect": "a perfect memory",
         "blind": "never a plan"}


def record(path: str, shape: world.Land, m: int, life: int = 0, seed: int = 90_000, **think) -> dict:
    """One life, watched: the land, the track, the map in the brain's head at three ticks,
    one thought of a far room late in life, and the yardsticks' meals in the same land."""
    model, task, _ = load(path, still=True)
    ticks = shape.life

    def born(arm):
        gen = torch.Generator().manual_seed(seed + 1000 * shape.side + life)
        looks, nxt, trees, start = world.land(task.d_out, shape, gen)
        return looks, nxt, trees, start, world.body_for(arm, model, m, looks, nxt, gen, **think), gen

    looks, nxt, trees, start, mind, gen = born("thinks")
    far, room_of = world.roads(nxt), {k: r for r, k in enumerate(world.keys(looks))}
    gone = torch.zeros(len(looks), 4, dtype=torch.bool)
    rec = {"looks": looks, "nxt": nxt, "trees": trees.tolist(), "side": shape.side, "ticks": ticks,
           "track": [], "maps": {}, "thought": None, "meals": {}}

    def know(t):
        says = mind.atlas(looks)
        rec["maps"][t] = {"gone": gone.clone(), "right": ((says > 0) == (looks[nxt] > 0)).all(-1)}

    rec["film"] = []                            # per tick: (room, fruit, the road thought of)
    plain = mind.think

    def thinking(wanted, g, trace=None):        # the same thought, its copies kept for the film
        trace = []
        way = plain(wanted, g, trace=trace)
        rec["last"] = [(hops, room_of.get(a), w, room_of.get(b)) for hops, a, w, b in trace] if way is not None else []
        return way

    mind.think = thinking

    def watch(t, room, fruit, way, planned):
        rec["film"].append((room, fruit, road(rec["last"], room, nxt) if planned else []))
        if t in (ticks // 8, ticks // 3):
            know(t)
        if rec["thought"] is None and t >= 2 * ticks // 3 and planned and int(far[room, fruit]) >= min(5, shape.side - 1):
            rec["thought"] = {"here": room, "want": fruit, "tick": t, "trace": rec["last"]}
        rec["track"].append(room)
        gone[room, way] = True

    with torch.no_grad():
        rec["meals"]["thinks"] = world.live(mind, looks, nxt, trees, start, ticks, gen, fruits=seed + life,
                                            watch=watch)["meals"]
        know(ticks)
        for arm in world.YARDSTICKS:
            looks, nxt, trees, start, body, gen = born(arm)
            rec["meals"][arm] = world.live(body, looks, nxt, trees, start, ticks, gen, fruits=seed + life)["meals"]
    return rec


def road(trace: list, here: int, nxt: torch.Tensor) -> list:
    """The rooms the road a thought found leads through in the land, from the body's room on
    ([]: the thought found none). It is the ways the brain means to go, followed in the real
    land: where the thought was wrong the road ends somewhere else than it meant."""
    ways = next((w for hops, _, w, _ in trace if hops == 0), None)
    if ways is None:
        return []
    rooms = [here]
    for way in ways:
        rooms.append(int(nxt[rooms[-1], way]))
    return rooms


def draw(rec: dict, name: str, cells: int) -> Image.Image:
    side, ticks, nxt = rec["side"], rec["ticks"], rec["nxt"]
    pane, gap, top = 330, 34, 96
    cs = (pane - 10) // side                    # a room, in pixels
    w, h = 4 * pane + 5 * gap, top + 2 * pane + 150
    img = Image.new("RGB", (w, h), PLANE)
    d = ImageDraw.Draw(img, "RGBA")
    f_title, f_head, f_text = font(24, True), font(15, True), font(13)
    d.text((gap, 22), f"One life in a land: {name}, {cells} cells, {side}x{side} rooms, {ticks} ticks", font=f_title, fill=INK)
    d.text((gap, 56), "Nobody leads it. It is told what the room with the fruit looks like; the way there it"
                      " has to think of, on what its own wandering left in its wires.", font=f_text, fill=MUTED)

    def centre(room, x0, y0):
        return x0 + (room % side + 0.5) * cs, y0 + (room // side + 0.5) * cs

    def land(col, row, title, note=""):
        """The land's frame, walls and trees; returns its corner."""
        x0, y0 = gap + col * (pane + gap), top + row * (pane + 70) + 24
        d.text((x0, y0 - 24), title, font=f_head, fill=INK2)
        d.rectangle([x0, y0, x0 + side * cs, y0 + side * cs], fill=SURFACE)
        for room in range(side * side):
            x, y = x0 + (room % side) * cs, y0 + (room // side) * cs
            for way, line in enumerate(([x, y, x + cs, y], [x + cs, y, x + cs, y + cs],
                                        [x, y + cs, x + cs, y + cs], [x, y, x, y + cs])):
                if int(nxt[room, way]) == room:
                    d.line(line, fill=INK2, width=2)
        for tree in rec["trees"]:
            x, y = centre(tree, x0, y0)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=INK, width=2)
        if note:
            d.text((x0, y0 + side * cs + 8), note, font=f_text, fill=MUTED)
        return x0, y0

    # top row: the map in its head at three ticks
    for col, (t, known) in enumerate(sorted(rec["maps"].items())):
        gone, right = known["gone"], known["right"]
        share = float(right[gone].float().mean()) if bool(gone.any()) else 0.0
        x0, y0 = land(col, 0, f"The map in its head at tick {t}",
                      f"ways gone {float(gone.float().mean()):.0%}, of them said right {share:.0%}")
        for room in range(side * side):
            x, y = centre(room, x0, y0)
            for way, (dr, dc) in enumerate(world.WAYS):
                ink = (AXIS if not gone[room, way] else PLUS if right[room, way] else WRONG)
                if int(nxt[room, way]) == room:             # a wall: a dot beside it
                    px, py = x + dc * cs * 0.34, y + dr * cs * 0.34
                    d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=ink)
                else:
                    d.line([x + dc * cs * 0.12, y + dr * cs * 0.12, x + dc * cs * 0.42, y + dr * cs * 0.42],
                           fill=ink, width=2)
    # one thought
    th = rec["thought"]
    if th:
        copies = [row for row in th["trace"] if row[0]]
        real = [(a, b) for _, a, _, b in copies if a is not None and b is not None and a != b]
        lost = sum(1 for _, _, _, b in copies if b is None)
        way = road(th["trace"], th["here"], nxt)
        x0, y0 = land(3, 0, f"One thought, tick {th['tick']}",
                      f"{len(copies)} copies; {lost} said no room of the land; road of {max(0, len(way) - 1)} steps")
        for a, b in real:
            d.line([centre(a, x0, y0), centre(b, x0, y0)], fill=PLUS + (110,), width=2)
        if way:
            d.line([centre(r, x0, y0) for r in way], fill=DOTS, width=3)
        x, y = centre(th["here"], x0, y0)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=DOTS, width=3)
        x, y = centre(th["want"], x0, y0)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=DOTS)
    # bottom row: where the body went
    for col, (title, part) in enumerate((("Where it went, the first third", rec["track"][:ticks // 3]),
                                         ("Where it went, the last third", rec["track"][2 * ticks // 3:]))):
        moves = sum(a != b for a, b in zip(part, part[1:]))
        x0, y0 = land(col, 1, title, f"{moves} steps from room to room, {len(part) - 1 - moves} into a wall")
        for a, b in zip(part, part[1:]):
            if a != b:
                d.line([centre(a, x0, y0), centre(b, x0, y0)], fill=PLUS + (28,), width=3)
    # meals by then
    x0, y0 = gap + 2 * (pane + gap), top + pane + 70 + 24
    cw, ch = 2 * pane + gap - 190, side * cs
    d.text((x0, y0 - 24), "Meals eaten by then, the same land", font=f_head, fill=INK2)
    most = max(len(m) for m in rec["meals"].values()) or 1
    d.line([x0, y0 + ch, x0 + cw, y0 + ch], fill=AXIS, width=1)
    for k in range(4):
        x = x0 + cw * k / 3
        d.text((x - (0 if k == 0 else 14), y0 + ch + 8), str(ticks * k // 3), font=f_text, fill=MUTED)
        if k:
            d.line([x, y0, x, y0 + ch], fill=AXIS, width=1)
    d.text((x0 + cw / 2 - 12, y0 + ch + 26), "tick", font=f_text, fill=MUTED)
    inks = {"oracle": MUTED, "thinks": PLUS, "perfect": INK2, "blind": MUTED}
    ends = []
    for arm in ("oracle", "perfect", "blind", "thinks"):
        meals = rec["meals"][arm]
        pts = [(x0, y0 + ch)] + [(x0 + cw * t / ticks, y0 + ch - ch * (i + 1) / most) for i, t in enumerate(meals)]
        pts.append((x0 + cw, pts[-1][1]))
        d.line(pts, fill=inks[arm], width=3 if arm == "thinks" else 2)
        ends.append([pts[-1][1], arm, len(meals)])
    ends.sort()
    for i in range(1, len(ends)):               # labels at the line ends, kept apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 18)
    for y, arm, n in ends:
        d.text((x0 + cw + 10, y - 9), f"{NAMES[arm]}: {n}", font=f_text, fill=INK if arm == "thinks" else INK2)
    d.text((gap, h - 34), "Blue: right. Red: wrong. Grey: never gone. Rings: trees. Orange: the body, the room it wants"
                          " and the road it thought of.", font=f_text, fill=MUTED)
    return img


def film(rec: dict, name: str, cells: int, first: int = 140, last: int = 140, tail: int = 12) -> list:
    """Frames of the life's first ticks and of its last: the land, the fruit, the body with
    the rooms it has just left, and the road it has thought of."""
    side, ticks, nxt = rec["side"], rec["ticks"], rec["nxt"]
    cs, pad, top = 360 // side, 20, 64
    size = (side * cs + 2 * pad, side * cs + top + 44)
    f_head, f_text = font(16, True), font(13)
    base = Image.new("RGB", size, PLANE)
    d = ImageDraw.Draw(base)
    d.rectangle([pad, top, pad + side * cs, top + side * cs], fill=SURFACE)
    for room in range(side * side):
        x, y = pad + (room % side) * cs, top + (room // side) * cs
        for way, line in enumerate(([x, y, x + cs, y], [x + cs, y, x + cs, y + cs],
                                    [x, y + cs, x + cs, y + cs], [x, y, x, y + cs])):
            if int(nxt[room, way]) == room:
                d.line(line, fill=INK2, width=2)
    centre = lambda room: (pad + (room % side + 0.5) * cs, top + (room // side + 0.5) * cs)
    for tree in rec["trees"]:
        x, y = centre(tree)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=MUTED, width=2)
    meals, frames = rec["meals"]["thinks"], []
    for part, span in (("newborn", range(0, min(first, ticks))), ("grown", range(max(0, ticks - last), ticks))):
        for t in span:
            room, fruit, way = rec["film"][t]
            img = base.copy()
            d = ImageDraw.Draw(img, "RGBA")
            d.text((pad, 14), f"{name}, {cells} cells: {part}", font=f_head, fill=INK)
            d.text((pad, 38), f"tick {t}   meals {sum(1 for m in meals if m <= t)}   "
                              + ("a road thought of" if way else "no road: a way at random"), font=f_text, fill=MUTED)
            went = [r for r, _, _ in rec["film"][max(span[0], t - tail):t + 1]]
            for i, (a, b) in enumerate(zip(went, went[1:])):
                if a != b:
                    d.line([centre(a), centre(b)], fill=PLUS + (40 + 200 * i // max(1, len(went) - 1),), width=3)
            if way:
                d.line([centre(r) for r in way], fill=DOTS + (200,), width=3)
            x, y = centre(fruit)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=DOTS)
            x, y = centre(room)
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=INK)
            d.text((pad, top + side * cs + 12), "white: the body   orange: the fruit, and the road in its head",
                   font=f_text, fill=MUTED)
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=32))
        frames.extend([frames[-1]] * 6)         # a pause between the parts
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genome")
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--cells", type=int, default=256)
    p.add_argument("--life", type=int, default=0, help="which of the exam's lands")
    p.add_argument("--seed", type=int, default=90_000)
    p.add_argument("--out", default="out/tissue/world.png")
    p.add_argument("--film", help="a film of the life's first and last ticks goes here (GIF)")
    p.add_argument("--budget", type=int, default=256, help="how many rooms a thought may think of")
    p.add_argument("--curious", type=float, default=0.0, help="by hand, see tissue.world")
    p.add_argument("--keep", action="store_true", help="by hand, see tissue.world")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    rec = record(a.genome, world.Land(a.side), a.cells, a.life, a.seed,
                 budget=a.budget, curious=a.curious, keep=a.keep)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    draw(rec, os.path.basename(a.genome).rsplit(".", 1)[0], a.cells).save(a.out)
    if a.film:
        frames = film(rec, os.path.basename(a.genome).rsplit(".", 1)[0], a.cells)
        frames[0].save(a.film, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
    print(f"{a.out}: meals " + ", ".join(f"{NAMES[k]} {len(v)}" for k, v in rec["meals"].items()))


if __name__ == "__main__":
    main()
