"""Exam 4: forks. What comes next depends on which way the body goes.

    python -m tissue.fork GENOME.pt ... [--ways 2] [--sizes 64 128 256 512] [--closed 4]
                                        [--numb] [--lives 16]

The world is ROOMS, each a distinct random item, and out of every room there are `ways`
ways. A way out of a room always leads to the same room. (Each way is a shuffle of the
rooms, the first of them one single round through them all, so every room is entered once
by every way, nothing is a dead end and the world hangs together. With one way the world
is one loop of exam 2.)

The brain has two senses. Its EYES see the item of the room it is in, and they may close.
Its BODY feels the way it has just gone, and that sense never closes: one more sensory
cell per way, +1 for the way gone and -1 for the others, all silent when the body was
carried somewhere (a jump).

A life is one stream of ticks and the brain is never reset:

  study   the eyes are open and the body is led out of every room by every way, `laps`
          times over (each lap a round that takes every way once, in an order of its own),
          so every link is seen once a lap and there is no jump at all;
  test    `blinks` times: the world jumps somewhere, the body walks on by ways drawn at
          random, the eyes see `open` rooms and then close for `closed` ticks.

Every tick the motor cells are read as "the room the world showed a tick ago", and while
the eyes were open a tick ago the teacher shows exactly that: the senses a tick late, as
in exam 2. With the eyes closed the j-th closed tick is HOP j: it can be right only if the
brain went on by itself AND went the way its body went.

Score: bit accuracy by hop. K = links = rooms x ways; K90 at a hop as before.
Yardsticks: a perfect memory of (room, way) -> room is right at every hop; `numb`, the best
a brain can do that does not feel its body, knows one of the rooms that follow a room and
is right about one hop in `ways`. The control for a tissue is the same thing: `feel=False`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

import torch

from tissue.assoc import k90
from tissue.train import load


@dataclass(frozen=True)
class Walk:
    """The shape of one life."""
    rooms: int
    ways: int = 2
    laps: int = 2        # study: times every way out of every room is taken
    blinks: int = 8      # test: times the eyes close
    open: int = 3        # rooms seen before each closing
    closed: int = 4      # ticks each closing lasts

    @property
    def links(self) -> int:
        return self.rooms * self.ways


def world(bits: int, walk: Walk, batch: int, gen: torch.Generator):
    """`batch` worlds: items (B, rooms, bits), all distinct within a world, and where each
    way leads, nxt (B, rooms, ways)."""
    n = 2 ** bits
    if walk.rooms > n:
        raise ValueError(f"{walk.rooms} distinct items do not fit in {bits} bits")
    idx = torch.rand(batch, n, generator=gen).argsort(dim=1)[:, :walk.rooms]
    items = ((idx.unsqueeze(-1) >> torch.arange(bits)) & 1).float() * 2 - 1
    nxt = torch.rand(batch, walk.rooms, walk.ways, generator=gen).argsort(dim=1)   # a shuffle per way
    ring = torch.rand(batch, walk.rooms, generator=gen).argsort(dim=1)     # way 0: one round of all
    nxt[:, :, 0].scatter_(1, ring, torch.roll(ring, -1, dims=1))
    return items, nxt


def tour(nxt: torch.Tensor, start: int, gen: torch.Generator) -> list:
    """A round from `start` back to it that takes every way out of every room once, in an
    order drawn at random: [(room reached, the way that led there)], the start left out."""
    n, a = nxt.shape
    left = [torch.randperm(a, generator=gen).tolist() for _ in range(n)]    # ways not yet taken
    to = nxt.tolist()
    stack, path = [(start, -1)], []
    while stack:
        room = stack[-1][0]
        if left[room]:
            way = left[room].pop()
            stack.append((to[room][way], way))
        else:
            path.append(stack.pop())
    return path[-2::-1]


def script(nxt: torch.Tensor, walk: Walk, gen: torch.Generator) -> list:
    """The world's side of a life, the same whoever lives it: per tick (kind, seen,
    room (B,), way (B,)) with kind in "study" | "echo" | "closed"; `way` is the way just
    gone, -1 when the body was carried."""
    b = nxt.shape[0]
    rows = torch.arange(b)
    start = torch.randint(walk.rooms, (b,), generator=gen)
    laps = [[step for _ in range(walk.laps) for step in tour(nxt[i], int(start[i]), gen)]
            for i in range(b)]
    ticks = [("study", True, start, torch.full((b,), -1))]
    for t in range(walk.laps * walk.links):
        ticks.append(("study", True, torch.tensor([lap[t][0] for lap in laps]),
                      torch.tensor([lap[t][1] for lap in laps])))
    for _ in range(walk.blinks):
        room = torch.randint(walk.rooms, (b,), generator=gen)
        ticks.append(("echo", True, room, torch.full((b,), -1)))
        for k in range(walk.open + walk.closed - 1):
            way = torch.randint(walk.ways, (b,), generator=gen)
            room = nxt[rows, room, way]
            seen = k < walk.open - 1
            ticks.append(("echo" if seen else "closed", seen, room, way))
    return ticks


def embodied(tissue, ways: int):
    """The same genome with `ways` more sensory cells, for the body's sense. A genome says
    nothing about how many sensory cells there are: their wires are drawn at birth."""
    if tissue.task.d_in != tissue.task.d_out:
        raise ValueError("the genome already has more senses than it says back")
    tissue.task = replace(tissue.task, d_in=tissue.task.d_in + ways)
    return tissue


def feel(way: torch.Tensor, ways: int) -> torch.Tensor:
    """What the body cells say of the way just gone (B,): +1 the cell of that way, -1 the
    others, all silent when the body was carried (way -1)."""
    felt = torch.zeros(way.shape[0], ways)
    gone = way >= 0
    felt[gone] = -1.0
    felt[gone.nonzero().flatten(), way[gone]] = 1.0
    return felt


BODIES = ("cells", "dendrites", "numb", "felt", "tribes")


class Dendrites:
    """BY HAND: a dendrite from the other interneurons per way. On a tick only the dendrite
    of the way just gone is heard and learns; when the body was carried (way -1) none is.
    The brain's own state holds whichever dendrite is live, so the cell and its rule are
    untouched. `numb`: one dendrite whatever the way (a brain that does not feel its body)."""

    def __init__(self, brain, state, ways: int, numb: bool = False):
        self.inner = next(i for i, blk in enumerate(brain.t.plan.blocks) if blk.pre == blk.post)
        self.numb = numb
        self.bank = torch.zeros(state[1][self.inner].shape[0], 1 if numb else ways,
                                *state[1][self.inner].shape[1:])
        self.rows = torch.arange(self.bank.shape[0])

    def tick(self, brain, state, cue, way, teach, flag: float, on: float, keep: bool = True):
        """One tick of the brain with the dendrite of `way` (B,) live. `keep=False`: what
        that dendrite learns on this tick is thrown away (a walk in thought only)."""
        gone = way >= 0
        which = torch.zeros_like(way[gone]) if self.numb else way[gone]
        live = torch.zeros_like(self.bank[:, 0])
        live[gone] = self.bank[self.rows[gone], which]
        state, out = brain.tick((state[0], {**state[1], self.inner: live}, state[2]), cue, teach, flag, on)
        if keep:
            self.bank[self.rows[gone], which] = state[1][self.inner][gone]
        return state, out


def run_life(brain, items: torch.Tensor, nxt: torch.Tensor, walk: Walk, gen: torch.Generator,
             body: str = "cells") -> dict:
    """Live the walk. {kind: (said, true)} for kind in "echo" | "closed", each (B, ticks of
    that kind, bits); "closed" is ordered blink by blink, hop by hop.

    `body`, how the way just gone reaches the brain:
      "cells"      on sensory cells of its own, beside the eyes' (`embodied`); what an
                   interneuron makes of them is up to its rule;
      "dendrites"  BY HAND: every interneuron has one dendrite from the other interneurons
                   per way, and on a tick only the dendrite of the way just gone is heard
                   and learns (none when the body was carried). The cell and its rule are
                   as they were: it never knows that its dendrite was changed for another;
      "numb"       not at all (the control);
      "tribes"     BY HAND, the ceiling of a brain whose CELLS lean to a way (one set of
                   wires): every interneuron belongs to one way, and on a tick only those of
                   the way just gone speak and learn (everyone when the body was carried,
                   and then no wire among interneurons changes);
      "felt"       on body cells of the brain's own plan (`Plan.body`), which every
                   interneuron reads on a dendrite of its own: nothing by hand but that
                   the sense is there. What a cell makes of it is its genome's affair."""
    if body not in BODIES:
        raise ValueError(f"body must be one of {BODIES}, not {body!r}")
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    ticks = script(nxt, walk, gen)              # the world first: the same for any brain
    state = brain.born(b, gen)
    silent = torch.zeros(b, d)
    said: dict = {"echo": [], "closed": []}
    true: dict = {"echo": [], "closed": []}
    before, before_seen, before_kind = silent, False, "study"
    dendrites = Dendrites(brain, state, walk.ways) if body == "dendrites" else None
    inner = next(i for i, blk in enumerate(brain.t.plan.blocks) if blk.pre == blk.post)
    tribe = torch.arange(brain.n["I"]) % walk.ways      # "tribes": the way each interneuron belongs to
    if body == "felt" and brain.t.plan.body != walk.ways:
        raise ValueError(f"the brain has {brain.t.plan.body} body cells and the world {walk.ways} ways")
    for kind, seen, room, way in ticks + [("end", False, None, None)]:
        item = silent if room is None else items[rows, room]
        cue = item if seen else silent
        way = torch.full((b,), -1) if way is None else way
        gone = way >= 0
        teach, flag = before if before_seen else silent, float(before_seen)
        if dendrites:
            state, out = dendrites.tick(brain, state, cue, way, teach, flag, float(seen))
        elif body == "felt":
            state, out = brain.tick(state, cue, teach, flag, float(seen), body=feel(way, walk.ways))
        elif body == "tribes":
            was = state[1][inner]
            state, out = brain.tick(state, cue, teach, flag, float(seen))
            awake = (tribe.unsqueeze(0) == way.unsqueeze(1)) | ~gone.unsqueeze(1)
            learnt = torch.where((awake & gone.unsqueeze(1)).unsqueeze(-1), state[1][inner], was)
            state = ({**state[0], "I": state[0]["I"] * awake}, {**state[1], inner: learnt}, state[2])
        else:
            if brain.n["S"] > d:
                felt = torch.zeros(b, brain.n["S"] - d)
                if body == "cells":
                    felt[gone] = -1.0
                    felt[rows[gone], way[gone]] = 1.0
                cue = torch.cat([cue, felt], dim=1)
            state, out = brain.tick(state, cue, teach, flag, float(seen))
        if before_kind in said:                 # what is said now is about the tick before
            said[before_kind].append(out)
            true[before_kind].append(before)
        before, before_seen, before_kind = item, seen, kind
    return {k: (torch.stack(said[k], 1), torch.stack(true[k], 1)) for k in said if said[k]}


def by_hop(out: dict, walk: Walk) -> list[float]:
    """Bit accuracy with the eyes closed, hop 1 .. `closed`."""
    said, true = out["closed"]
    right = ((said > 0) == (true > 0)).float().view(said.shape[0], walk.blinks, walk.closed, -1)
    return right.mean(dim=(0, 1, 3)).tolist()


def numb(items: torch.Tensor, nxt: torch.Tensor, walk: Walk, gen: torch.Generator) -> dict:
    """The best a memory can do that does not feel the body: of the rooms that follow a
    room it knows one (they were each seen as often), and goes on from the room it thinks
    it is in."""
    b = items.shape[0]
    rows = torch.arange(b)
    said: dict = {"echo": [], "closed": []}
    true: dict = {"echo": [], "closed": []}
    guess = None
    for kind, seen, room, _ in script(nxt, walk, gen):
        guess = room if seen else nxt[rows, guess, 0]
        if kind in said:
            said[kind].append(items[rows, guess])
            true[kind].append(items[rows, room])
    return {k: (torch.stack(said[k], 1), torch.stack(true[k], 1)) for k in said}


def k_grid(m: int, bits: int, ways: int) -> list[int]:
    """Links to examine a brain of M interneurons at: 8 up to 2 per interneuron and way."""
    top = min(2 * m * ways, ways * 2 ** bits)
    ks, k = [], 8.0
    while round(k / ways) * ways <= top:
        ks.append(max(2, round(k / ways)) * ways)
        k *= 2 ** 0.5
    return sorted(set(ks))


def curve(live, bits: int, ks: list[int], shape: Walk, seed: int = 70_000, lives: int = 16,
          stop_below: float | None = None) -> list:
    """`live(items, nxt, walk, gen) -> out` at each K links: [{"echo", "hop": [...]}]."""
    rows = []
    for k in ks:
        if stop_below is not None and rows and rows[-1]["hop"][0] < stop_below:
            break
        gen = torch.Generator().manual_seed(seed + k)
        rooms = k // shape.ways
        walk = replace(shape, rooms=rooms, blinks=max(shape.blinks, rooms))
        items, nxt = world(bits, walk, lives, gen)
        with torch.no_grad():
            out = live(items, nxt, walk, gen)
        said, true = out["echo"]
        rows.append({"echo": float(((said > 0) == (true > 0)).float().mean()), "hop": by_hop(out, walk)})
    return rows


def examine(path: str, sizes, shape: Walk, lives: int = 16, body: str = "cells", quick: bool = True,
            **switches) -> dict:
    """{m: {"ks", "acc": [[echo, hop 1, ...] per K], "k90": [by hop]}} for a found genome."""
    if body == "felt":                      # as many body cells as the world has ways
        switches["body"] = shape.ways
    model, task, _ = load(path, **switches)
    bits = task.d_out
    if body == "cells":
        model = embodied(model, shape.ways)
    out = {}
    for m in sizes:
        brain = model.grow(m)
        ks = k_grid(m, bits, shape.ways)
        rows = curve(lambda items, nxt, walk, g: run_life(brain, items, nxt, walk, g, body), bits, ks,
                     shape, lives=lives, stop_below=0.8 if quick else None)
        ks = ks[:len(rows)]
        accs = [[r["echo"]] + r["hop"] for r in rows]
        out[m] = {"ks": ks, "acc": accs,
                  "k90": [k90(ks, [a[h] for a in accs]) for h in range(shape.closed + 1)]}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--ways", type=int, default=2)
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--closed", type=int, default=4)
    p.add_argument("--sizes", type=int, nargs="*", default=[64, 128, 256, 512])
    p.add_argument("--lives", type=int, default=16)
    p.add_argument("--body", choices=BODIES, default="cells",
                   help="how the way just gone reaches the brain: on sensory cells of its own,"
                        " by hand as a dendrite per way, not at all (the control), or on the"
                        " body cells of the brain's own plan")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    shape = Walk(rooms=1, ways=a.ways, laps=a.laps, closed=a.closed)
    hops = [h for h in sorted({1, 2, a.closed // 2, a.closed}) if 1 <= h <= a.closed]
    floor = curve(lambda items, nxt, walk, g: numb(items, nxt, walk, g), 12, [32 * a.ways], shape)[0]
    print(f"{a.ways} ways. A memory that does not feel the body, by hop: "
          + " ".join(f"{x:.3f}" for x in floor["hop"]) + "; a perfect one: 1.000")
    tag = f".fork{a.ways}x{a.closed}" + (f".laps{a.laps}" if a.laps != 2 else "") + f".{a.body}"
    for path in a.genomes:
        res = examine(path, a.sizes, shape, a.lives, body=a.body)
        print(f"\n{path}  {tag.strip('.')}")
        print(f"{'M':>5} | K90 (links) at hop " + " ".join(f"{h:>6}" for h in hops)
              + " | acc at M/4 links by hop, echo (hop 0) first")
        for m, row in res.items():
            j = min(range(len(row["ks"])), key=lambda i: abs(row["ks"][i] - m / 4))
            print(f"{m:>5} |                    " + " ".join(f"{row['k90'][h]:6.1f}" for h in hops)
                  + " | " + " ".join(f"{x:.3f}" for x in row["acc"][j]))
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
