"""Exam 2: the stream. One unbroken life: no episodes, no rests, no teacher but the senses.

The world is a few LOOPS of distinct random items; after an item comes the next one on its
loop, for ever. A life is one stream of ticks and the brain is never reset:

  study   the eyes are open. Lap after lap, loop after loop (random order, random starting
          point), the brain is carried once round a loop. Where one loop is left for
          another the stream shows a pair that is not a link: a coincidence, seen once.
  idle    optional: the eyes close for a long while and the world goes on round its loop.
  test    `blinks` times: the world jumps somewhere, the eyes see `open` items, then close
          for `closed` ticks while the world goes on.

Every tick the motor cells are read as "what the world showed a tick ago". While the eyes
were open a tick ago the teacher shows exactly that, the senses delayed by one tick, so
nothing outside the stream teaches anything. With the eyes closed nobody shows anything:
the j-th closed tick is HOP j, and it can be right only if the brain went on by itself.

Score: bit accuracy (chance 0.5) of the echo while the eyes are open, and by hop while
they are closed. K = items in the world = links; K90 at a hop as in the earlier exams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tissue.assoc import Task


@dataclass(frozen=True)
class Life:
    """The shape of one life."""
    loops: int
    length: int = 8      # items in a loop
    laps: int = 2        # study: times each loop is walked round
    idle: int = 0        # ticks with the eyes closed between study and test
    blinks: int = 8      # test: times the eyes close
    open: int = 3        # items seen before each closing
    closed: int = 4      # ticks each closing lasts

    @property
    def items(self) -> int:
        return self.loops * self.length

    @property
    def ticks(self) -> int:
        return (self.laps * self.loops * (self.length + 1) + self.idle
                + self.blinks * (self.open + self.closed))


def world(task: Task, life: Life, batch: int, gen: torch.Generator) -> torch.Tensor:
    """`batch` worlds: items (B, loops, length, d), all distinct within a world."""
    n = 2 ** task.d_in
    if life.items > n:
        raise ValueError(f"{life.items} distinct items do not fit in {task.d_in} bits")
    idx = torch.rand(batch, n, generator=gen).argsort(dim=1)[:, :life.items]
    bits = ((idx.unsqueeze(-1) >> torch.arange(task.d_in)) & 1).float() * 2 - 1
    return bits.view(batch, life.loops, life.length, task.d_in)


def script(life: Life, batch: int, gen: torch.Generator) -> list:
    """The world's side of a life, the same whoever lives it: per tick
    (kind, seen, loop (B,), place (B,)) with kind in "study" | "idle" | "echo" | "closed"."""
    c, n = life.loops, life.length
    ticks = []
    loop = torch.zeros(batch, dtype=torch.long)
    place = torch.zeros(batch, dtype=torch.long)

    def jump(to_loop=None):
        nonlocal loop, place
        loop = torch.randint(c, (batch,), generator=gen) if to_loop is None else to_loop
        place = torch.randint(n, (batch,), generator=gen)

    def walk(kind, seen):
        nonlocal place
        place = (place + 1) % n
        ticks.append((kind, seen, loop, place))

    for _ in range(life.laps):
        order = torch.rand(batch, c, generator=gen).argsort(dim=1)      # each life its own order
        for i in range(c):
            jump(order[:, i])
            ticks.append(("study", True, loop, place))
            for _ in range(n):                  # once round, back to where it began
                walk("study", True)
    for _ in range(life.idle):
        walk("idle", False)
    for _ in range(life.blinks):
        jump()
        ticks.append(("echo", True, loop, place))
        for _ in range(life.open - 1):
            walk("echo", True)
        for _ in range(life.closed):
            walk("closed", False)
    return ticks


def run_life(brain, task: Task, items: torch.Tensor, life: Life, gen: torch.Generator,
             places: bool = False, told: str = "always") -> dict:
    """Live the stream. Returns {kind: (said, true)} for kind in "echo" | "closed" | "idle",
    each (B, ticks of that kind, d); "closed" is ordered blink by blink, hop by hop.

    `places`: every loop lies at a place of its own, drawn at random along a ring, and the
    brain is told the place the world is at on every tick, eyes open or closed (a place is
    felt, not seen). It says where, never what. `told`: when the brain is told the place:
    "always"; "seen", only while the eyes are open, so that with the eyes closed it has to
    keep the place itself; "study", only while the loops are first walked round, so that in
    the test it has to find where a thing lives from what it sees."""
    if told not in ("always", "seen", "study"):
        raise ValueError(f"told must be always, seen or study, not {told!r}")
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    ticks = script(life, b, gen)                # the world first: the same for any brain
    place_of = torch.rand(b, life.loops, generator=gen) if places else None
    felt = None
    state = brain.born(b, gen)
    silent = torch.zeros(b, d)
    said: dict = {"echo": [], "closed": [], "idle": []}
    true: dict = {"echo": [], "closed": [], "idle": []}
    before, before_seen, before_kind = silent, False, "study"
    for kind, seen, loop, place in ticks + [("end", False, None, None)]:
        item = silent if loop is None else items[rows, loop, place]
        if places and loop is not None:
            felt = place_of[rows, loop]
            if (told == "seen" and not seen) or (told == "study" and kind != "study"):
                felt = None
        elif told != "always":
            felt = None
        state, out = brain.tick(state, item if seen else silent,
                                before if before_seen else silent, float(before_seen), float(seen),
                                where=felt)
        if before_kind in said:                 # what is said now is about the tick before
            said[before_kind].append(out)
            true[before_kind].append(before)
        before, before_seen, before_kind = item, seen, kind
    return {k: (torch.stack(said[k], 1), torch.stack(true[k], 1)) for k in said if said[k]}


def by_hop(out: dict, life: Life) -> list[float]:
    """Bit accuracy with the eyes closed, hop 1 .. `closed`."""
    said, true = out["closed"]
    right = ((said > 0) == (true > 0)).float().view(said.shape[0], life.blinks, life.closed, -1)
    return right.mean(dim=(0, 1, 3)).tolist()


def acc(out: dict, kind: str) -> float:
    said, true = out[kind]
    return float(((said > 0) == (true > 0)).float().mean())


def idle_curve(out: dict, every: int = 8) -> list[float]:
    """Bit accuracy through a long closing, averaged over stretches of `every` ticks."""
    said, true = out["idle"]
    right = ((said > 0) == (true > 0)).float().mean(dim=(0, 2))
    return [float(right[i:i + every].mean()) for i in range(0, len(right), every)]


def ideal(items: torch.Tensor, life: Life, gen: torch.Generator) -> dict:
    """A perfect memory of the stream: for every item, the item most often seen right after
    it. The ceiling: all there is to know from what the eyes were shown. It needs two laps:
    after one, a link and a coincidence have each been seen once and nothing tells them apart."""
    b, c, n, d = items.shape
    k = c * n
    rows = torch.arange(b)
    flat = items.view(b, k, d)
    counts = torch.zeros(b, k, k)
    said: dict = {"echo": [], "closed": [], "idle": []}
    true: dict = {"echo": [], "closed": [], "idle": []}
    last, guess = None, None                    # last item seen; where the memory thinks it is
    for kind, seen, loop, place in script(life, b, gen):
        here = loop * n + place
        if seen:
            if last is not None:
                counts[rows, last, here] += 1
            last = guess = here
        else:
            last = None                         # nothing is linked across closed eyes
            guess = counts[rows, guess].argmax(dim=1)
        if kind in said:
            said[kind].append(flat[rows, guess])
            true[kind].append(flat[rows, here])
    return {key: (torch.stack(said[key], 1), torch.stack(true[key], 1)) for key in said if said[key]}


def k_grid(m: int, task: Task, length: int = 8) -> list[int]:
    """Items to examine a brain of M interneurons at: one loop up to 2 items per interneuron."""
    top = min(2 * m, 2 ** task.d_in)
    ks, k = [], float(length)
    while round(k / length) * length <= top:
        ks.append(round(k / length) * length)
        k *= 2 ** 0.5
    return sorted(set(ks))


def curve(live: Callable, task: Task, ks: list[int], shape: Life, seed: int = 30_000,
          stop_below: float | None = None, lives: int = 4):
    """`live(items, life, gen) -> out` at each K: [{"echo", "hop": [...], "idle": [...]}].
    `shape` gives everything about a life but the number of loops. With `stop_below` the
    loads beyond the first one whose hop 1 falls under it are not examined (K90 and the
    items held in thought are both decided before that). `lives`: the fewest lives a load is
    measured on (a heavy load is otherwise measured on four, and a long closing is one loop
    per life: four loops are a coarse measure of what is held)."""
    rows = []
    for k in ks:
        if stop_below is not None and rows and rows[-1]["hop"][0] < stop_below:
            break
        gen = torch.Generator().manual_seed(seed + k)
        loops = k // shape.length
        life = Life(loops, shape.length, shape.laps, shape.idle, max(shape.blinks, 2 * loops),
                    shape.open, shape.closed)
        items = world(task, life, max(lives, min(32, 1024 // k)), gen)
        with torch.no_grad():
            out = live(items, life, gen)
        rows.append({"echo": acc(out, "echo"), "hop": by_hop(out, life),
                     "idle": idle_curve(out) if "idle" in out else []})
    return rows
