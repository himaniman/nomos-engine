"""Exam 5: a goal. The brain is shown a room it wants and has to get there.

    python -m tissue.seek GENOME.pt ... [--rooms 32] [--ways 2] [--depth 4] [--sizes 256]
                                        [--arm thinks|numb|blind]

The world is the forked world of exam 4, studied the same way (the body led out of every
room by every way, twice). Then, `trials` times in the same unbroken life: the world jumps
somewhere, the eyes see the room, and a WANTED room is named, one that lies 1 to `depth`
steps away by the shortest road. The brain walks, eyes open, until it stands in the wanted
room or has taken `limit` steps.

All the brain knows of the world is in its wires, and the only way to ask them is to walk
them in the dark. THE THINKING, BY HAND: before every step a copy of the brain is walked
with its eyes closed along every string of `depth` ways, its body's sense fed as if it had
gone them; a string along which the copy SAYS the wanted room (all bits but one at most)
is a plan, the shortest plan wins, and its first way is taken by the real body. No plan:
a way at random. What a copy learns in thought is thrown away. So the map is the brain's
(the rule found for loops, the wires of this life); the search through it, the keeping of
the wanted room, the comparing and the choosing are mine.

Arms: `thinks` as above; `numb`, the same thinking in a brain that does not feel its body
(one dendrite whatever the way); `blind`, no thinking, a way at random at every step;
`ponders`, thinking with no copy: the brain itself closes its eyes, walks a string in
thought, opens them on the room it is in, and so for every string, and whatever its wires
learn in thought stays learnt. Sixteen strings four deep are 96 ticks of thought a step.
Two arms between them say which half of that costs: `ponders-still`, the brain walks in
thought itself but its wires are still while it does; `thinks-learns`, a copy walks but
what its wires learn is kept.

Per shortest distance to the wanted room: the share of trials that got there by a shortest
road (`straight`), the share that got there at all within the limit (`there`), and how
long the plan it first made was. A perfect map with the same thinking goes straight every
time.
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from tissue import fork
from tissue.train import load

ARMS = ("thinks", "numb", "blind", "ponders", "ponders-still", "thinks-learns")


def distances(nxt: torch.Tensor, most: int) -> torch.Tensor:
    """(B, rooms, rooms): the fewest steps from a room to a room, `most` + 1 if more."""
    b, n, a = nxt.shape
    step = torch.zeros(b, n, n)
    step.scatter_(2, nxt, 1.0)                  # step[b, i, j] > 0: one way leads from i to j
    far = torch.full((b, n, n), most + 1)
    far[:, torch.arange(n), torch.arange(n)] = 0
    reach = torch.eye(n).expand(b, n, n).clone()
    for k in range(1, most + 1):
        reach = ((reach @ step) > 0).float()
        far = torch.where((reach > 0) & (far > k), torch.full_like(far, k), far)
    return far


def imagine(brain, body: fork.Dendrites, state, ways: torch.Tensor, bits: int, keep: bool = False):
    """Walk the brain in the dark along `ways` (hops,), the same string for every life:
    (the state it ends in, what it said it would see after each hop, (B, hops, bits)).
    With `keep` off it is a copy that walks: the state given is not touched (a tick makes a
    new state) and nothing the dendrites learn in thought is written back."""
    b = state[0]["I"].shape[0]
    silent, said = torch.zeros(b, bits), []
    for way in ways.tolist() + [-1]:            # one more tick: what is said is about the tick before
        state, out = body.tick(brain, state, silent, torch.full((b,), way), silent, 0.0, 0.0, keep=keep)
        said.append(out)
    return state, torch.stack(said[1:], dim=1)


def seek(brain, items: torch.Tensor, nxt: torch.Tensor, walk: fork.Walk, gen: torch.Generator,
         depth: int = 4, trials: int = 16, limit: int | None = None, arm: str = "thinks",
         slack: int = 1) -> dict:
    """One life: study, then `trials` wanted rooms. {"far": (B, trials) the shortest road to
    the wanted room, "steps": (B, trials) steps taken (limit + 1: never got there), "plan":
    (B, trials) the length of the first plan made (0: none)}. `slack`: the bits of the wanted
    room a thought may get wrong and still count as saying it."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    limit = limit or 2 * depth + 2
    own, learns = arm in ("ponders", "ponders-still"), arm in ("ponders", "thinks-learns")
    study = fork.script(nxt, fork.Walk(walk.rooms, walk.ways, walk.laps, blinks=0), gen)
    far_all = distances(nxt, depth)
    strings = torch.tensor(list(itertools.product(range(walk.ways), repeat=depth)))
    state = brain.born(b, gen)
    body = fork.Dendrites(brain, state, walk.ways, numb=arm == "numb")
    silent = torch.zeros(b, d)
    before, seen_before = silent, False

    def look(room, way):                        # a real tick: the body has gone `way`, the eyes see `room`
        nonlocal state, before, seen_before
        item = items[rows, room]
        state, _ = body.tick(brain, state, item, way, before if seen_before else silent,
                             float(seen_before), 1.0)
        before, seen_before = item, True

    for _, _, room, way in study:
        look(room, way)
    out = {k: torch.zeros(b, trials, dtype=torch.long) for k in ("far", "steps", "plan")}
    for t in range(trials):
        room = torch.randint(walk.rooms, (b,), generator=gen)
        look(room, torch.full((b,), -1))
        want_far = 1 + torch.randint(depth, (b,), generator=gen)           # how far the wanted room lies
        ok = far_all[rows, room] == want_far.unsqueeze(1).float()
        ok = torch.where(ok.any(1, keepdim=True), ok, far_all[rows, room] == 1)     # none so far: a neighbour
        want = torch.multinomial(ok.float(), 1, generator=gen).squeeze(1)
        out["far"][:, t] = far_all[rows, room, want].long()
        there = torch.zeros(b, dtype=torch.bool)
        steps = torch.full((b,), limit + 1)
        for s in range(limit):
            way = torch.randint(walk.ways, (b,), generator=gen)             # no plan: a way at random
            if arm != "blind":
                best = torch.full((b,), depth + 1)
                for string in strings:
                    after, says = imagine(brain, body, state, string, d, keep=learns)
                    state = (after[0] if own else state[0], after[1] if learns else state[1], state[2])
                    if own:                     # no copy walked: it opens its eyes on the room again
                        seen_before = False
                        look(room, torch.full((b,), -1))
                    hit = ((says > 0) == (items[rows, want].unsqueeze(1) > 0)).sum(-1) >= d - slack
                    first = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.full((b,), depth + 1))
                    better = first < best
                    best = torch.where(better, first, best)
                    way = torch.where(better, string[0].expand(b), way)
                if s == 0:
                    out["plan"][:, t] = torch.where(best <= depth, best, torch.zeros_like(best))
            way = torch.where(there, torch.full_like(way, -1), way)         # arrived: it stays (carried nowhere)
            room = torch.where(there, room, nxt[rows, room, way.clamp(min=0)])
            look(room, way)
            now = (room == want) & ~there
            steps[now] = s + 1
            there |= now
            if bool(there.all()):
                break
        out["steps"][:, t] = steps
    return out


def thought(brain, items: torch.Tensor, nxt: torch.Tensor, walk: fork.Walk, gen: torch.Generator,
            depth: int = 4) -> dict:
    """One thought, kept for a picture: after the study the first life is put in a room and
    wants a room `depth` steps away. {"strings": (S, depth) the ways tried, "says": (S,
    depth, bits) what the copy said after each hop, "true": (S, depth) the room each hop
    really leads to, "here", "want": rooms, "items": (rooms, bits)}."""
    b, d = items.shape[0], items.shape[-1]
    rows = torch.arange(b)
    state = brain.born(b, gen)
    body = fork.Dendrites(brain, state, walk.ways)
    silent, before, seen_before = torch.zeros(b, d), None, False
    here = torch.randint(walk.rooms, (b,), generator=gen)
    far = distances(nxt, depth)
    study = fork.script(nxt, fork.Walk(walk.rooms, walk.ways, walk.laps, blinks=0), gen)
    for _, _, room, way in study + [("", True, here, torch.full((b,), -1))]:
        item = items[rows, room]
        state, _ = body.tick(brain, state, item, way, before if seen_before else silent,
                             float(seen_before), 1.0)
        before, seen_before = item, True
    ok = far[0, here[0]] == depth
    want = int(torch.multinomial(ok.float(), 1, generator=gen)) if bool(ok.any()) else int(nxt[0, here[0], 0])
    strings = torch.tensor(list(itertools.product(range(walk.ways), repeat=depth)))
    says, true = [], []
    for string in strings:
        says.append(imagine(brain, body, state, string, d)[1][0])
        room, road = int(here[0]), []
        for way in string.tolist():
            room = int(nxt[0, room, way])
            road.append(room)
        true.append(road)
    return {"strings": strings, "says": torch.stack(says), "true": torch.tensor(true),
            "here": int(here[0]), "want": want, "items": items[0]}


def by_distance(out: dict, depth: int, limit: int, trials: slice = slice(None)) -> dict:
    """{distance: {"n", "straight", "there", "plan", "planned"}} over the wanted rooms `trials`
    of every life (all of them, or the first few, or the last)."""
    table = {}
    for far in range(1, depth + 1):
        mask = out["far"][:, trials] == far
        n = int(mask.sum())
        if n:
            steps, plan = out["steps"][:, trials][mask], out["plan"][:, trials][mask].float()
            table[far] = {"n": n, "straight": float((steps == far).float().mean()),
                          "there": float((steps <= limit).float().mean()),
                          "plan": float(plan[plan > 0].mean()) if bool((plan > 0).any()) else 0.0,
                          "planned": float((plan > 0).float().mean())}
    return table


def examine(path: str, sizes, walk: fork.Walk, depth: int = 4, trials: int = 16, lives: int = 16,
            arm: str = "thinks", seed: int = 80_000, slack: int = 1, **switches) -> dict:
    model, task, _ = load(path, **switches)
    res = {}
    for m in sizes:
        gen = torch.Generator().manual_seed(seed + walk.rooms)
        items, nxt = fork.world(task.d_out, walk, lives, gen)
        with torch.no_grad():
            out = seek(model.grow(m), items, nxt, walk, gen, depth, trials, arm=arm, slack=slack)
        res[m] = by_distance(out, depth, 2 * depth + 2)
        for half, cols in (("first", slice(0, trials // 2)), ("last", slice(trials // 2, None))):
            for far, row in by_distance(out, depth, 2 * depth + 2, cols).items():
                res[m][far][half] = row["straight"]         # does a life of thinking wear the map?
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="+")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--ways", type=int, default=2)
    p.add_argument("--depth", type=int, default=4, help="how far a wanted room lies at most, and how deep a thought goes")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--lives", type=int, default=16)
    p.add_argument("--sizes", type=int, nargs="*", default=[256])
    p.add_argument("--arm", choices=ARMS, default="thinks")
    p.add_argument("--slack", type=int, default=1,
                   help="bits of the wanted room a thought may get wrong and still count as saying it")
    p.add_argument("--still", action="store_true",
                   help="by hand, in the kernel: with the eyes closed no wire changes")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    walk = fork.Walk(rooms=a.rooms, ways=a.ways)
    tag = f".seek{a.rooms}x{a.ways}.depth{a.depth}.{a.arm}" + (f".slack{a.slack}" if a.slack != 1 else "") + (
        ".still" if a.still else "")
    switches = {"still": True} if a.still else {}
    for path in a.genomes:
        res = examine(path, a.sizes, walk, a.depth, a.trials, a.lives, a.arm, slack=a.slack, **switches)
        print(f"\n{path}  {a.rooms} rooms, {a.ways} ways ({walk.links} links), {a.arm};"
              f" by the shortest road to the wanted room:")
        for m, table in res.items():
            print(f"{m:>5} cells | straight " + " ".join(f"{r['straight']:.2f}" for r in table.values())
                  + " | there " + " ".join(f"{r['there']:.2f}" for r in table.values())
                  + " | made a plan " + " ".join(f"{r['planned']:.2f}" for r in table.values())
                  + " | its length " + " ".join(f"{r['plan']:.1f}" for r in table.values())
                  + " | trials " + " ".join(str(r["n"]) for r in table.values())
                  + " | straight, first half " + " ".join(f"{r.get('first', 0):.2f}" for r in table.values())
                  + ", last half " + " ".join(f"{r.get('last', 0):.2f}" for r in table.values()))
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as f:
            json.dump({str(m): {str(k): v for k, v in t.items()} for m, t in res.items()}, f)


if __name__ == "__main__":
    main()
