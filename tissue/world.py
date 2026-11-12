"""Exam 6: a world. A brain is born into a walled land it has never seen and has to eat.

    python -m tissue.world GENOME.pt ... [--side 6] [--sizes 256] [--arm thinks|numb]
    python -m tissue.world --yardsticks [--side 6]

THE LAND is a square grid of rooms. Between two rooms that touch there is a wall or there is
none, and the land hangs together. Every room has a LOOK of its own, a random item, and no
two rooms look alike. The body goes north, east, south or west; going into a wall it stays
where it is. On a few rooms stands a TREE, and one tree at a time bears a fruit. A body that
comes to it eats, and another tree, drawn at random, bears the next.

THE BRAIN is the tissue of exam 5. Its eyes see the look of the room it stands in, they
never close, and its body feels the way it has just gone (BY HAND: a dendrite per way).
Nobody leads it: there is no study and there are no trials, one life from birth to the last
tick, and all it ever learns of the land it learns by going about in it.

THE WILL, BY HAND: it wants the room where the fruit is, and is told what that room looks
like (as in exam 5 a wanted room is named: a call of the fruit, if you like). Before every
step it THINKS; if the thought finds a road it takes the first way of it, if not, a way at
random. So a newborn, whose wires hold nothing, wanders; and what it wanders through is what
it later thinks with.

THE THINKING, BY HAND, on the map in the brain's own wires: copies of what its cells are
saying now walk in the dark, wires still, one hop along every way, then one hop on from
every room so reached, and so on (the nearest rooms first). What a copy says it would see
is its room; a room already thought of is not thought of again; the first copy to say the
wanted look ends the thought and the first way of its road is the plan.

Arms: `thinks` as above; `numb`, the same in a brain that does not feel its body.
Yardsticks, the same will with another memory: `blind` never has a plan and walks at
random; `perfect` remembers every way it has ever gone and thinks on that; `oracle` was
born knowing every way of the land. In one land the fruit goes from tree to tree in the
same order whoever lives there.

Per life: meals; meals in each third of it; and per TRIP (from one meal to the next) the
shortest road and the steps taken.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace

import torch

from tissue import fork
from tissue.train import load

WAYS = ((-1, 0), (0, 1), (1, 0), (0, -1))       # north, east, south, west: (row, column)
ARMS = ("thinks", "numb")
YARDSTICKS = ("blind", "perfect", "oracle")


@dataclass(frozen=True)
class Land:
    """The shape of a land and of a life in it."""
    side: int = 6
    walls: float = 0.3      # the share of the places between two rooms where a wall stands
    trees: int = 0          # 0: one to every six rooms
    ticks: int = 0          # the length of a life; 0: forty ticks to every room

    @property
    def rooms(self) -> int:
        return self.side * self.side

    @property
    def n_trees(self) -> int:
        return self.trees or max(2, self.rooms // 6)

    @property
    def life(self) -> int:
        return self.ticks or 40 * self.rooms


def land(bits: int, shape: Land, gen: torch.Generator):
    """One land: looks (rooms, bits), all distinct; nxt (rooms, 4), where each way leads (the
    room itself: a wall); trees (K,), the rooms with a tree; the room a body is born in."""
    s, n = shape.side, shape.rooms
    if n > 2 ** bits:
        raise ValueError(f"{n} distinct looks do not fit in {bits} bits")
    idx = torch.rand(2 ** bits, generator=gen).argsort()[:n]
    looks = ((idx.unsqueeze(-1) >> torch.arange(bits)) & 1).float() * 2 - 1
    gaps = [(r * s + c, r * s + c + 1) for r in range(s) for c in range(s - 1)]
    gaps += [(r * s + c, (r + 1) * s + c) for r in range(s - 1) for c in range(s)]
    gaps = [gaps[i] for i in torch.randperm(len(gaps), generator=gen).tolist()]
    root = list(range(n))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    open_, spare = set(), []
    for a, b in gaps:                           # a tree of open gaps first: the land hangs together
        if find(a) != find(b):
            root[find(a)] = find(b)
            open_.add((a, b))
        else:
            spare.append((a, b))
    walled = min(len(spare), round(shape.walls * len(gaps)))
    open_.update(spare[walled:])
    nxt = torch.arange(n).unsqueeze(1).repeat(1, 4)
    for a, b in open_:
        way = 1 if b == a + 1 else 2            # b lies east of a, or south of it
        nxt[a, way], nxt[b, (way + 2) % 4] = b, a
    trees = torch.randperm(n, generator=gen)[:shape.n_trees]
    return looks, nxt, trees, int(torch.randint(n, (1,), generator=gen))


def roads(nxt: torch.Tensor) -> torch.Tensor:
    """(rooms, rooms): the fewest steps from a room to a room."""
    n = nxt.shape[0]
    far = torch.full((n, n), n, dtype=torch.long)
    for a in range(n):
        far[a, a], edge, k = 0, [a], 0
        while edge:
            k += 1
            new = {int(b) for r in edge for b in nxt[r] if far[a, b] > k}
            for b in new:
                far[a, b] = k
            edge = list(new)
    return far


def keys(looks: torch.Tensor) -> list:
    """Looks (N, bits) as one number each: the bits by sign."""
    return ((looks > 0).long() << torch.arange(looks.shape[-1])).sum(-1).tolist()


def key(look: torch.Tensor) -> int:
    return keys(look.unsqueeze(0))[0]


class Blind:
    """A body with no memory at all: it never has a plan."""

    def see(self, look: torch.Tensor, way: int) -> None:
        pass

    def think(self, wanted: torch.Tensor, gen: torch.Generator, trace: list | None = None):
        return None


class Table(Blind):
    """A perfect memory of every way gone: (look, way) -> look. With `whole` it was born
    knowing every way of the land."""

    def __init__(self, whole: tuple | None = None):
        self.map: dict = {}
        self.at = None
        if whole:
            looks, nxt = whole
            ks = keys(looks)
            self.map = {(ks[r], w): ks[int(nxt[r, w])] for r in range(len(ks)) for w in range(4)}

    def see(self, look, way):
        now = key(look)
        if way >= 0 and self.at is not None:
            self.map[(self.at, way)] = now
        self.at = now

    def think(self, wanted, gen, trace=None):
        goal, seen = key(wanted), {self.at}
        edge = [(self.at, -1)]                  # (a room, the first way of the road to it)
        while edge:
            new, hits = [], []
            for room, first in edge:
                for w in range(4):
                    to = self.map.get((room, w))
                    if to is None or to in seen:
                        continue
                    f = w if first < 0 else first
                    if to == goal:
                        hits.append(f)
                    new.append((to, f))
            if hits:
                return hits[int(torch.randint(len(hits), (1,), generator=gen))]
            seen.update(to for to, _ in new)
            edge = list({to: (to, f) for to, f in new}.values())
        return None


class Mind(Blind):
    """The tissue in a body: it sees the look of its room, feels the way just gone and thinks
    in the dark. `depth`: how many hops a thought may go; `budget`: how many rooms it may
    think of; `slack`: the bits of the wanted look a thought may get wrong. `imagery`, BY
    HAND: in thought a copy is SHOWN what it has just said, as if its eyes saw that room (and
    no wire changes), so every hop of a thought starts from a sight and not from the hop
    before it. `mumble`, BY HAND: a copy that says its room more quietly than this (the mean
    size of the bits it says) is not followed: the brain speaks low of what it does not know."""

    def __init__(self, brain, gen: torch.Generator, numb: bool = False, depth: int = 24,
                 budget: int = 256, slack: int = 0, imagery: bool = False, mumble: float = 0.0):
        if not brain.t.plan.still:
            raise ValueError("a thought needs its wires still: load the genome with still=True")
        self.brain, self.depth, self.budget, self.slack = brain, depth, budget, slack
        self.imagery, self.mumble = imagery, mumble
        self.state = brain.born(1, gen)
        self.body = fork.Dendrites(brain, self.state, 4, numb=numb)
        self.bits = brain.n["O"]
        self.before = None                      # the look of the tick before
        deaf = copy.copy(brain.t)               # the examiner's twin: it sees and does not learn
        deaf.plan = replace(deaf.plan, learn=False)
        self.deaf = type(brain)(deaf, {"I": brain.n["I"]})

    def see(self, look, way):
        look = look.unsqueeze(0)
        teach = torch.zeros_like(look) if self.before is None else self.before
        self.state, _ = self.body.tick(self.brain, self.state, look, torch.tensor([way]), teach,
                                       float(self.before is not None), 1.0)
        self.before = look

    def hop(self, act: dict, way: int):
        """Every copy in `act` goes one hop in the dark by `way`: (what its cells say after,
        what it said it saw BEFORE the hop: the room it was in)."""
        f = act["I"].shape[0]
        bank = self.body.bank[:, 0 if self.body.numb else way]
        wires = {**self.state[1], self.body.inner: bank}
        silent = torch.zeros(f, self.bits)
        state, out = self.brain.tick((act, wires, self.state[2]), silent, silent, 0.0, 0.0)
        return state[0], out

    def think(self, wanted, gen, trace=None):
        """The first way of the shortest road the brain can think of to the `wanted` look, or
        None. `trace`: a list that gets (hops, the room a copy came from, the way, the room
        it said it reached) for every copy, rooms as `key`s, and last, if a road was found,
        (0, None, the ways of that road, None)."""
        act = {t: a.clone() for t, a in self.state[0].items()}      # one copy, in the room it is in
        first = torch.tensor([-1])              # per copy: the first way of its road
        came = [(key(self.before[0]), -1)]      # per copy: (the room it came from, by which way)
        seen, spent = {came[0][0]}, 0
        ways = [()] if trace is not None else None          # per copy: the ways of its road
        for hops in range(self.depth + 1):
            north, says = self.hop(act, 0)      # what a copy says of its room is the same whatever the way
            keep, rooms = [0], [came[0][0]]
            if hops:                            # only now is it known what room each copy is in
                rooms = keys(says)
                if trace is not None:
                    trace.extend((hops, a, w, b) for (a, w), b in zip(came, rooms))
                hit = ((says > 0) == (wanted > 0)).sum(-1) >= self.bits - self.slack
                if bool(hit.any()):
                    hits = hit.nonzero().flatten()
                    j = int(hits[int(torch.randint(len(hits), (1,), generator=gen))])
                    if trace is not None:
                        trace.append((0, None, ways[j], None))
                    return int(first[j])
                keep = []
                clear = (says.abs().mean(-1) >= self.mumble).tolist()
                for j, room in enumerate(rooms):
                    if clear[j] and room not in seen:
                        seen.add(room)
                        keep.append(j)
            spent += len(keep)
            if not keep or spent > self.budget:
                return None
            pick = torch.tensor(keep)           # only the copies in new rooms go on, by every way
            act = {t: a[pick] for t, a in act.items()}
            if self.imagery and hops:           # each is shown the room it said, and goes on from that sight
                shown = torch.where(says[pick] > 0, 1.0, -1.0)
                calm = {**self.state[1], self.body.inner: torch.zeros_like(self.body.bank[:, 0])}
                act = self.deaf.tick((act, calm, self.state[2]), shown, torch.zeros_like(shown), 0.0, 1.0)[0][0]
                kids = [self.hop(act, w)[0] for w in range(4)]
            else:
                kids = [{t: a[pick] for t, a in north.items()}] + [self.hop(act, w)[0] for w in (1, 2, 3)]
            act = {t: torch.cat([kid[t] for kid in kids]) for t in kids[0]}
            first = torch.cat([first[pick] if hops else torch.full((1,), w) for w in range(4)])
            came = [(rooms[j], w) for w in range(4) for j in keep]
            if ways is not None:
                ways = [ways[j] + (w,) for w in range(4) for j in keep]
        return None

    def atlas(self, looks: torch.Tensor):
        """The examiner asks, the brain is not touched: a copy is carried into every room,
        sees it and goes one hop in the dark by every way. (rooms, 4, bits): what it says it
        would see there."""
        n = looks.shape[0]
        act = {t: a.expand(n, -1) for t, a in self.state[0].items()}
        wires = {**self.state[1], self.body.inner: torch.zeros_like(self.body.bank[:, 0])}
        teach = (torch.zeros_like(looks) if self.before is None else self.before.expand(n, -1))
        state, _ = self.deaf.tick((act, wires, self.state[2]), looks, teach,
                                  float(self.before is not None), 1.0)
        says = []
        for w in range(4):
            there, _ = self.hop(state[0], w)
            says.append(self.hop(there, 0)[1])
        return torch.stack(says, dim=1)


def live(mind, looks: torch.Tensor, nxt: torch.Tensor, trees: torch.Tensor, start: int, ticks: int,
         gen: torch.Generator, fruits: int = 0, watch=None) -> dict:
    """One life. {"meals": [the tick of each], "trips": [(shortest road, steps taken, the tick
    it ended)], "planned": ticks on which a thought found a road, "walked": (rooms, 4) how
    often each way out of each room was gone}. `fruits`: the seed of the order in which the
    trees bear. `watch(tick, room, fruit, way, planned)` is called before every step."""
    far = roads(nxt)
    order = torch.Generator().manual_seed(fruits)       # the land's own dice, not the body's
    other = lambda room: int(trees[trees != room][torch.randint(int((trees != room).sum()), (1,), generator=order)])
    room, fruit, since = start, other(start), 0
    trip = int(far[room, fruit])
    out = {"meals": [], "trips": [], "planned": 0, "walked": torch.zeros(len(looks), 4)}
    mind.see(looks[room], -1)
    for t in range(ticks):
        if room == fruit:                       # it eats, and another tree bears
            out["meals"].append(t)
            out["trips"].append((trip, t - since, t))
            fruit, since = other(room), t
            trip = int(far[room, fruit])
        plan = mind.think(looks[fruit], gen)
        out["planned"] += plan is not None
        way = int(torch.randint(4, (1,), generator=gen)) if plan is None else plan
        if watch:
            watch(t, room, fruit, way, plan is not None)
        out["walked"][room, way] += 1
        room = int(nxt[room, way])
        mind.see(looks[room], way)
    return out


def body_for(arm: str, model, m: int, looks, nxt, gen, **think):
    """The memory an arm lives with."""
    if arm == "blind":
        return Blind()
    if arm in ("perfect", "oracle"):
        return Table((looks, nxt) if arm == "oracle" else None)
    return Mind(model.grow(m), gen, numb=arm == "numb", **think)


def known_map(mind: Mind, looks, nxt, walked) -> dict:
    """What the brain knows of its land at this tick, asked by the examiner: the share of
    ways it says the right room behind (every bit), among the ways it has gone and among
    those it never went; and how LOUD it speaks of each (the mean size of what it says)."""
    says = mind.atlas(looks)
    right = ((says > 0) == (looks[nxt] > 0)).all(-1)
    loud = says.abs().mean(-1)
    gone = walked > 0
    share = lambda x, mask: float(x[mask].float().mean()) if bool(mask.any()) else 0.0
    return {"gone": float(gone.float().mean()), "right_gone": share(right, gone),
            "right_never": share(right, ~gone), "loud_gone": share(loud, gone),
            "loud_never": share(loud, ~gone)}


BANDS = ((1, 2), (3, 4), (5, 8), (9, 99))      # trips by the shortest road to the wanted room


def banded(trips: list) -> dict:
    """Trips by the length of their shortest road: how many, the share that went by a
    shortest road, and the steps taken per step of it."""
    bands = {}
    for lo, hi in BANDS:
        rows = [(far, steps) for far, steps, _ in trips if lo <= far <= hi]
        if rows:
            bands[f"{lo}-{hi}" if hi < 99 else f"{lo}+"] = {
                "n": len(rows), "straight": sum(steps == far for far, steps in rows) / len(rows),
                "steps_per_far": sum(steps for _, steps in rows) / sum(far for far, _ in rows)}
    return bands


def summary(lives: list, ticks: int) -> dict:
    """Over the lives of one arm: meals a life and in each third of it; the trips by band,
    all of them and those that ended in the last third (`late`)."""
    n = len(lives)
    third = [sum(sum(1 for t in life["meals"] if k * ticks // 3 <= t < (k + 1) * ticks // 3) for life in lives) / n
             for k in range(3)]
    trips = [trip for life in lives for trip in life["trips"]]
    res = {"meals": sum(len(life["meals"]) for life in lives) / n, "thirds": third,
           "trips": banded(trips), "late": banded([trip for trip in trips if trip[2] >= 2 * ticks // 3]),
           "per_life": [len(life["meals"]) for life in lives],
           "late_per_life": [sum(1 for t in life["meals"] if t >= 2 * ticks // 3) for life in lives],
           "planned": sum(life["planned"] for life in lives) / (n * ticks)}
    maps = [life["map"] for life in lives if "map" in life]
    if maps:
        res["map"] = {k: sum(m[k] for m in maps) / len(maps) for k in maps[0]}
    return res


def examine(arm: str, shape: Land, lives: int = 16, path: str | None = None, m: int = 256,
            seed: int = 90_000, bits: int = 12, **think) -> dict:
    """`lives` lives of one arm, each in a land of its own: the same lands whatever the arm."""
    model = None
    if arm in ARMS:
        model, task, _ = load(path, still=True)
        bits = task.d_out
    rows = []
    for i in range(lives):
        gen = torch.Generator().manual_seed(seed + 1000 * shape.side + i)
        looks, nxt, trees, start = land(bits, shape, gen)
        with torch.no_grad():
            mind = body_for(arm, model, m, looks, nxt, gen, **think)
            life = live(mind, looks, nxt, trees, start, shape.life, gen, fruits=seed + i)
            if arm in ARMS:
                life["map"] = known_map(mind, looks, nxt, life["walked"])
        rows.append(life)
    return summary(rows, shape.life)


def show(name: str, res: dict) -> str:
    trips = lambda bands: " ".join(f"{band}: {r['straight']:.2f} x{r['steps_per_far']:.1f} ({r['n']})"
                                   for band, r in bands.items())
    line = (f"{name:>18} | meals {res['meals']:6.1f} | by third " + " ".join(f"{x:5.1f}" for x in res["thirds"])
            + f" | plan {res['planned']:.2f} | trips, straight and steps per step of the shortest road:"
            + f" {trips(res['trips'])} | in the last third: {trips(res['late'])}")
    if "map" in res:
        k = res["map"]
        line += (f" | ways gone {k['gone']:.2f}, right of them {k['right_gone']:.2f}, of the rest"
                 f" {k['right_never']:.2f}; loud {k['loud_gone']:.2f} / {k['loud_never']:.2f}")
    return line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("genomes", nargs="*")
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--walls", type=float, default=0.3)
    p.add_argument("--trees", type=int, default=0, help="0: one to every six rooms")
    p.add_argument("--ticks", type=int, default=0, help="the length of a life; 0: forty to a room")
    p.add_argument("--sizes", type=int, nargs="*", default=[256])
    p.add_argument("--lives", type=int, default=16)
    p.add_argument("--arm", choices=ARMS, default="thinks")
    p.add_argument("--yardsticks", action="store_true", help="blind, perfect and oracle in the same lands")
    p.add_argument("--depth", type=int, default=24, help="how many hops a thought may go")
    p.add_argument("--budget", type=int, default=256, help="how many rooms a thought may think of")
    p.add_argument("--slack", type=int, default=0, help="bits of the wanted look a thought may get wrong")
    p.add_argument("--mumble", type=float, default=0.0,
                   help="by hand: a copy that says its room more quietly than this is not followed")
    p.add_argument("--imagery", action="store_true",
                   help="by hand: in thought a copy is shown what it has just said, as if it saw it")
    p.add_argument("--seed", type=int, default=90_000)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    shape = Land(a.side, a.walls, a.trees, a.ticks)
    think = {"depth": a.depth, "budget": a.budget, "slack": a.slack, "imagery": a.imagery, "mumble": a.mumble}
    pilot = "" if a.seed == 90_000 else f".seed{a.seed}"        # not the exam's lands: a look
    print(f"{shape.side}x{shape.side} rooms ({4 * shape.rooms} ways), {shape.n_trees} trees,"
          f" {shape.life} ticks a life, {a.lives} lives")
    if a.yardsticks:
        res = {arm: examine(arm, shape, a.lives, seed=a.seed) for arm in YARDSTICKS}
        for arm, row in res.items():
            print(show(arm, row))
        with open(f"runs/tissue/world{shape.side}.yardsticks{pilot}.json", "w") as f:
            json.dump(res, f)
    tag = (f".world{shape.side}.{a.arm}" + (f".slack{a.slack}" if a.slack else "")
           + (".imagery" if a.imagery else "") + (f".mumble{a.mumble:g}" if a.mumble else "") + pilot)
    for path in a.genomes:
        res = {}
        for m in a.sizes:
            res[m] = examine(a.arm, shape, a.lives, path, m, seed=a.seed, **think)
            print(show(f"{path.split('/')[-1][:-3]} {m}", res[m]), flush=True)
        with open(path.rsplit(".", 1)[0] + tag + ".json", "w") as f:
            json.dump({str(m): row for m, row in res.items()}, f)


if __name__ == "__main__":
    main()
