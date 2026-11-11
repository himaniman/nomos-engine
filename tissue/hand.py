"""Hand-made memories for exam 0: what each size can hold when the rule is written by hand.

    python -m tissue.hand
    python -m tissue.hand --hop
    python -m tissue.hand --think [--cells 512 2048] [--wires 32 64] [--patch 64 --reach 32]
    python -m tissue.hand --index [--cells 512 2048 4096]

Nothing here is searched for. They are the yardstick a found genome is read against:

  direct   no interneurons: sensory -> motor wires, Hebb. Does not depend on size.
  hebb     M interneurons with random innate input wires and a threshold, Hebb on their
           wires to the motor cells (Marr's expansion). One multiply per wire per pair.
  delta    the same code, but a wire changes by the motor cell's ERROR, one pass.
  lstsq    the same code, the best possible linear read-out of it (not a local rule):
           the ceiling for this kind of interneuron.

Each uses the sparseness (share of interneurons a cue lights) that serves it best.

`--think` is the ceiling for a thought that holds (exam 2) when a cell has only K wires, drawn
at random from the other cells: every cell learns by itself, and long, to say its bit of the
next item's code from what its K sources said; then the brain is left alone for 64 ticks. It
is run with every cell speaking for every item and with few cells speaking per item.

`--index` is a memory of what lies where, by hand (`Index`): how many items it can tell the
place of, per cell of its own, on a ring of a given size.
"""

from __future__ import annotations

import math

import torch

from tissue.assoc import Task, curve, k90, k_grid

SPARSE = (0.5, 0.2, 0.1, 0.05, 0.02)
SIZES = (8, 16, 32, 64, 128, 256, 512)


def codes(cues: torch.Tensor, m: int, f: float, gen: torch.Generator) -> torch.Tensor:
    """What M threshold interneurons say about each cue, centred: (B, K, M)."""
    b, _, d = cues.shape
    wires = torch.randn(b, m, d, generator=gen)
    drive = torch.einsum("bmd,bkd->bkm", wires, cues) / math.sqrt(d)
    theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))
    return (drive > theta).float() - f


def hebb(c: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bko,bkm,bjm->bjo", answers, c, c)


def delta_wires(c: torch.Tensor, answers: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """One pass of the delta rule over the pairs in order: wires (B, answer bits, M)."""
    w = torch.zeros(c.shape[0], answers.shape[2], c.shape[2])
    for k in range(c.shape[1]):
        ck = c[:, k]
        err = answers[:, k] - torch.einsum("bom,bm->bo", w, ck)
        w = w + rate * err.unsqueeze(-1) * (ck / (ck * ck).sum(-1, keepdim=True)).unsqueeze(1)
    return w


def delta(c: torch.Tensor, answers: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    return torch.einsum("bom,bkm->bko", delta_wires(c, answers, rate), c)


def hop(items: torch.Tensor, m: int, f: float, gen: torch.Generator, rate: float = 1.0,
        exact: bool = False) -> torch.Tensor:
    """Exam 1 by hand, one hop: items (B, K, 2, d) are links first -> second.

    Wires between interneurons store code(first) -> code(second); the recalled code is
    cleaned by the interneuron's own threshold; a second set of wires, taught on every item
    seen, decodes a code back into the item. `exact`: both sets by least squares (not local).
    """
    b, _, _, d = items.shape
    wires = torch.randn(b, m, d, generator=gen)
    theta = math.sqrt(2) * torch.erfinv(torch.tensor(1 - 2 * f))

    def code(x):
        return (torch.einsum("bmd,bkd->bkm", wires, x) / math.sqrt(d) > theta).float() - f

    first, second = code(items[:, :, 0]), code(items[:, :, 1])
    seen, said = torch.cat([first, second], 1), torch.cat([items[:, :, 0], items[:, :, 1]], 1)
    if exact:
        recalled = first @ (torch.linalg.pinv(first) @ second)
        decode = (torch.linalg.pinv(seen) @ said).transpose(1, 2)
    else:
        recalled = delta(first, second, rate)
        decode = delta_wires(seen, said, rate)
    clean = (recalled > 0.5 - f).float() - f
    return torch.stack([torch.einsum("bom,bkm->bko", decode, first),      # hop 0, the echo
                        torch.einsum("bom,bkm->bko", decode, clean)], dim=2)


def lstsq(c: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return c @ (torch.linalg.pinv(c) @ answers)


def think(n: int, k: int, f: float, items: int, signed: bool = False, seed: int = 0,
          steps: int = 400, closing: int = 64) -> float:
    """A thought held by hand. N cells, K wires into each from random others, `items` on
    loops of 8. An item's code is fixed: the share `f` of cells with the largest random
    drive speak (1), the rest are silent (0, or -1 when `signed`: then silence is heard too,
    which is what a code where half the cells speak needs). Each cell fits its own K wires
    (logistic regression, `steps` passes over all the links: near the best any rule local
    to a cell could do). Left alone from an item's code for `closing` ticks, each tick the
    share `f` of cells driven hardest speak: an inhibition that never errs.

    Returns how much of the last lap's speaking cells are the right ones (1: all)."""
    g = torch.Generator().manual_seed(seed)
    a, loops = max(1, round(f * n)), items // 8
    p = loops * 8
    code = torch.zeros(p, n).scatter_(1, torch.randn(p, n, generator=g).topk(a, dim=1).indices, 1.0)
    after = torch.arange(p).view(loops, 8).roll(-1, dims=1).flatten()   # what follows on its loop
    sources = torch.randint(n, (n, k), generator=g)

    def heard(c):
        return (c * 2 - 1 if signed else c)[:, sources]                 # (items, n, k)

    with torch.enable_grad():
        w = torch.zeros(n, k, requires_grad=True)
        b = torch.zeros(n, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        x, y, rare = heard(code), code[after], torch.tensor((n - a) / a)
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                (x * w).sum(-1) + b, y, pos_weight=rare)
            opt.zero_grad()
            loss.backward()
            opt.step()
    w, b = w.detach(), b.detach()
    at = torch.arange(0, p, 8)[:32]                                     # one start on each loop
    state, right = code[at], []
    for t in range(closing):
        drive = (heard(state) * w).sum(-1) + b
        state = torch.zeros_like(drive).scatter_(1, drive.topk(a, dim=1).indices, 1.0)
        at = after[at]
        if t >= closing - 8:
            right.append(float((state * code[at]).sum(1).mean() / a))
    return sum(right) / len(right)


def think_in_places(n: int, k: int, items: int, reach: int, patch: int, seed: int = 0,
                    steps: int = 400, closing: int = 64) -> float:
    """A thought held by hand in a brain with places. N cells on a ring, K wires into each
    from different neighbours within `reach`; every loop of 8 items lies at a place drawn at
    random and only the `patch` cells nearest to it speak for its items (+1 or -1, at
    random), the rest are silent (0) and learn nothing of it. Each cell fits its own K wires
    as in `think`. Left alone on a loop, only that loop's patch speaks: a sense of place
    that never errs.

    Returns the share of the patch that says the right thing through the last lap."""
    g = torch.Generator().manual_seed(seed)
    loops = items // 8
    p = loops * 8
    span = torch.cat([torch.arange(-reach, 0), torch.arange(1, reach + 1)])
    pick = torch.rand(n, len(span), generator=g).argsort(-1)[:, :k]
    sources = (torch.arange(n).unsqueeze(1) + span[pick]) % n                       # (n, k)
    first = (torch.rand(loops, generator=g) * n).long()
    cells = (first.unsqueeze(1) + torch.arange(patch)) % n                          # (loops, patch)
    of = torch.arange(loops).repeat_interleave(8)                                   # the loop of an item
    code = torch.zeros(p, n)
    code.scatter_(1, cells[of], torch.randint(2, (p, patch), generator=g).float() * 2 - 1)
    after = torch.arange(p).view(loops, 8).roll(-1, dims=1).flatten()
    mine = cells[of]                                                                # (p, patch)

    def heard(c, who):                      # what the cells `who` (rows, patch) hear of c (rows, n)
        return c.gather(1, sources[who].flatten(1)).view(*who.shape, k)

    with torch.enable_grad():
        w = torch.zeros(n, k, requires_grad=True)
        b = torch.zeros(n, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        x, y = heard(code, mine), (code[after].gather(1, mine) > 0).float()
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                (x * w[mine]).sum(-1) + b[mine], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    w, b = w.detach(), b.detach()
    at = torch.arange(0, p, 8)[:32]
    who = cells[of[at]]
    state, right = code[at], []
    for t in range(closing):
        drive = (heard(state, who) * w[who]).sum(-1) + b[who]
        state = torch.zeros(len(at), n).scatter_(1, who, (drive > 0).float() * 2 - 1)
        at = after[at]
        if t >= closing - 8:
            right.append(float((state.gather(1, who) == code[at].gather(1, who)).float().mean()))
    return sum(right) / len(right)


def held_by_hand(n: int, k: int, f: float = 0.5, signed: bool = False, bar: float = 0.9,
                 reach: int = 0, patch: int = 0) -> int:
    """The largest load, on a grid growing by half, that `think` still holds at `bar`
    (`think_in_places` when a `patch` is given)."""
    def holds(items):
        if patch:
            return think_in_places(n, k, items, reach, patch) >= bar
        return think(n, k, f, items, signed) >= bar
    best, items = 0, 8
    while items <= 2 ** 13 and holds(items):
        best, items = items, 8 * max(items // 8 + 1, round(items * 1.5 / 8))
    return best


THINK = ((0.5, True), (0.25, False), (0.125, False), (0.0625, False))     # share speaking, signed


class Index:
    """A memory of what lies where, by hand. `cells` cells with random innate wires from
    the senses, each saying a wave of its dendrite (cos of `wave` times the drive, at a
    phase of its own): a code that is unlike for items that differ in a bit or two, which a
    threshold is not when an item is twelve bits (the search of rung 0 found such a
    non-monotonic code by itself). From them a few motor-like cells learn, by the delta
    rule, the place an item was seen at. The place is written in a Gray code, so
    that neighbouring stretches of the ring differ in one bit, as fine as `grain` cells of a
    ring of `ring` cells. Asked with an item it says a place, right or wrong.

    Its random numbers are its own (`seed`), so that a world lived with it is the same world
    as without.

    `pace` is for a brain that is never told a place and chooses one itself (`tape`): a
    thing never seen before is put `pace` cells further along the ring than the last new
    thing, like writing on a tape; a thing the index knows is where the index says."""

    def __init__(self, batch: int, cells: int, d: int, ring: int, grain: int = 16,
                 wave: float = 4.0, seed: int = 77, pace: int = 8):
        g = torch.Generator().manual_seed(seed)
        self.wires = torch.randn(batch, cells, d, generator=g)
        self.phase = torch.rand(batch, cells, generator=g) * 2 * math.pi
        self.wave, self.d = wave, d
        self.pace = pace / ring
        self.head = torch.zeros(batch)              # where the last new thing was put
        self.bits = max(1, math.ceil(math.log2(max(2, ring / grain))))
        self.w = torch.zeros(batch, self.bits, cells)

    def code(self, item: torch.Tensor) -> torch.Tensor:
        drive = torch.einsum("bmd,bd->bm", self.wires, item) / math.sqrt(self.d)
        return torch.cos(self.wave * drive + self.phase)

    def learn(self, item: torch.Tensor, place: torch.Tensor) -> None:
        """`place` (B,) is a share of the way round the ring."""
        q = (place * 2 ** self.bits).long().clamp(max=2 ** self.bits - 1)
        gray = q ^ (q >> 1)
        want = ((gray.unsqueeze(1) >> torch.arange(self.bits)) & 1).float() * 2 - 1
        c = self.code(item)
        err = want - torch.einsum("bom,bm->bo", self.w, c)
        self.w = self.w + err.unsqueeze(-1) * (c / (c * c).sum(-1, keepdim=True)).unsqueeze(1)

    def tape(self, item: torch.Tensor) -> torch.Tensor:
        """The place the brain itself gives to what it sees (and the index is taught it):
        where the index says if it is sure of every bit, else a little further along."""
        out = torch.einsum("bom,bm->bo", self.w, self.code(item))
        sure = (out.abs() > 0.5).all(dim=1)
        self.head = torch.where(sure, self.head, (self.head + self.pace) % 1.0)
        here = torch.where(sure, self.say(item), self.head)
        self.learn(item, here)
        return here

    def say(self, item: torch.Tensor) -> torch.Tensor:
        on = (torch.einsum("bom,bm->bo", self.w, self.code(item)) > 0).long()
        q = (on << torch.arange(self.bits)).sum(1)
        shift = 1
        while shift < self.bits:                    # out of the Gray code
            q = q ^ (q >> shift)
            shift *= 2
        return (q.float() + 0.5) / 2 ** self.bits


def index_right(ring: int, cells: int, items: int, laps: int = 2, lives: int = 8,
                within: int = 12, seed: int = 0) -> float:
    """The share of `items` whose place the index says within `within` cells, after it has
    been shown every item at its place `laps` times (loops of 8 items share a place)."""
    g = torch.Generator().manual_seed(seed)
    task = Task(d_in=12, d_out=12)
    loops = max(1, items // 8)
    idx = torch.rand(lives, 2 ** task.d_in, generator=g).argsort(dim=1)[:, :loops * 8]
    what = ((idx.unsqueeze(-1) >> torch.arange(task.d_in)) & 1).float() * 2 - 1     # (lives, items, d)
    where = torch.rand(lives, loops, generator=g).repeat_interleave(8, dim=1)
    index = Index(lives, cells, task.d_in, ring)
    for _ in range(laps):
        for k in torch.randperm(loops * 8, generator=g).tolist():
            index.learn(what[:, k], where[:, k])
    off = torch.stack([(index.say(what[:, k]) - where[:, k]).abs() for k in range(loops * 8)], dim=1)
    off = torch.minimum(off, 1 - off) * ring
    return float((off <= within).float().mean())


RULES = {
    "hebb": {"hebb": hebb},
    "delta": {f"rate {r}": (lambda c, a, r=r: delta(c, a, r)) for r in (1.0, 0.5, 0.25)},
    "lstsq": {"lstsq": lstsq},
}


def direct(cues, answers, gen):
    return hebb(cues, answers)


def expanded(rule, m: int, f: float):
    return lambda cues, answers, gen: rule(codes(cues, m, f, gen), answers)


def yardstick(task: Task = Task(), sizes=SIZES) -> dict:
    """K90 of every hand memory at every size, each at its best setting:
    {name: {m: (k90, sparseness, variant)}}."""
    ks = k_grid(16, task)
    out: dict = {"direct": {0: (k90(ks, curve(direct, task, ks)), None, None)}}
    for name, variants in RULES.items():
        out[name] = {}
        for m in sizes:
            ks = k_grid(m, task)
            out[name][m] = max((k90(ks, curve(expanded(rule, m, f), task, ks)), f, v)
                               for f in SPARSE for v, rule in variants.items())
    return out


def yardstick_hop(task: Task, sizes=SIZES) -> dict:
    """K90 of one hand-made hop at every size: {"delta" | "lstsq": {m: (k90, f, rate)}}."""
    from tissue import chain
    out: dict = {"delta": {}, "lstsq": {}}
    for m in sizes:
        ks = chain.k_grid(m, task)
        for name, rates in (("delta", (1.0, 0.5, 0.25)), ("lstsq", (None,))):
            best = (0.0, None, None)
            for f in SPARSE:
                for rate in rates:
                    done, accs = chain.curve(
                        lambda items, g, f=f, rate=rate: hop(items, m, f, g, rate or 1.0, rate is None),
                        task, ks, stop_below=0.9)
                    best = max(best, (k90(done, [a[1] for a in accs]), f, rate))
            out[name][m] = best
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hop", action="store_true", help="the yardstick for exam 1 (one hop)")
    p.add_argument("--index", action="store_true",
                   help="a hand-made memory of what lies where: how many items per cell")
    p.add_argument("--think", action="store_true",
                   help="the ceiling for a thought that holds with a few wires per cell")
    p.add_argument("--cells", type=int, nargs="*", default=[512, 2048])
    p.add_argument("--wires", type=int, nargs="*", default=[32, 64])
    p.add_argument("--patch", type=int, default=0,
                   help="with --think: a brain with places, a loop spoken for by this many cells")
    p.add_argument("--reach", type=int, default=32,
                   help="with --think --patch: wires come from within this many places")
    p.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.index:
        print("a memory of what lies where, by hand: the most items whose place it says within"
              " 12 cells, nine times in ten")
        print(f"{'ring':>6} {'bits':>5} | " + " | ".join(f"{x:>4} x ring" for x in (1, 2, 4, 8)) + "   (cells of the index)")
        for n in a.cells:
            row = []
            for x in (1, 2, 4, 8):
                best, items = 0, 8
                while items <= 4096 and index_right(n, x * n, items) >= 0.9:
                    best, items = items, 8 * max(items // 8 + 1, round(items * 1.5 / 8))
                row.append(best)
            bits = Index(1, 8, 12, n).bits
            print(f"{n:>6} {bits:>5} | " + " | ".join(f"{b:>11}" for b in row), flush=True)
        return
    if a.think and a.patch:
        print(f"items held in thought by hand in a brain with places: wires from within {a.reach}"
              f" places, a loop spoken for by a patch of {a.patch} cells")
        print(f"{'cells':>6} {'wires':>6} | {'held':>6} | per cell")
        for k in a.wires:
            for n in a.cells:
                h = held_by_hand(n, k, reach=a.reach, patch=a.patch)
                print(f"{n:>6} {k:>6} | {h:>6} | {h / n:.2f}", flush=True)
        return
    if a.think:
        print("items held in thought by hand (64 ticks alone), by the share of cells that speak")
        print(f"{'cells':>6} {'wires':>6} | " + " | ".join(
            f"{'half, signed' if signed else f'1/{round(1 / f)}':>12}" for f, signed in THINK))
        for k in a.wires:
            for n in a.cells:
                print(f"{n:>6} {k:>6} | " + " | ".join(
                    f"{held_by_hand(n, k, f, signed):>12}" for f, signed in THINK), flush=True)
        return
    if a.hop:
        y = yardstick_hop(Task(d_in=12, d_out=12), a.sizes)
        print(f"{'M':>5} | {'delta, one pass':>26} | {'least squares':>20}")
        for m in a.sizes:
            d, e = y["delta"][m], y["lstsq"][m]
            print(f"{m:>5} | {d[0]:7.1f} f={d[1]} rate {d[2]} | {e[0]:7.1f} f={e[1]}", flush=True)
        return
    y = yardstick(sizes=a.sizes)
    print(f"direct (no interneurons): K90 = {y['direct'][0][0]:.1f}")
    print(f"{'M':>5} | " + " | ".join(f"{r:>24}" for r in RULES))
    for m in a.sizes:
        print(f"{m:>5} | " + " | ".join(
            f"{y[r][m][0]:7.1f} f={y[r][m][1]:<4} {y[r][m][2]:<9}" for r in RULES))


if __name__ == "__main__":
    main()
