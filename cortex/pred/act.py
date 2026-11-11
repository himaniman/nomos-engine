"""The sheet that learned to predict its senses, given a body: does what it believes get it food?

    python -m cortex.pred.act --calibrate
    python -m cortex.pred.act --arms hand,blank,belief-s0,belief-s1,belief-s2 --worlds 32

Only the atlas's idea of what is SOLID changes between the arms; the wave, the frontier, the appetite
and every other hand-painted channel stay. HAND paints solids from the rays. BLANK knows a wall only
by bumping into it. BELIEF asks the sheet: the chance a ray is stopped within one cell, which nobody
ever showed it a map to learn (`NorthBrain.believed_solid`). Touch and doors that gave stay in all
arms: those are facts of the body. Worlds 1000.. are the ones the laws were examined on; the sheet
was trained on 2000-2059, picked on 2060-2075, and the test worlds stay closed.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from cortex import atlas as A
from cortex import atlas_lineage as L
from cortex.pred.bench import Data, GEOM
from cortex.pred.log import reduce_senses
from cortex.pred.model import build
from cortex.senses import N_TOUCH, TOUCH_CH
from cortex.world import FOOD_ENERGY, NIGHTCAP, World

OUT = Path("out/cortex/pred/act")
RUNS = Path("out/cortex/pred/runs")
THOUGHTS_PER_FRAME = 2                # as the brain was trained: 2 thoughts, 6 ticks a frame
SIDE = 32


def load_brain(seed: int, side: int = SIDE):
    b = build("northr", seed, side).eval()
    b.load_state_dict(torch.load(RUNS / f"northr-grow16_32-s{seed}" / "brain.pt", weights_only=True))
    return b


def parse_arm(arm: str) -> tuple[int, int]:
    """`beliefV-s1@64` = brain seed 1 on a sheet of 64: the same weights at any size."""
    body, _, side = arm.partition("@")
    return int(body.split("-s")[1]), int(side) if side else SIDE


class Sheet:
    """A NorthBrain frame by frame. The brain's own offset is all it knows of where it is."""

    def __init__(self, brain):
        self.b = brain
        self.s = torch.zeros(1, brain.ch, brain.side, brain.side)
        self.ay, self.ax = torch.zeros(1), torch.zeros(1)
        self.head = None
        self.last_act = torch.zeros(1, 3)

    @torch.no_grad()
    def frame(self, sens: np.ndarray, act: np.ndarray) -> None:
        """`act` = what the body did since the last frame. It moves the sheet, then the new senses
        are written. In training a frame heard the action that followed it; alive, that action has
        not been chosen yet, so it hears the one before."""
        b = self.b
        a = torch.tensor(act, dtype=torch.float32)[None]
        if self.head is not None:
            self.s, self.ay, self.ax, _ = b.jump(self.s, self.ay, self.ax, self.head, a)
        z = (torch.tensor(sens, dtype=torch.float32)[None] - b.norm_mean) / b.norm_std
        self.s, self.head = b.write(self.s, z, a, torch.ones(1, 1))

    @torch.no_grad()
    def belief(self, ry: np.ndarray, rx: np.ndarray) -> np.ndarray:
        """Would a ray stop at these offsets from the body?"""
        p = self.b.believed_solid(self.s, torch.tensor(ry.ravel(), dtype=torch.float32)[None],
                                  torch.tensor(rx.ravel(), dtype=torch.float32)[None], self.ay, self.ax)
        return p[0].numpy().reshape(ry.shape)


class SheetAtlas(A.Atlas):
    """The atlas, its idea of SOLID replaced. Touch and doors stay facts of the body."""

    def __init__(self, cfg, wcfg, rule, sheet: Sheet | None, threshold: float, blank: bool = False,
                 touch_memory: bool = False, veto: bool = False):
        super().__init__(cfg, wcfg, rule)
        self.sheet, self.threshold, self.blank = sheet, threshold, blank
        # touch_memory: what the body has bumped into stays known, as BLANK has it for free
        # veto: no wall where a ray has been seen to pass, as a body would not doubt its eyes
        self.touch_memory, self.veto = touch_memory, veto
        self.touched: set = set()     # world cells felt solid
        self.opened: set = set()      # world cells a press has opened
        self.since = np.zeros(3)
        self.count = 0
        ys, xs = np.mgrid[-self.R:self.R + 1, -self.R:self.R + 1]
        self.dy, self.dx = ys.astype(np.float32), xs.astype(np.float32)

    def gave(self, dy: int, dx: int, still: float) -> None:
        super().gave(dy, dx, still)
        if still == 0.0:
            self.opened.add((self.cell[0] + dy, self.cell[1] + dx))

    def _paint(self, vec, y, x, heading):
        before = self.state[A.SOLID].copy()   # scrolled already
        super()._paint(vec, y, x, heading)    # all the rest as ever
        st, R, S = self.state, self.R, self.side
        cy, cx = self.cell
        oy, ox = y - cy, x - cx
        if self.blank:
            solid = before
        elif self.thoughts % THOUGHTS_PER_FRAME == 0:
            n = self.wcfg.n_rays
            act = self.since / max(self.count, 1)
            self.sheet.frame(reduce_senses(vec, n), act)
            self.since, self.count = np.zeros(3), 0
            solid = (self.sheet.belief(self.dy - oy, self.dx - ox) > self.threshold).astype(np.float64)
            if self.veto:
                solid[st[A.FREE] > 0.5] = 0.0
            if self.touch_memory:
                for wy, wx in self.touched:
                    i, j = wy - cy + R, wx - cx + R
                    if 0 <= i < S and 0 <= j < S:
                        solid[i, j] = max(solid[i, j], 0.8)
        else:
            solid = before
        st[A.SOLID] = solid
        st[A.SOLID, R, R] = 0.0
        tlo, _ = self.layout["slices"]["touch"]
        for k in range(N_TOUCH):
            if vec[tlo + k * TOUCH_CH] > 0.0:
                a = heading + k * math.pi / 2
                ty = int(round(oy + math.sin(a))) + R
                tx = int(round(ox + math.cos(a))) + R
                if 0 <= ty < S and 0 <= tx < S and (ty, tx) != (R, R):
                    st[A.SOLID, ty, tx] = max(st[A.SOLID, ty, tx], 0.8)
                    if self.touch_memory:
                        self.touched.add((cy + ty - R, cx + tx - R))
        self.touched -= self.opened
        for wy, wx in self.opened:
            i, j = wy - cy + R, wx - cx + R
            if 0 <= i < S and 0 <= j < S:
                st[A.SOLID, i, j] = 0.0

    def act(self, vec, y, x, heading):
        cmd = super().act(vec, y, x, heading)
        self.since = self.since + np.array(cmd, dtype=np.float64)
        self.count += 1
        return cmd


class Forgetful(A.Atlas):
    """The hand atlas with parts of its memory wiped every thought: what is memory worth, and of what?"""

    KINDS = {"solid": (A.SOLID,), "free": (A.FREE,), "thing": (A.THING, A.HUE_C, A.HUE_S, A.BRIGHT),
             "map": (A.SOLID, A.FREE, A.THING, A.HUE_C, A.HUE_S, A.BRIGHT),
             "ghost": (A.GHOST,), "all": (A.SOLID, A.FREE, A.THING, A.HUE_C, A.HUE_S, A.BRIGHT, A.GHOST)}

    def __init__(self, cfg, wcfg, rule, kind: str):
        super().__init__(cfg, wcfg, rule)
        self.wipe = self.KINDS[kind]

    def _paint(self, vec, y, x, heading):
        for ch in self.wipe:
            self.state[ch] = 0.0
        super()._paint(vec, y, x, heading)


_brains: dict = {}


def life(args) -> dict:
    arm, seed, ticks, world, threshold = args
    torch.set_num_threads(1)
    prog = L.Program({})
    wcfg = L.WorldConfig(**L.WORLDS[world])
    w = World(seed=int(seed), cfg=wcfg)
    mind = A.AtlasMind(prog.config(), w.cfg, None)
    if arm.startswith("forget-"):
        mind.atlas = Forgetful(prog.config(), w.cfg, None, arm.split("-")[1])
    elif arm != "hand":
        sheet = None
        if arm.startswith(("belief", "strict")):
            key = parse_arm(arm)
            if key not in _brains:
                _brains[key] = load_brain(*key)
            sheet = Sheet(_brains[key])
        mind.atlas = SheetAtlas(prog.config(), w.cfg, None, sheet, threshold, blank=arm == "blank",
                                touch_memory=arm.startswith(("beliefT", "beliefV", "strictV")),
                                veto=arm.startswith(("beliefV", "strictV")))
    mind.attach(w)
    while w.agent.alive and w.tick < ticks:
        mind.step(w)
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"seed": int(seed), "meals": int(sum(w.agent.ate)), "ticks": int(w.tick), "immortal": int(w.agent.alive),
            "gained": gained, "night": int(w.agent.ate[NIGHTCAP]), "doors": int(w.deeds["door"])}


@torch.no_grad()
def calibrate(frames: int = 300, burn: int = 48, side: int = SIDE) -> dict:
    """The belief's cut into wall / not wall, chosen on TRAINING lives against the true map, the same
    way for every brain: where telling walls from the rest is best (Youden's J)."""
    train = Data("train")
    d = np.load("out/cortex/pred/data/train.npz")
    ego = d["ego"]
    R = ego.shape[-1] // 2
    f, l = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
    rng = np.random.default_rng(0)
    path = OUT / "calibration.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    at = "" if side == SIDE else f"@{side}"
    for seed in (0, 1, 2):
        b = load_brain(seed, side)
        scores, labels = [], []
        for _ in range(frames // 24 + 1):
            idx = rng.integers(len(train.lens), size=24)
            o = np.array([train.starts[i] + rng.integers(train.lens[i] - burn - 2) for i in idx])
            s = torch.tensor(np.array([train.sens[x:x + burn] for x in o]))
            a = torch.tensor(np.array([train.act[x:x + burn] for x in o]))
            keep: list = []
            b.run(s, a, torch.ones(len(o), burn), keep=keep)
            k = keep[-1]
            sn, cs = torch.sin(k["head"]), torch.cos(k["head"])
            ft = torch.tensor(f.ravel(), dtype=torch.float32)[None]
            lt = torch.tensor(l.ravel(), dtype=torch.float32)[None]
            ry, rx = sn[:, None] * ft + cs[:, None] * lt, cs[:, None] * ft - sn[:, None] * lt
            scores.append(b.believed_solid(k["s"], ry, rx, k["ay"], k["ax"]).numpy().ravel())
            labels.append(ego[o + burn - 1].reshape(len(o), -1).ravel())
        sc, la = np.concatenate(scores), np.concatenate(labels) > 0
        cuts = np.quantile(sc, np.linspace(0.05, 0.99, 95))
        j = [(sc[la] > c).mean() - (sc[~la] > c).mean() for c in cuts]
        best = int(np.argmax(j))
        out[f"belief-s{seed}{at}"] = float(cuts[best])
        print(f"s{seed}{at}: cut {cuts[best]:.4f}, walls above it {(sc[la] > cuts[best]).mean():.2f}, "
              f"the rest above it {(sc[~la] > cuts[best]).mean():.2f}", flush=True)
        # a stricter cut: a false wall closes a road, a missed wall only costs a bump
        strict = float(np.quantile(sc[~la], 0.90))
        out[f"strict-s{seed}{at}"] = strict
        print(f"    strict cut {strict:.4f}: walls above it {(sc[la] > strict).mean():.2f}, the rest 0.10", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def probes(arms: list[str], cuts: dict) -> None:
    """The ladder: arenas where what stands in the way decides (`corner`, `maze`)."""
    from cortex import probes as P
    torch.set_num_threads(4)
    prog = L.Program({})

    def driver_for(arm: str):
        made: dict = {}

        def drive(vec, w):
            if "atlas" not in made:
                if arm == "hand":
                    made["atlas"] = A.Atlas(prog.config(), w.cfg, None)
                else:
                    sheet = Sheet(load_brain(int(arm.split("-s")[1]))) if "-s" in arm else None
                    made["atlas"] = SheetAtlas(prog.config(), w.cfg, None, sheet, cuts.get(arm, 0.0), blank=arm == "blank")
            a = w.agent
            return made["atlas"].act(vec, a.y, a.x, a.heading)
        return drive
    names = None
    for arm in arms:
        p = P._suite(lambda arm=arm: driver_for(arm), P.TICKS)["profile"]
        names = names or list(p)
        if arm == arms[0]:
            print(f"{'':<12}" + "".join(f"{k:>10}" for k in names), flush=True)
        print(f"{arm:<12}" + "".join(f"{p[k]:>10.2f}" for k in names), flush=True)


def table(res: dict, arms: list[str]) -> None:
    n = min(len(res[a]) for a in arms)
    meals = {a: np.array([r["meals"] for r in res[a][:n]], float) for a in arms}
    print(f"{n} worlds\n{'arm':<12}{'meals':>7}{'se':>6}{'immortal':>10}{'doors':>7}   vs blank        vs hand")
    for a in arms:
        rows = res[a][:n]

        def vs(b):
            if b not in meals or a == b:
                return ""
            d = meals[a] - meals[b]
            return f"{d.mean():+6.1f} +- {d.std(ddof=1) / math.sqrt(n):.1f}"
        print(f"{a:<12}{meals[a].mean():>7.1f}{meals[a].std(ddof=1) / math.sqrt(n):>6.1f}"
              f"{sum(r['immortal'] for r in rows):>6}/{n:<3}{np.mean([r['doors'] for r in rows]):>7.1f}   "
              f"{vs('blank'):<15} {vs('hand')}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--side", type=int, default=SIDE, help="with --calibrate: the sheet's size")
    p.add_argument("--probes", action="store_true")
    p.add_argument("--arms", default="hand,blank,belief-s0,belief-s1,belief-s2")
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--first", type=int, default=1000)
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--world", default="default")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--tag", default="", help="keep these results apart, e.g. a confirmation on fresh worlds")
    args = p.parse_args(argv)
    if args.calibrate:
        calibrate(side=args.side)
        return 0
    cuts = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    if args.probes:
        probes(arms, cuts)
        return 0
    seeds = range(args.first, args.first + args.worlds)
    path = OUT / f"results-{args.world}-{args.ticks}{'-' + args.tag if args.tag else ''}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    with Pool(args.jobs) as pool:
        for arm in arms:
            have = {r["seed"] for r in res.get(arm, [])}
            cut = cuts.get(arm.replace("beliefT", "belief").replace("beliefV", "belief").replace("strictV", "strict"), 0.0)
            todo = [(arm, s, args.ticks, args.world, cut) for s in seeds if s not in have]
            if not todo:
                continue
            for r in pool.imap(life, todo, chunksize=1):
                res.setdefault(arm, []).append(r)
                path.write_text(json.dumps(res), encoding="utf-8")   # a stopped run goes on from here
            print(f"{arm}: {np.mean([r['meals'] for r in res[arm]]):.1f} meals", flush=True)
    res = {a: sorted(v, key=lambda r: r["seed"]) for a, v in res.items()}
    table(res, arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
