"""The observer: fixed exams a brain sits at the end of its life.

Not a curriculum. The world stays open and nothing here gates anything — a probe builds
its own small arena from its own fixed seed, drops a COPY of the brain into it, watches
for a while and throws the copy away. The live world is never touched
(docs/cortex-design.md, "the ladder is an OBSERVER").

Why this exists at all: at lambda = 1 a generation returns ONE life, and one life scored
by one number is noise. The first A/B of the signal layers went to the control arm on a
difference of one meal against three, which settled nothing. A probe suite turns a life
into a SKILL PROFILE — can it reach food, can it round a corner, will it open a door,
will it eat the purple one — and a profile is something you can actually reason about,
whether you are an evolutionary loop or an LLM.

Three rules the probes obey:

  * a DEEP COPY of the brain, taken after death, including the grown body, the cell
    states and everything learned within the life. The exam is of the developed,
    trained mind, not of the genome;
  * PLASTICITY FROZEN, or the exam becomes a training episode and the scores stop being
    comparable between lives. `learning=True` makes that a probe of its own — how fast
    does it adapt — rather than a contamination of all the others;
  * each probe gets a FRESH copy, so the order they run in cannot matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cortex import brain as B
from cortex.senses import ScentField, sense
from cortex.world import BERRY, NIGHTCAP, Config, FoodSpot, World
from cortex import tiles as T

ARENA = 33                      # a probe arena is small: one situation, no wandering
TICKS = 900


@dataclass
class Result:
    name: str
    score: float                # 0..1, comparable across probes
    detail: dict = field(default_factory=dict)
    path: list = field(default_factory=list)     # the agent's track, for the strip
    world: World | None = None


def _arena(seed: int = 0) -> World:
    """An empty walled room from the world's own vocabulary — same materials, same
    physics, same senses. A probe must not be a different game."""
    # spawn_room is the live world's guard against starting in a coffin; an arena is
    # smaller than that guard's floor, and we place the agent by hand anyway
    w = World(seed=seed, cfg=Config(size=ARENA, poison_weight=0.0, spawn_room=16))
    for y in range(ARENA):
        for x in range(ARENA):
            w.tiles.set(y, x, T.STONE if (y in (0, ARENA - 1) or x in (0, ARENA - 1))
                        else T.GROUND)
    for cy in range(ARENA // w.cfg.chunk + 1):
        for cx in range(ARENA // w.cfg.chunk + 1):
            w.food[(cy, cx)] = []
    w.agent.y, w.agent.x = ARENA // 2, 6.0
    w.agent.heading = 0.0
    w.agent.energy = w.cfg.start_energy
    w.damage.clear()
    w.deeds = {k: 0 for k in w.deeds}
    w.distance = 0.0
    return w


def _put(w: World, y: float, x: float, kind: int = BERRY) -> FoodSpot:
    spot = FoodSpot(y=y, x=x, kind=kind, renewable=False)
    w.food.setdefault((int(y) // w.cfg.chunk, int(x) // w.cfg.chunk), []).append(spot)
    return spot


def _run(w: World, driver, ticks: int, think_every: int = 3) -> list:
    """Drive the arena with anything that turns a sensor vector into a command — a
    copied brain, or a baseline policy. Being able to sit the REFLEX at the same exam
    is the whole point: a skill profile with nothing to compare it to is decoration."""
    field = ScentField(w.cfg.smell_radius)
    path, command = [], (0.0, 0.0, 0.0)
    while w.agent.alive and w.tick < ticks:
        if w.tick % think_every == 0:
            command = driver(sense(w, field), w)
        w.step(*command)
        if w.tick % 6 == 0:
            path.append((w.agent.y, w.agent.x))
    return path


class BrainDriver:
    """A copied brain at an exam. Frozen unless this is the learning probe."""

    def __init__(self, phenotype: B.Phenotype):
        self.p = phenotype
        self.last = None

    def __call__(self, vec, w):
        e = w.agent.energy
        reward = 0.0 if self.last is None else (e - self.last) / 12.0
        self.last = e
        cmd = self.p.act(vec, hormone=np.array(
            [2.0 * e / w.cfg.max_energy - 1.0, min(1.0, w.agent.poison / 30.0),
             max(-1.0, min(1.0, reward))]))
        if self.p.cfg.plastic or self.p.cfg.rule_plastic:
            self.p.learn(reward)
        return cmd


class PolicyDriver:
    """A baseline policy at the same exam, on the same terms — sensor vector in,
    command out, no access to the world."""

    def __init__(self, policy, layout, seed: int = 0):
        self.policy, self.rng = policy, np.random.default_rng(seed)
        self.state = {"layout": layout}

    def __call__(self, vec, w):
        return self.policy(vec, w, self.rng, self.state)


def _reach(w, spot, path, ticks) -> tuple[float, dict]:
    """Scored on CLOSING THE DISTANCE, not only on the meal. A brain that walks to its
    food and fails to press is doing most of the job, and a suite that cannot see the
    difference between that and standing still teaches us nothing."""
    d0 = math.hypot(spot.y - ARENA // 2, spot.x - 6.0)
    best = min((math.hypot(spot.y - y, spot.x - x) for y, x in path), default=d0)
    closed = max(0.0, min(1.0, (d0 - best) / max(1e-9, d0 - w.cfg.eat_reach)))
    ate = 0.0 if spot.ready else 1.0
    return 0.35 * closed + 0.65 * ate, {"closed": round(closed, 2), "ate": int(ate),
                                        "nearest": round(best, 1)}


# --- the probes ---------------------------------------------------------------------

def _maze(w) -> None:
    """A few short walls, no dead ends: enough terrain that a straight line will not do
    and the scent has to be followed round something."""
    for y in range(6, 20):
        w.tiles.set(y, 13, T.STONE)
    for x in range(13, 25):
        w.tiles.set(20, x, T.STONE)
    for y in range(22, ARENA - 2):
        w.tiles.set(y, 20, T.STONE)


def build(name: str) -> tuple[World, dict]:
    """Build a probe's arena WITHOUT running it — so the app can drop you (or a brain)
    into the same situation live, with the full HUD, instead of only ever seeing the
    aftermath as a track on a PNG."""
    w = _arena()
    spots = {}
    if name in ("corner", "door", "rubble"):
        for y in range(1, ARENA - 1):
            w.tiles.set(y, 16, T.STONE)
    if name == "corner":
        w.tiles.set(12, 16, T.GROUND)
        spots["food"] = _put(w, ARENA // 2, 20.0)
    elif name == "door":
        w.tiles.set(ARENA // 2, 16, T.DOOR)
        spots["food"] = _put(w, ARENA // 2, 22.0)
    elif name == "rubble":
        w.tiles.set(ARENA // 2, 16, T.RUBBLE)
        spots["food"] = _put(w, ARENA // 2, 22.0)
    elif name == "poison":
        spots["poison"] = _put(w, ARENA // 2, 17.0, NIGHTCAP)
        spots["food"] = _put(w, ARENA // 2 - 7, 23.0, BERRY)
    elif name == "open":
        spots["food"] = _put(w, ARENA // 2, 22.0)
    elif name == "near":                      # four cells dead ahead: the very floor
        spots["food"] = _put(w, ARENA // 2, 10.0)
    elif name == "side":                      # off the line: it has to turn
        spots["food"] = _put(w, ARENA // 2 - 9, 15.0)
    elif name == "behind":                    # out of sight entirely: smell only
        spots["food"] = _put(w, ARENA // 2 + 2, 3.0)
    elif name == "maze":
        _maze(w)
        spots["food"] = _put(w, 26.0, 26.0)
    elif name != "patience":
        raise KeyError(f"no probe called {name}: try {', '.join(NAMES)}")
    return w, spots


NAMES = ("near", "open", "side", "behind", "corner", "maze", "patience",
         "door", "rubble", "poison")


def probe_open(driver, ticks=TICKS):
    """Can it get to food it can see and smell, over open ground? The floor of the
    suite: everything else assumes this."""
    w, sp = build("open")
    spot = sp["food"]
    path = _run(w, driver, ticks)
    score, detail = _reach(w, spot, path, ticks)
    return Result("open", score, detail, path, w)


def probe_corner(driver, ticks=TICKS):
    """Food behind a wall with one gap: the straight line is blocked, the geodesic
    scent points the way round. Tests that it follows the WALK, not the sightline."""
    w, sp = build("corner")
    spot = sp["food"]
    path = _run(w, driver, ticks)
    score, detail = _reach(w, spot, path, ticks)
    detail["through_gap"] = int(any(x > 16.5 for _, x in path))
    return Result("corner", score, detail, path, w)


def probe_door(driver, ticks=TICKS):
    """Food behind a closed door. Only INTERACT opens it, so this asks whether the verb
    is ever used on something that is not a meal."""
    w, sp = build("door")
    spot = sp["food"]
    path = _run(w, driver, ticks)
    score, detail = _reach(w, spot, path, ticks)
    opened = w.tiles.material(ARENA // 2, 16) != T.DOOR
    detail["opened"] = int(opened)
    return Result("door", 0.5 * score + 0.5 * opened, detail, path, w)


def probe_rubble(driver, ticks=TICKS):
    """The same, behind breakable rubble: several presses in one place, which needs
    persistence rather than a single lucky button."""
    w, sp = build("rubble")
    spot = sp["food"]
    path = _run(w, driver, ticks)
    score, detail = _reach(w, spot, path, ticks)
    broke = w.tiles.material(ARENA // 2, 16) != T.RUBBLE
    detail["broke"] = int(broke)
    detail["chips"] = int(w.deeds["break"])
    return Result("rubble", 0.5 * score + 0.5 * broke, detail, path, w)


def probe_poison(driver, ticks=TICKS):
    """A nightcap dead ahead and a berry further off to the side. They smell IDENTICAL
    and differ only in hue, so following your nose takes you to the poison — the whole
    point being that only LOOKING can tell them apart.

    The first version put them equidistant, and everything scored exactly 0.50 by
    eating neither: two equal scent sources average to a bearing straight between them
    and the agent oscillates, which is the forage limit cycle reproduced in a test tube.
    True, but useless as a measurement, so the temptation is now asymmetric."""
    w, sp = build("poison")
    bad, good = sp["poison"], sp["food"]
    path = _run(w, driver, ticks)
    ate_good, ate_bad = int(not good.ready), int(not bad.ready)
    score = (1.0 if (ate_good and not ate_bad) else
             0.3 if (ate_good and ate_bad) else
             0.6 if not (ate_good or ate_bad) else 0.0)
    return Result("poison", score, {"good": ate_good, "poison": ate_bad}, path, w)


def probe_patience(driver, ticks=TICKS):
    """No food anywhere. Standing still costs half of walking, so the only sane play is
    to stop — and a brain that sprints into a wall for nine hundred ticks is telling us
    it has no idea what its energy is for. Scored on energy KEPT."""
    w, _ = build("patience")
    path = _run(w, driver, ticks)
    kept = w.agent.energy / w.cfg.start_energy
    floor = 1.0 - ticks * w.cfg.move_cost / w.cfg.start_energy
    score = max(0.0, min(1.0, (kept - floor) / max(1e-9, 1.0 - floor)))
    return Result("patience", score,
                  {"energy": round(w.agent.energy, 1), "walked": round(w.distance)},
                  path, w)


def _forage(name, ticks):
    """The whole of a foraging probe: build it, run it, score reaching and eating. The
    only difference between them is WHERE the food is and what is in the way."""
    w, sp = build(name)
    spot = sp["food"]
    path = _run(w, _forage.driver, ticks)
    score, detail = _reach(w, spot, path, ticks)
    return Result(name, score, detail, path, w)


def _forage_probe(name):
    def probe(driver, ticks=TICKS):
        _forage.driver = driver
        return _forage(name, ticks)
    probe.__name__ = f"probe_{name}"
    probe.__doc__ = f"Reach and eat the food in the `{name}` arena."
    return probe


probe_near = _forage_probe("near")
probe_side = _forage_probe("side")
probe_behind = _forage_probe("behind")
probe_maze = _forage_probe("maze")

# THE DEFAULT SUITE IS A FORAGING LADDER, and deliberately nothing else. An earlier
# version led with doors, breakables and poison; the strip showed every brain circling
# in place and scoring ~0.1 on five of six, so the whole suite carried one probe's worth
# of signal. A measurement has to have resolution WHERE ITS SUBJECT ACTUALLY IS. Eat and
# you have done well; the arenas only vary where the food is and what stands in the way,
# which is exactly the terrain-and-smell competence everything else will be built on.
BASIC = (probe_near, probe_open, probe_side, probe_behind, probe_corner, probe_maze,
         probe_patience)
# kept, and worth running once something can forage — but not while nothing can
ADVANCED = (probe_door, probe_rubble, probe_poison)
PROBES = BASIC


# --- running the suite ----------------------------------------------------------------

def snapshot(mind, learning: bool = False) -> B.Phenotype:
    """A deep copy of the brain as it stands: grown body, cell states, resource, and
    everything learned. Plasticity is OFF unless this is the learning probe, because an
    exam that teaches is not an exam."""
    cfg = B.Config(**vars(mind.phenotype.cfg))
    cfg.plastic = cfg.rule_plastic = bool(learning)
    cfg.develop = False                                   # a body does not grow mid-exam
    p = B.Phenotype(mind.genome, cfg, mind.layout)
    src = mind.phenotype
    p.mask, p.type_map = src.mask.copy(), src.type_map.copy()
    p.in_mask, p.out_mask = src.in_mask.copy(), src.out_mask.copy()
    p.organ_mod, p.resource = src.organ_mod.copy(), src.resource.copy()
    p._reindex()
    p.state = src.state.copy()
    p.dW = src.dW.copy()
    p._dW_rule = [d.copy() for d in src._dW_rule]
    if B._NUMBA:
        p._nb_dW = B._NbList(p._dW_rule)
    return p


def suite_for(which: str = "basic"):
    return {"basic": BASIC, "advanced": ADVANCED, "all": BASIC + ADVANCED}[which]


def run_suite(mind, ticks: int = TICKS, learning: bool = False,
              which: str = "basic") -> dict:
    """The whole exam for a BRAIN. Each probe gets its OWN fresh copy, so nothing one
    does can reach the next — the order they run in cannot matter."""
    return _suite(lambda: BrainDriver(snapshot(mind, learning)), ticks, which)


def run_suite_policy(policy, layout, ticks: int = TICKS, which: str = "basic") -> dict:
    """The same exam for a baseline. This is the calibration: a brain's profile means
    nothing until you know what twenty lines of reflex score on it."""
    return _suite(lambda: PolicyDriver(policy, layout), ticks, which)


def _suite(make_driver, ticks: int, which: str = "basic") -> dict:
    results = [fn(make_driver(), ticks) for fn in suite_for(which)]
    profile = {r.name: round(r.score, 3) for r in results}
    profile["mean"] = round(float(np.mean([r.score for r in results])), 3)
    return {"profile": profile, "results": results,
            "detail": {r.name: r.detail for r in results}}


def summarise(out: dict) -> str:
    p = out["profile"]
    body = "  ".join(f"{k} {v:.2f}" for k, v in p.items() if k != "mean")
    return f"skills  {body}   | mean {p['mean']:.2f}"


# --- watchable ------------------------------------------------------------------------

def render(out: dict, path: str = "out/cortex/probes.png", cell: int = 7):
    """The suite as one strip: each arena with the agent's TRACK drawn over it, its
    score and what it did. A probe that can only be read as a number is a probe we will
    not trust — half the value of an exam is seeing HOW it was failed."""
    import os
    import pygame
    from cortex.tiles import RGB
    from cortex.world import FOOD_RGB

    results = out["results"]
    pygame.init()
    pad, head = 8, 30
    side = ARENA * cell
    w = len(results) * (side + pad) + pad
    surf = pygame.Surface((w, side + head + pad * 2))
    surf.fill((12, 12, 16))
    font = pygame.font.SysFont("consolas", 12)
    big = pygame.font.SysFont("consolas", 14, bold=True)
    for i, r in enumerate(results):
        x0 = pad + i * (side + pad)
        mats = r.world.tiles.region(0, 0, ARENA, ARENA)
        img = pygame.surfarray.make_surface(np.transpose(RGB[mats], (1, 0, 2)))
        surf.blit(pygame.transform.scale(img, (side, side)), (x0, head))
        for spot in [s for lst in r.world.food.values() for s in lst]:
            col = tuple(int(q) for q in FOOD_RGB[spot.kind])
            c = (int(x0 + (spot.x + 0.5) * cell), int(head + (spot.y + 0.5) * cell))
            if spot.ready:
                pygame.draw.circle(surf, col, c, max(2, cell // 2))
            else:                                   # eaten: a hollow ring where it was
                pygame.draw.circle(surf, col, c, max(3, cell // 2), 1)
        for k in range(1, len(r.path)):             # the TRACK, fading in as it goes
            t = k / max(1, len(r.path) - 1)
            a = (int(60 + 190 * t), int(200 - 60 * t), int(230 - 120 * t))
            p0, p1 = r.path[k - 1], r.path[k]
            pygame.draw.line(surf, a,
                             (x0 + (p0[1] + 0.5) * cell, head + (p0[0] + 0.5) * cell),
                             (x0 + (p1[1] + 0.5) * cell, head + (p1[0] + 0.5) * cell), 2)
        if r.path:
            end = r.path[-1]
            pygame.draw.circle(surf, (255, 255, 255),
                               (int(x0 + (end[1] + 0.5) * cell),
                                int(head + (end[0] + 0.5) * cell)), 3)
        col = (120, 230, 150) if r.score > 0.66 else \
              (240, 210, 110) if r.score > 0.33 else (240, 110, 100)
        surf.blit(big.render(f"{r.name}  {r.score:.2f}", True, col), (x0, 4))
        surf.blit(font.render(" ".join(f"{k}{v}" for k, v in r.detail.items())[:34],
                              True, (130, 130, 145)), (x0, 18))
        pygame.draw.rect(surf, (40, 42, 52), pygame.Rect(x0, head, side, side), 1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pygame.image.save(surf, path)
    return path


# --- the command line -----------------------------------------------------------------

def main(argv=None) -> None:
    """    .\.venv\Scripts\python.exe -m cortex.probes --player reflex

    Sit a player at the exam, print the profile and write the strip. `--player brain`
    needs `--load-brain`; a brain is grown for `--grow` ticks in the live world first,
    because the exam is of a DEVELOPED mind and a bare seed is not one."""
    import argparse
    import json

    from cortex.baseline import POLICIES
    from cortex.senses import layout as _layout

    p = argparse.ArgumentParser(description="Run the probe suite and write the strip.")
    p.add_argument("--player", default="reflex",
                   choices=sorted(POLICIES) + ["brain"])
    p.add_argument("--load-brain", default="out/cortex/best_genome.json")
    p.add_argument("--grow", type=int, default=2500, help="ticks of life before the exam")
    p.add_argument("--world-seed", type=int, default=2)
    p.add_argument("--ticks", type=int, default=TICKS)
    p.add_argument("--out", default="out/cortex/probes.png")
    p.add_argument("--learning", action="store_true",
                   help="let the copy learn during the exam (a probe of adaptation)")
    args = p.parse_args(argv)

    if args.player == "brain":
        from cortex.agent import Mind, live
        genome = B.Genome.from_dict(json.load(open(args.load_brain, encoding="utf-8")))
        _, mind = live(genome, seed=args.world_seed, max_ticks=args.grow, think_every=3)
        print(f"grown: {mind.phenotype.cells()} cells, "
              f"{mind.phenotype.organs()[0]} sensory / {mind.phenotype.organs()[1]} effector")
        out = run_suite(mind, ticks=args.ticks, learning=args.learning)
    else:
        out = run_suite_policy(POLICIES[args.player], _layout(64), ticks=args.ticks)

    print(f"{args.player:<8} {summarise(out)}")
    for name, detail in out["detail"].items():
        print(f"   {name:<9} {detail}")
    print("strip ->", render(out, args.out))


if __name__ == "__main__":
    main()
