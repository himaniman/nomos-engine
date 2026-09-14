"""The LLM writes the growth program; a body develops from it, moves, and eats.

    .\\.venv\\Scripts\\python.exe -m cortex.lab --iterations 8

This is stage 6 of docs/cortex-design.md and the milestone that note states as
narrowly as it can be. The model is handed a SKILL PROFILE from the probe ladder, what
the life actually did, and a summary of the body that grew — and it writes back a
developmental program. Then the whole thing is grown again from a 24-cell seed and
sat at the same seven exams.

Three rules this loop is built on, all of them load-bearing:

**The model never sees a weight.** `docs/cortex-design.md` is blunt that handing a
model raw matrices is the one way to waste a good one. It writes `cortex/program.py`'s
named scalars — thresholds, urges, the pace of development, metabolism, chemistry,
axons, learning rates, the read-out — plus a few structural asks. Every candidate is
grown from the SAME random weight seed, so the only thing that varies between
generations is the program, and a difference in score is attributable to it.

**Everything is written down as a picture.** Each iteration leaves a HUD screenshot
(the world, the senses, the grown sheet) and a probe strip in the run directory, and a
contact sheet is rebuilt as they arrive. A run you can only read as numbers is a run
nobody looks at, and the project has been burnt by that before — the reactive limit
cycle in the forage world was spotted in a GIF, not in a table.

**rng is the control arm.** `--proposer rng` runs the identical loop with a random
program mutation in the model's chair. "Is an LLM actually better than rng at writing
growth programs" is a question this harness should be able to ANSWER, not assume.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

from cortex import brain as B
from cortex import probes
from cortex import program as PROG
from cortex.agent import Mind, fresh_genome
from cortex.world import Config as WorldConfig
from cortex.world import World
from nomos.llm import LLMError, make_client

# The reflex and the random walker are POLICIES, not brains — they do not develop, so
# nothing about growth or the read-out can move them. Both are deterministic; re-measure
# either in one command (`python -m cortex.probes --player reflex|random`) if you
# suspect drift, which is how the stale `random` row below was caught.
#
# Note what the random walker does to `maze`: it scores 1.00 there. A probe a blind
# stagger aces is not measuring the skill it is named for — the arena is small enough
# that wandering finds the food. `corner` is the honest terrain probe of the two.
BASELINES = {
    # re-measured 2026-09-14 after the probes started scoring the geodesic walk closed
    # (not the crow's flight) and the nose started reporting a bearing
    "reflex": {"near": 1.00, "open": 1.00, "side": 1.00, "behind": 1.00,
               "corner": 0.23, "maze": 0.22, "patience": 0.17, "mean": 0.66},
    "random": {"near": 0.35, "open": 0.16, "side": 0.33, "behind": 0.34,
               "corner": 0.23, "maze": 1.00, "patience": 0.00, "mean": 0.35},
}

SPEC = """\
You are designing the DEVELOPMENTAL PROGRAM of a brain that grows itself.

THE ORGANISM. One agent lives one long life in an open top-down world. It has three
commands: THRUST (0..1, forward), TURN (-1..1) and INTERACT (eat what is in reach,
else open the door / chip the breakable / shove the block in front). It dies of
starvation and nothing else. Its senses are properties, never labels — the world never
says "food": there are colour/distance rays for sight, four odorant channels for
smell, four touch directions, and interoception (energy, speed, heading, poison, age).

THE BRAIN is not a network you write. It is a SHEET of cells on a 64x64 lattice that
grows from a 24-cell SEED. Every cell:
  * holds 14 state channels and updates them from its own 3x3 neighbourhood, twenty or
    so times per thought (a signal crosses ONE CELL per sub-tick — this is why where
    things sit on the sheet matters);
  * runs the update rule of its TYPE (a few types, laid out as regions in the seed; a
    daughter inherits her mother's type);
  * emits, from that same little rule, seven MORPHOGENETIC requests: divide (and in
    which direction), die, become a sensory organ, become an effector, sprout an axon.

Three signal layers at three speeds: ELECTRIC (the 3x3 neighbourhood, every sub-tick),
CHEMICAL (a few channels that diffuse and decay slowly — the only carrier with a
spatial GRADIENT, which is what a cell reads its own POSITION from, and what a growing
axon follows), HORMONAL (a global vector from the body — hunger first — that reaches
every cell at once and swings every rule's gain).

DEVELOPMENT fires on a rare development tick, never per sub-tick. Each such tick the
tissue reads what its cells have been asking for and grants some of it, subject to:
  * a THRESHOLD — how loudly a cell must ask before anything happens;
  * an INNATE URGE added to what the cell asked for (this is what makes an embryo grow
    before it has learnt anything);
  * RESOURCE — supply enters at the body's BORDER in proportion to how full the agent
    is, seeps inward and decays. A division spends it. This is why a body has a SHAPE
    and not a disc: the deep interior is fed last;
  * a BUDGET — a ceiling on births per tick, and deaths as a fraction of it.
A cell that becomes a sensory organ takes the modality of whichever CHEMICAL is
strongest where it stands (`organ_map` says which chemical means which sense). That is
the French-flag model: concentration decides fate, so an "eye region" is something
that can FORM rather than something anyone places.

Life has two phases: an EMBRYO window where development is frequent and its cells are
CHEAP, then adulthood where structure changes rarely and cells cost full price. Either
way TISSUE COSTS ENERGY EVERY TICK, and that cost is the only thing arguing for a small
brain.

BIGGER IS NOT BETTER, and this is the failure mode to avoid. The lattice holds at most
4096 cells. A run before yours raised every growth lever it could find, generation after
generation, until the body hit that ceiling — and every score fell. At 4096 cells the
agent walked twenty-seven cells in a whole life: a brain too large to act, paying for
tissue that never earns anything back. A signal still crosses ONE CELL PER SUB-TICK, so
a wider body is a body whose senses are further from its effectors, not a smarter one.
If you want more capability, look at where tissue sits and what it costs before you look
at how much of it there is.

YOU WRITE THE PROGRAM, NOT THE WEIGHTS. The connection weights are random and are held
FIXED across every candidate, so whatever changes in the score is attributable to your
program and nothing else. You will never be shown a weight and should never ask for
one.

THE MEASUREMENT is a ladder of seven probes. Each drops a COPY of the brain, exactly
as its life left it, into a small arena with plasticity frozen, and scores 0..1:
  near     food four cells dead ahead
  open     food far away over open ground
  side     food off the line of travel — it has to TURN
  behind   food out of sight entirely: smell only
  corner   food behind a wall with one gap
  maze     food past a few short walls
  patience no food at all; standing still is half price, so the score is energy kept
The first four are only "reach the food"; they differ in where it is. The reflex
baseline aces those and stumbles on terrain, which is what makes it the bar.

WHAT TO REPLY WITH. Reason first, in prose, then ONE fenced json block:

```json
{
  "hypothesis": "one sentence: what you are changing and what you expect it to do",
  "program": {"grow_thresh": 0.18, "sense_bias": 0.30},
  "subticks": 24,
  "organ_map": [1, 7, 8],
  "seed_types": 3
}
```

Only "program" is required, and only the fields you want to CHANGE: your program is
applied ON TOP OF the best one so far, so everything you omit keeps the value that
program has, and you will be shown that program each turn. "subticks" (8..48) is how
many rounds the sheet relaxes
per thought, so it is how far a signal can travel before the command is read.
"organ_map" says which sensory modality each chemical region becomes: 1-6 are sight
ray channels, 7 odour, 8 touch, 9 interoception. "seed_types" (1..6) is how many cell
types the seed starts with.

Change a FEW things at a time and say why. A generation that moves twenty knobs
teaches nothing, whichever way it goes.
"""

FIRST_TURN = """\
Here is the current program, with every field's range and what it does:

{program}

Grown from it, over {n_seeds} world seeds:

{report}

For comparison, on the same seven exams:

{baselines}

A note on where this stands, so you do not have to rediscover it. A fresh brain
circles roughly in place. The command is read by pooling the loudest `pool_frac` of
the effector cells; pooling over ALL of them was measured to shrink the TURN command's
spread by 2-3x, because the few cells that heard something get averaged with the many
that did not. And `thrust_rest` is what a sheet holding NOTHING commands — at 0.5 a
silent brain drifts forward, which flatters any probe whose food is straight ahead.

State a hypothesis and write the next program.
"""


# --- growing and scoring a candidate --------------------------------------------------

def build_genome(spec: dict, brain_seed: int, wcfg: WorldConfig | None = None,
                 base: dict | None = None):
    """A genome from a proposal. The WEIGHTS come from `brain_seed` and nothing the
    model writes touches them — that is what makes a generation's difference
    attributable to the program.

    `base` is the program being EDITED, and it matters more than it looks. Without it
    every proposal was applied to the defaults, so a field the model chose not to
    mention silently reverted — the model believed it was refining one program and was
    in fact drawing independent samples around the default. The prompt promises that
    omitted fields keep their value; this is where that promise is kept."""
    bcfg = B.Config()
    if spec.get("seed_types"):
        bcfg.seed_types = int(min(bcfg.max_types, max(1, int(spec["seed_types"]))))
    g = fresh_genome(seed=brain_seed, wcfg=wcfg, bcfg=bcfg)
    if base:
        g.program.update(base)
    unknown = g.program.update(spec.get("program") or {})
    if spec.get("subticks"):
        g.subticks = int(min(bcfg.subticks_max,
                             max(bcfg.subticks_min, int(spec["subticks"]))))
    if spec.get("organ_map"):
        om = [int(v) for v in spec["organ_map"]][:len(g.organ_map)]
        if om:
            g.organ_map = np.array(
                [min(bcfg.organ_mods, max(1, v)) for v in om]
                + list(g.organ_map[len(om):]), dtype=np.int64)
    return g, unknown


def evaluate(genome, seeds, ticks: int = 4000, probe_ticks: int = 500) -> dict:
    """One life per world seed, then the exam on what each life left behind.

    Fitness leans on the LADDER, because the ladder is the milestone, with meals as a
    small additive term so that a brain which passes exams and starves in the world
    still ranks below one that does both. The profile is reported whole regardless —
    at one sample per generation a single number is not something to reason from."""
    lives, minds = [], []
    for seed in seeds:
        w = World(seed=int(seed), cfg=WorldConfig())
        mind = Mind(genome)
        mind.attach(w)
        while w.agent.alive and w.tick < ticks:
            mind.step(w)
        lives.append({"ticks": w.tick, "meals": int(sum(w.agent.ate)),
                      "distance": round(w.distance, 1),
                      "cells": mind.phenotype.cells(),
                      "sensory": int((mind.phenotype.in_mask
                                      & mind.phenotype.mask).sum()),
                      "effector": int((mind.phenotype.out_mask
                                       & mind.phenotype.mask).sum()),
                      "axons": len(mind.phenotype.axon_cells),
                      "born": mind.phenotype.births, "died": mind.phenotype.deaths})
        minds.append(mind)
    suites = [probes.run_suite(m, ticks=probe_ticks) for m in minds]
    keys = list(suites[0]["profile"])
    skills = {k: round(float(np.mean([s["profile"][k] for s in suites])), 3)
              for k in keys}
    body = {k: round(float(np.mean([lf[k] for lf in lives])), 1)
            for k in lives[0]}
    return {"skills": skills, "body": body, "lives": lives,
            "suite": suites[0], "mind": minds[0],
            "fitness": round(skills["mean"] + 0.01 * body["meals"], 4)}


def _profile_table(skills: dict) -> str:
    keys = [k for k in skills if k != "mean"]
    return ("  " + "".join(f"{k:>9}" for k in keys) + f"{'mean':>9}\n"
            + "  " + "".join(f"{skills[k]:>9.2f}" for k in keys)
            + f"{skills['mean']:>9.2f}")


def _baseline_table() -> str:
    keys = [k for k in BASELINES["reflex"] if k != "mean"]
    out = ["  " + f"{'':<8}" + "".join(f"{k:>9}" for k in keys) + f"{'mean':>9}"]
    for name, p in BASELINES.items():
        out.append("  " + f"{name:<8}" + "".join(f"{p[k]:>9.2f}" for k in keys)
                   + f"{p['mean']:>9.2f}")
    return "\n".join(out)


def report(res: dict) -> str:
    """What a generation looked like, for the model and for the journal. The BODY is in
    here on purpose: a program that grows 900 cells of which 800 are ears explains its
    own score, and the model cannot see the sheet."""
    b = res["body"]
    cap = B.Config().max_cells
    # cells are shown AGAINST THE CEILING because a body at the ceiling is a specific
    # failure, not a big success, and "2098 cells" does not say which one it is
    at_cap = "  AT THE CEILING" if b["cells"] >= cap * 0.98 else ""
    return (_profile_table(res["skills"]) + "\n\n"
            + f"  body     {b['cells']:.0f}/{cap} cells{at_cap}"
              f"  {b['sensory']:.0f} sensory"
              f"  {b['effector']:.0f} effector  {b['axons']:.0f} axons"
              f"  ({b['born']:.0f} born, {b['died']:.0f} died)\n"
            + f"  life     {b['ticks']:.0f} ticks  {b['meals']:.1f} meals"
              f"  {b['distance']:.0f} cells walked")


def leaderboard(board: list[dict], top: int = 5) -> str:
    best = sorted(board, key=lambda e: -e["fitness"])[:top]
    lines = ["LEADERBOARD (best first)", ""]
    for e in best:
        sk = e["skills"]
        lines.append(f"  #{e['iter']:<3} fit {e['fitness']:.3f}  mean {sk['mean']:.2f}"
                     f"  meals {e['body']['meals']:.1f}"
                     f"  cells {e['body']['cells']:.0f}")
        lines.append(f"       {e.get('hypothesis', '')[:150]}")
        lines.append("       " + "  ".join(f"{k} {v:.2f}" for k, v in sk.items()
                                           if k != "mean"))
    return "\n".join(lines) + "\n"


# --- pictures -------------------------------------------------------------------------

def shot(mind, path: Path, seed: int = 2, ticks: int = 400) -> bool:
    """A HUD screenshot of the grown brain, driving. Same App a human plays in, drawn
    headless: the world with the agent's TRACK behind it, the sense panel, and the
    sheet itself. This is the picture the user reads a run from."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame

        from cortex.app import App
        pygame.init()
        app = App(seed=seed, view=45, cell=8, size=256, mind=mind)
        keys: dict = {}
        for _ in range(ticks):
            app.frame(keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(app.draw(), str(path))
        return True
    except Exception as e:                       # a missing picture must never stop a run
        print(f"  (screenshot failed: {type(e).__name__}: {e})", file=sys.stderr)
        return False


def gallery(entries: list[dict], path: Path, width: int = 480) -> bool:
    """Every iteration's HUD, in order, with its score under it — rebuilt each turn so
    a long run can be watched while it runs rather than read after it."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame
        pygame.init()
        shots = [(e, pygame.image.load(str(e["shot"]))) for e in entries
                 if e.get("shot") and Path(e["shot"]).exists()]
        if not shots:
            return False
        head, pad, cols = 26, 8, min(3, len(shots))
        scale = width / shots[0][1].get_width()
        cell_h = int(shots[0][1].get_height() * scale) + head
        rows = (len(shots) + cols - 1) // cols
        surf = pygame.Surface((cols * (width + pad) + pad,
                               rows * (cell_h + pad) + pad))
        surf.fill((12, 12, 16))
        font = pygame.font.SysFont("consolas", 13, bold=True)
        small = pygame.font.SysFont("consolas", 11)
        best = max(e["fitness"] for e, _ in shots)
        for i, (e, img) in enumerate(shots):
            x = pad + (i % cols) * (width + pad)
            y = pad + (i // cols) * (cell_h + pad)
            top = e["fitness"] >= best
            col = (140, 240, 190) if top else (200, 200, 215)
            surf.blit(font.render(f"#{e['iter']}  fit {e['fitness']:.3f}"
                                  + ("   <- best" if top else ""), True, col), (x, y))
            sk = e["skills"]
            surf.blit(small.render(
                f"mean {sk['mean']:.2f}  meals {e['body']['meals']:.1f}"
                f"  {e['body']['cells']:.0f} cells", True, (130, 130, 145)), (x, y + 14))
            surf.blit(pygame.transform.smoothscale(
                img, (width, cell_h - head)), (x, y + head))
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surf, str(path))
        return True
    except Exception as e:
        print(f"  (gallery failed: {type(e).__name__}: {e})", file=sys.stderr)
        return False


# --- the proposers --------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """The one fenced json block. Falls back to the last {...} span, because a model
    that forgets its fence has still done the work."""
    import re
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m[-1] if m else None
    if blob is None:
        i, j = text.find("{"), text.rfind("}")
        blob = text[i:j + 1] if 0 <= i < j else None
    if not blob:
        return None
    try:
        out = json.loads(blob)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


class RngProposer:
    """The control arm: the same loop with a random program mutation in the model's
    chair. Without this the loop can show that the LLM improves things and still not
    show that the LLM is what improved them."""

    def __init__(self, seed: int = 0, prob: float = 0.25, scale: float = 0.15):
        self.rng = np.random.default_rng(seed)
        self.prob, self.scale = prob, scale
        self.model = f"rng(seed={seed})"
        self.best: dict = {}

    def propose(self, board: list[dict]) -> tuple[dict, str]:
        if board:
            self.best = max(board, key=lambda e: e["fitness"])["spec"].get("program", {})
        child = PROG.Program(self.best).mutate(self.rng, self.prob, self.scale)
        return ({"program": child.to_dict()},
                "random jitter of the best program so far")


# --- the loop -------------------------------------------------------------------------

def run_lab(client, iterations: int = 8, seeds=(0, 1), ticks: int = 4000,
            probe_ticks: int = 500, brain_seed: int = 0,
            out_dir: str = "out/cortex/lab", temperature: float = 1.0,
            shots: bool = True) -> dict:
    run = Path(out_dir) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    journal = run / "journal.md"
    model = getattr(client, "model", "?")
    journal.write_text(
        f"# cortex lab — {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Proposer: `{model}` | brain seed {brain_seed} (weights FIXED across "
        f"candidates) | world seeds {list(seeds)} | {ticks} ticks/life\n\n"
        f"Baselines on the same ladder:\n\n```\n{_baseline_table()}\n```\n",
        encoding="utf-8")

    def log(text: str) -> None:
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(text)

    board: list[dict] = []

    def record_entry(i: int, spec: dict, hyp: str, res: dict,
                     changed: dict | None = None) -> dict:
        entry = {"iter": i, "spec": spec, "hypothesis": hyp, "changed": changed or {},
                 "skills": res["skills"], "body": res["body"],
                 "fitness": res["fitness"]}
        strip = run / f"iter{i:02d}_probes.png"
        try:
            probes.render(res["suite"], str(strip))
            entry["strip"] = str(strip)
        except Exception as e:
            print(f"  (strip failed: {e})", file=sys.stderr)
        if shots:
            png = run / f"iter{i:02d}_brain.png"
            if shot(res["mind"], png):
                entry["shot"] = str(png)
        board.append(entry)
        gallery(board, run / "gallery.png")
        rel = lambda p: Path(p).name                                   # noqa: E731
        log(f"\n---\n\n## Iteration {i}"
            f"{'  (best so far)' if entry['fitness'] >= max(e['fitness'] for e in board) else ''}"
            f"\n\n**Hypothesis:** {hyp}\n\n"
            # what MOVED, not the whole vector: a generation reporting thirty numbers
            # teaches nothing next to one reporting the three that changed
            f"**Changed:** `{json.dumps(entry['changed']) if i else 'the defaults'}`\n\n"
            f"```\n{report(res)}\n```\n\n"
            + (f"![brain {i}]({rel(entry['shot'])})\n\n" if entry.get("shot") else "")
            + (f"![probes {i}]({rel(entry['strip'])})\n" if entry.get("strip") else ""))
        return entry

    # GENERATION 0 is the default program, evaluated on exactly the same terms. Without
    # it "better" has no referent, and the model's first turn has nothing to read.
    print(f"[gen 0] growing the default program...", file=sys.stderr)
    g0, _ = build_genome({}, brain_seed)
    res0 = evaluate(g0, seeds, ticks, probe_ticks)
    record_entry(0, {"program": g0.program.to_dict()}, "the default program", res0)
    print(f"[gen 0] fit {res0['fitness']:.3f}\n{report(res0)}", file=sys.stderr)

    is_llm = hasattr(client, "chat")
    messages = [{"role": "system", "content": SPEC},
                {"role": "user", "content": FIRST_TURN.format(
                    program=g0.program.as_prompt(), n_seeds=len(seeds),
                    report=report(res0), baselines=_baseline_table())}]

    done = 0
    while done < iterations:
        if is_llm:
            # RETRY, because the two failures actually seen are both transient: a
            # thinking model returning empty content (a known deepseek-v4 quirk, see
            # CLAUDE.md) and a timed-out connection. Losing a fifty-minute run to
            # either is the wrong trade.
            reply, err = None, None
            for attempt in range(3):
                try:
                    reply = client.chat(messages, temperature=temperature)
                    break
                except LLMError as e:
                    err = e
                    print(f"[gen {done + 1}] {e} (attempt {attempt + 1}/3)",
                          file=sys.stderr)
            if reply is None:
                log(f"\n---\n\n## Iteration {done + 1} — the model errored\n\n{err}\n")
                break
            messages.append({"role": "assistant", "content": reply})
            spec = extract_json(reply)
            if spec is None or not isinstance(spec.get("program"), dict):
                log(f"\n---\n\n## Iteration {done + 1} (no program found)\n\n{reply}\n")
                messages.append({"role": "user", "content":
                                 "I could not find a json block with a \"program\" "
                                 "object in it. Resend just that block."})
                continue
            hyp = str(spec.get("hypothesis", ""))[:400]
        else:
            spec, hyp = client.propose(board)
            reply = ""

        done += 1
        # EDIT THE BEST, not the defaults. (1 + lambda) with lambda = 1, which is the
        # lineage scheme the rest of the project already runs on, and it is what makes
        # "change a few things at a time" a coherent instruction: a proposal is a step
        # from a known program rather than a fresh draw around the default.
        base = max(board, key=lambda e: e["fitness"])["spec"]["program"]
        genome, unknown = build_genome(spec, brain_seed, base=base)
        print(f"[gen {done}/{iterations}] growing...", file=sys.stderr)
        res = evaluate(genome, seeds, ticks, probe_ticks)
        # what the genome ACTUALLY carries, not what was asked for: values are clamped
        # into range on the way in, and a run must journal what it ran
        changed = {k: v for k, v in genome.program.to_dict().items() if base.get(k) != v}
        spec = dict(spec, program=genome.program.to_dict())
        entry = record_entry(done, spec, hyp, res, changed)
        best = max(board, key=lambda e: e["fitness"])
        print(f"[gen {done}] fit {entry['fitness']:.3f}"
              + ("  <- best" if best is entry else f"  (best #{best['iter']} "
                                                   f"{best['fitness']:.3f})")
              + f"\n{report(res)}", file=sys.stderr)
        if is_llm:
            note = (f"Fields I did not recognise and dropped: {unknown}\n\n"
                    if unknown else "")
            nxt = max(board, key=lambda e: e["fitness"])
            moved = json.dumps(changed) if changed else "nothing (it matched the base)"
            messages.append({"role": "user", "content":
                             f"{note}What actually changed: {moved}\n\n"
                             f"Your program grew this:\n\n{report(res)}\n\n"
                             f"{leaderboard(board)}\n"
                             # be explicit about WHICH program the next edit lands on.
                             # Leaving this implicit is how a model ends up believing it
                             # is refining one thing while it edits another.
                             f"Your next program will be written on top of iteration "
                             f"{nxt['iter']}, the best so far. That is:\n\n"
                             f"{PROG.Program(nxt['spec']['program']).summary()}\n\n"
                             "Say what you conclude, then write the next program. "
                             "Change a few things and say why."})

    best = max(board, key=lambda e: e["fitness"])
    (run / "best_program.json").write_text(
        json.dumps(best["spec"], indent=2), encoding="utf-8")
    # ...and the whole GENOME the winning program builds, so the run's answer can be
    # WATCHED (`cortex.app --load-brain`) and not only read. Written to a stable path
    # as well as into the run, because that is the one an F5 config can point at.
    best_genome, _ = build_genome(best["spec"], brain_seed)
    blob = json.dumps(best_genome.to_dict())
    (run / "best_genome.json").write_text(blob, encoding="utf-8")
    stable = Path(out_dir) / "best_genome.json"
    stable.write_text(blob, encoding="utf-8")
    log(f"\n---\n\n## Result\n\nBest: iteration {best['iter']}, "
        f"fitness {best['fitness']:.3f}\n\n```\n{_profile_table(best['skills'])}\n```\n\n"
        f"![gallery](gallery.png)\n")
    return {"best": best, "board": board, "dir": str(run)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cortex.lab", description="An LLM writes the brain's growth program.")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--seeds", default="0,1", help="world seeds a life rotates over")
    p.add_argument("--ticks", type=int, default=4000, help="cap on one life")
    p.add_argument("--probe-ticks", type=int, default=500)
    p.add_argument("--brain-seed", type=int, default=0,
                   help="the FIXED random weights every candidate is grown with")
    p.add_argument("--out", default="out/cortex/lab")
    p.add_argument("--proposer", choices=("llm", "rng"), default="llm",
                   help="rng is the control arm: same loop, random program mutation")
    p.add_argument("--model", default=None, help="override NOMOS_LLM_MODEL")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--no-shots", action="store_true", help="skip HUD screenshots")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.proposer == "rng":
        client = RngProposer(seed=args.brain_seed)
    else:
        try:
            client = make_client(model=args.model)
        except LLMError as e:
            print(f"cannot reach the model: {e}", file=sys.stderr)
            return 2
    out = run_lab(client, iterations=args.iterations, seeds=seeds, ticks=args.ticks,
                  probe_ticks=args.probe_ticks, brain_seed=args.brain_seed,
                  out_dir=args.out, temperature=args.temperature,
                  shots=not args.no_shots)
    b = out["best"]
    print(f"\nbest: iteration {b['iter']}, fitness {b['fitness']:.3f}")
    print(_profile_table(b["skills"]))
    print(f"\nrun -> {out['dir']}   (journal.md, gallery.png, best_program.json)")
    print(f"watch it: python -m cortex.app --load-brain "
          f"{Path(args.out) / 'best_genome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
