"""The LLM writes the atlas's growth program; the map is grown from it and lives.

    .\\.venv\\Scripts\\python.exe -m cortex.atlas_lab --iterations 8 --jobs 8
    .\\.venv\\Scripts\\python.exe -m cortex.atlas_lab --proposer rng ...   # the control

`cortex/lab.py` put a model in charge of a brain that could not be taught, and its eight
generations optimised body size while the score was noise. This is the same loop on a
brain that works: the model writes `cortex/atlas_lineage.FIELDS` — a dozen named
scalars, radius first — and every candidate is lived on the SAME eight worlds for the
same number of ticks, in parallel, so a score is exact and a difference between two
programs is a difference between the programs. The reflex policy's meals on those
worlds are in the prompt as the bar.

The model never sees a weight and never sees the rule; the rule is the wave (or a
found one, `--rule trained.npy`). What it gets is what a designer would want: the
knobs with their ranges and what each does, the leaderboard, and for each past
candidate its meals, lifetime, how many nightcaps it ate and how much its map cost.
`--proposer rng` is the control arm: the identical loop with `Program.mutate` in the
model's chair.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

from cortex import atlas as A
from cortex import atlas_lineage as L
from cortex.lab import extract_json
from nomos.llm import LLMError, make_client

SPEC = """\
You are designing the GROWTH PROGRAM of a map-brain: a cellular automaton whose lattice
is a map of the world around a foraging agent. The agent sits at the centre; every sense
paints where it refers to (walls seen by rays, morsels seen, smells as sources out along
their bearing); one local rule spreads a value wave from sources through non-solid cells;
the agent steps toward the brightest neighbour of its own cell, and presses INTERACT to
eat what is within reach. Unknown cells beside known ones are faint sources (that is
how it explores). Painted evidence fades, so the map is a memory. Every map cell costs
energy every tick, so a bigger map has to earn its keep. Food kinds: berry 12 energy,
fruit 22, mushroom 32 (rare, in caves), nightcap -30 (poison: smells like a berry,
differs only in hue; the brain learns to refuse it after one burn).

You write the program's scalars. Reply with a short hypothesis (one or two sentences:
what you are changing and why you expect it to eat more) and then ONE fenced json block
with the fields you want to set, e.g.

```json
{"radius": 10, "decay": 0.05}
```

Fields you do not mention keep the current best program's values. Out-of-range values
are clamped. Think like an ecologist with a budget: the score is what a life EATS."""

FIRST_TURN = """\
The program and its knobs:

{fields}

Generation 0 is the current defaults, lived on {n_worlds} worlds for up to {ticks}
ticks each:

{report}

The reflex policy (no brain, no map cost) on the same worlds: {bar}

Propose the first change."""

NEXT_TURN = """\
Result of your proposal (generation {gen}):

{report}

Leaderboard (best first):
{board}

Propose the next change."""


def report(res: dict, prog: L.Program) -> str:
    return (f"fitness {res['fitness']:.1f}  meals {res['meals']:.1f}  lifetime "
            f"{res['ticks']:.0f} ticks  nightcaps {res.get('night', 0):.1f}  "
            f"map {(2 * prog.values['radius'] + 1) ** 2} cells\n  program: {prog}")


def _life(args) -> dict:
    prog_d, seed, ticks, world, theta = args
    from cortex.world import FOOD_ENERGY, NIGHTCAP
    rule = None
    if theta is not None:
        from cortex.wave import NeuralRule
        rule = NeuralRule(np.asarray(theta))
    w, mind = A.live(int(seed), int(ticks), L.Program(prog_d).config(),
                     wcfg=L.WorldConfig(**L.WORLDS[world]), rule=rule)
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"ticks": w.tick, "meals": int(sum(w.agent.ate)), "gained": gained,
            "night": int(w.agent.ate[NIGHTCAP]), "fitness": gained + 0.002 * w.tick}


def evaluate(pool, prog: L.Program, seeds, ticks: int, world: str, theta) -> dict:
    args = [(prog.to_dict(), s, ticks, world, theta) for s in seeds]
    rows = pool.map(_life, args) if pool else [_life(a) for a in args]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def reflex_bar(seeds, ticks: int, world: str) -> str:
    from cortex import baseline
    rows = [baseline.play("reflex", int(s), ticks, cfg=L.WorldConfig(**L.WORLDS[world]))
            for s in seeds]
    meals = np.mean([sum(w.agent.ate) for w in rows])
    life = np.mean([w.tick for w in rows])
    return f"meals {meals:.1f}, lifetime {life:.0f} ticks"


class RngProposer:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.model = f"rng(seed={seed})"

    def propose(self, best: L.Program) -> tuple[dict, str]:
        child = best.mutate(self.rng, prob=0.3, scale=0.2)
        return child.diff(best) and {k: v[1] for k, v in child.diff(best).items()} or {}, \
            "random jitter of the best program so far"


def run_lab(client, iterations: int = 8, seeds=tuple(range(8)), ticks: int = 8000,
            world: str = "default", jobs: int = 8, out_dir: str = "out/cortex/atlas_lab",
            temperature: float = 1.0, theta=None) -> dict:
    from multiprocessing import Pool
    run = Path(out_dir) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    journal = run / "journal.md"
    model = getattr(client, "model", "?")
    pool = Pool(jobs) if jobs > 1 else None

    def log(text: str) -> None:
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(text)

    try:
        bar = reflex_bar(seeds, ticks, world)
        journal.write_text(
            f"# atlas lab — {dt.datetime.now():%Y-%m-%d %H:%M}\n\nProposer: `{model}` | "
            f"world {world} | seeds {list(seeds)} | {ticks} ticks/life | reflex: {bar}\n",
            encoding="utf-8")
        board: list[dict] = []

        def record(i: int, prog: L.Program, hyp: str, res: dict, changed: dict) -> dict:
            entry = {"iter": i, "program": prog.to_dict(), "hypothesis": hyp,
                     "changed": changed, **res}
            board.append(entry)
            best = max(board, key=lambda e: e["fitness"])
            log(f"\n---\n\n## Iteration {i}{'  (best so far)' if best is entry else ''}\n\n"
                f"**Hypothesis:** {hyp}\n\n**Changed:** `{json.dumps(changed)}`\n\n"
                f"```\n{report(res, prog)}\n```\n")
            with (run / "board.json").open("w", encoding="utf-8") as fh:
                json.dump(board, fh, indent=1)
            with (run / "best_program.json").open("w", encoding="utf-8") as fh:
                json.dump(best["program"], fh, indent=1)
            return entry

        print("[gen 0] the default program...", file=sys.stderr)
        p0 = L.Program()
        r0 = evaluate(pool, p0, seeds, ticks, world, theta)
        record(0, p0, "the default program", r0, {})
        print(f"[gen 0] {report(r0, p0)}", file=sys.stderr)

        is_llm = hasattr(client, "chat")
        fields = "\n".join(f"  {k:<12} = {L.DEFAULTS[k]:<8} [{v[1]}..{v[2]}]  {v[4]}"
                           for k, v in L.FIELDS.items())
        messages = [{"role": "system", "content": SPEC},
                    {"role": "user", "content": FIRST_TURN.format(
                        fields=fields, n_worlds=len(seeds), ticks=ticks,
                        report=report(r0, p0), bar=bar)}]
        for i in range(1, iterations + 1):
            best = max(board, key=lambda e: e["fitness"])
            base = L.Program(best["program"])
            if is_llm:
                try:
                    reply = client.chat(messages, temperature=temperature)
                except LLMError as e:
                    log(f"\n\n**LLM error at iteration {i}:** {e}\n")
                    print(f"[gen {i}] LLM error: {e}", file=sys.stderr)
                    break
                spec = extract_json(reply) or {}
                hyp = reply.split("```")[0].strip()[:600] or "(no hypothesis given)"
                messages.append({"role": "assistant", "content": reply})
            else:
                spec, hyp = client.propose(base)
            prog = L.Program({**base.to_dict(), **spec})
            changed = prog.diff(base)
            print(f"[gen {i}] {hyp[:100]}...", file=sys.stderr)
            res = evaluate(pool, prog, seeds, ticks, world, theta)
            record(i, prog, hyp, res, {k: v[1] for k, v in changed.items()})
            print(f"[gen {i}] {report(res, prog)}", file=sys.stderr)
            if is_llm:
                lines = [f"  gen {e['iter']}: fitness {e['fitness']:.1f}  meals {e['meals']:.1f}"
                         f"  lifetime {e['ticks']:.0f}  changed {json.dumps(e['changed'])}"
                         for e in sorted(board, key=lambda e: -e["fitness"])[:6]]
                messages.append({"role": "user", "content": NEXT_TURN.format(
                    gen=i, report=report(res, prog), board="\n".join(lines))})
    finally:
        if pool:
            pool.close()
            pool.join()
    best = max(board, key=lambda e: e["fitness"])
    return {"best": best, "board": board, "dir": str(run)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="An LLM writes the atlas's growth program.")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--world", choices=sorted(L.WORLDS), default="default")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--proposer", choices=("llm", "rng"), default="llm")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rule", default="", help="a found rule (.npy) instead of the wave")
    p.add_argument("--out", default="out/cortex/atlas_lab")
    args = p.parse_args(argv)
    theta = np.load(args.rule) if args.rule else None
    if args.proposer == "rng":
        client = RngProposer()
    else:
        try:
            client = make_client(model=args.model)
        except LLMError as e:
            print(f"cannot reach the model: {e}", file=sys.stderr)
            return 2
    out = run_lab(client, args.iterations, A._seeds(args.seeds), args.ticks, args.world,
                  args.jobs, args.out, args.temperature, theta)
    b = out["best"]
    print(f"\nbest: iteration {b['iter']}, fitness {b['fitness']:.1f}, meals {b['meals']:.1f}")
    print(f"run -> {out['dir']}  (journal.md, board.json, best_program.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
