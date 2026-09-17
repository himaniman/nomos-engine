"""The LLM writes the atlas's RULE — as a law of nature in the nomos DSL.

    .\\.venv\\Scripts\\python.exe -m cortex.law_lab --iterations 8 --jobs 8
    .\\.venv\\Scripts\\python.exe -m cortex.law_lab --proposer rng ...   # the control

`cortex/atlas_lab.py` lets the model turn twelve knobs of a rule it never sees. Here
the model sees the rule and rewrites it: the wave that plans the agent's path is a
`.nomos` law (`cortex/atlas_law.py`), the same language the scientist writes physics in
and the brain lab wrote snake brains in, so the mutation operator is a model editing
program text — channels can be added, the wave can be made anisotropic, hysteretic,
distance-aware, anything the DSL can say. Generation 0 is the hand wave transcribed
(`laws/atlas_wave.nomos`). Every candidate lives the same eight worlds for the same
ticks with the same growth program, so a score is exact and a difference between two
laws is a difference between the laws. `--proposer rng` is the control: the same loop
with `atlas_law.jitter` — random moves of the best law's constants, structure frozen —
in the model's chair. `--holdout` re-lives the finalists on worlds nobody selected on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from cortex import atlas as A
from cortex import atlas_law as LAW
from cortex import atlas_lineage as L
from cortex.atlas_lab import reflex_bar
from nomos.dsl import NomosError
from nomos.llm import LLMError, make_client
from nomos.scientist import extract_law

SPEC = """\
You are writing the RULE of a map-brain: a cellular automaton whose lattice is a map of
the world around a foraging agent, written as a law in the nomos DSL below. The agent
sits at the centre cell; the map scrolls under it as it walks. Every sense paints where
it refers to. One local rule — yours — turns what is painted into a `value` field, and
the agent steps toward the brightest of the eight neighbours of its own cell (ties go
to the least turn; a neighbour that is solid counts for 12% of its value; if nothing
anywhere is above zero it stands still). When it is led INTO a solid cell it presses
INTERACT, which opens doors and chips rubble. When food is within reach it eats.

The world: open terrain with walls, caves, doors and rubble. Food kinds: berry 12
energy, fruit 22, mushroom 32 (rare, in caves), nightcap -30 (a poison that smells like
a berry; the brain learns to refuse it, and refused things arrive in `bad`). Living
costs energy every tick and every map cell costs a little; death is by starvation only.
The score is what a life EATS, on eight fixed worlds, up to {ticks} ticks each.

THE HARNESS. The lattice is the {side}x{side} map padded by two rings: the inner ring is
the horizon (free, sourceless — a way round the outside), the outer ring is forced to
zero after every step, so nothing wraps. Each thought (every 3 world ticks) the harness
paints the INPUT channels, steps your law {passes} times, and reads `value`. Your
`value` and any other channel you declare PERSIST between thoughts and scroll with the
map (cells that scroll into view are zero).

INPUT channels, repainted before every thought (declare the ones you read; writing
them is pointless), all in [0, 1] except source, agent at the centre cell:
  source   what the senses want: smell painted as a source out along its bearing,
           seen morsels weighted by appetite, faint (0.12) marks on unknown cells that
           touch known-free ones (the frontier: how the agent explores)
  solid    how solid a cell is known to be (seen wall, felt wall)
  free     how surely a cell is known free (seen through, walked)
  thing    a seen morsel is here
  bright   its brightness
  scent    the painted smell alone
  bad      a remembered refusal (poison eaten here, or a thing refused)
  here     1 at the centre cell: where the agent is on its own map
OUTPUT: `value`, clipped to [0, 4]. Reference law (generation 0), the Bellman wave:

```nomos
{reference}
```

posx()/posy() give a cell's position in [0,1) on the lattice — the agent is at
(0.5, 0.5), so abs(posx()-0.5) is a distance from the agent. time() counts steps.

THE nomos DSL. A law declares channels and rules:

    channel <name> [: init <expr>]      # per-cell float64 field, default 0
    rule <name>:
        when <expr>                     # nonzero = true, per cell
        set <channel> = <expr>          # overwrite where true
        add <channel> = <expr>          # accumulate where true

Synchronous update: every expression reads the snapshot from the start of the step.
Rules apply in declaration order; per channel the last matching `set` wins, `add`s
accumulate. NaN -> 0, values clipped. Operators: or and not == != < <= > >= + - * / %
and unary minus. Built-ins: sum8 avg8 max8 min8 (Moore, centre excluded), sum4 avg4
max4 min4 (von Neumann), shift(f, dx, dy) (the field moved: a cell sees f(x-dx, y-dy)),
argmax8(key, value) (the value of the neighbour with the largest key), posx() posy()
time(), abs sign floor round sqrt exp log sin cos pi(), min(a,b) max(a,b)
clamp(x,lo,hi) where(cond,a,b), rand() bernoulli(p) uniform(a,b) normal(mu,sigma)
(seeded, reproducible). No power operator: write constants out.

Reply with a short hypothesis (two or three sentences: what you are changing in the
rule and why the agent will eat more) and then ONE fenced ```nomos block with the WHOLE
law. Think like a physicist of this little world: what should a planning wave do that
Bellman does not — remember, commit, discount distance, distrust old evidence, weigh
the frontier, avoid what burned it? A law that fails to parse is sent back to you with
the error. Do not touch the growth program: only the law."""

FIRST_TURN = """\
Generation 0, the reference wave, lived on {n_worlds} worlds for up to {ticks} ticks:

{report}

The reflex policy (no brain, no map cost) on the same worlds: {bar}

Propose the first law."""

NEXT_TURN = """\
Result of your law (generation {gen}):

{report}

Leaderboard (best first):
{board}

Propose the next law. You may build on any law on the board, or start fresh."""


def report(res: dict) -> str:
    return (f"fitness {res['fitness']:.1f}  meals {res['meals']:.1f}  lifetime "
            f"{res['ticks']:.0f} ticks  nightcaps {res.get('night', 0):.1f}  "
            f"immortal {res.get('immortal', 0):.0f}/{res.get('n', 0):.0f}")


def _life(args) -> dict:
    text, seed, ticks, world, prog_d = args
    from cortex.world import FOOD_ENERGY, NIGHTCAP
    rule = LAW.LawRule(text, seed)
    w, mind = A.live(int(seed), int(ticks), L.Program(prog_d).config(),
                     wcfg=L.WorldConfig(**L.WORLDS[world]), rule=rule)
    gained = float(sum(n * FOOD_ENERGY[k] for k, n in enumerate(w.agent.ate)))
    return {"ticks": w.tick, "meals": int(sum(w.agent.ate)), "gained": gained,
            "night": int(w.agent.ate[NIGHTCAP]), "fitness": gained + 0.002 * w.tick,
            "immortal": int(w.agent.alive)}


def evaluate(pool, text: str, seeds, ticks: int, world: str, prog: L.Program) -> dict:
    args = [(text, s, ticks, world, prog.to_dict()) for s in seeds]
    rows = pool.map(_life, args) if pool else [_life(a) for a in args]
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["immortal"] = float(sum(r["immortal"] for r in rows))
    out["n"] = float(len(rows))
    out["per_seed"] = [r["meals"] for r in rows]
    return out


class FileProposer:
    """A physicist's laws, one file per generation, through the same loop and the
    same holdout as the model's."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.model = "files"

    def propose(self, best_text: str) -> tuple[str, str]:
        path = self.paths.pop(0)
        text = Path(path).read_text(encoding="utf-8")
        first = text.strip().splitlines()[0].lstrip("# ").strip()
        return text, f"{path}: {first}"


class RngProposer:
    """The control: the best law's constants jittered, its structure frozen."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.model = f"rng(seed={seed})"

    def propose(self, best_text: str) -> tuple[str, str]:
        return LAW.jitter(best_text, self.rng), "random jitter of the best law's constants"


def run_lab(client, iterations: int = 8, seeds=tuple(range(8)), ticks: int = 8000,
            world: str = "default", jobs: int = 8, out_dir: str = "out/cortex/law_lab",
            temperature: float = 1.0, program: dict | None = None,
            holdout=()) -> dict:
    from multiprocessing import Pool
    run = Path(out_dir) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    journal = run / "journal.md"
    model = getattr(client, "model", "?")
    pool = Pool(jobs) if jobs > 1 else None
    prog = L.Program(program or {})
    reference = Path(LAW.REFERENCE).read_text(encoding="utf-8")

    def log(text: str) -> None:
        with journal.open("a", encoding="utf-8") as fh:
            fh.write(text)

    try:
        bar = reflex_bar(seeds, ticks, world)
        journal.write_text(
            f"# law lab — {dt.datetime.now():%Y-%m-%d %H:%M}\n\nProposer: `{model}` | "
            f"world {world} | seeds {list(seeds)} | {ticks} ticks/life | program "
            f"{prog} | reflex: {bar}\n", encoding="utf-8")
        board: list[dict] = []

        def record(i: int, text: str, hyp: str, res: dict) -> dict:
            entry = {"iter": i, "law": text, "hypothesis": hyp, **res}
            board.append(entry)
            best = max(board, key=lambda e: e["fitness"])
            (run / f"law{i:02d}.nomos").write_text(text, encoding="utf-8")
            log(f"\n---\n\n## Iteration {i}{'  (best so far)' if best is entry else ''}\n\n"
                f"**Hypothesis:** {hyp}\n\n```\n{report(res)}\n  meals per world: "
                f"{res['per_seed']}\n```\n\n```nomos\n{text}\n```\n")
            with (run / "board.json").open("w", encoding="utf-8") as fh:
                json.dump(board, fh, indent=1)
            (run / "best.nomos").write_text(best["law"], encoding="utf-8")
            return entry

        print("[gen 0] the reference wave...", file=sys.stderr)
        r0 = evaluate(pool, reference, seeds, ticks, world, prog)
        record(0, reference, "the reference wave (Bellman through non-solid cells)", r0)
        print(f"[gen 0] {report(r0)}", file=sys.stderr)

        is_llm = hasattr(client, "chat")
        side = 2 * prog.values["radius"] + 1
        messages = [{"role": "system", "content": SPEC.format(
                        ticks=ticks, side=side, passes=prog.values["subticks"],
                        reference=reference.strip())},
                    {"role": "user", "content": FIRST_TURN.format(
                        n_worlds=len(seeds), ticks=ticks, report=report(r0), bar=bar)}]
        attempts = 0
        i = 1
        while i <= iterations:
            best = max(board, key=lambda e: e["fitness"])
            if is_llm:
                try:
                    reply = client.chat(messages, temperature=temperature)
                except LLMError as e:
                    log(f"\n\n**LLM error at iteration {i}:** {e}\n")
                    print(f"[gen {i}] LLM error: {e}", file=sys.stderr)
                    break
                messages.append({"role": "assistant", "content": reply})
                text = extract_law(reply)
                hyp = reply.split("```")[0].strip()[:700] or "(no hypothesis given)"
                if text is None:
                    attempts += 1
                    log(f"\n---\n\n## Attempt (no law found)\n\n{reply}\n")
                    messages.append({"role": "user", "content":
                                     "No ```nomos block found — resend the whole law fenced."})
                    if attempts > 3 * iterations:
                        break
                    continue
                try:
                    LAW.check(text)
                except NomosError as e:
                    attempts += 1
                    log(f"\n---\n\n## Attempt (invalid law)\n\n**Error:** {e}\n\n"
                        f"```nomos\n{text}\n```\n")
                    print(f"[gen {i}] invalid law: {e}", file=sys.stderr)
                    messages.append({"role": "user", "content":
                                     f"That law did not load: {e}\nFix it and resend the "
                                     "whole law (it must declare `value`)."})
                    if attempts > 3 * iterations:
                        break
                    continue
            else:
                text, hyp = client.propose(best["law"])
            print(f"[gen {i}] {hyp[:100]}...", file=sys.stderr)
            res = evaluate(pool, text, seeds, ticks, world, prog)
            record(i, text, hyp, res)
            print(f"[gen {i}] {report(res)}", file=sys.stderr)
            if is_llm:
                lines = [f"  gen {e['iter']}: fitness {e['fitness']:.1f}  meals {e['meals']:.1f}"
                         f"  lifetime {e['ticks']:.0f}  — {e['hypothesis'][:120]}"
                         for e in sorted(board, key=lambda e: -e["fitness"])[:6]]
                messages.append({"role": "user", "content": NEXT_TURN.format(
                    gen=i, report=report(res), board="\n".join(lines))})
            i += 1

        if holdout:
            log("\n---\n\n## Holdout\n\n")
            best = max(board, key=lambda e: e["fitness"])
            for name, text in (("reference", reference), (f"best (gen {best['iter']})",
                                                          best["law"])):
                res = evaluate(pool, text, holdout, ticks, world, prog)
                log(f"- {name}: {report(res)}  meals per world {res['per_seed']}\n")
                print(f"[holdout] {name}: {report(res)}", file=sys.stderr)
    finally:
        if pool:
            pool.close()
            pool.join()
    best = max(board, key=lambda e: e["fitness"])
    return {"best": best, "board": board, "dir": str(run)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="An LLM writes the atlas's rule as a nomos law.")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--holdout", default="", help="seeds the finalists are re-lived on")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--world", choices=sorted(L.WORLDS), default="default")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--proposer", choices=("llm", "rng", "files"), default="llm")
    p.add_argument("--laws", nargs="*", default=[], help="with --proposer files")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--program", default="", help="a growth program json (default: defaults)")
    p.add_argument("--out", default="out/cortex/law_lab")
    args = p.parse_args(argv)
    program = json.load(open(args.program, encoding="utf-8")) if args.program else None
    if args.proposer == "rng":
        client = RngProposer()
    elif args.proposer == "files":
        client = FileProposer(args.laws)
        args.iterations = len(args.laws)
    else:
        try:
            client = make_client(model=args.model)
        except LLMError as e:
            print(f"cannot reach the model: {e}", file=sys.stderr)
            return 2
    out = run_lab(client, args.iterations, A._seeds(args.seeds), args.ticks, args.world,
                  args.jobs, args.out, args.temperature, program,
                  A._seeds(args.holdout) if args.holdout else ())
    b = out["best"]
    print(f"\nbest: iteration {b['iter']}, fitness {b['fitness']:.1f}, meals {b['meals']:.1f}")
    print(f"run -> {out['dir']}  (journal.md, board.json, best.nomos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
