"""The automated scientist loop.

An LLM plays physicist: it proposes a hypothesis and a law, the engine runs it,
the observer reports, and the model reads the result, concludes, and proposes
the next experiment. Everything is logged to a markdown lab journal and to the
SQLite archive. This is the manual loop of Experiments 1-4 handed to an API.

The point (per the project's philosophy) is NOT to maximize a score. It is to
build understanding — explain what happened, form causal hypotheses, explore.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from nomos import archive
from nomos.dsl import NomosError
from nomos.llm import LLMError, make_client
from nomos.sim import digest, simulate

DSL_SPEC = """\
You are a physicist exploring artificial worlds. You do not move cells; you
write the LAWS a grid world lives by, then read what emerges and reason about it.

THE WORLD. A toroidal (edges wrap) grid. Each cell holds several named float
channels. Every step, all rules read a SNAPSHOT of the world at the start of the
step and update synchronously. Values are auto-regularized (NaN->0, clipped to
+/-1e6); a world may die but never crashes.

LAW SYNTAX (a .nomos program):
    channel <name>            # a per-cell float field, default init 0
    channel <name>: init <expr>
    rule <name>:
        when <expr>           # condition; nonzero = true
        set <chan> = <expr>   # overwrite where condition holds
        add <chan> = <expr>   # accumulate where condition holds
Rules apply in declaration order; per channel the last matching `set` wins and
`add`s accumulate. A rule needs >=1 set/add.

EXPRESSIONS. Operators (loose->tight): or, and, not, == != < <= > >=, + -, * / %,
unary -. Comparisons/booleans give 0/1 fields. Functions:
  neighbourhood (center excluded, wraps): sum8 avg8 max8 min8 (Moore),
    sum4 avg4 max4 min4 (von Neumann)
  inheritance: argmax8(key,value)/argmax4 = value of the neighbour with the
    largest key. argmax8(rand()*alive, g) copies a random living neighbour's g.
    For LINKED inheritance of several genes from ONE parent, use a shared random
    key channel pk (set pk = rand() each step): argmax8(pk*alive, a) and
    argmax8(pk*alive, b) then pick the same neighbour.
  space/time: posx() posy() in [0,1); time() = step number
  math: abs sign floor round sqrt exp log sin cos pi()
        min(a,b) max(a,b) clamp(x,lo,hi) where(cond,a,b)
  random (seeded): rand() bernoulli(p) uniform(a,b) normal(mu,sigma)

EXAMPLE (Game of Life):
  channel alive: init bernoulli(0.2)
  rule birth: when alive == 0 and sum8(alive) == 3
    set alive = 1
  rule death: when alive == 1 and (sum8(alive) < 2 or sum8(alive) > 3)
    set alive = 0

WHAT THE OBSERVER TELLS YOU (after each run): a verdict (frozen/static/alive/
boiling), activity level and trend, and per channel: spread, whether it became
organized along an axis (gradient), connected-blob structure, how its entropy
drifted, and self-recurrence (oscillation). Plus the strongest correlations
between channels. Use these to judge your prediction.

HOW TO REPLY, every turn, in this order:
1. HYPOTHESIS: one or two sentences — the causal idea you are testing.
2. A law in a fenced ```nomos code block. Keep it small; change ONE thing at a
   time from the previous law so you can attribute the effect.
3. PREDICTION: one sentence — what you expect the observer to report, concretely.
Do not optimize a number. Seek understanding: persistence, structure, self-
organization, memory, life-like dynamics, surprises.
"""

FIRST_TURN = ("Begin. Propose your first hypothesis, law, and prediction. "
              "A good first world is one that stays alive without dying out or boiling.")

_LAW_RE = re.compile(r"```(?:nomos|text)?\s*\n(.*?)```", re.DOTALL)


def extract_law(reply: str) -> str | None:
    m = _LAW_RE.search(reply)
    if m:
        return m.group(1).strip()
    # fallback: a bare block that clearly contains a law
    if "channel" in reply and "rule" in reply:
        start = reply.index("channel")
        return reply[start:].strip()
    return None


def _follow_up(report_digest: str) -> str:
    return (f"Observation from your law:\n{report_digest}\n\n"
            "Was your prediction right? State what you concluded in one or two "
            "sentences, then propose the next hypothesis, law, and prediction "
            "(change one thing at a time).")


def _bad_law_msg(err: str) -> str:
    return (f"That law did not run: {err}\n"
            "Fix it and resend a corrected law in a ```nomos block (same reply "
            "format: hypothesis, law, prediction).")


class Journal:
    def __init__(self, path: Path, model: str, params: dict):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Auto lab journal — {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
                  f"Physicist model: `{model}`  |  world: {params}\n")
        path.write_text(header, encoding="utf-8")

    def append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)


def run_loop(client, iterations: int, size: int, steps: int, seed: int,
             archive_path: str | None, journal_path: Path,
             temperature: float = 1.0) -> Journal:
    params = {"size": size, "steps": steps, "seed": seed}
    journal = Journal(journal_path, getattr(client, "model", "?"), params)
    messages = [{"role": "system", "content": DSL_SPEC},
                {"role": "user", "content": FIRST_TURN}]

    completed = 0
    while completed < iterations:
        reply = client.chat(messages, temperature=temperature)
        messages.append({"role": "assistant", "content": reply})
        law = extract_law(reply)

        if law is None:
            note = "No parseable law in the reply; asking again."
            print(f"[iter {completed + 1}] {note}", file=sys.stderr)
            journal.append(f"\n---\n\n## Attempt {completed + 1} (no law found)\n\n{reply}\n")
            messages.append({"role": "user",
                             "content": "I could not find a law in a ```nomos block. "
                                        "Please resend with the law fenced."})
            continue

        try:
            report = simulate(law, size=size, steps=steps, seed=seed)
        except NomosError as e:
            print(f"[iter {completed + 1}] invalid law: {e}", file=sys.stderr)
            journal.append(f"\n---\n\n## Attempt {completed + 1} (law error)\n\n"
                           f"{reply}\n\n**Engine error:** {e}\n")
            messages.append({"role": "user", "content": _bad_law_msg(str(e))})
            continue

        completed += 1
        dig = digest(report)
        run_id = None
        if archive_path:
            run_id = archive.record(archive_path, report, law, law_name=f"auto-{completed}")
        print(f"[iter {completed}/{iterations}] {report['verdict']}"
              + (f"  (archive #{run_id})" if run_id else ""), file=sys.stderr)
        journal.append(
            f"\n---\n\n## Experiment {completed}\n\n{reply}\n\n"
            f"```nomos\n{law}\n```\n\n**Observer:**\n```\n{dig}\n```\n"
            + (f"\n_archived as run #{run_id}._\n" if run_id else "")
        )
        messages.append({"role": "user", "content": _follow_up(dig)})

    return journal


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nomos.scientist",
                                description="Automated LLM scientist loop over artificial physics.")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--archive", default="runs.db", help="SQLite archive path (or 'none')")
    p.add_argument("--journal", default=None, help="journal markdown path (default journal/auto-<ts>.md)")
    p.add_argument("--model", default=None, help="override NOMOS_LLM_MODEL")
    p.add_argument("--base-url", default=None, dest="base_url")
    p.add_argument("--dry-run", action="store_true",
                   help="use a built-in mock physicist (no API calls)")
    args = p.parse_args(argv)

    journal_path = Path(args.journal) if args.journal else Path(
        f"journal/auto-{dt.datetime.now():%Y%m%d-%H%M%S}.md")
    archive_path = None if args.archive.lower() == "none" else args.archive

    if args.dry_run:
        mock = [
            "HYPOTHESIS: a mild majority rule self-organizes into stable patches.\n"
            "```nomos\nchannel a: init bernoulli(0.5)\n"
            "rule flip: when 1 set a = where(sum8(a) >= 4, 1, 0)\n```\n"
            "PREDICTION: activity falls to near zero as patches freeze.",
        ] * max(args.iterations, 1)
        client = make_client(mock=mock)
    else:
        try:
            client = make_client(model=args.model, base_url=args.base_url)
        except LLMError as e:
            print(f"nomos.scientist: {e}", file=sys.stderr)
            return 1

    try:
        journal = run_loop(client, args.iterations, args.size, args.steps, args.seed,
                           archive_path, journal_path, temperature=args.temperature)
    except LLMError as e:
        print(f"nomos.scientist: LLM error: {e}", file=sys.stderr)
        return 1
    print(f"journal: {journal.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
