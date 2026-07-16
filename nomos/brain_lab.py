"""LLM-driven brain evolution: the model itself writes the RULES between runs.

Where nomos/evolve.py mutates a few numeric weights, here the LLM is the
mutation operator — it proposes whole brain laws (new channels, new rules, new
mechanisms), we measure how well each plays Snake, and feed the score back so it
can iterate. This is the project's thesis made concrete: an LLM searching the
space of cellular-automaton laws, guided by a real external objective, keeping a
lab journal of what it tried. Selection is explicit via a leaderboard of the
best brains carried in the prompt.

Deterministic given the game side (fixed seeds/cadence); the LLM is the only
stochastic part, exactly as in nomos/scientist.py.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from nomos.dsl import NomosError, parse_program
from nomos.embodied import play, require_embodied
from nomos.engine import World
from nomos.llm import LLMError, make_client
from nomos.render import save_gif, split_images
from nomos.scientist import Journal, extract_hypothesis, extract_law

BRAIN_SPEC = """\
You are designing the BRAIN of a small creature that plays Snake. The brain is a
cellular automaton: a toroidal grid of named float channels that update every
physics step under LAWS you write. You do not steer the snake directly — you
write the physics, and the grid's own activity is read out as the snake's moves.

LAW SYNTAX (a .nomos program):
    channel <name>[: init <expr>]        # a per-cell float field, default 0
    rule <name>:
        when <expr>                      # nonzero = applies
        set <chan> = <expr>              # overwrite where the condition holds
        add <chan> = <expr>              # accumulate where it holds
Rules run in declaration order every step, reading a SNAPSHOT from the start of
the step. Values auto-regularize (NaN->0, clipped +/-1e6); a brain never crashes.

EXPRESSIONS: or and not, == != < <= > >=, + - * / %, unary -.
Functions: neighbourhood (centre excluded, wraps) sum8 avg8 max8 min8 (Moore),
sum4 avg4 min4 max4 (von Neumann); abs sign floor round sqrt exp log sin cos
pi(); min(a,b) max(a,b) clamp(x,lo,hi) where(cond,a,b); posx() posy() time();
rand() bernoulli(p) uniform(a,b) normal(mu,sigma).

THE EMBODIMENT — your interface to the game (all on the SAME grid, egocentric to
the snake's head, which sits at the grid centre):
- `food` (SENSOR, you MUST declare it, read-only): a bump pointing toward the
  food; BRIGHTER when the food is near, dimmer when far. The harness rewrites it
  every step, so treat it as a given input field, not scratch space.
- `danger` (SENSOR, optional): if you declare it, the harness fills it with the
  snake's OWN BODY, egocentric to the head — so you can steer away from it.
- `motor` (MOTOR, you MUST declare it): after your rules run, the harness looks
  at where the NON-NEGATIVE mass of `motor` sits relative to the grid centre and
  moves the snake that way (up/down/left/right). Silent motor = keep heading.
  So: push motor mass toward where you want to go, away from where you don't.
The game ticks once every several physics steps, giving the grid time to settle.
Eating food scores a point and grows the snake; running into your own body ends
the run. The board wraps — there are no walls, only the food and your body.

YOUR GOAL: write a brain law that makes the snake score as HIGH as possible
(eat a lot, live long). You will be told each brain's score and shown a
leaderboard of the best so far; build on what works, and try genuinely new
mechanisms when progress stalls — extra channels for memory or planning, a
refractory trail to avoid your body, lateral inhibition to sharpen the target.

HOW TO REPLY, every turn, in this order:
1. HYPOTHESIS: one or two sentences — the idea about brain design you are testing.
2. A brain law in a fenced ```nomos block (must declare `food` and `motor`).
3. PREDICTION: one sentence — what score/behavior you expect and why.
"""

FIRST_TURN = ("Begin. Design your first Snake brain. A simple working start is to "
              "let the food signal flow into the motor field so its mass sits toward "
              "the food; then improve from there.")


def _leaderboard_block(board: list[dict], top: int = 5) -> str:
    if not board:
        return ""
    lines = ["LEADERBOARD — best brains so far (highest score first). Beat them:"]
    for i, e in enumerate(sorted(board, key=lambda x: x["score"], reverse=True)[:top]):
        lines.append(f"\n#{i + 1}  score={e['score']:.2f}  survival={e['survival']:.0f}"
                     + (f"  ({e['hypothesis']})" if e["hypothesis"] else "")
                     + f"\n```nomos\n{e['law']}\n```")
    return "\n".join(lines) + "\n\n"


def _score_line(res: dict) -> str:
    per = ", ".join(f"{s}" for s in res["scores"])
    return (f"score={res['score']:.2f} (per seed: {per})  "
            f"survival={res['survival']:.0f} ticks")


def evaluate(law_text: str, seeds, world_size, game_size, moves, ticks) -> dict:
    """Average game performance of a brain over several boards."""
    scores, survivals = [], []
    for s in seeds:
        r = play(law_text, world_size=world_size, game_size=game_size, seed=s,
                 moves=moves, ticks_per_move=ticks)
        scores.append(r["score"])
        survivals.append(r["steps_alive"])
    return {"score": sum(scores) / len(scores), "scores": scores,
            "survival": sum(survivals) / len(survivals)}


def _render_best(law_text, gif_path: Path, world_size, game_size, moves, ticks,
                 show_seed=0, fps=6, scale=6) -> bool:
    r = play(law_text, world_size=world_size, game_size=game_size, seed=show_seed,
             moves=moves, ticks_per_move=ticks, collect_frames=True)
    if not r["life_frames"]:
        return False
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(split_images(r["life_frames"], r["game_frames"], scale=scale),
             str(gif_path), fps=fps)
    return True


def run_brain_lab(client, iterations: int, journal_path: Path, eval_seeds=(0, 1, 2),
                  world_size: int = 48, game_size: int = 16, moves: int = 300,
                  ticks: int = 8, gif_dir: Path | None = None, show_seed: int = 0,
                  temperature: float = 1.0) -> dict:
    params = {"world": world_size, "board": game_size, "moves": moves,
              "ticks": ticks, "eval_seeds": list(eval_seeds)}
    journal = Journal(journal_path, getattr(client, "model", "?"), params)
    messages = [{"role": "system", "content": BRAIN_SPEC},
                {"role": "user", "content": FIRST_TURN}]

    board: list[dict] = []
    best = None
    completed = 0
    while completed < iterations:
        reply = client.chat(messages, temperature=temperature)
        messages.append({"role": "assistant", "content": reply})
        law = extract_law(reply)

        if law is None:
            journal.append(f"\n---\n\n## Attempt {completed + 1} (no law found)\n\n{reply}\n")
            messages.append({"role": "user", "content": "No ```nomos block found — "
                             "resend the brain law fenced."})
            continue
        try:
            program = parse_program(law)
            require_embodied(program)
            World(program, world_size, eval_seeds[0])   # trigger static validation early
        except NomosError as e:
            journal.append(f"\n---\n\n## Attempt {completed + 1} (invalid)\n\n{reply}\n\n"
                           f"**Error:** {e}\n")
            messages.append({"role": "user", "content": f"That brain did not load: {e}\n"
                             "Fix it and resend (must declare `food` and `motor`)."})
            continue

        completed += 1
        res = evaluate(law, eval_seeds, world_size, game_size, moves, ticks)
        hyp = extract_hypothesis(reply)
        entry = {"iter": completed, "law": law, "hypothesis": hyp, **res}
        board.append(entry)
        is_best = best is None or res["score"] > best["score"]
        if is_best:
            best = entry

        gif_link = None
        if gif_dir is not None:
            gif_path = gif_dir / f"brain{completed:02d}-score{res['score']:.0f}.gif"
            try:
                if _render_best(law, gif_path, world_size, game_size, moves, ticks,
                                show_seed=show_seed):
                    try:
                        gif_link = gif_path.relative_to(journal_path.parent).as_posix()
                    except ValueError:
                        gif_link = gif_path.as_posix()
            except Exception as e:
                print(f"[brain {completed}] gif failed: {e}", file=sys.stderr)

        print(f"[brain {completed}/{iterations}] {_score_line(res)}"
              + ("  <- best" if is_best else ""), file=sys.stderr)
        journal.append(
            f"\n---\n\n## Brain {completed}{'  (best so far)' if is_best else ''}\n\n"
            f"{reply}\n\n```nomos\n{law}\n```\n\n**Game:** {_score_line(res)}\n"
            + (f"\n![brain {completed}]({gif_link})\n" if gif_link else ""))
        messages.append({"role": "user", "content":
                         f"Your brain scored: {_score_line(res)}.\n\n"
                         + _leaderboard_block(board)
                         + "Propose the next brain — beat the leaderboard. State what "
                         "you concluded, then hypothesis, law, prediction."})

    return {"best": best, "leaderboard": sorted(board, key=lambda x: x["score"],
                                                reverse=True)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="nomos.brain_lab",
                                description="LLM designs Snake brains across runs.")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--eval-seeds", type=int, default=3, dest="eval_seeds")
    p.add_argument("--world-size", type=int, default=48, dest="world_size")
    p.add_argument("--game-size", type=int, default=16, dest="game_size")
    p.add_argument("--moves", type=int, default=300)
    p.add_argument("--ticks", type=int, default=8)
    p.add_argument("--show-seed", type=int, default=0, dest="show_seed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--model", default=None)
    p.add_argument("--journal", default=None)
    p.add_argument("--gif-dir", default=None, dest="gif_dir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    journal_path = Path(args.journal) if args.journal else Path(
        f"journal/brain-{dt.datetime.now():%Y%m%d-%H%M%S}.md")
    if args.gif_dir is None:
        gif_dir = journal_path.parent / f"{journal_path.stem}-gifs"
    elif args.gif_dir.lower() == "none":
        gif_dir = None
    else:
        gif_dir = Path(args.gif_dir)

    if args.dry_run:
        brain = ("HYPOTHESIS: food into motor.\n```nomos\nchannel food\n"
                 "channel motor: init 0\nrule think:\n  when 1\n"
                 "  set motor = 0.5*avg8(motor) + food\n```\nPREDICTION: eats.")
        client = make_client(mock=[brain] * max(args.iterations, 1))
    else:
        try:
            client = make_client(model=args.model)
        except LLMError as e:
            print(f"nomos.brain_lab: {e}", file=sys.stderr)
            return 1

    try:
        out = run_brain_lab(client, args.iterations, journal_path,
                            eval_seeds=tuple(range(args.eval_seeds)),
                            world_size=args.world_size, game_size=args.game_size,
                            moves=args.moves, ticks=args.ticks, gif_dir=gif_dir,
                            show_seed=args.show_seed, temperature=args.temperature)
    except LLMError as e:
        print(f"nomos.brain_lab: LLM error: {e}", file=sys.stderr)
        return 1
    b = out["best"]
    if b:
        print(f"best brain: score={b['score']:.2f} (iter {b['iter']})")
    print(f"journal: {journal_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
