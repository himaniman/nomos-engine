"""Session records — what gives a score a meaning.

A brain that eats 3 berries is neither good nor bad until you know what a human does
on the same seed, and what a coin-flipping walker does. docs/cortex-design.md makes
this a precondition rather than a nicety: at lambda = 1 the LLM gets ONE life per
turn, so that life had better return a profile it can reason about.

Every session — human, baseline policy or brain — appends one JSON line to
`out/cortex/sessions.jsonl`. Same schema for all of them, so they are directly
comparable, and `summarise()` prints the table.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from cortex.world import FOOD_NAMES, World

PATH = os.path.join("out", "cortex", "sessions.jsonl")


def record(w: World, who: str, note: str = "", path: str = PATH) -> dict:
    """Summarise a finished life and append it to the log. Returns the record."""
    a = w.agent
    fed = {FOOD_NAMES[k]: n for k, n in enumerate(a.ate) if n}
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "who": who, "note": note, "seed": w.seed, "size": w.cfg.size,
           "ticks": w.tick, "alive": a.alive,
           "energy": round(a.energy, 2),
           "gained": round(sum(float(n) * _energy(k) for k, n in enumerate(a.ate)), 1),
           "ate": fed, "meals": int(sum(a.ate)),
           "distance": round(w.distance, 1),
           "deeds": dict(w.deeds)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _energy(kind: int) -> float:
    from cortex.world import FOOD_ENERGY
    return float(FOOD_ENERGY[kind])


def load(path: str = PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(path: str = PATH) -> str:
    """One row per player: how long they lived and how well they ate, averaged over
    their sessions. This is the table the brain has to earn a line in."""
    rows = load(path)
    if not rows:
        return "no sessions recorded yet"
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["who"]].append(r)
    out = [f"{'who':<14}{'runs':>5}{'ticks':>9}{'meals':>7}{'gained':>8}"
           f"{'cells':>8}  diet"]
    for who, rs in sorted(by.items(), key=lambda kv: -_mean(kv[1], "ticks")):
        diet: dict[str, int] = defaultdict(int)
        for r in rs:
            for k, n in r["ate"].items():
                diet[k] += n
        out.append(f"{who:<14}{len(rs):>5}{_mean(rs, 'ticks'):>9.0f}"
                   f"{_mean(rs, 'meals'):>7.1f}{_mean(rs, 'gained'):>8.1f}"
                   f"{_mean(rs, 'distance'):>8.0f}  "
                   + " ".join(f"{k[:4]}{v}" for k, v in sorted(diet.items())))
    return "\n".join(out)


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))
