"""A night in the lab: the model designs STRUCTURE, the machine tunes CONSTANTS, a control
arm keeps both honest — hours long, unattended, on a slow local model.

    .\\.venv\\Scripts\\python.exe -m cortex.night_lab --hours 6 --dir out/cortex/night/run1
    .\\.venv\\Scripts\\python.exe -m cortex.night_lab --mock --hours 0.05 ...   # dry run

`cortex/law_lab.py` asks a model for a whole law each turn and scores it once. That fits
a fast API; it fits a local CPU model badly, and two lessons of this project say it
fits the problem badly too: the snake's best brain came from an LLM's ARCHITECTURE with
its constants tuned by evolution (either alone lost), and in the law lab rng's train win
was luck on holdout. The measurements that shaped this loop (2026-09-23, qwen3.6-35b-a3b
on the llm VM): a design turn is ~40-95 s without thinking and ~10 min with it, while a
law lives 16 worlds x 8000 ticks in ~16 s on this machine. The MODEL is the scarce
resource, so every one of its ideas is worth dozens of cheap evaluations.

So three things run at once:
  * the MODEL, in a background thread, one request at a time (the server has one slot,
    shared — before each call the loop waits while somebody else is being served). Its
    jobs: DESIGN a new structure (every `deep_every`-th turn with thinking on), rewrite
    its THEORY (short notes carried in every prompt instead of a growing chat history,
    which a 10 tok/s model could not afford), and OBSERVE — it is shown a picture of one
    life of the current champion on the world it does worst on, with the track drawn,
    and says what the agent does wrong. The prompt is rebuilt each turn around a
    constant system text, so the server's prefix cache serves most of it.
  * the LLM ARM's tuner: every structure the model writes becomes an ISLAND; random
    jitter of its constants, (1+1) on the training worlds, `tune` evaluations first,
    then round-robin over the best three. A law that only changes numbers of an existing
    island is caught by its structure key and told so — constants are the machine's job.
  * the CONTROL ARM: the same (1+1) jitter from the best hand law (`atlas_ghost`), the
    same number of evaluations as the LLM arm's tuner (the two alternate). The question
    the night answers is whether a model's STRUCTURE beats the best known structure
    with the same tuning budget, on worlds nobody selected on (the HOLDOUT).

Everything lands in the run directory: `journal.md` (human), `state.json` (resume:
running the same command again continues where it stopped), `status.json` (heartbeat),
`observe/*.png`, `reasoning/*.txt`, and at the end `summary.md`, the champion's law and
a GIF of its life.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from cortex import atlas as A
from cortex import atlas_law as LAW
from cortex import atlas_lineage as L
from cortex import law_lab as LL
from nomos.dsl import NomosError
from nomos.llm import LLMError, embed, make_client
from nomos.scientist import extract_law

TRAPS = """

SYNTAX TRAPS (the parser rejects these): there is NO `let`, NO variables, NO
`if/then/else`, NO `pass`, NO power operator — inline every expression, use
where(cond, a, b) for a choice, and a helper CHANNEL (with its own rule) for anything
you want to name or remember. Each rule has exactly one `when` line and one or more
`set`/`add` lines. Comments start with #. Declare every channel you read.
Several `set` lines on ONE channel do NOT chain: every line reads the snapshot from the
start of the step and only the LAST matching set counts, so `set c = x` followed by
`set c = 0.9 * max8(c)` throws the first line away. To build a quantity in stages, give
each stage its own channel and its own rule (the next rule sees it on the next pass).
min() and max() take two or more arguments."""

NIGHT = """

THIS IS A LONG NIGHT OF WORK. You design STRUCTURE; a machine tunes the constants of
every law you write (random search on the training worlds), so a law that only changes
numbers wastes your turn. Change what the rule DOES: a new channel, a new rule, a
different way the wave spreads, remembers, forgets or commits. Keep every channel
bounded (a quantity that only grows ends clipped at 1e6 and blinds the wave). Between
design turns you will sometimes be asked to rewrite your THEORY (short notes you keep
for yourself), or to look at a picture of one life and say what went wrong.

KNOWN CLIFFS (measured, each scored ~0-5 meals): the wave must be a max() of DISCOUNTED
terms — any factor >= 1 on a neighbour term, or a SUM of neighbour terms (0.95*max4(v) +
0.9*max8(v)), saturates the whole map at the cap, every neighbour looks equal and the
agent stops. A value that is flat (all zero) also stops it. After each law you are told
how its value field looked on a small test map with one source.

Be brief: at most one short comment line per rule, the whole reply under 70 lines.

When you design, reply in exactly this shape:
HYPOTHESIS: two or three sentences — the mechanism and why it makes the agent eat more.
PREDICTION: one number, the mean meals per life you expect on the training worlds.
```nomos
the whole law
```"""

DESIGN_TURN = """\
THEORY (your own notes):
{theory}
{observer}
BOARD — laws after the machine tuned their constants. train = the {n_train} worlds
selection uses; holdout = {n_hold} other worlds nobody selects on (the honest number):
{board}
{feedback}
THE BEST LAW SO FAR ({best_name}):
```nomos
{best_law}
```
{second}{starving}{novelty}
Write a NEW STRUCTURE{deep}. HYPOTHESIS, PREDICTION, then the whole law."""

THEORY_TURN = """\
Your current THEORY:
{theory}

Evidence since you last wrote it (each: your idea -> what the tuned law ate on the
training worlds, vs the best law before it):
{evidence}

The board now:
{board}
{observer}
Rewrite the THEORY in at most 150 words: which mechanisms help this agent eat, which
hurt, why, and what is still untried. Plain text, no law, no code."""

OBSERVER = """\
You are the OBSERVER in an artificial-life lab. You look at a picture of one life of a
foraging agent and describe its BEHAVIOUR plainly, like a field biologist. The agent's
brain is a map-shaped cellular automaton: a rule (the law below) spreads a value wave
from what it smells and sees, and the agent steps toward the brightest neighbour cell.
You never write laws or code; you report what you see and name the one failing you
would fix first."""

OBSERVE_TURN = """\
This is ONE LIFE of the current best law ({name}) on training world {world}, the
world it does worst on. The picture: the world from above (grey = walls/rock, dark =
open ground, coloured dots = food still growing at the end). The agent's TRACK runs
from cyan (early) to red (late); white squares = meals eaten; green square = start;
{end}.

Numbers: {stats}

The law it runs:
```nomos
{law}
```

This is NOT a design turn. Say in at most five sentences what the agent does in this
life and where it fails (circling, pacing a wall, stuck in a room, ignoring visible
food, never exploring, ...), and name ONE mechanism in the law that would fix it.
Plain sentences only: no HYPOTHESIS, no law, no code."""


# --- a life with a track, and its picture ------------------------------------------------

def track_life(text: str, seed: int, ticks: int, world: str = "default",
               prog: dict | None = None, frames: list | None = None) -> dict:
    """One life of a law with its path recorded (every 4 ticks) and its meals marked.
    Same world and brain as `law_lab._life` — only the bookkeeping differs."""
    from cortex.world import World
    rule = LAW.LawRule(text, seed)
    w = World(seed=int(seed), cfg=L.WorldConfig(**L.WORLDS[world]))
    mind = A.AtlasMind(L.Program(prog or {}).config(), w.cfg, rule)
    mind.attach(w)
    a = w.agent
    pts, meals, eaten = [(a.y, a.x)], [], 0
    while a.alive and w.tick < ticks:
        mind.step(w)
        if w.tick % 4 == 0:
            pts.append((a.y, a.x))
        if w.deeds["eat"] > eaten:
            eaten = w.deeds["eat"]
            meals.append((a.y, a.x, w.tick))
        if frames is not None and w.tick % 12 == 0:
            frames.append(A.frame(w, mind.atlas))
    pts.append((a.y, a.x))
    cells = {(int(y), int(x)) for y, x in pts}
    tail = pts[-125:]                                   # the last ~500 ticks
    ty, tx = [p[0] for p in tail], [p[1] for p in tail]
    stats = {"meals": int(sum(a.ate)), "ticks": int(w.tick), "alive": bool(a.alive),
             "distance": round(float(w.distance), 1), "cells_visited": len(cells),
             "last500_box": f"{max(ty) - min(ty):.0f}x{max(tx) - min(tx):.0f}",
             "doors": w.deeds["door"], "breaks": w.deeds["break"],
             "idle_presses": w.deeds["idle"], "nightcaps": int(a.ate[3])}
    return {"world": w, "pts": pts, "meals": meals, "stats": stats}


def track_image(life: dict, max_px: int = 640) -> np.ndarray:
    from cortex.tiles import RGB
    from cortex.world import FOOD_RGB
    w, pts = life["world"], np.array(life["pts"])
    pad = 8
    y0, x0 = int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad
    h = int(pts[:, 0].max()) - y0 + pad + 1
    wd = int(pts[:, 1].max()) - x0 + pad + 1
    side = max(h, wd, 40)
    y0 -= (side - h) // 2
    x0 -= (side - wd) // 2
    img = (RGB[w.tiles.region(y0, x0, side, side)].astype(np.float64) * 0.55)
    for s in w.nearby_food(y0 + side / 2, x0 + side / 2, side * 0.75):
        yy, xx = int(s.y) - y0, int(s.x) - x0
        if 0 <= yy < side and 0 <= xx < side:
            img[yy, xx] = FOOD_RGB[s.kind]
    sc = max(1, max_px // side)
    img = np.repeat(np.repeat(img, sc, axis=0), sc, axis=1)
    n = len(pts)
    for i in range(n - 1):
        t = i / max(1, n - 2)
        col = np.array([255 * t, 200 * (1 - t) + 60 * t, 255 * (1 - t) + 60 * t])
        (ay, ax), (by, bx) = pts[i], pts[i + 1]
        for k in range(8):
            f = k / 8
            py = int(((ay + (by - ay) * f) - y0) * sc + sc / 2)
            px = int(((ax + (bx - ax) * f) - x0) * sc + sc / 2)
            if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                img[max(0, py - sc // 4):py + sc // 4 + 1,
                    max(0, px - sc // 4):px + sc // 4 + 1] = col

    def square(y, x, col, r):
        py, px = int((y - y0) * sc + sc / 2), int((x - x0) * sc + sc / 2)
        img[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = col

    r = max(2, sc)
    for my, mx, _ in life["meals"]:
        square(my, mx, (255, 255, 255), r)
    square(pts[0, 0], pts[0, 1], (40, 255, 60), r + 1)
    ey, ex = pts[-1]
    square(ey, ex, (255, 230, 40) if life["stats"]["alive"] else (255, 0, 0), r + 1)
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG")
    return b.getvalue()


# --- small helpers -----------------------------------------------------------------------

def overwritten(text: str) -> list[str]:
    """Channels a law `set`s more than once. Every set reads the snapshot and the
    last matching one wins, so all but the last line are thrown away — the model's most
    common silent mistake."""
    try:
        prog = LAW.LawRule(text).program
    except Exception:
        return []
    count: dict[str, int] = {}
    for r in prog.rules:
        for a in r.actions:
            if a[0] == "set":
                count[a[1]] = count.get(a[1], 0) + 1
    return [f"`{c}` is set {n} times" for c, n in count.items() if n > 1]


def diagnose(text: str, side: int = 25, thoughts: int = 6, passes: int = 6) -> str:
    """How a law's value field looks on an open test map with ONE source five cells
    east of the agent — a cheap mirror for the model, which cannot see its field: is it
    saturated at the cap, flat, and does the agent's best neighbour point at the
    source?"""
    try:
        rule = LAW.LawRule(text)
        state = np.zeros((A.N_CH + rule.hidden, side, side))
        state[A.FREE] = 1.0
        c = side // 2
        src = np.zeros((side, side))
        src[c, c + 5] = 1.0
        for _ in range(thoughts):
            rule.step(state, src, np.ones((side, side)), passes)
        v = state[A.VALUE]
        cap = float((v >= LAW.VALUE_CAP - 1e-6).mean())
        nb = {(dy, dx): v[c + dy, c + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if dy or dx}
        top = max(nb.values())
        best = [k for k, x in nb.items() if x >= top - 1e-9]
        if float(v.max()) <= 1e-9:
            verdict = "FLAT: all zero, the agent would stand still"
        elif cap > 0.5:
            verdict = f"SATURATED: {cap:.0%} of cells at the cap, every direction looks equal"
        elif len(best) > 3:
            verdict = f"no clear direction ({len(best)} neighbours tie)"
        elif (0, 1) in best:
            verdict = "healthy: the best neighbour points at the source"
        else:
            verdict = f"points AWAY from the source (best neighbour {best[0]})"
        return (f"value on the test map: max {float(v.max()):.2f}, mean "
                f"{float(v.mean()):.2f}, {cap:.0%} at the cap -> {verdict}")
    except Exception as e:
        return f"value on the test map: the law failed ({type(e).__name__}: {e})"


_COMMENT = re.compile(r"#.*")


def structure_key(text: str) -> str:
    """A law with its numbers, comments and spacing removed: two laws with one key
    differ only in constants."""
    body = []
    for line in text.splitlines():
        line = _COMMENT.sub("", line).strip()
        if line:
            body.append(re.sub(r"\s+", " ", LAW._NUM.sub("#", line)))
    return hashlib.sha1("\n".join(body).encode()).hexdigest()[:12]


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


_PRED = re.compile(r"PREDICTION\s*[:=]\s*~?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_HYP = re.compile(r"HYPOTHESIS\s*[:=]\s*(.+?)(?:\n\s*PREDICTION|```|$)", re.I | re.S)


def parse_design(reply: str) -> tuple[str, float | None]:
    m = _HYP.search(reply)
    hyp = (m.group(1) if m else reply.split("```")[0]).strip()
    p = _PRED.search(reply)
    return re.sub(r"\s+", " ", hyp)[:700] or "(no hypothesis given)", \
        (float(p.group(1)) if p else None)


class Job(threading.Thread):
    """One model job in the background. A daemon, so a hung request never keeps the
    process alive past the deadline."""

    def __init__(self, kind: str, fn):
        super().__init__(daemon=True)
        self.kind, self.fn = kind, fn
        self.result, self.error = None, None
        self.t0 = time.time()

    def run(self):
        try:
            self.result = self.fn()
        except Exception as e:                                  # surfaced by the loop
            self.error = f"{type(e).__name__}: {e}"


# --- the lab -----------------------------------------------------------------------------

class NightLab:
    def __init__(self, client, run: Path, hours: float = 6.0, train=range(16),
                 holdout=range(16, 32), ticks: int = 8000, world: str = "default",
                 jobs: int = 16, tune: int = 5, deep_every: int = 5, theory_every: int = 6,
                 observe_every: int = 8, holdout_every: float = 30.0, seed: int = 0,
                 use_embed: bool = True, grace: float = 25.0, final=(),
                 margin: float = 0.02):
        self.client, self.run = client, run
        self.hours, self.train, self.holdout = hours, list(train), list(holdout)
        # FINAL worlds: lived only once, at the end, by a handful of finalists. The model
        # sees holdout numbers, so holdout is weakly selected on; these nobody ever sees
        self.final = list(final)
        # a tuned child must beat its parent's train fitness by this fraction: the
        # probe night showed (1+1) jitter overfitting the training worlds (a structure
        # at holdout 23.1 as written fell to 16.9 after 13 accepted "improvements")
        self.margin = margin
        self.ticks, self.world, self.jobs, self.tune = ticks, world, jobs, tune
        self.deep_every, self.theory_every = deep_every, theory_every
        self.observe_every, self.holdout_every = observe_every, holdout_every
        self.use_embed, self.grace = use_embed, grace
        self.prog = L.Program({})
        self.rng = np.random.default_rng(seed)
        side = 2 * self.prog.values["radius"] + 1
        ref = Path(LAW.REFERENCE).read_text(encoding="utf-8").strip()
        self.system = LL.SPEC.format(ticks=ticks, side=side, passes=self.prog.values["subticks"],
                                     reference=ref) + TRAPS + NIGHT
        for sub in ("observe", "reasoning", "laws"):
            (run / sub).mkdir(parents=True, exist_ok=True)
        self.pool = None
        self.state_path = run / "state.json"
        if self.state_path.exists():
            self.s = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.s["resumed"] = self.s.get("resumed", 0) + 1
            self.rng = np.random.default_rng(seed + 1000 * self.s["resumed"])
        else:
            self.s = {"started": dt.datetime.now().isoformat(timespec="seconds"),
                      "elapsed": 0.0, "resumed": 0, "islands": [], "control": None,
                      "theory": "", "theories": [], "observations": [], "turns": [],
                      "feedback": "", "evals": {"llm": 0, "control": 0, "holdout": 0},
                      "timeline": [], "design_turns": 0, "since_theory": 0,
                      "since_observe": 0, "rr": 0, "arm": 0, "last_holdout": 0.0,
                      "observed_champion": None, "done": False, "llm_seconds": 0.0,
                      "waited": 0.0, "model": getattr(client, "model", "?")}
        self.t_start = time.time() - self.s["elapsed"]
        self.doing = "starting"

    # --- bookkeeping ---------------------------------------------------------------------
    def elapsed(self) -> float:
        return time.time() - self.t_start

    def log(self, text: str) -> None:
        with (self.run / "journal.md").open("a", encoding="utf-8") as fh:
            fh.write(text)

    def save(self) -> None:
        self.s["elapsed"] = self.elapsed()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s), encoding="utf-8")
        tmp.replace(self.state_path)

    def heartbeat(self) -> None:
        best = self.champion()
        c = self.s["control"]
        st = {"now": dt.datetime.now().isoformat(timespec="seconds"),
              "elapsed_h": round(self.elapsed() / 3600, 3), "hours": self.hours,
              "doing": self.doing, "design_turns": self.s["design_turns"],
              "islands": len(self.s["islands"]), "evals": self.s["evals"],
              "champion": best and {"id": best["id"], "train_meals": best["train"]["meals"],
                                    "holdout_meals": (best.get("holdout") or {}).get("meals")},
              "control": c and {"train_meals": c["train"]["meals"],
                                "holdout_meals": (c.get("holdout") or {}).get("meals")},
              "done": self.s["done"]}
        (self.run / "status.json").write_text(json.dumps(st, indent=1), encoding="utf-8")

    # --- evaluation ----------------------------------------------------------------------
    def _pool(self):
        if self.pool is None:
            from multiprocessing import Pool
            self.pool = Pool(self.jobs)
        return self.pool

    def evaluate(self, text: str, seeds, timeout: float = 900.0) -> dict | None:
        """Mean over lives; None if the law crashed or hung (the pool is rebuilt)."""
        args = [(text, s, self.ticks, self.world, self.prog.to_dict()) for s in seeds]
        try:
            if self.jobs <= 1:
                rows = [LL._life(a) for a in args]
            else:
                rows = self._pool().map_async(LL._life, args).get(timeout=timeout)
        except Exception as e:
            self.log(f"\n> evaluation failed ({type(e).__name__}: {str(e)[:200]}); "
                     "pool rebuilt\n")
            if self.pool:
                self.pool.terminate()
            self.pool = None
            return None
        out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["immortal"] = int(sum(r["immortal"] for r in rows))
        out["per_seed"] = [r["meals"] for r in rows]
        return out

    # --- the board -----------------------------------------------------------------------
    def born(self) -> list[dict]:
        return [i for i in self.s["islands"] if not i.get("given")]

    def champion(self) -> dict | None:
        """The model's best structure by train — None until one of them eats at least
        half as well as the ghost (a zero-meal law is not worth a look or a GIF)."""
        bar = 0.5 * self.island(1)["train"]["fitness"] if len(self.s["islands"]) > 1 else 0
        b = [i for i in self.born() if i["train"] and i["train"]["fitness"] >= bar]
        return max(b, key=lambda i: i["train"]["fitness"]) if b else None

    def ranked(self, islands=None) -> list[dict]:
        pool = [i for i in (islands if islands is not None else self.s["islands"]) if i["train"]]
        return sorted(pool, key=lambda i: -i["train"]["fitness"])

    def island(self, iid: int) -> dict:
        return next(i for i in self.s["islands"] if i["id"] == iid)

    @staticmethod
    def _hold(i: dict) -> str:
        h = i.get("holdout")
        if not h:
            return "holdout  -  "
        stale = "" if h.get("of") == digest(i["best"]) else "*"
        return f"holdout {h['meals']:4.1f}{stale}"

    def board(self, n: int = 6) -> str:
        lines = []
        for i in self.ranked(self.born())[:n]:
            lines.append(f"  #{i['id']:<3} train {i['train']['meals']:4.1f} meals  "
                         f"{self._hold(i)}  (tuned {i['evals']}x) | {i['hypothesis'][:110]}")
        for i in self.s["islands"]:
            if i.get("given"):
                lines.append(f"  #{i['id']:<3} train {i['train']['meals']:4.1f} meals  "
                             f"{self._hold(i)}  (hand-written) | {i['hypothesis'][:110]}")
        c = self.s["control"]
        if c:
            h = c.get("holdout")
            lines.append(f"  CONTROL: the ghost law, constants tuned by random search "
                         f"{c['evals']}x: train {c['train']['meals']:.1f} meals, holdout "
                         f"{h['meals']:.1f}" if h else
                         f"  CONTROL: the ghost law tuned {c['evals']}x: train "
                         f"{c['train']['meals']:.1f} meals")
        return "\n".join(lines)

    # --- startup -------------------------------------------------------------------------
    def seed_islands(self) -> None:
        if self.s["islands"]:
            return
        self.log(f"# night lab — {self.s['started']}\n\nModel `{self.s['model']}` | "
                 f"{self.hours} h | world {self.world} | train {self.train} | holdout "
                 f"{self.holdout} | {self.ticks} ticks/life | tune {self.tune} | deep "
                 f"every {self.deep_every} | theory every {self.theory_every} | observe "
                 f"every {self.observe_every}\n")
        for iid, (path, hyp) in enumerate((
                (LAW.REFERENCE, "the reference Bellman wave"),
                ("laws/atlas_ghost.nomos", "sources linger (ghost), erased under the agent"))):
            text = Path(path).read_text(encoding="utf-8")
            self.doing = f"anchor {path}"
            self.heartbeat()
            tr = self.evaluate(text, self.train)
            ho = self.evaluate(text, self.holdout)
            ho["of"] = digest(text)
            self.s["islands"].append({"id": iid, "given": True, "law": text, "best": text,
                                      "hypothesis": hyp, "train": tr, "holdout": ho,
                                      "first": tr, "evals": 0, "tune_left": 0,
                                      "key": structure_key(text), "turn": 0})
            self.log(f"\n- anchor #{iid} `{path}`: train {tr['meals']:.1f} meals "
                     f"{tr['per_seed']}, holdout {ho['meals']:.1f}\n")
        g = self.s["islands"][1]
        self.s["control"] = {"best": g["best"], "train": g["train"], "holdout": g["holdout"],
                             "evals": 0, "accepted": 0}
        self.save()

    # --- the machine's side: tuning ------------------------------------------------------
    def _child(self, text: str) -> str:
        for _ in range(8):
            c = LAW.jitter(text, self.rng)
            if c != text:
                return c
        return LAW.jitter(text, self.rng, prob=1.0)

    def tune_step(self) -> bool:
        """One tuning evaluation for whichever arm is behind, so their budgets stay
        equal (the control may lead by three while the model has no structure yet).
        False = nothing to do."""
        born = [i for i in self.born() if i["train"]]
        c = self.s["control"]
        tuned = sum(i["evals"] for i in born)
        if born and tuned <= c["evals"]:
            arm = "llm"
        elif c["evals"] < tuned + 3:
            arm = "control"
        elif born:
            arm = "llm"
        else:
            return False
        if arm == "control":
            child = self._child(c["best"])
            self.doing = "control arm: tuning"
            self.heartbeat()
            r = self.evaluate(child, self.train)
            c["evals"] += 1
            self.s["evals"]["control"] += 1
            if r and r["fitness"] > c["train"]["fitness"] * (1 + self.margin):
                c.update(best=child, train=r)
                c["accepted"] += 1
            return True
        fresh = [i for i in born if i["tune_left"] > 0]
        if fresh:
            isl = max(fresh, key=lambda i: i["train"]["fitness"])
        else:
            top = self.ranked(born)[:3]
            isl = top[self.s["rr"] % len(top)]
            self.s["rr"] += 1
        child = self._child(isl["best"])
        self.doing = f"llm arm: tuning #{isl['id']}"
        self.heartbeat()
        r = self.evaluate(child, self.train)
        isl["evals"] += 1
        isl["tune_left"] = max(0, isl["tune_left"] - 1)
        self.s["evals"]["llm"] += 1
        if r and r["fitness"] > isl["train"]["fitness"] * (1 + self.margin):
            isl.update(best=child, train=r)
            isl["accepted"] = isl.get("accepted", 0) + 1
        return True

    def holdout_check(self, final: bool = False) -> None:
        todo = self.ranked(self.born())[:5 if final else 3]
        for isl in todo:
            if (isl.get("holdout") or {}).get("of") != digest(isl["best"]):
                self.doing = f"holdout #{isl['id']}"
                self.heartbeat()
                h = self.evaluate(isl["best"], self.holdout)
                if h:
                    h["of"] = digest(isl["best"])
                    isl["holdout"] = h
                self.s["evals"]["holdout"] += 1
        c = self.s["control"]
        if (c.get("holdout") or {}).get("of") != digest(c["best"]):
            self.doing = "holdout control"
            h = self.evaluate(c["best"], self.holdout)
            if h:
                h["of"] = digest(c["best"])
                c["holdout"] = h
            self.s["evals"]["holdout"] += 1
        best = self.champion()
        self.s["timeline"].append({
            "h": round(self.elapsed() / 3600, 2), "design_turns": self.s["design_turns"],
            "champion": best and best["id"],
            "champ_train": best and best["train"]["meals"],
            "champ_hold": best and (best.get("holdout") or {}).get("meals"),
            "control_train": c["train"]["meals"],
            "control_hold": (c.get("holdout") or {}).get("meals"),
            "evals": dict(self.s["evals"])})
        t = self.s["timeline"][-1]
        self.log(f"\n### Holdout checkpoint at {t['h']:.2f} h\n\n```\n{self.board(8)}\n```\n")
        self.s["last_holdout"] = self.elapsed()
        self.save()

    # --- the model's side ----------------------------------------------------------------
    def _chat(self, messages, **kw) -> str:
        waited = 0.0
        busy = getattr(self.client, "busy", None)
        while busy and busy() and waited < 900:
            time.sleep(15)
            waited += 15
        self.s["waited"] += waited
        reply = self.client.chat(messages, **kw)
        last = getattr(self.client, "last", {}) or {}
        self.s["llm_seconds"] += float(last.get("seconds", 0.0))
        return reply

    def _observer_text(self) -> str:
        obs = self.s["observations"]
        if not obs:
            return ""
        o = obs[-1]
        return (f"\nOBSERVER (a look at one life of #{o['island']} on world {o['world']}):\n"
                f"{o['text'][:700]}\n")

    def design_messages(self, deep: bool) -> list[dict]:
        born = self.ranked(self.born())
        allr = self.ranked()
        best = allr[0]
        second = ""
        runner = next((i for i in allr[1:] if not i.get("given")), None) \
            if best.get("given") else next((i for i in allr[1:]), None)
        if runner is not None and runner["train"]["meals"] > 0:
            label = "hand-written" if runner.get("given") else "yours, tuned"
            second = (f"\nRUNNER-UP (#{runner['id']}, {label}):\n```nomos\n"
                      f"{runner['best']}\n```\n")
        per = best["train"]["per_seed"]
        weak = [f"world {s} ({m})" for s, m in zip(self.train, per) if m < 5]
        starving = (f"\nWorlds where the best law starves (meals < 5): {', '.join(weak)}.\n"
                    if weak else "")
        novelty = self.s.get("novelty", "")
        user = DESIGN_TURN.format(
            theory=self.s["theory"] or "(none yet — you will be asked to write it)",
            observer=self._observer_text(), n_train=len(self.train),
            n_hold=len(self.holdout), board=self.board(),
            feedback=("\nYOUR LAST PROPOSAL: " + self.s["feedback"] + "\n")
            if self.s["feedback"] else "",
            best_name=f"#{best['id']}, yours, tuned" if not best.get("given")
            else f"#{best['id']}, hand-written",
            best_law=best["best"].strip(), second=second, starving=starving,
            novelty=("\n" + novelty + "\n") if novelty else "",
            deep=" (this is a DEEP turn: think it through, aim for a bold mechanism)"
            if deep else "")
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": user}]

    def design_job(self, deep: bool):
        messages = self.design_messages(deep)

        def fn():
            out = {"deep": deep, "attempts": [], "reasoning": ""}
            msgs = list(messages)
            t0 = time.time()
            for attempt in range(3):
                think = deep and attempt == 0
                try:
                    reply = self._chat(msgs, temperature=0.6 if think else 0.7,
                                       max_tokens=9000 if think else 3000, thinking=think)
                except LLMError as e:
                    if think and "empty" in str(e):     # thinking ate the budget
                        reply = self._chat(msgs, temperature=0.7, max_tokens=3000,
                                           thinking=False)
                    else:
                        raise
                last = getattr(self.client, "last", {}) or {}
                if last.get("reasoning"):
                    out["reasoning"] += last["reasoning"] + "\n\n"
                out.setdefault("replies", []).append(reply)
                law = extract_law(reply)
                if law is None:
                    err = "no ```nomos block found"
                else:
                    try:
                        LAW.check(law)
                        d = diagnose(law)
                        if attempt < 2 and "healthy" not in d:
                            ow = overwritten(law)
                            err = (f"it loads, but it cannot steer — {d}. The agent would "
                                   "not move toward anything (see KNOWN CLIFFS)")
                            if ow:
                                err += (". Likely cause: " + ", ".join(ow) + " — only the "
                                        "LAST set of a channel counts, the earlier lines are "
                                        "thrown away. Give each stage its own channel and "
                                        "rule, or fold them into ONE expression")
                        else:
                            out.update(reply=reply, law=law, seconds=time.time() - t0)
                            return out
                    except (NomosError, Exception) as e:
                        err = f"{type(e).__name__}: {e}"
                out["attempts"].append({"reply": reply, "error": err})
                msgs = msgs + [{"role": "assistant", "content": reply},
                               {"role": "user", "content":
                                f"That law was sent back: {err}\nFix exactly that and resend "
                                "the WHOLE law in one ```nomos block (HYPOTHESIS and "
                                "PREDICTION lines may stay as they were)."}]
            out.update(reply=None, law=None, seconds=time.time() - t0)
            return out
        return Job("deep" if deep else "design", fn)

    def theory_job(self):
        ev = []
        for t in self.s["turns"][-(self.theory_every + 4):]:
            if t.get("island") is not None and t.get("result"):
                ev.append(f"  #{t['island']}: {t['hypothesis'][:160]} -> {t['result']}")
            elif t.get("error"):
                ev.append(f"  (turn {t['turn']}: the law never loaded: {t['error'][:100]})")
        user = THEORY_TURN.format(theory=self.s["theory"] or "(none yet)",
                                  evidence="\n".join(ev) or "  (nothing new)",
                                  board=self.board(), observer=self._observer_text())
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": user}]

        def fn():
            return {"text": self._chat(messages, temperature=0.6, max_tokens=700,
                                       thinking=False).strip()}
        return Job("theory", fn)

    def observe_job(self):
        best = self.champion() or self.island(1)
        per = best["train"]["per_seed"]
        k = int(np.argmin(per))
        seed, text = self.train[k], best["best"]
        name = f"#{best['id']}"
        n = len(self.s["observations"])
        path = self.run / "observe" / f"obs{n:02d}_island{best['id']}_world{seed}.png"

        def fn():
            life = track_life(text, seed, self.ticks, self.world, self.prog.to_dict())
            png = png_bytes(track_image(life))
            path.write_bytes(png)
            st = life["stats"]
            end = ("yellow square = where it stood when time ran out" if st["alive"]
                   else "red square = where it starved")
            stats = ", ".join(f"{k} {v}" for k, v in st.items())
            prompt = OBSERVE_TURN.format(name=name, world=seed, end=end, stats=stats,
                                         law=text.strip())
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {
                           "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]
            reply = self._chat([{"role": "system", "content": OBSERVER},
                                {"role": "user", "content": content}],
                               temperature=0.5, max_tokens=500, thinking=False)
            note = re.sub(r"```.*?(```|$)", "", reply, flags=re.S).strip()
            return {"text": note[:1200], "island": best["id"], "world": seed,
                    "image": str(path.relative_to(self.run)), "stats": st}
        return Job("observe", fn)

    def next_job(self):
        champ = self.champion()
        new_champ = champ is not None and champ["id"] != self.s["observed_champion"]
        if self.s["since_theory"] >= self.theory_every:
            return self.theory_job()
        if self.s["since_observe"] >= self.observe_every or (
                new_champ and self.s["since_observe"] >= self.observe_every // 2):
            return self.observe_job()
        deep = self.deep_every > 0 and self.s["design_turns"] % self.deep_every == \
            self.deep_every - 1
        return self.design_job(deep)

    # --- handling what the model did -----------------------------------------------------
    def handle(self, job: Job) -> None:
        if job.error:
            self.log(f"\n> model job `{job.kind}` failed: {job.error[:300]}\n")
            self.s["turns"].append({"turn": len(self.s["turns"]), "kind": job.kind,
                                    "error": job.error[:300], "h": self.elapsed() / 3600})
            time.sleep(30)
            return
        r = job.result
        if job.kind == "theory":
            self.s["theory"] = r["text"][:1500]
            self.s["theories"].append({"h": round(self.elapsed() / 3600, 2),
                                       "turn": self.s["design_turns"], "text": r["text"]})
            self.s["since_theory"] = 0
            self.log(f"\n---\n\n## Theory (after {self.s['design_turns']} design turns)\n\n"
                     f"{r['text']}\n")
            return
        if job.kind == "observe":
            self.s["observations"].append(r)
            self.s["since_observe"] = 0
            self.s["observed_champion"] = r["island"]
            self.log(f"\n---\n\n## Observer: #{r['island']} on world {r['world']}\n\n"
                     f"![]({r['image']})\n\n`{r['stats']}`\n\n{r['text']}\n")
            return
        self.handle_design(job, r)

    def handle_design(self, job: Job, r: dict) -> None:
        turn = self.s["design_turns"]
        self.s["design_turns"] += 1
        self.s["since_theory"] += 1
        self.s["since_observe"] += 1
        if r.get("reasoning"):
            (self.run / "reasoning" / f"turn{turn:03d}.txt").write_text(
                r["reasoning"], encoding="utf-8")
        rec = {"turn": turn, "kind": job.kind, "h": round(self.elapsed() / 3600, 3),
               "seconds": round(r.get("seconds", 0.0), 1), "repairs": len(r["attempts"])}
        tag = " (DEEP)" if r["deep"] else ""
        if r["law"] is None:
            err = r["attempts"][-1]["error"] if r["attempts"] else "?"
            rec["error"] = err
            self.s["turns"].append(rec)
            self.s["feedback"] = (f"it never loaded after {len(r['attempts'])} tries "
                                  f"({err[:160]}). Keep to the DSL.")
            self.log(f"\n---\n\n## Turn {turn}{tag}: no valid law after "
                     f"{len(r['attempts'])} tries\n\nLast error: `{err}`\n")
            return
        # a repair reply often carries only the law: the idea is in the first reply
        hyp, pred = "(no hypothesis given)", None
        for rep_ in r.get("replies") or [r["reply"]]:
            h, p_ = parse_design(rep_)
            if hyp == "(no hypothesis given)" and h != "(no hypothesis given)":
                hyp = h
            pred = pred if pred is not None else p_
        law = r["law"]
        rec.update(hypothesis=hyp, prediction=pred)
        key = structure_key(law)
        twin = next((i for i in self.s["islands"] if i["key"] == key), None)
        self.doing = f"evaluating turn {turn}"
        self.heartbeat()
        res = self.evaluate(law, self.train)
        self.s["evals"]["llm"] += 1
        ref = self.champion() or self.island(1)
        if res is None:
            rec["error"] = "crashed or hung in evaluation"
            self.s["turns"].append(rec)
            self.s["feedback"] = "it crashed or hung when lived. Keep it simple and bounded."
            self.log(f"\n---\n\n## Turn {turn}{tag}: crashed in evaluation\n\n"
                     f"```nomos\n{law}\n```\n")
            return
        rec["actual"] = res["meals"]
        if twin is None:
            # a BEHAVIOURAL twin: new text, and the agent does exactly the same thing on
            # every world — the new parts changed nothing (usually a chained `set`)
            twin = next((i for i in self.s["islands"]
                         if sum(res["per_seed"]) > 0
                         and i["first"]["per_seed"] == res["per_seed"]
                         and abs(i["first"]["fitness"] - res["fitness"]) < 1e-6), None)
            if twin is not None:
                rec.update(island=twin["id"], twin=True, behaviour=True,
                           result=f"{res['meals']:.1f} meals (behaves exactly as "
                                  f"#{twin['id']} as first written)")
                self.s["feedback"] = (
                    f"it BEHAVES EXACTLY like #{twin['id']}: the same meals on every "
                    f"world ({res['per_seed']}). Your new parts changed nothing the agent "
                    "does — check that what you added actually reaches `value` (a chained "
                    "`set` keeps only its last line; a term that is always smaller inside "
                    "a max() never wins).")
                self.s["turns"].append(rec)
                self.log(f"\n---\n\n## Turn {turn}{tag}: behavioural twin of "
                         f"#{twin['id']}\n\n**Hypothesis:** {hyp}\n\n```nomos\n{law}\n```\n")
                return
        if twin is not None:
            better = res["fitness"] > twin["train"]["fitness"]
            if better and not twin.get("given"):
                twin.update(best=law, train=res)
            rec.update(island=twin["id"], twin=True,
                       result=f"{res['meals']:.1f} meals (same structure as #{twin['id']})")
            self.s["feedback"] = (
                f"it has the SAME STRUCTURE as #{twin['id']} — only numbers differ. It "
                f"ate {res['meals']:.1f} meals (#{twin['id']} tuned: "
                f"{twin['train']['meals']:.1f}). Constants are the machine's job: change "
                "what the rule does.")
            self.s["turns"].append(rec)
            self.log(f"\n---\n\n## Turn {turn}{tag}: constants-only twin of #{twin['id']}\n\n"
                     f"**Hypothesis:** {hyp}\n\nPrediction {pred}, got {res['meals']:.1f}\n")
            return
        iid = len(self.s["islands"])
        poor = res["fitness"] < 0.5 * self.island(1)["train"]["fitness"]
        isl = {"id": iid, "law": law, "best": law, "hypothesis": hyp, "prediction": pred,
               "train": res, "first": res, "holdout": None, "evals": 0,
               "tune_left": 1 if poor else self.tune, "key": key, "turn": turn,
               "deep": r["deep"], "accepted": 0}
        (self.run / "laws" / f"island{iid:03d}.nomos").write_text(
            f"# {hyp}\n\n{law}\n", encoding="utf-8")
        near = self.novelty(isl, hyp)
        self.s["islands"].append(isl)
        dm = res["meals"] - ref["train"]["meals"]
        rec.update(island=iid, result=f"{res['meals']:.1f} meals ({dm:+.1f} vs the best "
                   f"#{ref['id']} at {ref['train']['meals']:.1f})")
        self.s["turns"].append(rec)
        per = res["per_seed"]
        diag = diagnose(law)
        isl["diagnosis"] = diag
        self.s["feedback"] = (
            f"#{iid} ate {res['meals']:.1f} meals per life before tuning (you predicted "
            f"{pred if pred is not None else '?'}); best before it #{ref['id']} "
            f"{ref['train']['meals']:.1f}. Per world: {per}. Immortal "
            f"{res['immortal']}/{len(per)}. {diag}. The machine is now tuning its constants.")
        self.log(f"\n---\n\n## Turn {turn}{tag} -> island #{iid}"
                 f"{' (poor: 1 tuning try)' if poor else ''}\n\n**Hypothesis:** {hyp}\n\n"
                 f"Prediction {pred}, got **{res['meals']:.1f}** meals, fitness "
                 f"{res['fitness']:.0f}, per world {per}; model {rec['seconds']} s, "
                 f"repairs {rec['repairs']}{near}\n\n{diag}\n\n```nomos\n{law}\n```\n")

    def novelty(self, isl: dict, hyp: str) -> str:
        self.s["novelty"] = ""
        if not self.use_embed:
            return ""
        try:
            v = np.asarray(embed([hyp])[0], dtype=np.float64)
        except LLMError:
            return ""
        v /= max(1e-9, float(np.linalg.norm(v)))
        isl["emb"] = [round(float(x), 5) for x in v]
        best, sim = None, -1.0
        for o in self.born():
            if o.get("emb"):
                s = float(np.dot(v, np.asarray(o["emb"])))
                if s > sim:
                    best, sim = o, s
        isl["nearest"] = best and [best["id"], round(sim, 3)]
        if best is not None and sim > 0.85:
            self.s["novelty"] = (f"NOTE: your last idea read almost like #{best['id']}'s "
                                 f"(similarity {sim:.2f}). Try a different mechanism.")
        return f"; nearest idea #{best['id']} sim {sim:.2f}" if best is not None else ""

    # --- the loop ------------------------------------------------------------------------
    def loop(self) -> None:
        self.seed_islands()
        deadline = self.hours * 3600
        job: Job | None = None
        while True:
            now = self.elapsed()
            if job is None and now < deadline:
                job = self.next_job()
                job.start()
                self.log(f"\n> {dt.datetime.now():%H:%M} model job `{job.kind}` started\n")
            if job is not None and not job.is_alive():
                self.handle(job)
                job = None
                self.save()
                continue
            if job is not None and now > deadline + self.grace * 60:
                self.log(f"\n> abandoned a `{job.kind}` job past the grace period\n")
                job = None
            if job is None and now >= deadline:
                break
            if now - self.s["last_holdout"] > self.holdout_every * 60:
                self.holdout_check()
            if not self.tune_step():
                self.doing = f"waiting for the model ({job.kind if job else '-'})"
                self.heartbeat()
                time.sleep(2)
                continue
            self.save()
            self.heartbeat()
        self.finish()

    # --- the morning -----------------------------------------------------------------
    def finish(self) -> None:
        self.doing = "final holdout"
        self.heartbeat()
        self.holdout_check(final=True)
        if self.final and "final" not in self.s:
            self.final_check()
        self.s["done"] = True
        self.save()
        self.summary()
        self.doing = "done"
        self.heartbeat()
        if self.pool:
            self.pool.close()
            self.pool.join()

    def final_check(self) -> None:
        """The verdict: finalists on worlds nobody has selected on, not even through
        the model's eyes — the anchors, the control, the model's best structures by train
        and by holdout (tuned), and the holdout champions as the model first wrote them."""
        born = self.ranked(self.born())
        by_hold = sorted([i for i in born if i.get("holdout")],
                         key=lambda i: -i["holdout"]["fitness"])
        cands = [("reference wave (#0)", self.island(0)["best"]),
                 ("ghost, hand-written (#1)", self.island(1)["best"]),
                 (f"CONTROL: ghost tuned {self.s['control']['evals']}x",
                  self.s["control"]["best"])]
        seen = set()
        for i in by_hold[:3] + born[:2]:
            if i["id"] not in seen:
                seen.add(i["id"])
                cands.append((f"#{i['id']} tuned {i['evals']}x", i["best"]))
        for i in by_hold[:2]:
            if i["law"] != i["best"]:
                cands.append((f"#{i['id']} as written", i["law"]))
        rows = []
        for name, text in cands:
            self.doing = f"final: {name}"
            self.heartbeat()
            r = self.evaluate(text, self.final)
            rows.append({"name": name, "meals": r["meals"] if r else -1.0,
                         "immortal": r["immortal"] if r else 0,
                         "fitness": r["fitness"] if r else 0.0,
                         "per_seed": r["per_seed"] if r else []})
        self.s["final"] = rows
        self.log("\n### FINAL worlds (never seen by selection or by the model)\n\n```\n"
                 + self.final_table() + "\n```\n")
        self.save()

    def final_table(self) -> str:
        return "\n".join(f"{r['name']:<34} {r['meals']:5.1f} meals  immortal "
                         f"{r['immortal']}/{len(self.final)}" for r in self.s.get("final", []))

    def summary(self) -> None:
        s = self.s
        born = self.ranked(self.born())
        held = sorted([i for i in born if i.get("holdout")],
                      key=lambda i: -i["holdout"]["fitness"])
        # the champion is the model's best by HOLDOUT: train is what tuning overfits
        champ = held[0] if held else (born[0] if born else None)
        c = s["control"]
        g0, g1 = self.island(0), self.island(1)

        def row(name, tr, ho):
            h = f"{ho['meals']:5.1f}  {ho['immortal']:>2}/{len(self.holdout)}" if ho else "   -"
            return f"{name:<34} {tr['meals']:5.1f}  {tr['immortal']:>2}/{len(self.train)}   {h}"
        lines = [f"{'':<34} train  immortal   holdout  immortal",
                 row("reference wave (#0)", g0["train"], g0["holdout"]),
                 row("ghost, hand-written (#1)", g1["train"], g1["holdout"]),
                 row(f"CONTROL: ghost tuned {c['evals']}x", c["train"], c.get("holdout"))]
        for i in born[:5]:
            lines.append(row(f"#{i['id']} (turn {i['turn']}, tuned {i['evals']}x)",
                             i["train"], i.get("holdout")))
        turns = [t for t in s["turns"] if t.get("kind") in ("design", "deep")]
        valid = [t for t in turns if t.get("island") is not None]
        first = [t for t in valid if t.get("repairs") == 0]
        twins = [t for t in valid if t.get("twin")]
        deep = [t for t in turns if t["kind"] == "deep"]
        preds = [(t["prediction"], t["actual"]) for t in valid
                 if t.get("prediction") is not None and t.get("actual") is not None]
        cal = ""
        if len(preds) >= 4:
            half = len(preds) // 2
            e1 = np.mean([abs(p - a) for p, a in preds[:half]])
            e2 = np.mean([abs(p - a) for p, a in preds[half:]])
            cal = (f"\nPrediction error (|predicted - actual| meals, before tuning): first "
                   f"half {e1:.1f}, second half {e2:.1f} over {len(preds)} predictions — "
                   f"{'the theory got better at predicting' if e2 < e1 else 'no gain in prediction'}.\n")
        tl = "\n".join(
            f"  {t['h']:5.2f} h  turns {t['design_turns']:>3}  champion "
            f"#{t['champion']} train {t['champ_train'] or 0:5.1f} hold "
            f"{t['champ_hold'] if t['champ_hold'] is not None else '-':>5}  |  control train "
            f"{t['control_train']:5.1f} hold {t['control_hold'] if t['control_hold'] is not None else '-'}"
            for t in s["timeline"])
        out = [f"# Night lab summary — {s['started']}\n",
               f"Model `{s['model']}`, {s['elapsed'] / 3600:.2f} h, {len(turns)} design turns "
               f"({len(deep)} deep), model busy {s['llm_seconds'] / 3600:.2f} h, waited for "
               f"other users {s['waited'] / 60:.0f} min, resumed {s['resumed']}x.\n",
               "```\n" + "\n".join(lines) + "\n```\n",
               (f"\n**FINAL worlds** ({len(self.final)} worlds nobody selected on, not "
                f"even the model through the prompt):\n\n```\n{self.final_table()}\n```\n")
               if s.get("final") else "",
               f"Design turns: {len(valid)} valid ({len(first)} first try, "
               f"{len(valid) - len(first)} after repair), {len(turns) - len(valid)} never "
               f"loaded, {len(twins)} constants-only twins. Evaluations: {s['evals']}.\n", cal,
               "\n## Timeline\n\n```\n" + tl + "\n```\n"]
        if champ:
            out += [f"\n## Champion #{champ['id']}\n\n**Hypothesis:** {champ['hypothesis']}\n\n"
                    f"As written: {champ['first']['meals']:.1f} meals; tuned "
                    f"{champ['evals']}x: {champ['train']['meals']:.1f}.\n\n```nomos\n"
                    f"{champ['best']}\n```\n"]
            (self.run / "champion.nomos").write_text(
                f"# night lab champion #{champ['id']}: {champ['hypothesis']}\n\n{champ['best']}\n",
                encoding="utf-8")
        (self.run / "control.nomos").write_text(c["best"], encoding="utf-8")
        if s["theories"]:
            out += [f"\n## Theory, first version ({s['theories'][0]['h']} h)\n\n"
                    f"{s['theories'][0]['text']}\n",
                    f"\n## Theory, last version ({s['theories'][-1]['h']} h)\n\n"
                    f"{s['theories'][-1]['text']}\n"]
        if s["observations"]:
            out.append("\n## Observations\n")
            for o in s["observations"]:
                out.append(f"\n**#{o['island']} on world {o['world']}** — ![]({o['image']})\n\n"
                           f"{o['text']}\n")
        if champ:
            try:
                frames = []
                per = champ["train"]["per_seed"]
                seed = self.train[int(np.argsort(per)[len(per) // 2])]
                track_life(champ["best"], seed, min(self.ticks, 4000), self.world,
                           self.prog.to_dict(), frames=frames)
                A.save_gif(frames, str(self.run / "champion.gif"))
                out.append(f"\nGIF: `champion.gif` (world {seed}, first 4000 ticks).\n")
            except Exception as e:                              # never lose the summary
                out.append(f"\n(GIF failed: {e})\n")
        (self.run / "summary.md").write_text("".join(out), encoding="utf-8")


# --- a mock model for dry runs --------------------------------------------------------------

class NightMock:
    """Plays all three roles well enough to exercise the loop offline."""

    model = "mock"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.last = {}
        self.n = 0

    def chat(self, messages, **kw) -> str:
        self.n += 1
        self.last = {"seconds": 0.01, "reasoning": "mock thoughts" if kw.get("thinking") else ""}
        user = messages[-1]["content"]
        text = user if isinstance(user, str) else user[0]["text"]
        if "Rewrite the THEORY" in text:
            return "Lingering sources help; pure decay changes do not."
        if "ONE LIFE" in text:
            return "It paces a wall. A memory of visited cells would push it on."
        self.designs = getattr(self, "designs", 0) + 1
        if self.designs % 4 == 2:
            return "HYPOTHESIS: broken.\nPREDICTION: 3\n```nomos\nchannel value\nrule r:\n  let x = 1\n```"
        k = float(self.rng.uniform(0.9, 0.99))
        extra = "" if self.designs % 4 != 3 else "\nchannel trail\nrule trail:\n  when 1\n  set trail = max(here, 0.9 * trail)\n"
        return (f"HYPOTHESIS: a ghost that fades at {k:.2f}.\nPREDICTION: 20\n```nomos\n"
                f"channel source\nchannel solid\nchannel here\nchannel value\nchannel ghost"
                f"{extra}\nrule linger:\n  when 1\n  set ghost = max(source, {k:.3f} * ghost) * "
                f"(1 - here)\n\nrule wave:\n  when 1\n  set value = max(ghost, max(0.94 * "
                f"max4(value), 0.916 * max8(value))) * (1 - solid * (1 - leak))\n```")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A night of model-designed laws, machine-tuned.")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--dir", default="", help="run directory (existing = resume)")
    p.add_argument("--train", default="0-19")
    p.add_argument("--holdout", default="20-35")
    p.add_argument("--final", default="36-67", help="seeds lived only at the end")
    p.add_argument("--margin", type=float, default=0.02,
                   help="a tuned child must beat its parent by this fraction")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--world", choices=sorted(L.WORLDS), default="default")
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--tune", type=int, default=5)
    p.add_argument("--deep-every", type=int, default=5)
    p.add_argument("--theory-every", type=int, default=6)
    p.add_argument("--observe-every", type=int, default=8)
    p.add_argument("--holdout-every", type=float, default=30.0, help="minutes")
    p.add_argument("--grace", type=float, default=25.0, help="minutes past the deadline "
                   "to wait for a running model job")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--mock", action="store_true", help="offline dry run")
    args = p.parse_args(argv)
    run = Path(args.dir or f"out/cortex/night/{dt.datetime.now():%Y%m%d-%H%M%S}")
    run.mkdir(parents=True, exist_ok=True)
    if args.mock:
        client = NightMock(args.seed)
    else:
        try:
            client = make_client()
        except LLMError as e:
            print(f"cannot reach the model: {e}", file=sys.stderr)
            return 2
    lab = NightLab(client, run, args.hours, A._seeds(args.train), A._seeds(args.holdout),
                   args.ticks, args.world, args.jobs, args.tune, args.deep_every,
                   args.theory_every, args.observe_every, args.holdout_every, args.seed,
                   use_embed=not (args.no_embed or args.mock), grace=args.grace,
                   final=A._seeds(args.final) if args.final else (), margin=args.margin)
    if lab.s["done"]:
        print(f"{run} is finished — see summary.md")
        return 0
    try:
        lab.loop()
    finally:
        lab.save()
        if lab.pool:
            lab.pool.terminate()
    print(f"run -> {run}  (summary.md, journal.md, champion.nomos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
