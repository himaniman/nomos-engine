"""The atlas's rule as a LAW OF NATURE in the nomos DSL.

    .\\.venv\\Scripts\\python.exe -m cortex.atlas_law laws/atlas_wave.nomos --ladder
    .\\.venv\\Scripts\\python.exe -m cortex.atlas_law laws/atlas_wave.nomos --lives 0-7

`cortex/atlas.py` hard-codes the wave in numpy; `cortex/wave.py` finds it as a per-cell
perceptron. This is the third form: the SAME rule written in the project's own language,
the DSL the scientist and the brain lab write laws in — `channel` and `rule`, `max8` and
`where`. The lattice of the map is a nomos world; each thought the harness paints the
senses into read-only channels, steps the world `subticks` times, and reads `value`
back. Every other channel the law declares is its own hidden state, kept between
thoughts and scrolled with the map like everything else.

That closes a loop the project was built for: a rule an LLM can WRITE (not knobs of a
rule, the rule), validated by the engine's own parser, stepped by the same
`World.step()` that runs the gliders in `tests/`. `laws/atlas_wave.nomos` is the hand
wave transcribed; it reproduces `Atlas._wave` to floating-point equality (there is a
test), so a law that differs from it differs on purpose.

The contract a law lives by — the harness's side:
  * INPUTS, written fresh before every thought (a law may read them, writing them is
    pointless): `source` what the senses want, `solid` how solid a cell is (0..1),
    `free` how surely it is known free, `thing` a seen morsel, `bright` its
    brightness, `scent` the smell painted along its bearing, `bad` a remembered
    refusal. All in the agent-centred frame, agent at the centre cell.
  * OUTPUT: `value` — the command is read from the eight neighbours of the centre
    (the brightest wins, ties go to the least turn). It persists between thoughts
    too, so a law can build on last thought's wave.
  * The lattice is the map padded by TWO rings: the inner one is the horizon — free,
    sourceless, a way round the outside of the map; the outer one is held at zero
    after every step, so the engine's torus never lets a wave wrap from one edge of
    the map to the other.
  * Config.decay and Config.leak are NOT applied: they are the law's business now.
    (The read-out still discounts a solid neighbour by Config.leak.)
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from cortex import atlas as A
from nomos.dsl import NomosError, parse_program, walk_names
from nomos.engine import World

# what the harness paints each thought, in the atlas's own channels
INPUTS = ("source", "solid", "free", "thing", "bright", "scent", "bad", "here")
_INPUT_CH = {"solid": A.SOLID, "free": A.FREE, "thing": A.THING, "bright": A.BRIGHT,
             "scent": A.SCENT, "bad": A.BAD}
OUTPUT = "value"
PAD = 2
VALUE_CAP = 4.0             # a runaway law is clipped, not trusted

REFERENCE = "laws/atlas_wave.nomos"


class LawRule:
    """A nomos law in the rule's seat. Picklable by its text, so `atlas.lives` can
    hand it to a pool."""

    def __init__(self, text: str, seed: int = 0):
        self.text = text
        self.seed = int(seed)
        self.program = parse_program(text)
        names = self.program.channel_names()
        # the inputs exist whether or not the law says so: a law that reads `bad`
        # without declaring it is reading what the harness paints, not a typo (the
        # first live model forgot the declaration two proposals out of two)
        used = {n for r in self.program.rules
                for e in [r.when] + [a[2] for a in r.actions] for n in walk_names(e)}
        missing = [n for n in INPUTS if n in used and n not in names]
        if missing:
            decl = "".join(f"channel {n}\n" for n in missing)
            self.program = parse_program(decl + text)
            names = self.program.channel_names()
        if OUTPUT not in names:
            raise NomosError(f"the law must declare channel {OUTPUT!r}")
        self.extra = [n for n in names if n != OUTPUT and n not in INPUTS]
        self.hidden = len(self.extra)
        self.world: World | None = None

    # --- pickling: the text is the rule --------------------------------------------
    def __getstate__(self):
        return {"text": self.text, "seed": self.seed}

    def __setstate__(self, d):
        self.__init__(d["text"], d["seed"])

    def _lattice(self, side: int) -> World:
        if self.world is None or self.world.shape[0] != side + 2 * PAD:
            self.world = World(self.program, side + 2 * PAD, self.seed)
        return self.world

    def step(self, state: np.ndarray, src: np.ndarray, passable: np.ndarray,
             passes: int) -> None:
        S = state.shape[1]
        w = self._lattice(S)
        st = w.state
        sl = slice(PAD, PAD + S)
        # paint the senses; the padding is blank (the horizon knows nothing)
        for name in INPUTS:
            if name in st:
                st[name][:] = 0.0
                if name == "source":
                    st[name][sl, sl] = src
                elif name == "here":
                    st[name][PAD + S // 2, PAD + S // 2] = 1.0
                else:
                    st[name][sl, sl] = np.clip(state[_INPUT_CH[name]], 0.0, 1.0)
        st[OUTPUT][:] = 0.0
        st[OUTPUT][sl, sl] = state[A.VALUE]
        for i, name in enumerate(self.extra):
            st[name][:] = 0.0
            st[name][sl, sl] = state[A.N_CH + i]
        for _ in range(int(passes)):
            w.step()
            for arr in w.state.values():          # the outer ring: nothing beyond
                arr[0, :] = arr[-1, :] = 0.0
                arr[:, 0] = arr[:, -1] = 0.0
        state[A.VALUE] = np.clip(w.state[OUTPUT][sl, sl], 0.0, VALUE_CAP)
        for i, name in enumerate(self.extra):
            state[A.N_CH + i] = w.state[name][sl, sl]


def load(path: str = REFERENCE, seed: int = 0) -> LawRule:
    with open(path, encoding="utf-8") as fh:
        return LawRule(fh.read(), seed)


def check(text: str, side: int = 9) -> LawRule:
    """Parse, validate, and run a few blank passes so a law that dies at runtime dies
    here, with its error, and not in a pool worker."""
    rule = LawRule(text)
    state = np.zeros((A.N_CH + rule.hidden, side, side))
    src = np.zeros((side, side))
    src[side // 2 + 2, side // 2] = 1.0
    rule.step(state, src, np.ones((side, side)), 4)
    if not np.isfinite(state[A.VALUE]).all():
        raise NomosError("the law produced non-finite values")
    return rule


_NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")


def jitter(text: str, rng: np.random.Generator, prob: float = 0.3,
           scale: float = 0.2) -> str:
    """The control arm's mutation: every numeric literal in the law moves by a
    log-normal factor with probability `prob`. The structure never changes — that is
    the point; the model's edits can."""
    def sub(m):
        if rng.random() >= prob:
            return m.group(0)
        v = float(m.group(0)) * float(np.exp(rng.normal(0.0, scale)))
        return f"{v:.4g}"
    # only inside rule bodies: leave `init` lines and comments alone
    out = []
    for line in text.splitlines():
        body = line.split("#", 1)[0]
        if body.lstrip().startswith(("set", "add", "when")):
            line = _NUM.sub(sub, body) + ("#" + line.split("#", 1)[1] if "#" in line else "")
        out.append(line)
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the atlas on a nomos law.")
    p.add_argument("law", nargs="?", default=REFERENCE)
    p.add_argument("--ladder", action="store_true")
    p.add_argument("--lives", default="", help="seeds, e.g. 0-7")
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--gif", default="", help="record one life (first seed) here")
    args = p.parse_args(argv)
    rule = load(args.law)
    print(f"law {args.law}: hidden channels {rule.extra}")
    if args.ladder:
        from cortex import probes
        out = probes._suite(lambda: A.Driver(rule=rule), 600)
        print("law     " + probes.summarise(out))
    if args.lives:
        rows = A.lives(A._seeds(args.lives), args.ticks, jobs=args.jobs, rule=rule)
        print(A.table(rows, "law"))
    if args.gif:
        frames = []
        A.live(A._seeds(args.lives or "0")[0], args.ticks, frames=frames, rule=rule)
        A.save_gif(frames, args.gif)
        print(f"gif -> {args.gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
