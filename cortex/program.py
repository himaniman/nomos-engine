"""The GROWTH PROGRAM: the part of a genome that has a name.

Until now a cortex genome was, honestly, a wall of weights — `rules`, `P_in`, `P_out`,
`P_horm`, plus the masks. Everything that actually governs how a body DEVELOPS (how
loud a cell must be to divide, how fast the chemistry diffuses, how supply reaches the
tissue, how often development fires at all) lived in `brain.Config`: shared by every
lineage, identical in every generation, untouched by `mutate`. Two consequences nobody
had written down:

  * **evolution could not reach development.** It jittered weights and nothing else, so
    every brain in every lineage grew under exactly the same developmental physics;
  * **there was nothing for an LLM to write.** `docs/cortex-design.md` is blunt about
    this — do not hand a model raw weight matrices — and the model's slot in the loop
    was waiting on precisely the adapter this module is.

So: a few dozen NAMED scalars, each with a range and a sentence saying what it does.
Small enough to read, small enough to write, and now genuinely heritable. `Config`
keeps the defaults and the long explanations; a `Program` is the heritable OVERRIDE
layer over it, and `Config.merged(program)` is what a phenotype actually runs on.

Adding a field here is what makes it evolvable and LLM-writable. That is the whole
contract, and it is why the table below is the interesting part of the file.
"""

from __future__ import annotations

import numpy as np

# name -> (default, lo, hi, integer?, what it does)
#
# The docs are written FOR THE MODEL, in the second person of a designer: what does
# turning this up do to the body that grows. They go into the prompt verbatim.
FIELDS: dict[str, tuple] = {
    # --- what a cell must say before the tissue acts on it -------------------------
    "grow_thresh":   (0.25, 0.02, 1.20, False,
                      "how loudly a cell must ask before it divides. Lower = a bigger,"
                      " faster-growing body"),
    "die_thresh":    (0.25, 0.02, 1.20, False,
                      "how loudly a cell must ask before it apoptoses. Lower = more"
                      " turnover, tissue that sculpts itself instead of only adding"),
    "organ_thresh":  (0.30, 0.02, 1.20, False,
                      "how loudly a cell must ask before it becomes a sensory organ or"
                      " an effector. Lower = many organs, few interneurons to think"
                      " with"),
    "sprout_thresh": (0.30, 0.02, 1.20, False,
                      "how loudly a cell must ask before it grows an axon"),
    # --- the innate urges: what a cell asks for before it has learnt anything ------
    # These used to be baked into each rule's output bias at birth. Same arithmetic
    # (the morph tail is a raw mean, so a bias and an offset are identical), but as a
    # named scalar it can be read, written and mutated instead of being smeared across
    # every cell type and jittered as if it were a weight.
    "grow_bias":     (0.55, -0.50, 1.50, False,
                      "the seed's innate urge to divide. An embryo grows unless told"
                      " not to; at 0 a fresh genome never grows and development is a"
                      " no-op"),
    "die_bias":      (0.00, -1.00, 1.00, False,
                      "innate urge to die. Positive makes tissue transient"),
    "sense_bias":    (0.10, -0.50, 1.20, False,
                      "innate urge to become a sensory organ. Too high and the body is"
                      " all ears with nothing between them"),
    "act_bias":      (0.12, -0.50, 1.20, False,
                      "innate urge to become an effector — a cell the command is read"
                      " from"),
    "sprout_bias":   (0.26, -0.50, 1.20, False,
                      "innate urge to send an axon out across the sheet"),
    # --- how fast, and how much at a time -----------------------------------------
    "embryo_ticks":     (900, 0, 6000, True,
                         "world ticks of embryo: development is frequent and its cells"
                         " are FREE. This is the head start that lets a brain exist"
                         " before the world starts charging for it"),
    "dev_every_embryo": (8, 1, 200, True,
                         "world ticks between development ticks while an embryo"),
    "dev_every":        (120, 4, 2000, True,
                         "...and once adult. Structural plasticity, slow and charged"),
    "max_born_per_dev": (5, 1, 60, True,
                         "absolute ceiling on births per development tick. A percentage"
                         " alone compounds into thousands of cells"),
    "max_change_frac":  (0.05, 0.005, 0.50, False,
                         "...and the same ceiling as a share of the living body"),
    "die_ratio":       (0.40, 0.0, 2.0, False,
                        "deaths allowed per development tick, as a share of the birth"
                        " budget. At 1.0 the sheet churns and never becomes anything"),
    # --- metabolism: growth is paid for out of what the agent ate ------------------
    "resource_gain":   (0.55, 0.0, 3.0, False,
                        "supply fed in at the body's BORDER each development tick,"
                        " scaled by how full the agent is"),
    "resource_mix":    (0.25, 0.0, 0.9, False,
                        "how much supply seeps to a neighbour per pass. High values"
                        " carry it off the border and pool it in the interior, which is"
                        " exactly where there is no room to divide"),
    "resource_spread": (1, 0, 6, True, "seepage passes per development tick"),
    "resource_decay":  (0.10, 0.0, 0.8, False,
                        "supply not spent is lost, so it is a flow and not a store"),
    "divide_cost":     (0.35, 0.02, 3.0, False, "supply one division spends"),
    "cell_cost":  (0.000015, 0.0, 0.002, False,
                   "energy per cell per world tick, charged to the AGENT once adult."
                   " This is the only thing arguing for a small brain"),
    # --- the chemical layer: what a cell reads its own position from ---------------
    "chem_mix":   (0.35, 0.0, 0.95, False,
                   "share of the neighbourhood average a chemical channel takes per"
                   " diffusion pass. High = broad, flat gradients; low = sharp, local"),
    "chem_decay": (0.04, 0.0, 0.5, False,
                   "chemical fade per pass, so a plume dissipates instead of piling up"),
    "chem_every": (2, 1, 20, True,
                   "sub-ticks per diffusion pass. Chemistry must be SLOWER than thought"
                   " or it is just another electric channel"),
    # --- axons: the long-range topology -------------------------------------------
    "axon_min_len": (3, 2, 20, True,
                     "shorter than this and the 3x3 neighbourhood already covers it"),
    "axon_max_len": (22, 3, 60, True, "how far a growth cone may walk"),
    "axon_cost":    (0.5, 0.0, 3.0, False, "supply one axon spends — dearer than a cell"),
    "axon_w0":      (0.35, 0.0, 1.5, False, "the weight a new wire is born with"),
    "axon_decay":   (0.02, 0.0, 0.5, False,
                     "weight lost per development tick: use it or lose it"),
    "axon_lr":      (0.02, 0.0, 0.4, False, "reward-gated Hebbian rate on a wire"),
    # --- thinking and the read-out -------------------------------------------------
    "think_every": (2, 1, 32, True,
                    "world ticks per THOUGHT, the command held in between. Thinking is"
                    " expensive and one tick of a 0.35-cell step changes little"),
    # 0.25 is measured, not chosen: at 1.0 the TURN command's spread is 2-3x smaller on
    # every brain tried, because a mean over every effector divides the few cells that
    # heard something by the many that did not. See docs/lab-journal.md 2026-07-24.
    "pool_frac":   (0.25, 0.02, 1.0, False,
                    "share of effector cells the command is read from, LOUDEST first."
                    " At 1.0 the answer is divided by the size of the crowd, which"
                    " lands hardest on TURN"),
    "thrust_rest": (0.50, 0.01, 0.95, False,
                    "what a sheet holding NOTHING commands as forward thrust. High"
                    " values mean a brain with no opinion sprints, which is free marks"
                    " on any probe whose food is straight ahead"),
    "hormone_gain": (0.6, 0.0, 3.0, False,
                     "how hard the body's state (hunger first) swings every cell's rule"
                     " gain. This is the whole-brain mode switch"),
    # --- learning within one life --------------------------------------------------
    "plastic_lr":     (0.03, 0.0, 0.3, False,
                       "reward-modulated Hebbian rate on the motor read-out"),
    "rule_lr":        (0.006, 0.0, 0.1, False, "...and on each cell type's output layer"),
    "plastic_elig":   (0.5, 0.0, 0.99, False,
                       "how long an action stays creditable for a reward that arrives"
                       " later. Poison is the case this exists for"),
    "plastic_forget": (0.002, 0.0, 0.1, False, "what learning loses per tick"),
}

INT_FIELDS = {k for k, v in FIELDS.items() if v[3]}
DEFAULTS = {k: (int(v[0]) if v[3] else float(v[0])) for k, v in FIELDS.items()}


def clamp(name: str, value):
    """A value forced into its declared range and type. Every path in and out of a
    program goes through this — mutation, the LLM, a hand-written JSON file — so a
    program can never be out of range, whatever wrote it."""
    default, lo, hi, is_int, _ = FIELDS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULTS[name]
    if not np.isfinite(v):
        return DEFAULTS[name]
    v = min(float(hi), max(float(lo), v))
    return int(round(v)) if is_int else float(v)


class Program:
    """A growth program: the named, heritable half of a genome.

    Behaves like a read-only mapping (`p["grow_thresh"]`, `p.values`), because the
    thing that consumes it is `Config.merged`, and the things that produce it are
    mutation and an LLM. Unknown keys are DROPPED rather than raising: a model that
    invents a field should lose the field, not the generation."""

    def __init__(self, values: dict | None = None):
        self.values = dict(DEFAULTS)
        if values:
            self.update(values)

    def update(self, values: dict) -> list[str]:
        """Write what is recognised, report what was not. The report is what the LLM
        loop echoes back to the model, so a hallucinated knob is a visible mistake
        rather than a silent no-op."""
        unknown = []
        for k, v in values.items():
            if k in FIELDS:
                self.values[k] = clamp(k, v)
            else:
                unknown.append(k)
        return unknown

    def __getitem__(self, k: str):
        return self.values[k]

    def get(self, k: str, default=None):
        return self.values.get(k, default)

    def __contains__(self, k: str) -> bool:
        return k in self.values

    def copy(self) -> "Program":
        return Program(self.values)

    def to_dict(self) -> dict:
        return dict(self.values)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Program":
        return cls(d or {})

    def mutate(self, rng, prob: float = 0.25, scale: float = 0.20) -> "Program":
        """A jittered copy. RELATIVE to the field's range, not to its value, so a knob
        currently sitting at zero can still move — an absolute-fraction jitter would
        pin any field that ever reached its floor, which is the quiet way a search
        space collapses."""
        out = self.copy()
        for k, (default, lo, hi, is_int, _) in FIELDS.items():
            if rng.random() >= prob:
                continue
            span = float(hi) - float(lo)
            out.values[k] = clamp(k, out.values[k] + rng.normal(0.0, scale * span))
        return out

    def diff(self, other: "Program") -> dict:
        """What changed, for the journal and for the LLM's own history. A generation
        that reports 'these three knobs moved' teaches more than one reporting a new
        vector of thirty numbers."""
        return {k: (other.values[k], self.values[k])
                for k in FIELDS if other.values[k] != self.values[k]}

    def summary(self, per_line: int = 3) -> str:
        """Compact, for the HUD."""
        items = [f"{k} {self._fmt(k)}" for k in FIELDS]
        return "\n".join("  ".join(items[i:i + per_line])
                         for i in range(0, len(items), per_line))

    def _fmt(self, k: str) -> str:
        v = self.values[k]
        return str(v) if k in INT_FIELDS else (f"{v:.5f}" if abs(v) < 0.01 else f"{v:.3f}")

    def as_prompt(self) -> str:
        """The program as the LLM sees it: value, range, and what it does. Verbose on
        purpose — a model that knows what a knob MEANS proposes differently from one
        matching numbers to a score."""
        lines = []
        for k, (default, lo, hi, is_int, doc) in FIELDS.items():
            rng_s = f"{int(lo)}..{int(hi)}" if is_int else f"{lo:g}..{hi:g}"
            lines.append(f"  {k:<18} = {self._fmt(k):<10} [{rng_s}]  {doc}")
        return "\n".join(lines)


def spec() -> str:
    """The field table alone, for a prompt that has no program to show yet."""
    return Program().as_prompt()
