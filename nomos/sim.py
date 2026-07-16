"""Run a law to completion and return the observer report — the one function
the scientist loop calls. Kept separate from the CLI so both share identical
simulation semantics.
"""

from __future__ import annotations

import numpy as np

from nomos.dsl import parse_program
from nomos.engine import World
from nomos.metrics import summarize


def _observe_frame(world: World, cap: int, channels=None) -> dict:
    h, w = world.shape
    sy, sx = max(1, h // cap), max(1, w // cap)
    names = channels if channels is not None else world.state
    return {name: world.state[name][::sy, ::sx].copy() for name in names}


def render_channels(program, limit: int = 3) -> list[str]:
    """Channels to draw: first `limit` declared. One -> colormap, 2-3 -> RGB."""
    return [c.name for c in program.channels[:limit]]


def simulate(law_text: str, size: int = 96, steps: int = 300, seed: int = 1,
             obs_cap: int = 48, samples: int = 40,
             frames_out: dict | None = None, render_cap: int = 96) -> dict:
    """Parse, run, and observe a law. Raises NomosError on a malformed law.

    If `frames_out` is given, it is filled in place with {channel: [frame, ...]}
    for the first few channels, sampled at the same cadence — ready to hand to
    nomos.render for a GIF, without bloating the observer report.
    """
    program = parse_program(law_text)
    world = World(program, size=size, seed=seed)
    every = max(1, steps // samples)
    history = [_observe_frame(world, obs_cap)]

    rchans = render_channels(program) if frames_out is not None else []
    if frames_out is not None:
        for name, arr in _observe_frame(world, render_cap, rchans).items():
            frames_out[name] = [arr]

    activity = []
    for step in range(1, steps + 1):
        activity.append(world.step())
        if step % every == 0:
            history.append(_observe_frame(world, obs_cap))
            if frames_out is not None:
                for name, arr in _observe_frame(world, render_cap, rchans).items():
                    frames_out[name].append(arr)
    report = summarize(world, activity, history=history)
    return report


def digest(report: dict) -> str:
    """Compress an observer report into the short text a scientist reads.

    Deliberately lossy and cheap on tokens: verdict, activity, the salient
    per-channel facts (spread, axis gradient, blob structure, entropy drift),
    and the strongest channel correlations.
    """
    lines = [f"verdict: {report['verdict']}"]
    act = report["activity"]
    per = act.get("periodicity")
    lines.append(f"activity: mean={act['mean']:.3f} final={act['final']:.3f} "
                 f"trend={act['trend']}"
                 + (f" period={per['period']}(str {per['strength']})" if per else ""))
    dyn = report.get("dynamics", {})
    for name, ch in report["channels"].items():
        parts = [f"mean={ch['mean']:.3f}", f"std={ch['std']:.3f}",
                 f"range=[{ch['min']:.3f},{ch['max']:.3f}]"]
        if "coherence" in ch:
            parts.append(f"coherence={ch['coherence']:+.2f}")
        if "gradient" in ch:
            parts.append(f"gradient={ch['gradient']['axis']}({ch['gradient']['corr']:+.2f})")
        if "components" in ch and ch["components"]["count"]:
            c = ch["components"]
            parts.append(f"blobs={c['count']}(max {c['max_size']})")
        d = dyn.get(name)
        if d:
            parts.append(f"entropy {d['entropy_start']:.1f}->{d['entropy_end']:.1f}")
            if "recurrence_lag" in d:
                parts.append(f"recur@lag{d['recurrence_lag']}={d['recurrence_corr']:+.2f}")
        lines.append(f"  {name}: " + " ".join(parts))
    corrs = [c for c in report.get("correlations", []) if abs(c[2]) >= 0.3]
    if corrs:
        lines.append("correlations: " + ", ".join(f"{a}~{b}={c:+.2f}" for a, b, c in corrs[:4]))
    return "\n".join(lines)
