"""The observer: compute a run's report from its history.

These are deliberately cheap proxies. The report is meant to be read by a
scientist (human or LLM) alongside the GIF — it should describe, not judge.
"""

from __future__ import annotations

import zlib

import numpy as np


def _quantize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def gzip_ratio(a: np.ndarray) -> float:
    """Compressed/raw size of the quantized field. ~0 uniform, ~1 noise."""
    q = _quantize(a).tobytes()
    return len(zlib.compress(q, 6)) / len(q)


def entropy_bits(a: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy of the value histogram, in bits (max = log2(bins))."""
    hist, _ = np.histogram(a, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def channel_report(a: np.ndarray) -> dict:
    return {
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "entropy_bits": round(entropy_bits(a), 3),
        "gzip_ratio": round(gzip_ratio(a), 4),
    }


def summarize(world, activity: list[float]) -> dict:
    """Full observer report for a finished run."""
    n = len(activity)
    tail = activity[max(n - max(n // 10, 1), 0):]
    act = np.array(activity) if activity else np.zeros(1)
    report = {
        "steps": n,
        "size": list(world.shape),
        "seed": world.seed,
        "activity": {
            "mean": round(float(act.mean()), 6),
            "final": round(float(np.mean(tail)), 6) if activity else 0.0,
            "trend": _trend(act),
        },
        "channels": {name: channel_report(arr) for name, arr in world.state.items()},
        "verdict": _verdict(float(np.mean(tail)) if activity else 0.0, world),
    }
    return report


def _trend(act: np.ndarray) -> str:
    if len(act) < 20:
        return "too_short"
    first, last = act[: len(act) // 4].mean(), act[-len(act) // 4:].mean()
    if last < 1e-6:
        return "died_out"
    if last < first * 0.5:
        return "decaying"
    if last > first * 1.5:
        return "growing"
    return "steady"


def _verdict(final_activity: float, world) -> str:
    """One human-readable line: the observer's first impression."""
    flat = all(v.std() < 1e-9 for v in world.state.values())
    if flat and final_activity < 1e-6:
        return "frozen: every channel is uniform and nothing changes"
    if final_activity < 1e-6:
        return "static: structure remains but nothing changes anymore"
    if final_activity > 0.9:
        return "boiling: almost every cell changes every step"
    return "alive: the world keeps changing at a moderate rate"
