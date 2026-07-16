"""The observer: compute a run's report from its history.

Design goal of v2: the observer should *notice* things unprompted — that a
world oscillates, that structures persist, that a channel became organized
along an axis, that two channels track each other — without being told which
channel means what. Metrics are cheap proxies; the report describes, it does
not judge.

A run may pass `history`: a list of {channel: 2d array} frames sampled at some
cadence (optionally downsampled). Temporal metrics need it; static metrics work
from the final `world.state` alone.
"""

from __future__ import annotations

import zlib

import numpy as np


# ---- primitives ----

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


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of two fields, or None if either is constant."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spatial_autocorr(a: np.ndarray) -> float | None:
    """Mean one-cell neighbour correlation (Moran's-I-like), on the torus.

    Distinguishes *coherence* from mere disorder, which gzip cannot: a smooth
    field (waves, blobs, gradients) has neighbours that track each other, so
    this is ~1; spatial noise gives ~0; a checkerboard that flips sign each cell
    gives <0. None if the field is constant.
    """
    a = a.astype(np.float64)
    if a.std() < 1e-12:
        return None
    cs = [c for shift in ((0, 1), (1, 0))
          if (c := pearson(a, np.roll(a, shift, axis=(0, 1)))) is not None]
    return round(float(np.mean(cs)), 3) if cs else None


# ---- structure: connected components on a torus (no scipy) ----

_MOORE = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def components(mask: np.ndarray, max_iter: int = 256) -> dict:
    """Count 8-connected foreground blobs by toroidal label propagation.

    Each foreground cell starts with a unique id; ids flood to the max among
    neighbours until stable. Matches the engine's wrap-around topology.
    """
    if not mask.any():
        return {"count": 0, "max_size": 0, "mean_size": 0.0}
    labels = np.where(mask, np.arange(mask.size).reshape(mask.shape), -1).astype(np.int64)
    for _ in range(max_iter):
        nxt = labels
        for dy, dx in _MOORE:
            shifted = np.roll(labels, (dy, dx), axis=(0, 1))
            nxt = np.where(mask & (shifted > nxt), shifted, nxt)
        if np.array_equal(nxt, labels):
            break
        labels = nxt
    _, counts = np.unique(labels[mask], return_counts=True)
    return {"count": int(counts.size),
            "max_size": int(counts.max()),
            "mean_size": round(float(counts.mean()), 1)}


# ---- temporal: periodicity, persistence, recurrence ----

def dominant_period(series: list[float]) -> dict | None:
    """Strongest non-DC frequency in a 1D series -> its period and share."""
    x = np.asarray(series, dtype=np.float64)
    if len(x) < 8:
        return None
    x = x - x.mean()
    if x.std() < 1e-12:
        return None
    spec = np.abs(np.fft.rfft(x))
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if k == 0 or spec[k] <= 0:
        return None
    return {"period": round(len(x) / k, 2),
            "strength": round(float(spec[k] / (spec.sum() + 1e-12)), 3)}


def temporal_signature(frames: list[np.ndarray], max_lag: int = 12) -> dict | None:
    """How the field at time t predicts itself later.

    persistence = lag-1 self-correlation (does state carry over one step);
    recurrence  = the lag>0 with the highest self-correlation, revealing
    oscillation (a period-2 blinker peaks at lag 2, not lag 1).
    """
    flats = [f.ravel().astype(np.float64) for f in frames]
    n = len(flats)
    if n < 3:
        return None
    persistence = None
    best = None
    for lag in range(1, min(max_lag, n - 1) + 1):
        cs = [c for t in range(n - lag)
              if (c := pearson(flats[t], flats[t + lag])) is not None]
        if not cs:
            continue
        m = float(np.mean(cs))
        if lag == 1:
            persistence = round(m, 3)
        if best is None or m > best[1]:
            best = (lag, m)
    if best is None:
        return None
    return {"persistence": persistence,
            "recurrence_lag": best[0],
            "recurrence_corr": round(best[1], 3)}


# ---- reports ----

def _axis_alignment(a: np.ndarray) -> dict | None:
    """Is the field organized along x or y? (correlation with a coordinate ramp)"""
    h, w = a.shape
    rx = np.broadcast_to(np.arange(w) / w, a.shape)
    ry = np.broadcast_to((np.arange(h) / h)[:, None], a.shape)
    cx, cy = pearson(a, rx), pearson(a, ry)
    cands = [("x", cx), ("y", cy)]
    cands = [(ax, c) for ax, c in cands if c is not None]
    if not cands:
        return None
    axis, corr = max(cands, key=lambda t: abs(t[1]))
    if abs(corr) < 0.3:
        return None
    return {"axis": axis, "corr": round(corr, 3)}


def channel_report(a: np.ndarray) -> dict:
    rep = {
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "entropy_bits": round(entropy_bits(a), 3),
        "gzip_ratio": round(gzip_ratio(a), 4),
    }
    coh = spatial_autocorr(a)
    if coh is not None:
        rep["coherence"] = coh
    grad = _axis_alignment(a)
    if grad is not None:
        rep["gradient"] = grad
    if a.std() > 1e-9:
        thresh = (float(a.min()) + float(a.max())) / 2
        rep["components"] = components(a > thresh)
    return rep


def _pair_correlations(state: dict[str, np.ndarray], top: int = 5) -> list:
    names = list(state)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = pearson(state[names[i]], state[names[j]])
            if c is not None:
                pairs.append([names[i], names[j], round(c, 3)])
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:top]


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
    # final_activity is magnitude-based (mean |Δ| / spatial range); see World.step.
    flat = all(v.std() < 1e-9 for v in world.state.values())
    if flat and final_activity < 1e-4:
        return "frozen: every channel is uniform and nothing changes"
    if final_activity < 1e-3:
        return "static: structure remains but barely changes anymore"
    if final_activity < 0.03:
        return "settled: small persistent fluctuations around a steady state"
    if final_activity > 0.22:
        return "boiling: the world changes drastically every step"
    return "alive: the world keeps changing at a moderate rate"


def summarize(world, activity: list[float], history: list[dict] | None = None) -> dict:
    """Full observer report for a finished run.

    history: optional list of {channel: 2d frame} sampled during the run;
    enables the temporal (dynamics) section.
    """
    n = len(activity)
    tail = activity[max(n - max(n // 10, 1), 0):]
    act = np.array(activity) if activity else np.zeros(1)
    final = float(np.mean(tail)) if activity else 0.0

    report = {
        "steps": n,
        "size": list(world.shape),
        "seed": world.seed,
        "activity": {
            "mean": round(float(act.mean()), 6),
            "final": round(final, 6),
            "trend": _trend(act),
            "periodicity": dominant_period(activity),
        },
        "channels": {name: channel_report(arr) for name, arr in world.state.items()},
        "correlations": _pair_correlations(world.state),
        "verdict": _verdict(final, world),
    }

    if history:
        dynamics = {}
        for name in world.state:
            frames = [h[name] for h in history if name in h]
            if len(frames) < 3:
                continue
            sig = temporal_signature(frames)
            entry = {
                "gzip_start": round(gzip_ratio(frames[0]), 4),
                "gzip_end": round(gzip_ratio(frames[-1]), 4),
                "entropy_start": round(entropy_bits(frames[0]), 3),
                "entropy_end": round(entropy_bits(frames[-1]), 3),
            }
            if sig is not None:
                entry.update(sig)
            dynamics[name] = entry
        report["dynamics"] = dynamics

    return report
