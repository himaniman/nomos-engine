"""MAP-Elites-style behavior map over the run archive.

Instead of chasing one number, runs are placed on a discrete map of *behavior
regimes*: activity (how much the world changes) x structure (how spatially
organized it is) x order (how temporally self-similar). The scientist's goal
becomes coverage — reach regimes no law has produced yet — which is the
"search for laws favourable to complexity" made operational.

Cells are computed from stored observer reports on the fly, so the binning can
evolve without touching the archive schema.
"""

from __future__ import annotations

from nomos import archive

# Bin edges and labels per axis. len(labels) == len(edges) + 1.
# ACTIVITY edges mirror the observer's verdict thresholds (metrics._verdict).
ACTIVITY_EDGES = [1e-4, 1e-3, 0.03, 0.22]
ACTIVITY_LABELS = ["frozen", "static", "settled", "alive", "boiling"]
# STRUCTURE is spatial coherence (neighbour autocorrelation), NOT gzip: it tells
# a coherent wave/blob field apart from noise, which gzip could not. "uniform"
# is the degenerate flat field (no coherence to measure); otherwise higher
# coherence == more organized. Edges are on coherence, mapped high->low below.
COHERENCE_EDGES = [0.35, 0.6, 0.85]             # noisy < textured < structured < ordered
STRUCTURE_LABELS = ["uniform", "ordered", "structured", "textured", "noisy"]
ORDER_EDGES = [0.25, 0.5, 0.8]                  # max temporal self-correlation
ORDER_LABELS = ["chaotic", "drifting", "recurrent", "locked"]

AXES = ["activity", "structure", "order"]
AXIS_LABELS = [ACTIVITY_LABELS, STRUCTURE_LABELS, ORDER_LABELS]
AXIS_BOUNDS = [(0, len(ACTIVITY_LABELS) - 1),
               (0, len(STRUCTURE_LABELS) - 1),
               (0, len(ORDER_LABELS) - 1)]
TOTAL_CELLS = len(ACTIVITY_LABELS) * len(STRUCTURE_LABELS) * len(ORDER_LABELS)


def _bin(x: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def _structure(report: dict) -> tuple[int, float]:
    """(STRUCTURE_LABELS index, coherence used). Index 0 is the flat 'uniform'
    field; otherwise bin the most-coherent channel high->low into
    ordered/structured/textured/noisy. Falls back to gzip for legacy runs
    recorded before the observer measured coherence."""
    chans = report.get("channels", {})
    cohs = [c["coherence"] for c in chans.values() if c.get("coherence") is not None]
    if not cohs:
        stds = [c.get("std", 0.0) for c in chans.values()]
        if not chans or all(s < 1e-9 for s in stds):
            return 0, 0.0                       # uniform (flat)
        gzs = [c["gzip_ratio"] for c in chans.values() if "gzip_ratio" in c]
        coh = 1.0 - (sum(gzs) / len(gzs)) if gzs else 0.0   # rough legacy proxy
    else:
        coh = max(cohs)                         # the most-organized channel wins
    # coherence rises 0..3; map to labels noisy(4)/textured(3)/structured(2)/ordered(1)
    rank = _bin(coh, COHERENCE_EDGES)
    return [4, 3, 2, 1][rank], coh


def descriptor(report: dict) -> dict:
    """Raw behavior coordinates of a run, from its observer report."""
    act = float(report.get("activity", {}).get("final", 0.0))
    _, coherence = _structure(report)
    recs = [d["recurrence_corr"] for d in report.get("dynamics", {}).values()
            if "recurrence_corr" in d]
    # a world that no longer changes is trivially self-similar
    order = max(recs) if recs else (1.0 if act < 1e-3 else 0.0)
    return {"activity": act, "structure": coherence, "order": order}


def cell(report: dict) -> tuple[int, int, int]:
    act = float(report.get("activity", {}).get("final", 0.0))
    recs = [d["recurrence_corr"] for d in report.get("dynamics", {}).values()
            if "recurrence_corr" in d]
    order = max(recs) if recs else (1.0 if act < 1e-3 else 0.0)
    return (_bin(act, ACTIVITY_EDGES),
            _structure(report)[0],
            _bin(order, ORDER_EDGES))


def cell_label(c: tuple[int, int, int]) -> str:
    return (f"activity={ACTIVITY_LABELS[c[0]]} structure={STRUCTURE_LABELS[c[1]]} "
            f"order={ORDER_LABELS[c[2]]}")


def grid(db_path: str) -> dict[tuple[int, int, int], list[dict]]:
    """cell -> occupants [{id, law_name, verdict}], oldest first, from the archive."""
    cells: dict[tuple[int, int, int], list[dict]] = {}
    for e in reversed(archive.memory(db_path, limit=100000)):
        c = cell(e["report"])
        cells.setdefault(c, []).append(
            {"id": e["id"], "law_name": e["law_name"], "verdict": e["verdict"]})
    return cells


# ---- illumination: steer toward empty cells adjacent to occupied ones ----

def neighbours(cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Cells one axis-step away (±1 on a single axis, within bounds)."""
    out = []
    for i, (lo, hi) in enumerate(AXIS_BOUNDS):
        for d in (-1, 1):
            v = cell[i] + d
            if lo <= v <= hi:
                nc = list(cell)
                nc[i] = v
                out.append(tuple(nc))
    return out


def frontier(occupied) -> dict[tuple, list[tuple]]:
    """Empty cells adjacent to an occupied one -> the occupied seed(s) next to them."""
    occ = set(occupied)
    targets: dict[tuple, list[tuple]] = {}
    for oc in occ:
        for nb in neighbours(oc):
            if nb not in occ:
                targets.setdefault(nb, []).append(oc)
    return targets


def choose_target(occupied, attempts: dict | None = None, rng=None):
    """Pick (target_cell, seed_cell): an empty frontier cell to aim at and an
    occupied neighbour to seed from. Prefers well-supported, least-attempted
    targets. Returns None when the frontier is empty (map full or unseeded)."""
    front = frontier(occupied)
    if not front:
        return None
    attempts = attempts or {}
    scored = [(len(seeds) - attempts.get(t, 0), t, seeds) for t, seeds in front.items()]
    best = max(s for s, _, _ in scored)
    cands = sorted((t, seeds) for s, t, seeds in scored if s == best)
    t, seeds = cands[rng.randrange(len(cands))] if rng else cands[0]
    seeds = sorted(seeds)
    seed = seeds[rng.randrange(len(seeds))] if rng else seeds[0]
    return t, seed


def axis_move(seed: tuple, target: tuple):
    """(axis_name, from_label, to_label, 'up'|'down') for the one differing axis."""
    for i in range(3):
        if seed[i] != target[i]:
            return (AXES[i], AXIS_LABELS[i][seed[i]], AXIS_LABELS[i][target[i]],
                    "up" if target[i] > seed[i] else "down")
    return None


def elite_laws(db_path: str, limit: int = 100000) -> dict[tuple, str]:
    """cell -> most recent law text that produced it, from the archive."""
    out: dict[tuple, str] = {}
    for e in reversed(archive.memory(db_path, limit=limit)):
        out[cell(e["report"])] = e["law"]
    return out


def coverage_summary(db_path: str) -> str:
    """Text summary of the behavior map for the scientist's prompt."""
    cells = grid(db_path)
    if not cells:
        return ""
    lines = [f"BEHAVIOR MAP — {len(cells)}/{TOTAL_CELLS} regimes visited "
             f"(axes: activity {'/'.join(ACTIVITY_LABELS)}; "
             f"structure {'/'.join(STRUCTURE_LABELS)}; "
             f"order {'/'.join(ORDER_LABELS)}):"]
    for c in sorted(cells):
        occ = cells[c]
        lines.append(f"  {cell_label(c)} — {len(occ)} run(s), "
                     f"e.g. #{occ[0]['id']} {occ[0]['law_name']}")
    return "\n".join(lines)
