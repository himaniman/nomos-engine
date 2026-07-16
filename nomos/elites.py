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
STRUCTURE_EDGES = [0.05, 0.3, 0.6, 0.85]        # mean gzip ratio: 0 uniform, 1 noise
STRUCTURE_LABELS = ["uniform", "ordered", "structured", "textured", "noisy"]
ORDER_EDGES = [0.25, 0.5, 0.8]                  # max temporal self-correlation
ORDER_LABELS = ["chaotic", "drifting", "recurrent", "locked"]

AXES = ["activity", "structure", "order"]
TOTAL_CELLS = len(ACTIVITY_LABELS) * len(STRUCTURE_LABELS) * len(ORDER_LABELS)


def _bin(x: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def descriptor(report: dict) -> dict:
    """Raw behavior coordinates of a run, from its observer report."""
    act = float(report.get("activity", {}).get("final", 0.0))
    chans = report.get("channels", {})
    gzs = [c["gzip_ratio"] for c in chans.values() if "gzip_ratio" in c]
    structure = sum(gzs) / len(gzs) if gzs else 0.0
    recs = [d["recurrence_corr"] for d in report.get("dynamics", {}).values()
            if "recurrence_corr" in d]
    # a world that no longer changes is trivially self-similar
    order = max(recs) if recs else (1.0 if act < 1e-3 else 0.0)
    return {"activity": act, "structure": structure, "order": order}


def cell(report: dict) -> tuple[int, int, int]:
    d = descriptor(report)
    return (_bin(d["activity"], ACTIVITY_EDGES),
            _bin(d["structure"], STRUCTURE_EDGES),
            _bin(d["order"], ORDER_EDGES))


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
