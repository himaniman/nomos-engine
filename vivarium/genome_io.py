"""Persist evolved NEAT genomes to disk and load them back as a seed pool.

The workflow this unlocks: evolve competent brains in an EASY (open) world, save
the longest-lived genomes, then seed a HARD (walled) world from them — random
brains can't bootstrap evolution inside a maze, but evolved foragers dropped in
live at once. The genomes are stored as plain JSON (a list of node/connection
graphs), so they're inspectable and are exactly what an external LLM will later
read and improve (DESIGN.md stage 5)."""

from __future__ import annotations

import json
from pathlib import Path

from vivarium import nca, neat

_BRAINS = {"neat": neat.Genome, "nca": nca.Genome}


def save_genomes(path: str, genomes: list) -> None:
    """Save a list of genomes (NEAT or NCA) as JSON. Each genome's to_dict carries a
    "brain" tag so load can rebuild the right type; a mixed list is fine."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "vivarium-2", "genomes": [g.to_dict() for g in genomes]}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_genomes(path: str) -> list:
    """Load genomes saved by `save_genomes` into a list (a seed pool). Dispatches on
    each genome's "brain" tag; legacy files without it are NEAT."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_BRAINS[d.get("brain", "neat")].from_dict(d) for d in data["genomes"]]
