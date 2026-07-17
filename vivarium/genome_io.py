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

from vivarium.neat import Genome


def save_genomes(path: str, genomes: list[Genome]) -> None:
    """Save a list of NEAT genomes as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "vivarium-neat-1", "genomes": [g.to_dict() for g in genomes]}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_genomes(path: str) -> list[Genome]:
    """Load genomes saved by `save_genomes` into a list (a seed pool)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Genome.from_dict(d) for d in data["genomes"]]
