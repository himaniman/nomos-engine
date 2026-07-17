"""Persist evolved genomes to disk and load them back as a seed pool.

The workflow this unlocks: evolve a competent population in an EASY (open) world,
save its longest-lived genomes, then seed a HARD (walled) world from them — random
brains can't bootstrap evolution inside a maze, but evolved foragers dropped into
one live at once. The same serialized genomes are what an external LLM will later
read and improve (DESIGN.md stage 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_genomes(path: str, genomes: list[np.ndarray]) -> None:
    """Save a list of equal-length genomes as a single .npz array."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, genomes=np.asarray(genomes, dtype=np.float64))


def load_genomes(path: str) -> list[np.ndarray]:
    """Load genomes saved by `save_genomes` into a list (a seed pool)."""
    with np.load(path) as data:
        return [row.copy() for row in data["genomes"]]
