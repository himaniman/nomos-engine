"""Island-model evolution: many worlds evolve in parallel, then exchange migrants.

Each ISLAND is an independent headless vivarium run (its own seed). After it
runs, we take its top-K most successful genomes (longevity + fecundity, via the
hall-of-fame). Between ROUNDS the islands exchange migrants: the top genomes from
all islands are merged into a pool, and each next island is seeded from a RANDOM
SUBSET of that pool (partial migration — full migration would collapse every
island to the same solution; partial keeps them exploring different regions).

This is the natural way to scale the project: the runs are pure functions, so
islands are embarrassingly parallel. Here they run as local processes
(ProcessPoolExecutor — process-level, so the GIL is irrelevant); the SAME
`run_island` maps onto one Kubernetes Job per island, with this controller's
merge/redistribute step between rounds (see DESIGN.md). Deterministic given
(rounds, islands, seeds, ticks).

    python -m vivarium.island --rounds 3 --islands 10 --ticks 10000 --top 3 \
        --out vivarium/pools/islands.json
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from vivarium.config import Config
from vivarium.genome_io import load_genomes, save_genomes
from vivarium.neat import Genome
from vivarium.world import World


def run_island(job: dict) -> list[dict]:
    """Evolve one island and return its top-K genomes (as dicts, for pickling).
    Top-level so ProcessPoolExecutor can ship it to a worker / a k8s Job can call
    the same logic."""
    cfg = replace(Config(**job["config"]), seed=job["seed"])
    pool = [Genome.from_dict(d) for d in job["pool"]] or None
    world = World(cfg, genome_pool=pool)
    for _ in range(job["ticks"]):
        world.step()
    return [g.to_dict() for g in world.top_genomes(job["top_k"])]


def _seed_pools(merged: list[dict], n: int, subset: int,
                rng: random.Random) -> list[list[dict]]:
    """Partial migration: each island gets a random subset of the merged pool."""
    if not merged:
        return [[] for _ in range(n)]
    k = max(1, min(subset, len(merged)))
    return [rng.sample(merged, k) for _ in range(n)]


def evolve_islands(rounds: int, n_islands: int, ticks: int, top_k: int = 3,
                   subset: int = 6, workers: int | None = None,
                   config: dict | None = None, base_pool: list[Genome] | None = None,
                   rng_seed: int = 0, on_round=None) -> list[Genome]:
    """Run the island model and return the final merged pool of top genomes."""
    rng = random.Random(rng_seed)
    config = config or {}
    merged = [g.to_dict() for g in base_pool] if base_pool else []

    for r in range(rounds):
        pools = _seed_pools(merged, n_islands, subset, rng)
        jobs = [{"config": config, "seed": r * 1000 + i, "ticks": ticks,
                 "top_k": top_k, "pool": pools[i]} for i in range(n_islands)]
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(run_island, jobs))
        else:
            results = [run_island(j) for j in jobs]
        merged = [g for island in results for g in island]      # gather all migrants
        if on_round is not None:
            on_round(r, merged)

    return [Genome.from_dict(d) for d in merged]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.island",
                                description="island-model evolution of vivarium brains")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--islands", type=int, default=10)
    p.add_argument("--ticks", type=int, default=10000)
    p.add_argument("--top", type=int, default=3, dest="top_k",
                   help="genomes taken from each island per round")
    p.add_argument("--subset", type=int, default=6,
                   help="migrants each next island is seeded with (partial migration)")
    p.add_argument("--workers", type=int, default=1, help="parallel island processes")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--wall-density", type=float, default=0.0, dest="wall_density",
                   help="0 (open) is easiest to bootstrap; walls need a seed pool")
    p.add_argument("--load", default=None, help="start from this genome pool (.json)")
    p.add_argument("--out", required=True, help="write the final merged pool here (.json)")
    a = p.parse_args(argv)

    cfg = {"size": a.size, "wall_density": a.wall_density, "init_population": 80}
    base = load_genomes(a.load) if a.load else None

    def on_round(r, merged):
        hidden = [Genome.from_dict(d).complexity()[0] for d in merged]
        print(f"[round {r}] merged pool={len(merged)} hidden(min/max)="
              f"{min(hidden)}/{max(hidden)}", flush=True)

    final = evolve_islands(a.rounds, a.islands, a.ticks, top_k=a.top_k,
                           subset=a.subset, workers=a.workers, config=cfg,
                           base_pool=base, on_round=on_round)
    save_genomes(a.out, final)
    print(f"wrote {len(final)} genomes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
