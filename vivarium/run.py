"""Headless driver: run a vivarium for N ticks and report the ecology over time.

    python -m vivarium.run --ticks 2000 --every 100

A pure function of (seed, Config), so the printed history is reproducible.
"""

from __future__ import annotations

import argparse

from vivarium.config import Config
from vivarium.genome_io import load_genomes, save_genomes
from vivarium.world import World


def run_headless(config: Config, ticks: int, every: int = 100,
                 log=None, genome_pool=None, save_path: str | None = None) -> list[dict]:
    world = World(config, genome_pool=genome_pool)
    history = []
    for _ in range(ticks):
        world.step()
        if world.tick % every == 0:
            snap = world.snapshot()
            history.append(snap)
            if log is not None:
                log(snap)
    if save_path:
        save_genomes(save_path, world.hall_genomes() or [world._seed_genome()])
    return history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.run",
                                description="headless vivarium ecology")
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--brain", choices=("neat", "nca"), default="neat",
                   help="brain model: neat graph or nca cellular-automaton")
    p.add_argument("--wall-density", type=float, default=None, dest="wall_density",
                   help="override inner-wall density (0 = open world)")
    p.add_argument("--save-genomes", default=None, dest="save_genomes",
                   help="save the longest-lived genomes here at the end (.npz)")
    p.add_argument("--load-genomes", default=None, dest="load_genomes",
                   help="seed the population from this genome pool (.npz)")
    a = p.parse_args(argv)
    kw = {} if a.wall_density is None else {"wall_density": a.wall_density}
    cfg = Config(size=a.size, seed=a.seed, init_population=a.population,
                 brain=a.brain, **kw)
    pool = load_genomes(a.load_genomes) if a.load_genomes else None

    def log(s):
        print(f"t={s['tick']:>6}  pop={s['population']:>4}  food={s['food']:>4}  "
              f"corpses={s['corpses']:>3}  births={s['births']:>5}  deaths={s['deaths']:>5}")

    run_headless(cfg, a.ticks, a.every, log=log, genome_pool=pool,
                 save_path=a.save_genomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
