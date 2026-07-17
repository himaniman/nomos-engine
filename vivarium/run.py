"""Headless driver: run a vivarium for N ticks and report the ecology over time.

    python -m vivarium.run --ticks 2000 --every 100

A pure function of (seed, Config), so the printed history is reproducible.
"""

from __future__ import annotations

import argparse

from vivarium.config import Config
from vivarium.world import World


def run_headless(config: Config, ticks: int, every: int = 100,
                 log=None) -> list[dict]:
    world = World(config)
    history = []
    for _ in range(ticks):
        world.step()
        if world.tick % every == 0:
            snap = world.snapshot()
            history.append(snap)
            if log is not None:
                log(snap)
    return history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.run",
                                description="headless vivarium ecology")
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population", type=int, default=40)
    a = p.parse_args(argv)
    cfg = Config(size=a.size, seed=a.seed, init_population=a.population)

    def log(s):
        print(f"t={s['tick']:>6}  pop={s['population']:>4}  food={s['food']:>4}  "
              f"corpses={s['corpses']:>3}  births={s['births']:>5}  deaths={s['deaths']:>5}")

    run_headless(cfg, a.ticks, a.every, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
