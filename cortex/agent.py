"""Wiring a brain into the world: sense -> think -> act -> learn.

One agent, one long life. The brain drives `World.step(thrust, turn, interact)` —
the same call the keyboard and the baselines use, so a brain's session record is
directly comparable with a human's.

Two different time scales live here and both matter:

  * `subticks` — how many rounds the sheet RELAXES inside one thought. This is how far
    a signal can travel from an organ to an effector before the command is read, so it
    bounds how deep the computation can be. Heritable.
  * `think_every` — how many WORLD ticks pass per thought, with the command held in
    between. Thinking is expensive and the world is fast; a big brain that thought
    every tick would run at a crawl for no gain, since nothing much changes in one
    tick of a 0.35-cell step. The nomos/forage worlds settled the same way.

REWARD is the change in ENERGY, nothing else. It needs no shaping: eating is a large
positive, walking a small negative, poison a slow drip of negative that arrives AFTER
the meal — which is exactly the credit-assignment problem the eligibility trace exists
to solve.
"""

from __future__ import annotations

import numpy as np

from cortex import brain as B
from cortex.senses import ScentField, layout, sensor_size
from cortex.world import Config as WorldConfig
from cortex.world import World


class Mind:
    """A brain plus everything it needs to keep sensing the same world."""

    def __init__(self, genome: B.Genome, cfg: B.Config | None = None,
                 think_every: int = 2, reward_scale: float = 12.0):
        self.cfg = cfg or B.Config()
        self.genome = genome
        self.layout = layout(_rays_of(genome))
        self.phenotype = B.Phenotype(genome, self.cfg, self.layout)
        self.think_every = int(think_every)
        self.reward_scale = float(reward_scale)
        self.field: ScentField | None = None
        self.command = (0.0, 0.0, 0.0)
        self.thoughts = 0
        self.last_reward = 0.0
        self.sensors: np.ndarray | None = None

    def attach(self, w: World) -> None:
        self.field = ScentField(w.cfg.smell_radius)

    def step(self, w: World) -> None:
        """One WORLD tick. The brain is consulted every `think_every` ticks and its
        command is held in between; learning happens every tick, because the energy
        that arrives between thoughts is just as much a consequence of them."""
        if self.field is None:
            self.attach(w)
        if w.tick % self.think_every == 0:
            self.sensors = np.asarray(_sense(w, self.field))
            self.command = self.phenotype.act(self.sensors)
            self.thoughts += 1
        before = w.agent.energy
        w.step(*self.command)
        self.last_reward = (w.agent.energy - before) / self.reward_scale
        self.phenotype.learn(self.last_reward)


def _sense(w: World, field: ScentField):
    from cortex.senses import sense
    return sense(w, field)


def _rays_of(genome: B.Genome) -> int:
    """The ray count a genome's input width implies — so a brain and a world can never
    silently disagree about what slot means what."""
    from cortex.senses import INTERO, N_TOUCH, ODOUR_CH, RAY_CH, TOUCH_CH
    from cortex.world import N_ODOUR
    static = N_ODOUR * ODOUR_CH + N_TOUCH * TOUCH_CH + INTERO
    return (genome.n_in - static) // RAY_CH


def fresh_genome(seed: int = 0, wcfg: WorldConfig | None = None,
                 bcfg: B.Config | None = None) -> B.Genome:
    wcfg = wcfg or WorldConfig()
    return B.random_genome(sensor_size(wcfg.n_rays), np.random.default_rng(seed),
                           bcfg or B.Config())


def live(genome: B.Genome, seed: int = 0, max_ticks: int = 20000,
         wcfg: WorldConfig | None = None, bcfg: B.Config | None = None,
         think_every: int = 2) -> tuple[World, Mind]:
    """One whole life, headless. A pure function of (genome, seed, limits)."""
    w = World(seed=seed, cfg=wcfg or WorldConfig())
    mind = Mind(genome, bcfg, think_every=think_every)
    mind.attach(w)
    while w.agent.alive and w.tick < max_ticks:
        mind.step(w)
    return w, mind
