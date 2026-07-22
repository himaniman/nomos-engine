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
                 think_every: int | None = None, reward_scale: float = 12.0):
        self.genome = genome
        self.layout = layout(_rays_of(genome))
        self.phenotype = B.Phenotype(genome, cfg, self.layout)
        # the phenotype resolved the genome's program against the ambient config; use
        # THAT, so a Mind and its brain can never disagree about how fast it thinks
        self.cfg = self.phenotype.cfg
        self.think_every = int(self.cfg.think_every if think_every is None
                               else think_every)
        self.reward_scale = float(reward_scale)
        self.field: ScentField | None = None
        self.command = (0.0, 0.0, 0.0)
        self.thoughts = 0
        self.last_reward = 0.0
        self.last_dev = {"born": 0, "died": 0}
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
            self.command = self.phenotype.act(self.sensors, hormone=self.hormone(w))
            self.thoughts += 1
        before = w.agent.energy
        w.step(*self.command)
        self.last_reward = (w.agent.energy - before) / self.reward_scale
        self.phenotype.learn(self.last_reward)
        self._develop(w)

    def _develop(self, w: World) -> None:
        """Growth on its own slow clock, and paid for.

        TWO PHASES, straight out of biology and out of arithmetic. In the EMBRYO the
        tissue develops often and its cells are free, so a brain gets to exist before
        the world starts charging for it — without that, growth is a tax you take out
        before you can forage and no lineage would ever pay it. Afterwards development
        is rare and every cell costs energy every tick, so a bigger brain has to earn
        its keep by eating more. That cost IS the selection pressure against bloat;
        nothing else in the system argues for a small brain."""
        c = self.phenotype.cfg
        if not c.develop:
            return
        embryo = w.tick < c.embryo_ticks
        every = c.dev_every_embryo if embryo else c.dev_every
        if w.tick % every == 0:
            self.last_dev = self.phenotype.develop(
                energy_frac=max(0.0, w.agent.energy / w.cfg.max_energy), embryo=embryo)
        if not embryo:
            w.agent.energy -= c.cell_cost * self.phenotype.cells()


    def hormone(self, w: World) -> np.ndarray:
        """What the BODY broadcasts to the whole sheet: how full it is, how poisoned,
        and how the last moments went. Unlike the senses this reaches every cell
        whether or not it is an organ — it is a bloodstream, not a wire — and its
        first component also swings every rule's gain, so a starving brain is
        physically not the same computer as a fed one.

        Note this is NOT new information: hunger is already in interoception. The
        difference is that a sense arrives at ONE organ and has to be carried, while a
        hormone arrives everywhere at once and changes how everything computes. That
        distinction is the whole point of having the layer."""
        a = w.agent
        return np.array([2.0 * a.energy / w.cfg.max_energy - 1.0,   # fed <-> starving
                         min(1.0, a.poison / 30.0),
                         max(-1.0, min(1.0, self.last_reward))])


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
