"""The v1 brain: a small FIXED-topology MLP whose weights ARE the genome.

Sensors -> hidden (tanh) -> outputs (action logits). The genome is a flat weight
vector; reproduction clones it with Gaussian mutation. This is deliberately the
simplest thing that EVOLVES — heredity + variation — so the ecology can select
on it from day one. Stage 3 (DESIGN.md) swaps this for an evolvable NEAT-lite
topology behind the same `act()` interface, so nothing else has to change.
"""

from __future__ import annotations

import numpy as np

# action ids returned by act()
FORWARD, TURN_LEFT, TURN_RIGHT, REPRODUCE = 0, 1, 2, 3
N_ACTIONS = 4


def input_size(n_rays: int) -> int:
    """Vision: 3 typed channels (food, obstacle, agent) per ray, + 1
    interoceptive energy input, + 1 bias."""
    return n_rays * 3 + 2


class Brain:
    """A fixed 1-hidden-layer MLP. `genome` is the flat weight vector; identical
    genomes give identical behaviour (determinism)."""

    def __init__(self, n_in: int, n_hidden: int, genome: np.ndarray):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.genome = genome
        # unpack the flat vector into W1,b1,W2,b2
        i = 0
        self.w1 = genome[i:(i := i + n_in * n_hidden)].reshape(n_in, n_hidden)
        self.b1 = genome[i:(i := i + n_hidden)]
        self.w2 = genome[i:(i := i + n_hidden * N_ACTIONS)].reshape(n_hidden, N_ACTIONS)
        self.b2 = genome[i:(i := i + N_ACTIONS)]

    @staticmethod
    def genome_size(n_in: int, n_hidden: int) -> int:
        return n_in * n_hidden + n_hidden + n_hidden * N_ACTIONS + N_ACTIONS

    @classmethod
    def random(cls, n_in: int, n_hidden: int, rng: np.random.Generator,
               forward_bias: float = 0.0) -> "Brain":
        """A random genome. `forward_bias` lifts the FORWARD output bias so a
        fresh brain TENDS TO MOVE rather than spin in place — without it a random
        MLP usually picks one constant action and the population is full of
        degenerate spinners that never forage, so evolution can't bootstrap.
        A mild motor prior (real animals have one); sensing still overrides it."""
        g = rng.normal(0.0, 1.0, size=cls.genome_size(n_in, n_hidden))
        g[-N_ACTIONS + FORWARD] += forward_bias      # b2[FORWARD] slot
        return cls(n_in, n_hidden, g)

    def act(self, inputs: np.ndarray) -> int:
        """Forward pass -> the chosen action id (argmax of the output logits)."""
        h = np.tanh(inputs @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return int(np.argmax(out))

    def child_genome(self, rng: np.random.Generator, strength: float) -> np.ndarray:
        """A mutated copy of this genome — asexual reproduction with variation."""
        return self.genome + rng.normal(0.0, strength, size=self.genome.shape)
