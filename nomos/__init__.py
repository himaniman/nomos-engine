"""Nomos: artificial physics playground.

Laws of nature are written in a small DSL, a grid world lives by them,
an observer reports what emerged.
"""

from nomos.dsl import parse_program, parse_expression, NomosError
from nomos.engine import World

__all__ = ["parse_program", "parse_expression", "NomosError", "World"]
