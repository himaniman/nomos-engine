"""Heritable VISION genes, shared by both brain genomes (neat, nca).

Each creature carries how many vision rays it fans (`n_rays`) and how far each ray
reaches (`ray_len`, in cells) as EVOLVABLE traits, bounded by the Config. The brain's
input vector is sized for `n_rays_max` ray slots (see aqua.world.sensor_size); a genome
that fans fewer rays just leaves the extra slots at zero. Ray signal strength falls off
with distance, so a longer ray only lets faraway food/agents register faintly — cheap to
allow and useful to evolve (far-sighted vs. sharp-but-short is a real trade-off, paid for
by nothing but the reach itself). These genes ride along harmlessly on the grid vivarium
(which fans a fixed fan from the Config); they only steer the continuous `aqua` world.
"""

from __future__ import annotations

DEFAULT_N_RAYS = 5       # used when a genome predates these genes (old saved pools)
DEFAULT_RAY_LEN = 12.0


def _bounds(cfg):
    return (int(getattr(cfg, "n_rays_min", 3)), int(getattr(cfg, "n_rays_max", 12)),
            float(getattr(cfg, "ray_length_min", 6)), float(getattr(cfg, "ray_length_max", 24)))


def random_vision(rng, cfg):
    """Fresh (n_rays, ray_len) drawn uniformly from the allowed ranges."""
    rmin, rmax, lmin, lmax = _bounds(cfg)
    return int(rng.integers(rmin, rmax + 1)), float(rng.uniform(lmin, lmax))


def drift_vision(g, rng, cfg) -> None:
    """In-place heritable drift of a child's vision genes: occasionally nudge the ray
    COUNT by one or jitter the ray LENGTH, clamped to the configured bounds."""
    rmin, rmax, lmin, lmax = _bounds(cfg)
    if rng.random() < float(getattr(cfg, "vision_mut_prob", 0.12)):
        if rng.random() < 0.5:
            step = 1 if rng.random() < 0.5 else -1
            g.n_rays = int(min(rmax, max(rmin, int(g.n_rays) + step)))
        else:
            g.ray_len = float(min(lmax, max(lmin, float(g.ray_len) + float(rng.normal(0.0, 2.0)))))


def cross_vision(g_fit, g_other, rng):
    """Each vision gene inherited from one parent (like diet/carnivore)."""
    n_rays = g_fit.n_rays if rng.random() < 0.5 else g_other.n_rays
    ray_len = g_fit.ray_len if rng.random() < 0.5 else g_other.ray_len
    return int(n_rays), float(ray_len)
