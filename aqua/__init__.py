"""aqua — the CONTINUOUS-world variant of the vivarium ecology.

Same living world (multi-agent, energy/metabolism, ray vision, food niches with
drifting halos, diet, predation, speciation by colour, hall-of-fame + extinction
reseed from the best) and the SAME brains (vivarium's NCA / NEAT genomes are
imported unchanged — evolved pools load directly), but the physics is SMOOTH:
agents have float position, a continuous heading, a continuous speed, and their
motor read-out is read continuously from the brain's logits (thrust + turn-rate)
instead of a 4-way argmax. Food and walls stay on a grid for now; agents live in
float space over them, collide as circles, and see by ray-marching.

Reproducibility is preserved: a run is a pure function of (Config, seed).
"""
