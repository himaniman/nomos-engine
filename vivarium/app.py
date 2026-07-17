"""Real-time pygame view of a vivarium (DESIGN.md stage 2).

    python -m vivarium.app --size 96 --seed 1

Global camera over the whole world. Agents are coloured by GENOME (a stable hue
hash, so lineages/families share a colour and you watch them spread and compete);
brightness tracks energy; a short spoke shows heading. Food = green dots, corpses
= dim brown blocks, walls = grey. A side panel shows the ecology HUD and, when
you click an agent, that agent's brain (its MLP, edges coloured by weight) and
gene stats.

Controls: SPACE pause/resume · S single-step · +/- speed · click an agent to
inspect · R reseed-from-hall · ESC/Q quit.

The drawing is split from the loop (`draw` returns a Surface) so it can be
rendered headless (SDL dummy) for tests/screenshots without a display.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib

import numpy as np
import pygame

from vivarium.config import Config
from vivarium.neat import Genome
from vivarium.world import World

_CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"


def _genome_hash(genome: Genome) -> int:
    return int(hashlib.md5(genome.key()).hexdigest(), 16)


def genome_color(genome: Genome) -> tuple[int, int, int]:
    """A stable hue from the genome — a lineage shares a colour; mutation shifts
    it slightly, so families are visible and drift shows."""
    h = (_genome_hash(genome) % 3600) / 3600.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def genome_name(genome: Genome) -> str:
    """A 6-letter pronounceable-ish name for a genome, so lineages are nameable
    in the stats. Stable per genome, changes when the structure/weights change."""
    n = _genome_hash(genome)
    letters = []
    for _ in range(6):
        letters.append(_CONSONANTS[n % len(_CONSONANTS)])
        n //= len(_CONSONANTS)
    return "".join(letters)


def draw(world: World, cell_px: int, panel_w: int, font, small,
         selected: int | None = None, hud: dict | None = None) -> pygame.Surface:
    """Render one frame to a Surface: the world on the left, HUD + (optional)
    inspected-agent panel on the right."""
    W, H = world.w * cell_px, world.h * cell_px
    surf = pygame.Surface((W + panel_w, H))
    surf.fill((12, 12, 16))

    # walls
    wall_ys, wall_xs = np.where(world.walls)
    for y, x in zip(wall_ys, wall_xs):
        surf.fill((60, 60, 68), (x * cell_px, y * cell_px, cell_px, cell_px))
    # corpses (decaying obstacles)
    cy, cx = np.where(world.corpse > 0)
    for y, x in zip(cy, cx):
        surf.fill((70, 40, 30), (x * cell_px, y * cell_px, cell_px, cell_px))
    # food
    fy, fx = np.where(world.food)
    fr = max(1, cell_px // 2)
    for y, x in zip(fy, fx):
        pygame.draw.circle(surf, (60, 200, 90),
                           (x * cell_px + cell_px // 2, y * cell_px + cell_px // 2), fr)
    # agents
    for a in world.agents.values():
        col = genome_color(a.brain.genome)
        bright = 0.35 + 0.65 * min(1.0, a.energy / world.cfg.max_energy)
        col = tuple(int(ch * bright) for ch in col)
        px, py = a.x * cell_px, a.y * cell_px
        surf.fill(col, (px, py, cell_px, cell_px))
        if a.id == selected:
            pygame.draw.rect(surf, (255, 255, 255), (px, py, cell_px, cell_px), 1)
        dy, dx = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[a.heading]
        cxp, cyp = px + cell_px // 2, py + cell_px // 2
        pygame.draw.line(surf, (0, 0, 0), (cxp, cyp),
                         (cxp + dx * cell_px // 2, cyp + dy * cell_px // 2), 1)

    _draw_panel(surf, world, W, panel_w, H, font, small, selected, hud)
    return surf


def _draw_panel(surf, world, x0, panel_w, H, font, small, selected, hud=None):
    surf.fill((20, 20, 26), (x0, 0, panel_w, H))
    s = world.snapshot()
    lines = [f"tick {s['tick']}", f"pop  {s['population']}", f"food {s['food']}",
             f"births {s['births']}", f"deaths {s['deaths']}"]
    if hud is not None:
        lines.append(f"fps {hud['fps']:.0f}  x{hud['speed']}  {hud['sps']:.0f}/s")
    for i, t in enumerate(lines):
        surf.blit(font.render(t, True, (220, 220, 230)), (x0 + 12, 10 + i * 20))
    y = 10 + len(lines) * 20 + 10

    # top-3 longest-lived lineages by name, with a colour swatch
    surf.blit(small.render("leaders (age / name / net):", True, (150, 160, 180)),
              (x0 + 12, y)); y += 18
    for age, g in world.hall[:3]:
        nh, nc = g.complexity()
        pygame.draw.rect(surf, genome_color(g), (x0 + 12, y + 2, 10, 10))
        surf.blit(small.render(f"{age:>4} {genome_name(g)}  h{nh} c{nc}", True,
                               (210, 215, 230)), (x0 + 28, y)); y += 17
    y += 8

    a = world.agents.get(selected) if selected is not None else None
    if a is None:
        return
    g = a.brain.genome
    nh, nc = g.complexity()
    pygame.draw.rect(surf, genome_color(g), (x0 + 12, y + 3, 12, 12))
    surf.blit(font.render(genome_name(g), True, (235, 235, 245)), (x0 + 30, y))
    y += 24
    for t in [f"agent #{a.id}", f"energy {a.energy:.1f}", f"age {a.age}",
              f"hidden {nh}  conns {nc}"]:
        surf.blit(small.render(t, True, (200, 210, 230)), (x0 + 12, y)); y += 18
    _draw_net(surf, g, x0 + 12, y + 8, panel_w - 24, H - (y + 8) - 12)


def _draw_net(surf, genome, x, y, w, h):
    """Draw the NEAT graph: inputs on the left, outputs on the right, hidden nodes
    stacked in the middle; edges coloured by weight sign, thicker for magnitude."""
    n_in, n_out = genome.n_in, genome.n_out
    hidden = [i for i, k in genome.nodes.items() if k == "hidden"]
    def col_x(kind):
        return {"in": x, "out": x + w, "hidden": x + w // 2}[kind]
    order = {"in": list(range(n_in)),
             "out": [n_in + j for j in range(n_out)],
             "hidden": hidden}
    pos = {}
    for kind, ids in order.items():
        for j, nid in enumerate(ids):
            yy = y + int((j + 0.5) / max(1, len(ids)) * h)
            pos[nid] = (col_x(kind), yy)
    m = max((abs(c["w"]) for c in genome.conns if c["enabled"]), default=1.0) + 1e-6
    for c in genome.conns:
        if not c["enabled"] or c["in"] not in pos or c["out"] not in pos:
            continue
        col = (90, 160, 255) if c["w"] >= 0 else (255, 110, 90)
        pygame.draw.line(surf, col, pos[c["in"]], pos[c["out"]],
                         1 + int(2 * abs(c["w"]) / m))
    for nid, (px, py) in pos.items():
        kind = genome.nodes[nid]
        cc = {"in": (140, 220, 150), "out": (240, 200, 120),
              "hidden": (210, 210, 230)}[kind]
        pygame.draw.circle(surf, cc, (px, py), 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.app", description="watch a vivarium")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=60)
    p.add_argument("--cell", type=int, default=7, help="pixels per cell")
    p.add_argument("--load-genomes", default=None, dest="load_genomes",
                   help="seed the world from an evolved genome pool (.json)")
    a = p.parse_args(argv)

    cfg = Config(size=a.size, seed=a.seed, init_population=a.population)
    pool = None
    if a.load_genomes:
        from vivarium.genome_io import load_genomes
        pool = load_genomes(a.load_genomes)
    world = World(cfg, genome_pool=pool)
    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    panel_w = 260
    screen = pygame.display.set_mode((world.w * a.cell + panel_w, world.h * a.cell))
    pygame.display.set_caption("vivarium")
    clock = pygame.time.Clock()

    # `speed` = simulation steps per RENDERED frame. Rendering is the bottleneck,
    # so running many steps per draw (and, at high speed, uncapping the frame
    # rate) lets the simulation outrun the display for fast-forward. The HUD shows
    # the measured FPS, the speed, and the resulting steps/sec.
    paused, speed, selected = False, 1, None
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_s:
                    world.step()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(1024, speed * 2)
                elif e.key == pygame.K_MINUS:
                    speed = max(1, speed // 2)
                elif e.key == pygame.K_r:
                    world._reseed()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                gx, gy = mx // a.cell, my // a.cell
                if 0 <= gy < world.h and 0 <= gx < world.w and world.occ[gy, gx] >= 0:
                    selected = int(world.occ[gy, gx])
        if not paused:
            for _ in range(speed):
                world.step()
        if selected is not None and selected not in world.agents:
            selected = None
        fps = clock.get_fps()
        hud = {"fps": fps, "speed": speed, "sps": fps * (speed if not paused else 0)}
        screen.blit(draw(world, a.cell, panel_w, font, small, selected, hud), (0, 0))
        pygame.display.flip()
        clock.tick(60 if speed <= 2 else 0)          # uncap the frame rate when fast
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
