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

from vivarium.brain import N_ACTIONS
from vivarium.config import Config
from vivarium.world import World

_ACTION_NAMES = ["fwd", "left", "right", "breed"]


def genome_color(genome: np.ndarray) -> tuple[int, int, int]:
    """A stable hue from the genome — near-identical genomes (a lineage) share a
    colour; mutation shifts it slightly, so families are visible and drift shows."""
    key = np.round(genome, 1).tobytes()
    h = int(hashlib.md5(key).hexdigest(), 16) % 3600 / 3600.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw(world: World, cell_px: int, panel_w: int, font, small,
         selected: int | None = None) -> pygame.Surface:
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

    _draw_panel(surf, world, W, panel_w, H, font, small, selected)
    return surf


def _draw_panel(surf, world, x0, panel_w, H, font, small, selected):
    surf.fill((20, 20, 26), (x0, 0, panel_w, H))
    s = world.snapshot()
    lines = [f"tick {s['tick']}", f"pop  {s['population']}", f"food {s['food']}",
             f"corpses {s['corpses']}", f"births {s['births']}", f"deaths {s['deaths']}"]
    for i, t in enumerate(lines):
        surf.blit(font.render(t, True, (220, 220, 230)), (x0 + 12, 12 + i * 22))

    a = world.agents.get(selected) if selected is not None else None
    if a is None:
        return
    y = 12 + len(lines) * 22 + 16
    for t in [f"agent #{a.id}", f"energy {a.energy:.1f}", f"age {a.age}",
              f"genes {a.brain.genome.size}"]:
        surf.blit(small.render(t, True, (200, 210, 230)), (x0 + 12, y)); y += 18
    _draw_net(surf, a.brain, x0 + 12, y + 8, panel_w - 24, 190)


def _draw_net(surf, brain, x, y, w, h):
    """Draw the fixed MLP: input column, hidden column, output column, edges
    coloured by weight sign and thickened by magnitude."""
    cols = [brain.n_in, brain.n_hidden, N_ACTIONS]
    xs = [x + int(w * f) for f in (0.0, 0.5, 1.0)]
    pos = []
    for ci, n in enumerate(cols):
        col = []
        for j in range(n):
            yy = y + int((j + 0.5) / n * h)
            col.append((xs[ci], yy))
        pos.append(col)
    for (mat, a_i, b_i) in ((brain.w1, 0, 1), (brain.w2, 1, 2)):
        m = float(np.abs(mat).max()) + 1e-6
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                wv = mat[i, j]
                c = (90, 160, 255) if wv >= 0 else (255, 110, 90)
                pygame.draw.line(surf, c, pos[a_i][i], pos[b_i][j],
                                 1 + int(2 * abs(wv) / m))
    for ci, col in enumerate(pos):
        for j, (px, py) in enumerate(col):
            pygame.draw.circle(surf, (230, 230, 240), (px, py), 4)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.app", description="watch a vivarium")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=60)
    p.add_argument("--cell", type=int, default=7, help="pixels per cell")
    a = p.parse_args(argv)

    cfg = Config(size=a.size, seed=a.seed, init_population=a.population)
    world = World(cfg)
    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    panel_w = 260
    screen = pygame.display.set_mode((world.w * a.cell + panel_w, world.h * a.cell))
    pygame.display.set_caption("vivarium")
    clock = pygame.time.Clock()

    paused, steps_per_frame, selected = False, 1, None
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
                    steps_per_frame = min(64, steps_per_frame * 2)
                elif e.key == pygame.K_MINUS:
                    steps_per_frame = max(1, steps_per_frame // 2)
                elif e.key == pygame.K_r:
                    world._reseed()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                gx, gy = mx // a.cell, my // a.cell
                if 0 <= gy < world.h and 0 <= gx < world.w and world.occ[gy, gx] >= 0:
                    selected = int(world.occ[gy, gx])
        if not paused:
            for _ in range(steps_per_frame):
                world.step()
        if selected is not None and selected not in world.agents:
            selected = None
        screen.blit(draw(world, a.cell, panel_w, font, small, selected), (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
