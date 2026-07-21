"""Play the cortex world yourself — top-down, agent-centred, keyboard.

    .\\.venv\\Scripts\\python.exe -m cortex.app --seed 1

This exists BEFORE the brain on purpose (docs/cortex-world.md): a human at the
keyboard is the baseline that gives a brain's score a meaning, and the fastest
detector of a world that is boring or trivially solvable.

The keyboard drives `World.step(thrust, turn, interact)` — the very call the brain
will use. Layout is already the agreed one: the world on the left, a panel on the
right that will grow the sensor radar and then the brain inspector.

Controls: W/Up thrust · A/D or Left/Right turn · E interact (inert this stage) ·
SPACE pause · P screenshot · H help · ESC/Q quit.

`draw()` returns a Surface so the view renders headless (SDL dummy) in tests.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pygame

from cortex.tiles import RGB
from cortex.world import Config, World

PANEL_W = 300
BG = (12, 12, 16)
FG = (210, 210, 220)
DIM = (120, 120, 132)

HELP = ["W / Up      thrust", "A D / Left Right   turn", "E           interact (inert)",
        "SPACE       pause", "P           screenshot", "H           this help",
        "ESC / Q     quit"]


class App:
    def __init__(self, seed: int = 1, view: int = 65, cell: int = 12,
                 size: int | None = 512):
        self.world = World(seed=seed, cfg=Config(size=size))
        self.view = int(view) | 1                 # odd, so the agent sits dead centre
        self.cell = int(cell)
        self.paused = False
        self.show_help = False
        self.fps = 0.0
        self.w = self.view * self.cell + PANEL_W
        self.h = self.view * self.cell
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 17, bold=True)

    # --- input --------------------------------------------------------------------
    def action(self, keys) -> tuple[float, float, float]:
        """Keyboard -> the SAME action tuple the brain emits. One code path."""
        thrust = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        turn = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            turn -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            turn += 1.0
        interact = 1.0 if keys[pygame.K_e] else 0.0
        return thrust, turn, interact

    # --- render -------------------------------------------------------------------
    def draw(self) -> pygame.Surface:
        surf = pygame.Surface((self.w, self.h))
        surf.fill(BG)
        self._draw_world(surf)
        self._draw_panel(surf)
        return surf

    def _draw_world(self, surf: pygame.Surface) -> None:
        """The camera window, drawn as a small pixel image scaled up — one pixel per
        tile — then blitted at a FRACTIONAL offset so the camera follows the float
        agent smoothly instead of snapping cell by cell."""
        a, v, cs = self.world.agent, self.view, self.cell
        half = v // 2
        # top-left cell of a window one cell larger than the view, so the fractional
        # scroll offset always has a spare row/column to reveal
        y0, x0 = int(math.floor(a.y)) - half - 1, int(math.floor(a.x)) - half - 1
        mats = self.world.tiles.region(y0, x0, v + 2, v + 2)
        img = RGB[mats]                                        # (h, w, 3) uint8
        # pygame surfaces are (x, y); numpy gives (y, x) — swap before make_surface
        tile_surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        tile_surf = pygame.transform.scale(tile_surf, ((v + 2) * cs, (v + 2) * cs))
        # the agent's fractional position inside its cell decides the sub-cell scroll
        offy = (a.y - math.floor(a.y)) * cs
        offx = (a.x - math.floor(a.x)) * cs
        world_rect = pygame.Rect(0, 0, v * cs, v * cs)
        surf.set_clip(world_rect)
        # world Y maps to screen (Y + 0.5 - y0)*cs + blit_y; solving for the agent
        # landing on the view centre gives blit = -(1 + frac)*cs on each axis
        surf.blit(tile_surf, (-cs - offx, -cs - offy))
        # the agent is always at the exact centre of the world view
        cy = cx = v * cs // 2
        rad = max(4, int(self.world.cfg.radius * cs))
        hx = cx + math.cos(a.heading) * cs * 2.0
        hy = cy + math.sin(a.heading) * cs * 2.0
        pygame.draw.line(surf, (255, 80, 80), (cx, cy), (hx, hy), 2)
        pygame.draw.circle(surf, (10, 10, 14), (cx, cy), rad + 2)     # outline, so the
        pygame.draw.circle(surf, (250, 250, 255), (cx, cy), rad)      # body always pops
        pygame.draw.circle(surf, (90, 170, 255), (cx, cy), max(1, rad // 2))
        surf.set_clip(None)
        pygame.draw.rect(surf, (40, 40, 48), world_rect, 1)

    def _draw_panel(self, surf: pygame.Surface) -> None:
        a, w = self.world.agent, self.world
        x = self.view * self.cell + 14
        y = 12

        def line(text, colour=FG, font=None):
            nonlocal y
            surf.blit((font or self.font).render(text, True, colour), (x, y))
            y += 20

        line("cortex — stage 1", (150, 200, 255), self.font_big)
        y += 6
        line(f"tick   {w.tick}")
        line(f"pos    {a.y:7.2f} {a.x:7.2f}")
        line(f"head   {math.degrees(a.heading):6.1f} deg")
        line(f"speed  {a.speed:.3f}")
        line(f"energy {a.energy:7.2f}", (120, 230, 140) if a.energy > 25 else (240, 120, 110))
        line(f"age    {a.age}")
        line(f"seed   {w.seed}   size {w.cfg.size}")
        line(f"fps    {self.fps:.0f}")
        fy, fx = w.facing_cell()
        line(f"facing {fy} {fx}", DIM)
        if not a.alive:
            y += 6
            line("DEAD (energy)", (250, 90, 90), self.font_big)
        if self.paused:
            y += 6
            line("PAUSED", (250, 210, 90))
        y += 10
        line("H — help", DIM)
        if self.show_help:
            y += 4
            for h in HELP:
                line(h, DIM)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play the cortex world (stage 1).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--view", type=int, default=65, help="camera window in cells")
    p.add_argument("--cell", type=int, default=12, help="pixels per cell")
    p.add_argument("--size", type=int, default=512, help="world side; 0 = unbounded")
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args(argv)

    pygame.init()
    app = App(seed=args.seed, view=args.view, cell=args.cell,
              size=(args.size or None))
    screen = pygame.display.set_mode((app.w, app.h))
    pygame.display.set_caption(f"cortex — seed {args.seed}")
    clock = pygame.time.Clock()
    t0, frames = time.time(), 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    app.paused = not app.paused
                elif ev.key == pygame.K_h:
                    app.show_help = not app.show_help
                elif ev.key == pygame.K_p:
                    os.makedirs("out/cortex", exist_ok=True)
                    name = f"out/cortex/t{app.world.tick:07d}_seed{args.seed}.png"
                    pygame.image.save(app.draw(), name)
                    print("saved", name)
        if not app.paused:
            app.world.step(*app.action(pygame.key.get_pressed()))
        screen.blit(app.draw(), (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)
        frames += 1
        if frames % 20 == 0:
            now = time.time()
            app.fps = 20.0 / max(1e-6, now - t0)
            t0 = now
    pygame.quit()


if __name__ == "__main__":
    main()
