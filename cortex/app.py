"""Play the cortex world yourself — top-down, agent-centred, keyboard.

    .\\.venv\\Scripts\\python.exe -m cortex.app --seed 2

This exists BEFORE the brain on purpose (docs/cortex-world.md): a human at the
keyboard is the baseline that gives a brain's score a meaning, and the fastest
detector of a world that is boring or trivially solvable.

The keyboard drives `World.step(thrust, turn, interact)` — the very call the brain
will use. Layout is the agreed one: the world on the left, the SENSES on the right
(and later the brain).

The sense panel is drawn from the SENSOR VECTOR ALONE — never from the world. That is
deliberate: press TAB and the map goes dark, leaving only what the agent itself
receives. If a human cannot play like that, the brain has no chance, and we would
rather learn it now than after wiring four thousand cells to a blind eye.

Controls: W/Up thrust · A/D or Left/Right turn · E interact (ONE verb — eats the
food in reach, else opens/closes the door, chips the breakable or shoves the block
in front) · TAB blind mode · SPACE pause · P screenshot · H help · ESC/Q quit.

`draw()` returns a Surface so the view renders headless (SDL dummy) in tests.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
import time
from collections import deque

import numpy as np
import pygame

from cortex.brainpanel import BRAIN_W, BrainPanel
from cortex.senses import (INTERO, N_TOUCH, ODOUR_CH, RAY_CH, TOUCH_CH, ScentField,
                           layout, sense)
from cortex.session import record, summarise
from cortex.tiles import HP, NAMES, RGB
from cortex.world import FOOD_NAMES, FOOD_RGB, N_ODOUR, Config, World

PANEL_W = 360
BG = (12, 12, 16)
FG = (210, 210, 220)
DIM = (120, 120, 132)
ACCENT = (150, 200, 255)
GRID = (44, 46, 56)
# four neutral, distinguishable colours for the odorants. NOT the food colours: the
# agent is not told which smell belongs to which morsel, and neither is the player.
ODOUR_COL = ((235, 110, 110), (235, 200, 110), (120, 220, 190), (170, 150, 240))
INTERO_NAMES = ("energy", "speed", "turn", "sin h", "cos h", "age", "poison", "bias")
TOUCH_NAMES = ("front", "right", "back", "left")
# THE TRACK: how far back the agent's own path is remembered, and how often it is
# sampled. 600 samples every 3 ticks is ~1800 ticks of history — long enough that a
# circler closes a visible loop, short enough that a forager's line does not fill the
# screen with everywhere it has ever been.
TRAIL_LEN, TRAIL_EVERY = 600, 3
TRAIL_NEW, TRAIL_OLD = (120, 235, 190), (60, 70, 110)

HELP = ["W / Up          thrust", "A D / Left Right    turn",
        "E               interact:", "                eat / open / break / push",
        "TAB             blind mode (senses only)",
        "B               brain drives / you drive",
        "V               brain view: type/activity/hormone/organs",
        "T               animate the thinking sub-ticks",
        "R               the track it has walked (fading)",
        "- / =           world speed (1/20x .. 8x)",
        "[ / ]           slower / faster sub-tick animation",
        "click a cell    show its perceptron",
        "SPACE           pause", "P               screenshot", "H               this help",
        "ESC / Q         quit"]


def _hsv_rgb(hue_cos: float, hue_sin: float, sat: float, val: float):
    """Rebuild a colour from the two circular hue channels the eye actually reports."""
    hue = (math.atan2(hue_sin, hue_cos) / (2 * math.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, max(0.0, min(1.0, sat)), max(0.0, min(1.0, val)))
    return int(r * 255), int(g * 255), int(b * 255)


class App:
    def __init__(self, seed: int = 2, view: int = 65, cell: int = 12,
                 size: int | None = 512, mind=None):
        self.world = World(seed=seed, cfg=Config(size=size))
        self.field = ScentField(self.world.cfg.smell_radius)
        self.layout = layout(self.world.cfg.n_rays)
        self.vec = sense(self.world, self.field)
        # a brain, if one is playing. `auto` says who is driving right now — the same
        # world, the same action tuple, so you can take the wheel off it and hand it
        # back mid-life and compare like for like.
        self.mind = mind
        self.auto = mind is not None
        self.brain_panel = BrainPanel(self) if mind is not None else None
        if mind is not None:
            mind.attach(self.world)
        self.view = int(view) | 1                 # odd, so the agent sits dead centre
        self.cell = int(cell)
        self.paused = False
        # SPEED as a ladder, not a multiplier on the frame rate: the display stays at a
        # steady 30 fps and the WORLD runs slower or faster inside it. Slowing down is
        # the point — a thought is twenty sub-ticks and the brain re-thinks every few
        # world ticks, so at 1x the animation is showing frames from a different thought
        # each time and reads as flicker rather than a wave.
        self.speeds = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        self.speed_i = 4
        self._accum = 0.0
        self.blind = False
        # THE TRACK. Where it has actually BEEN, fading with age. The probe strips have
        # always drawn one and it is the single most legible thing on them — circling
        # in place, oscillating between two smells, walking a wall look like nothing in
        # a number and are unmistakable as a line. The live HUD had no equivalent, so
        # the same failures were invisible in the world they actually happen in.
        self.trail: deque = deque(maxlen=TRAIL_LEN)
        self.trail_on = True
        self.show_help = False
        self.fps = 0.0
        self.w = self.view * self.cell + PANEL_W + (BRAIN_W if mind else 0)
        self.h = self.view * self.cell
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 11)
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

    @property
    def speed(self) -> float:
        return self.speeds[self.speed_i]

    def slower(self) -> None:
        self.speed_i = max(0, self.speed_i - 1)

    def faster(self) -> None:
        self.speed_i = min(len(self.speeds) - 1, self.speed_i + 1)

    def frame(self, keys) -> int:
        """Advance the world by however much this frame is worth. Returns the number of
        ticks actually taken, so a fractional speed simply skips most frames."""
        if self.paused:
            return 0
        self._accum += self.speed
        n = 0
        while self._accum >= 1.0:
            self._accum -= 1.0
            self.step(keys)
            n += 1
        if self.brain_panel is not None and self.brain_panel.trace_on:
            self.brain_panel.advance_trace()
        return n

    def step(self, keys) -> None:
        """One world tick, driven by whoever has the wheel. In brain mode the panel
        shows the sensor vector the BRAIN was handed — stale between thoughts, which
        is honest: that is what it is acting on."""
        if self.auto and self.mind is not None:
            self.mind.step(self.world)
            if self.mind.sensors is not None:
                self.vec = self.mind.sensors
        else:
            self.world.step(*self.action(keys))
            self.vec = sense(self.world, self.field)
        if self.world.tick % TRAIL_EVERY == 0:
            a = self.world.agent
            self.trail.append((a.y, a.x))

    # --- render -------------------------------------------------------------------
    def draw(self) -> pygame.Surface:
        surf = pygame.Surface((self.w, self.h))
        surf.fill(BG)
        if self.blind:
            self._draw_blind(surf)
        else:
            self._draw_world(surf)
        self._draw_panel(surf)
        if self.brain_panel is not None:
            self.brain_panel.draw(surf, self.view * self.cell + PANEL_W)
        return surf

    def _draw_blind(self, surf: pygame.Surface) -> None:
        rect = pygame.Rect(0, 0, self.view * self.cell, self.view * self.cell)
        pygame.draw.rect(surf, (7, 7, 10), rect)
        pygame.draw.rect(surf, (40, 40, 48), rect, 1)
        txt = self.font_big.render("BLIND — playing by the senses alone", True, (70, 70, 86))
        surf.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - 10))
        sub = self.font.render("TAB to look again", True, (52, 52, 64))
        surf.blit(sub, (rect.centerx - sub.get_width() // 2, rect.centery + 16))

    def _draw_world(self, surf: pygame.Surface) -> None:
        """The camera window, drawn as a small pixel image scaled up — one pixel per
        tile — then blitted at a FRACTIONAL offset so the camera follows the float
        agent smoothly instead of snapping cell by cell."""
        a, v, cs = self.world.agent, self.view, self.cell
        half = v // 2
        y0, x0 = int(math.floor(a.y)) - half - 1, int(math.floor(a.x)) - half - 1
        mats = self.world.tiles.region(y0, x0, v + 2, v + 2)
        img = RGB[mats].astype(np.float64)                     # (h, w, 3)
        # DAMAGE is visible: a chipped tile darkens toward its break point. The agent's
        # eye reads the same `damage_frac`, so player and brain see one world.
        for (dy, dx) in self.world.damage:
            i, j = dy - y0, dx - x0
            if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
                img[i, j] *= 1.0 - 0.55 * self.world.damage_frac(dy, dx)
        img = img.astype(np.uint8)
        # pygame surfaces are (x, y); numpy gives (y, x) — swap before make_surface
        tile_surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        tile_surf = pygame.transform.scale(tile_surf, ((v + 2) * cs, (v + 2) * cs))
        offy = (a.y - math.floor(a.y)) * cs
        offx = (a.x - math.floor(a.x)) * cs
        world_rect = pygame.Rect(0, 0, v * cs, v * cs)
        surf.set_clip(world_rect)
        # world Y maps to screen (Y + 0.5 - y0)*cs + blit_y; solving for the agent
        # landing on the view centre gives blit = -(1 + frac)*cs on each axis
        surf.blit(tile_surf, (-cs - offx, -cs - offy))
        cy = cx = v * cs // 2
        self._draw_trail(surf, cy, cx, cs)
        for s in self.world.nearby_food(a.y, a.x, half * 1.5):
            sy, sx = cy + (s.y - a.y) * cs, cx + (s.x - a.x) * cs
            if not world_rect.collidepoint(sx, sy):
                continue
            col = tuple(int(q) for q in FOOD_RGB[s.kind])
            pygame.draw.circle(surf, (14, 14, 18), (int(sx), int(sy)), max(3, cs // 3) + 1)
            pygame.draw.circle(surf, col, (int(sx), int(sy)), max(2, cs // 3))
        if self.world.nearby_food(a.y, a.x, self.world.cfg.eat_reach):
            pygame.draw.circle(surf, (90, 230, 130), (cx, cy),
                               int(self.world.cfg.eat_reach * cs), 1)
        fy, fx = self.world.facing_cell()
        ty, tx = cy + (fy - a.y) * cs, cx + (fx - a.x) * cs
        pygame.draw.rect(surf, (90, 90, 110),
                         pygame.Rect(int(tx - cs / 2), int(ty - cs / 2), cs, cs), 1)
        rad = max(4, int(self.world.cfg.radius * cs))
        hx = cx + math.cos(a.heading) * cs * 2.0
        hy = cy + math.sin(a.heading) * cs * 2.0
        pygame.draw.line(surf, (255, 80, 80), (cx, cy), (hx, hy), 2)
        pygame.draw.circle(surf, (10, 10, 14), (cx, cy), rad + 2)     # outline, so the
        pygame.draw.circle(surf, (250, 250, 255), (cx, cy), rad)      # body always pops
        pygame.draw.circle(surf, (90, 170, 255), (cx, cy), max(1, rad // 2))
        surf.set_clip(None)
        pygame.draw.rect(surf, (40, 40, 48), world_rect, 1)

    def _draw_trail(self, surf, cy: int, cx: int, cs: int) -> None:
        """The path already walked, oldest end faded out.

        Drawn onto a per-pixel-alpha overlay rather than blended toward the background
        colour, because the ground it crosses is not one colour — a track that fades
        into stone would still be opaque over grass. Age is carried by BOTH alpha and
        hue, so the direction of travel is readable in a still screenshot, which is the
        form these mostly get looked at in."""
        if not self.trail_on or len(self.trail) < 2:
            return
        a = self.world.agent
        side = self.view * cs
        layer = pygame.Surface((side, side), pygame.SRCALPHA)
        pts = list(self.trail)
        n = len(pts)
        prev = None
        for i, (ty, tx) in enumerate(pts):
            sy, sx = cy + (ty - a.y) * cs, cx + (tx - a.x) * cs
            age = i / max(1, n - 1)                    # 0 oldest .. 1 newest
            if prev is not None:
                # a jump means the pair straddles a teleport or a paused stretch —
                # joining them would draw a line through terrain never crossed
                if abs(sy - prev[0]) < side / 2 and abs(sx - prev[1]) < side / 2:
                    col = tuple(int(o + (nw - o) * age)
                                for o, nw in zip(TRAIL_OLD, TRAIL_NEW))
                    pygame.draw.line(layer, col + (int(25 + 190 * age * age),),
                                     prev, (sx, sy), max(1, cs // 6))
            prev = (sx, sy)
        surf.blit(layer, (0, 0))

    # --- the sense panel ------------------------------------------------------------
    def _draw_panel(self, surf: pygame.Surface) -> None:
        a, w = self.world.agent, self.world
        x = self.view * self.cell + 14
        pw = PANEL_W - 28
        y = 10

        def line(text, colour=FG, font=None, dy=19):
            nonlocal y
            surf.blit((font or self.font).render(text, True, colour), (x, y))
            y += dy

        # --- energy: it is LIFE, so it gets the most legible widget
        frac = max(0.0, min(1.0, a.energy / w.cfg.max_energy))
        col = (120, 230, 140) if a.energy > 40 else \
              (240, 200, 90) if a.energy > 15 else (240, 100, 96)
        pygame.draw.rect(surf, (34, 34, 42), pygame.Rect(x, y, pw, 13))
        pygame.draw.rect(surf, col, pygame.Rect(x, y, int(pw * frac), 13))
        pygame.draw.rect(surf, (70, 70, 84), pygame.Rect(x, y, pw, 13), 1)
        if a.poison > 0:
            pygame.draw.rect(surf, (190, 90, 210), pygame.Rect(x, y + 13, int(
                pw * min(1.0, a.poison / 30.0)), 3))
        y += 18
        eaten = " ".join(f"{FOOD_NAMES[k][:3]}{n}" for k, n in enumerate(a.ate) if n)
        line(f"E:{a.energy:6.1f}  t{w.tick}  {eaten}", col)

        aff = w.afforded()
        line("E: " + (aff if aff else "—"),
             (150, 240, 170) if aff else DIM, self.font_big)
        line(w.last_event or "", DIM)
        if not a.alive:
            line("DEAD — starved", (250, 90, 90), self.font_big)
        elif self.paused:
            line("PAUSED", (250, 210, 90))
        else:
            y += 4

        y = self._draw_radar(surf, x, y, pw)
        y = self._draw_odour(surf, x, y, pw)
        y = self._draw_touch_intero(surf, x, y, pw)

        sp = self.speed
        line(f"fps {self.fps:.0f}   speed {('x%g' % sp) if sp >= 1 else ('1/%gx' % (1 / sp))}"
             f"   seed {w.seed}   {NAMES[w.tiles.material(*w.facing_cell())]}",
             (250, 210, 130) if sp != 1.0 else DIM, self.small, dy=15)
        line("TAB blind   H help", DIM, self.small, dy=15)
        if self.show_help:
            for h in HELP:
                line(h, DIM, self.small, dy=14)

    def _draw_radar(self, surf, x, y, pw) -> int:
        """VISION, drawn from the sensor vector alone: a fan of contacts, angle = ray,
        radius = how far it hit, colour = what it looked like. An old radar scope with
        coloured returns — which is exactly what a ray sensor is."""
        n = self.world.cfg.n_rays
        half = self.world.cfg.fov / 2
        rays = self.vec[:n * RAY_CH].reshape(n, RAY_CH)
        # the fan only covers the FIELD OF VIEW, so it is a wedge, not a disc: its
        # lateral reach is R*sin(half) and its vertical reach is R
        R = int(min((pw / 2) / max(0.2, math.sin(half)), (self.h - y) * 0.40))
        if R < 20:
            return y
        ccx, ccy = x + pw // 2, y + R
        arc0, arc1 = math.pi / 2 - half, math.pi / 2 + half   # pygame arcs: 0 = east, ccw
        reach = self.world.cfg.ray_reach
        # RADIAL SCALE IS sqrt(d/reach), not d/reach. A true-scale scope crushes the
        # first few cells into the apex, and the first few cells are where everything
        # that matters happens — food is taken at 1.1, walls are hit at 0.5.
        for cells in (1, 4, 10, 20):                          # rings labelled in CELLS
            rr = int(R * math.sqrt(min(1.0, cells / reach)))
            pygame.draw.arc(surf, GRID, pygame.Rect(ccx - rr, ccy - rr, 2 * rr, 2 * rr),
                            arc0, arc1, 1)
            surf.blit(self.small.render(str(cells), True, (60, 62, 74)),
                      (ccx + 2, ccy - rr - 1))
        for edge in (-half, half):
            pygame.draw.line(surf, GRID, (ccx, ccy),
                             (ccx + math.sin(edge) * R, ccy - math.cos(edge) * R))
        step = 2 * half / n
        for i in range(n):
            rel = -half + (i + 0.5) * step
            near = float(rays[i, 4])
            sx, sy = math.sin(rel), -math.cos(rel)
            if near <= 0.001:                                 # nothing out there
                pygame.draw.line(surf, (28, 30, 38),
                                 (ccx + sx * (R - 4), ccy + sy * (R - 4)),
                                 (ccx + sx * R, ccy + sy * R))
                continue
            dist_r = math.sqrt(max(0.0, 1.0 - near)) * R      # nearness -> radius
            px, py = ccx + sx * dist_r, ccy + sy * dist_r
            colour = _hsv_rgb(rays[i, 0], rays[i, 1], rays[i, 2], rays[i, 3])
            pygame.draw.line(surf, tuple(int(q * 0.30) for q in colour),
                             (ccx, ccy), (px, py))
            size = 2 if rays[i, 5] > 0.5 else 3               # solid = a hard pip,
            pygame.draw.circle(surf, colour, (int(px), int(py)), size)    # soft = a blob
        pygame.draw.circle(surf, (240, 240, 250), (ccx, ccy), 3)
        surf.blit(self.small.render("sight   (rings = cells, sqrt scale)", True, DIM),
                  (x, ccy - 10))
        return ccy + 6

    def _draw_odour(self, surf, x, y, pw) -> int:
        """SMELL: one row per odorant — how strong, and which way the walkable path
        goes (the needle, not the sightline)."""
        lo, _ = self.layout["slices"]["odour"]
        surf.blit(self.small.render("smell   (bar = how strong, dial = which way to walk)",
                                    True, DIM), (x, y))
        y += 14
        dial = 11
        bw = pw - 2 * dial - 10
        for j in range(N_ODOUR):
            amp, fwd, lat = self.vec[lo + j * ODOUR_CH: lo + j * ODOUR_CH + 3]
            pygame.draw.rect(surf, (30, 30, 38), pygame.Rect(x, y + 2, bw, 10))
            pygame.draw.rect(surf, ODOUR_COL[j],
                             pygame.Rect(x, y + 2, int(bw * min(1.0, float(amp))), 10))
            cxx, cyy = x + bw + 8 + dial, y + 7
            pygame.draw.circle(surf, GRID, (cxx, cyy), dial, 1)
            if amp > 0.001:                                  # forward is UP on the dial
                pygame.draw.line(surf, ODOUR_COL[j], (cxx, cyy),
                                 (cxx + float(lat) * dial, cyy - float(fwd) * dial), 2)
            y += 2 * dial + 2
        return y + 4

    def _draw_touch_intero(self, surf, x, y, pw) -> int:
        tlo, _ = self.layout["slices"]["touch"]
        ilo, _ = self.layout["slices"]["intero"]
        surf.blit(self.small.render("touch", True, DIM), (x, y))
        surf.blit(self.small.render("interoception", True, DIM), (x + 84, y))
        y += 13
        # touch: a little compass of what is pressed against the body, front up
        cxx, cyy, s = x + 30, y + 30, 20
        for k in range(N_TOUCH):
            press = float(self.vec[tlo + k * TOUCH_CH])
            hue = float(self.vec[tlo + k * TOUCH_CH + 1])
            ang = -math.pi / 2 + k * math.pi / 2
            col = _hsv_rgb(math.cos(hue * 2 * math.pi), math.sin(hue * 2 * math.pi),
                           0.45, 0.35 + 0.65 * press) if press > 0 else (32, 32, 40)
            px, py = cxx + math.cos(ang) * s, cyy + math.sin(ang) * s
            pygame.draw.circle(surf, col, (int(px), int(py)), 9)
            surf.blit(self.small.render(TOUCH_NAMES[k][0].upper(), True,
                                        (16, 16, 20) if press > 0 else (70, 70, 84)),
                      (int(px) - 3, int(py) - 7))
        pygame.draw.circle(surf, (200, 200, 210), (cxx, cyy), 4)
        # interoception: eight small labelled bars, zero in the middle
        bx, lw, row = x + 84, 40, 13
        bw = pw - 84 - lw
        for k in range(INTERO):
            v = float(self.vec[ilo + k])
            by = y + k * row
            surf.blit(self.small.render(INTERO_NAMES[k], True, DIM), (bx, by))
            pygame.draw.rect(surf, (30, 30, 38), pygame.Rect(bx + lw, by + 1, bw, 9))
            mid = bx + lw + bw // 2
            width = int(abs(v) * bw / 2)
            pygame.draw.rect(surf, (110, 170, 230),
                             pygame.Rect(mid if v >= 0 else mid - width, by + 1, width, 9))
            pygame.draw.line(surf, GRID, (mid, by + 1), (mid, by + 10))
        return y + max(INTERO * row, 62) + 6


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play the cortex world.")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--view", type=int, default=65, help="camera window in cells")
    p.add_argument("--cell", type=int, default=12, help="pixels per cell")
    p.add_argument("--size", type=int, default=512, help="world side; 0 = unbounded")
    p.add_argument("--blind", action="store_true", help="start in senses-only mode")
    p.add_argument("--player", default="human", help="name for the session log")
    p.add_argument("--brain", action="store_true", help="let a fresh brain play")
    p.add_argument("--brain-seed", type=int, default=0, help="which fresh brain")
    p.add_argument("--load-brain", default="", help="a saved genome (json) to play")
    p.add_argument("--think-every", type=int, default=2, help="world ticks per thought")
    p.add_argument("--no-reflex", action="store_true",
                   help="strip the INNATE ARCS — the smell drive, the pioneer tracts "
                        "and the read-out — and watch the same brain without them. "
                        "This is the ablation arm, made watchable")
    p.add_argument("--probe", default="", help="drop into a PROBE arena instead of the "
                                               "world: open/corner/door/rubble/poison/patience")
    p.add_argument("--no-log", action="store_true", help="do not record the session")
    p.add_argument("--fps", type=int, default=30, help="display rate; speed is separate")
    args = p.parse_args(argv)

    pygame.init()
    mind = None
    if args.brain or args.load_brain:
        from cortex import brain as B
        from cortex.agent import Mind, fresh_genome
        if args.load_brain:
            import json
            genome = B.Genome.from_dict(json.load(open(args.load_brain, encoding="utf-8")))
        else:
            genome = fresh_genome(seed=args.brain_seed)
        if args.no_reflex:
            from cortex import reflex
            reflex.silence(genome)     # the tracts never grow: `_pioneer` needs the gain
            genome.program.update({"reflex_gain": 0.0})
        mind = Mind(genome, think_every=args.think_every)
        if args.player == "human":
            args.player = f"brain-{args.brain_seed}"
            if args.no_reflex:
                args.player += "-noreflex"
    app = App(seed=args.seed, view=args.view, cell=args.cell, size=(args.size or None),
              mind=mind)
    if args.probe:
        # the same arena the observer builds, but LIVE and with the whole HUD — so a
        # probe can be walked through by hand instead of only read as a track on a PNG
        from cortex import probes as _probes
        app.world, _ = _probes.build(args.probe)
        app.field = ScentField(app.world.cfg.smell_radius)
        if mind is not None:
            mind.attach(app.world)
        app.vec = sense(app.world, app.field)
        pygame.display.set_caption(f"cortex — probe: {args.probe}")
    app.blind = args.blind
    screen = pygame.display.set_mode((app.w, app.h))
    pygame.display.set_caption(f"cortex — seed {args.seed}")
    clock = pygame.time.Clock()
    t0, frames = time.time(), 0
    running, logged = True, False

    def log(note):
        """A human life is the calibration every other number is read against, so it
        is recorded on death AND on quit — a session you walked away from still says
        how long you lasted."""
        nonlocal logged
        if logged or args.no_log or app.world.tick < 50:
            return
        logged = True
        rec = record(app.world, args.player,
                     note=note + (" blind" if app.blind else ""))
        print(f"session: {rec['ticks']} ticks, {rec['meals']} meals, "
              f"{rec['gained']:+.0f} energy eaten, {rec['distance']:.0f} cells walked")
        print(summarise())

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and app.brain_panel is not None:
                app.brain_panel.click(ev.pos)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE and app.brain_panel is not None                         and app.brain_panel.picked is not None:
                    app.brain_panel.picked = None        # unpick before quitting
                elif ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    app.paused = not app.paused
                elif ev.key == pygame.K_TAB:
                    app.blind = not app.blind
                elif ev.key == pygame.K_b and app.mind is not None:
                    app.auto = not app.auto          # take the wheel, or give it back
                elif ev.key == pygame.K_v and app.brain_panel is not None:
                    app.brain_panel.cycle_view()
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    app.slower()                         # 1/20x .. 8x world speed
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    app.faster()
                elif ev.key == pygame.K_LEFTBRACKET and app.brain_panel is not None:
                    app.brain_panel.slower()             # sub-tick animation rate
                elif ev.key == pygame.K_RIGHTBRACKET and app.brain_panel is not None:
                    app.brain_panel.faster()
                elif ev.key == pygame.K_t and app.mind is not None:
                    bp = app.brain_panel
                    bp.trace_on = not bp.trace_on    # watch a thought cross the sheet
                    bp.trace_i = 0
                    app.mind.phenotype.record = bp.trace_on
                elif ev.key == pygame.K_r:
                    app.trail_on = not app.trail_on      # the track it has walked
                elif ev.key == pygame.K_h:
                    app.show_help = not app.show_help
                elif ev.key == pygame.K_p:
                    os.makedirs("out/cortex", exist_ok=True)
                    name = f"out/cortex/t{app.world.tick:07d}_seed{args.seed}.png"
                    pygame.image.save(app.draw(), name)
                    print("saved", name)
        app.frame(pygame.key.get_pressed())
        if not app.world.agent.alive:
            log("starved")
        screen.blit(app.draw(), (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)
        frames += 1
        if frames % 20 == 0:
            now = time.time()
            app.fps = 20.0 / max(1e-6, now - t0)
            t0 = now
    log("quit")
    pygame.quit()


if __name__ == "__main__":
    main()
