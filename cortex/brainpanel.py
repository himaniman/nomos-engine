"""The brain column of the HUD — everything about the sheet that can be looked at.

What it shows, and why each of them: the TISSUE itself (shape is the whole point of a
cellular brain, so you have to be able to see it), what its cells ARE (type) and what
they are DOING (state), where the senses enter and the commands leave (organs), the
two CLOCKS and how they line up with the world's, what is being commanded right now,
and the WEIGHTS that plasticity is moving within the life.

Press T and the sheet animates its SUB-TICKS: one frame per round of relaxation, so
you watch a thought cross the tissue from an organ to an effector. The trace comes
from the same arithmetic as an ordinary act, so watching an agent think never changes
what it thinks.

The sheet is drawn as a small image and scaled up — the same trick the world view
uses. At a few thousand cells, one pygame rect per cell would cost more than the
thinking does.
"""

from __future__ import annotations

import numpy as np
import pygame

BRAIN_W = 470
FG = (210, 210, 220)
DIM = (120, 120, 132)
ACCENT = (150, 200, 255)
GRID = (44, 46, 56)

# one colour per cell TYPE, so differentiation is visible at a glance
TYPE_COL = ((90, 160, 250), (250, 150, 90), (140, 230, 140), (230, 130, 220),
            (240, 220, 120), (120, 220, 230))
# one per sensory MODALITY, matching cortex.senses: 1..6 are the per-ray channels,
# 7 odour, 8 touch, 9 interoception, 0 = hears the whole vector
MOD_COL = ((235, 235, 235), (250, 90, 90), (250, 150, 80), (245, 215, 90),
           (140, 230, 130), (110, 200, 250), (170, 140, 245),
           (235, 110, 200), (120, 220, 200), (255, 180, 140))
MOD_NAME = ("all", "hue+", "hue-", "sat", "light", "near", "solid",
            "smell", "touch", "inner")
VIEWS = ("type", "activity", "chemistry", "hormone", "organs")


def _heat(W) -> np.ndarray:
    """A weight matrix as an image: red positive, blue negative, dark for a weight
    doing nothing. Scaled to its own largest magnitude, so a small rule is as legible
    as a big one."""
    A = np.asarray(W, dtype=np.float64)
    n = A / max(1e-9, float(np.abs(A).max()))
    img = np.zeros(A.shape + (3,), dtype=np.float64)
    img[..., 0] = np.clip(n, 0, 1) * 250
    img[..., 2] = np.clip(-n, 0, 1) * 250
    img[..., 1] = np.abs(n) * 40
    return img.astype(np.uint8)


class BrainPanel:
    def __init__(self, app):
        self.app = app
        self.view = 0
        self.trace_on = False
        self.trace_i = 0
        # The animation runs on its OWN clock, not the world's. A thought is ~20
        # sub-ticks and the world runs at 60 fps, so tying them together flashes the
        # whole relaxation past in a third of a second and you see nothing.
        self.trace_every = 6            # rendered frames per sub-tick shown
        self._frames = 0
        self.picked = None              # (y, x) of the cell being inspected
        self._sheet_rect = None         # where the sheet was drawn, for hit-testing
        self._sheet_scale = 1
        self._sheet_origin = (0, 0)

    def cycle_view(self) -> str:
        self.view = (self.view + 1) % len(VIEWS)
        return VIEWS[self.view]

    def slower(self) -> None:
        self.trace_every = min(60, self.trace_every + 2)

    def faster(self) -> None:
        self.trace_every = max(1, self.trace_every - 2)

    def advance_trace(self) -> None:
        self._frames += 1
        if self._frames % max(1, self.trace_every):
            return
        frames = len(self.app.mind.phenotype.trace) if self.app.mind else 0
        self.trace_i = (self.trace_i + 1) % max(1, frames)

    def click(self, pos) -> bool:
        """Pick the cell under the cursor, so its own little perceptron can be shown."""
        if self._sheet_rect is None or not self._sheet_rect.collidepoint(pos):
            return False
        oy, ox = self._sheet_origin
        y = oy + (pos[1] - self._sheet_rect.top) // self._sheet_scale
        x = ox + (pos[0] - self._sheet_rect.left) // self._sheet_scale
        g = self.app.mind.genome
        self.picked = (int(y), int(x)) if g.mask[int(y), int(x)] else None
        return True

    # --- the whole column -------------------------------------------------------------
    def draw(self, surf, x: int) -> None:
        mind = self.app.mind
        g, p = mind.genome, mind.phenotype
        pw = BRAIN_W - 28
        x += 14
        y = 10
        small, f, big = self.app.small, self.app.font, self.app.font_big

        def line(text, colour=FG, font=None, dy=16):
            nonlocal y
            surf.blit((font or small).render(text, True, colour), (x, y))
            y += dy

        sens, eff = g.organs()
        line("brain", ACCENT, big, dy=22)
        line(f"cells {g.cells():<5} types {g.types_used()}/{g.n_types:<3}"
             f" depth {g.max_depth()}  width {g.max_width()}", FG, f, dy=18)
        line(f"organs {sens} sensory / {eff} effector    {g.params() / 1000:.1f}k params",
             FG, f, dy=18)
        # TIME: the brain's two clocks, against the world's one
        line(f"time  {g.subticks} subticks per thought, "
             f"1 thought per {mind.think_every} world ticks", (170, 190, 220))
        line(f"      {mind.thoughts} thoughts so far "
             f"= {mind.thoughts * g.subticks} sheet updates", (140, 150, 175))
        y += 4

        # the sheet takes whatever vertical room is left after the panels below it —
        # otherwise picking a cell pushes the command bars off the bottom of the window
        reserved = 200 + (170 if self.picked is not None else 0) + 13 * g.n_types
        y = self._draw_sheet(surf, x, y, pw, max(60, self.app.h - y - reserved))
        if self.picked is not None:
            y = self._draw_cell(surf, x, y, pw)
        y = self._draw_types(surf, x, y, pw)
        y = self._draw_command(surf, x, y, pw)
        y = self._draw_weights(surf, x, y, pw)
        line(f"view {VIEWS[self.view]}   V cycle   T watch it think   click a cell",
             DIM, dy=14)
        if self.trace_on:
            line(f"sub-tick {self.trace_i + 1}/{max(1, len(p.trace))}   "
                 f"[ / ] speed ({self.trace_every} frames each)", (250, 220, 130), dy=14)

    # --- the tissue -------------------------------------------------------------------
    def _draw_sheet(self, surf, x, y, pw, max_h: int = 300) -> int:
        mind = self.app.mind
        g, p = mind.genome, mind.phenotype
        ys, xs = np.nonzero(g.mask)
        if not len(ys):
            return y
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h, w = y1 - y0 + 1, x1 - x0 + 1
        img = np.zeros((h, w, 3))
        state = p.trace[self.trace_i] if (self.trace_on and p.trace) else p.state
        mode = VIEWS[self.view]
        self._chem = g.channel_map(p.cfg)["chemical"]
        for yy, xx in zip(ys, xs):
            i, j = yy - y0, xx - x0
            cell = state[yy, xx]
            if mode == "type":                    # what a cell IS, lit by what it does
                base = np.array(TYPE_COL[int(g.type_map[yy, xx]) % len(TYPE_COL)])
                img[i, j] = base * (0.28 + 0.72 * min(1.0, float(np.abs(cell).mean()) * 2.2))
            elif mode == "activity":
                a = min(1.0, float(np.abs(cell).mean()) * 2.5)
                img[i, j] = np.array((28, 38, 58)) * (1 - a) + np.array((250, 240, 160)) * a
            elif mode == "chemistry":             # the diffusing layer: three
                c0, c1 = self._chem                # channels as RGB, so a gradient
                ch = cell[c0:c1]                   # reads as a colour ramp. Scaled to
                img[i, j] = [abs(float(ch[k])) if k < len(ch) else 0.0   # the FRAME's
                             for k in range(3)]    # own range, or a saturated sheet is
                                                   # a white rectangle telling you
                                                   # nothing about where the plume is
            elif mode == "hormone":               # the channel that gates every rule
                hv = float(np.tanh(cell[-1]))
                img[i, j] = (np.array((250, 120, 90)) * max(0.0, hv)
                             + np.array((90, 150, 250)) * max(0.0, -hv)
                             + np.array((34, 34, 44)) * (1 - abs(hv)))
            else:
                if g.in_mask[yy, xx]:
                    img[i, j] = MOD_COL[int(g.organ_mod[yy, xx]) % len(MOD_COL)]
                elif g.out_mask[yy, xx]:
                    img[i, j] = (255, 255, 255)
                else:
                    img[i, j] = (52, 54, 66)
        if mode == "chemistry":                   # normalise the frame, then paint
            img *= 255.0 / max(1e-6, float(img.max()))
        scale = max(1, min(pw // max(1, w), int(max_h) // max(1, h)))
        sheet = pygame.surfarray.make_surface(
            np.transpose(img.clip(0, 255).astype(np.uint8), (1, 0, 2)))
        sheet = pygame.transform.scale(sheet, (w * scale, h * scale))
        surf.blit(sheet, (x, y))
        if mode != "organs" and scale >= 4:       # mark where signal enters and leaves
            for yy, xx in zip(ys, xs):
                px, py = x + (xx - x0) * scale, y + (yy - y0) * scale
                if g.out_mask[yy, xx]:
                    pygame.draw.rect(surf, (255, 255, 255),
                                     pygame.Rect(px, py, scale, scale), 1)
                elif g.in_mask[yy, xx]:
                    k = int(g.organ_mod[yy, xx]) % len(MOD_COL)
                    pygame.draw.rect(surf, MOD_COL[k],
                                     pygame.Rect(px, py, max(2, scale // 2),
                                                 max(2, scale // 2)))
        self._sheet_rect = pygame.Rect(x, y, w * scale, h * scale)
        self._sheet_scale, self._sheet_origin = scale, (y0, x0)
        if self.picked is not None and g.mask[self.picked]:      # ring the picked cell
            py = y + (self.picked[0] - y0) * scale
            px = x + (self.picked[1] - x0) * scale
            pygame.draw.rect(surf, (255, 255, 120),
                             pygame.Rect(px - 1, py - 1, scale + 2, scale + 2), 1)
        pygame.draw.rect(surf, (52, 54, 66), pygame.Rect(x, y, w * scale, h * scale), 1)
        y += h * scale + 5
        if mode == "organs":                      # legend, only for what is present
            cx = x
            for k, name in enumerate(MOD_NAME):
                if not (g.in_mask & g.mask & (g.organ_mod == k)).any():
                    continue
                pygame.draw.rect(surf, MOD_COL[k], pygame.Rect(cx, y + 2, 7, 7))
                surf.blit(self.app.small.render(name, True, DIM), (cx + 9, y))
                cx += 16 + self.app.small.size(name)[0]
                if cx > x + pw - 46:
                    cx, y = x, y + 12
            y += 14
        return y + 2

    # --- one cell's own little network ----------------------------------------------
    def _draw_cell(self, surf, x, y, pw) -> int:
        """The PERCEPTRON behind the picked cell: what it is, what it currently holds in
        each channel, and the layer stack it computes with. Every cell of a type shares
        that stack — differentiation is by TYPE, not per cell — so this answers "what is
        THIS cell running", which the type colours alone cannot tell you."""
        g, p = self.app.mind.genome, self.app.mind.phenotype
        cy, cx = self.picked
        k = int(g.type_map[cy, cx])
        role = ("sense " + MOD_NAME[int(g.organ_mod[cy, cx]) % len(MOD_NAME)]
                if g.in_mask[cy, cx] else
                "effector" if g.out_mask[cy, cx] else "interneuron")
        pygame.draw.rect(surf, TYPE_COL[k % len(TYPE_COL)], pygame.Rect(x, y + 1, 8, 8))
        surf.blit(self.app.small.render(
            f"cell ({cy},{cx})  type {k}  {role}   ESC unpicks", True, FG), (x + 12, y))
        y += 14
        # what it is holding right now, coloured by which LAYER each channel belongs to
        cmap = g.channel_map(p.cfg)
        state = (p.trace[self.trace_i] if (self.trace_on and p.trace) else p.state)[cy, cx]
        bw = max(3, min(14, (pw - 70) // max(1, g.C)))
        for c in range(g.C):
            v = float(state[c])
            col = ((120, 200, 255) if c < cmap["chemical"][0]
                   else (140, 240, 180) if c < cmap["hormone"][0] else (250, 170, 120))
            hgt = int(min(1.0, abs(v)) * 22)
            pygame.draw.rect(surf, (30, 30, 38), pygame.Rect(x + c * bw, y, bw - 1, 46))
            pygame.draw.rect(surf, col,
                             pygame.Rect(x + c * bw, y + 23 - (hgt if v > 0 else 0),
                                         bw - 1, max(1, hgt)))
        pygame.draw.line(surf, GRID, (x, y + 23), (x + g.C * bw, y + 23))
        for i, (name, col) in enumerate((("electric", (120, 200, 255)),
                                         ("chem", (140, 240, 180)),
                                         ("hormone", (250, 170, 120)))):
            surf.blit(self.app.small.render(name, True, col),
                      (x + g.C * bw + 6, y + i * 13))
        y += 50
        # the layer stack itself: 9C -> H -> ... -> C, one heat map per layer
        rule = g.rules[k]
        surf.blit(self.app.small.render(
            f"its perceptron: {9 * g.C} in -> "
            + " -> ".join(str(int(L["W"].shape[0])) for L in rule), True, DIM), (x, y))
        y += 13
        cxp, height = x, 40
        for i, L in enumerate(rule):
            W = np.asarray(L["W"], dtype=np.float64)
            if p.cfg.rule_plastic and i == len(rule) - 1 and k < len(p._dW_rule):
                W = W + p._dW_rule[k].T
            wpx = max(10, min(pw // len(rule) - 6, 120))
            img = pygame.surfarray.make_surface(np.transpose(_heat(W), (1, 0, 2)))
            surf.blit(pygame.transform.scale(img, (wpx, height)), (cxp, y))
            pygame.draw.rect(surf, (70, 72, 88), pygame.Rect(cxp, y, wpx, height), 1)
            cxp += wpx + 5
        return y + height + 8

    # --- what kinds of cell there are ----------------------------------------------------
    def _draw_types(self, surf, x, y, pw) -> int:
        """One row per CELL TYPE — the different little perceptrons the tissue has
        differentiated into. Beyond count and size, each row says what the type is
        LIKE, read straight off its rule:

          self   how much of its input weight is on its OWN state rather than the
                 neighbours. A type that mostly listens to itself is a MEMORY cell —
                 it holds rather than relays.
          keep   how strongly its output feeds its own channels back. High keep with
                 high self is a latch; low both is a reflex relay.

        These are descriptions, not labels: nothing forces a memory type to exist, and
        the point of showing the numbers is to see whether one appears."""
        g, p = self.app.mind.genome, self.app.mind.phenotype
        surf.blit(self.app.small.render(
            "cell types   (self = listens to itself, keep = feeds itself back)",
            True, DIM), (x, y))
        y += 13
        counts = np.bincount(g.type_map[g.mask], minlength=g.n_types)
        C = g.C
        for k, rule in enumerate(g.rules):
            W0 = np.asarray(rule[0]["W"], dtype=np.float64)     # (H, 9C)
            mass = np.abs(W0).sum()
            self_mass = np.abs(W0[:, 4 * C:5 * C]).sum()        # slot 4 = the centre
            selfness = float(self_mass / max(1e-9, mass) * 9.0)  # 1.0 = its fair share
            # KEEP: linearise the loop own-state -> hidden -> own-output and compare the
            # diagonal (channel c drives channel c) against the rest (channels mix).
            # Above 1 the type PRESERVES what it holds; below 1 it transforms it.
            Wl = np.asarray(rule[-1]["W"], dtype=np.float64)     # (C, H) after the stack
            if Wl.shape[1] == W0.shape[0]:
                A = np.abs(Wl @ W0[:, 4 * C:5 * C])             # (C, C)
                d = float(np.mean(np.diag(A)))
                off = float((A.sum() - np.trace(A)) / max(1, A.size - A.shape[0]))
                keep = d / max(1e-9, off)
            else:
                keep = float("nan")
            pygame.draw.rect(surf, TYPE_COL[k % len(TYPE_COL)],
                             pygame.Rect(x, y + 1, 8, 8))
            surf.blit(self.app.small.render(
                f"{counts[k] if k < len(counts) else 0:>4} cells   d{len(rule)} "
                f"w{int(rule[0]['W'].shape[0]):<3} self {selfness:4.2f}  keep {keep:5.2f}",
                True, FG), (x + 12, y))
            y += 12
        return y + 4

    # --- what it is commanding ---------------------------------------------------------
    def _draw_command(self, surf, x, y, pw) -> int:
        mind = self.app.mind
        p = mind.phenotype
        surf.blit(self.app.small.render("command   (read out from the effectors)",
                                        True, DIM), (x, y))
        y += 14
        for k, name in enumerate(("thrust", "turn", "interact")):
            v = float(mind.command[k])
            surf.blit(self.app.small.render(name, True, DIM), (x, y))
            bx, bw = x + 54, pw - 104
            pygame.draw.rect(surf, (30, 30, 38), pygame.Rect(bx, y + 1, bw, 9))
            if k == 1:                            # turn is signed, zero in the middle
                mid = bx + bw // 2
                width = int(abs(v) * bw / 2)
                pygame.draw.rect(surf, (250, 190, 110),
                                 pygame.Rect(mid if v >= 0 else mid - width, y + 1,
                                             width, 9))
                pygame.draw.line(surf, GRID, (mid, y + 1), (mid, y + 10))
            else:
                pygame.draw.rect(surf, (120, 230, 150),
                                 pygame.Rect(bx, y + 1,
                                             int(bw * max(0.0, min(1.0, v))), 9))
            surf.blit(self.app.small.render(f"{v:+.2f}", True, FG), (bx + bw + 6, y))
            y += 12
        surf.blit(self.app.small.render(
            f"reward {mind.last_reward:+.3f}    learned |dW| {p.plastic_norm():.3f}",
            True, (170, 190, 220)), (x, y))
        return y + 16

    # --- the weights -------------------------------------------------------------------
    def _draw_weights(self, surf, x, y, pw) -> int:
        """Each type rule's OUTPUT layer — the part plasticity moves within a life —
        with the learned delta folded in, plus the motor readout at the right."""
        g, p = self.app.mind.genome, self.app.mind.phenotype
        surf.blit(self.app.small.render("weights   (rule output layers | readout)",
                                        True, DIM), (x, y))
        y += 14
        cx, top, height = x, y, 44
        for k, rule in enumerate(g.rules):
            W = np.asarray(rule[-1]["W"], dtype=np.float64)
            if p.cfg.rule_plastic and k < len(p._dW_rule):
                W = W + p._dW_rule[k].T
            s = pygame.surfarray.make_surface(np.transpose(_heat(W), (1, 0, 2)))
            wpx = max(8, min(44, int(W.shape[1])))
            s = pygame.transform.scale(s, (wpx, height))
            surf.blit(s, (cx, top))
            pygame.draw.rect(surf, TYPE_COL[k % len(TYPE_COL)],
                             pygame.Rect(cx, top, wpx, height), 1)
            cx += wpx + 4
            if cx > x + pw - 70:
                break
        out = np.asarray(g.P_out, dtype=np.float64) + (p.dW if p.cfg.plastic else 0.0)
        s = pygame.surfarray.make_surface(np.transpose(_heat(out), (1, 0, 2)))
        s = pygame.transform.scale(s, (24, height))
        surf.blit(s, (x + pw - 24, top))
        pygame.draw.rect(surf, (200, 200, 210),
                         pygame.Rect(x + pw - 24, top, 24, height), 1)
        return top + height + 8
