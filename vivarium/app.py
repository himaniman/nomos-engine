"""Real-time pygame view of a vivarium (DESIGN.md stage 2).

    python -m vivarium.app --size 96 --seed 1

Global camera over the whole world. Agents are coloured by GENOME (a stable hue
hash, so lineages/families share a colour and you watch them spread and compete);
brightness tracks energy; a short spoke shows heading. Food = green dots, corpses
= dim brown blocks, walls = grey. A side panel shows the ecology HUD and, when
you click an agent, that agent's brain (its MLP, edges coloured by weight) and
gene stats.

Controls: SPACE pause/resume · S single-step · +/- speed · click an agent to
inspect · L follow the longest-lived agent (auto) · [ / ] food rate · , / . food
ceiling · P screenshot now · T animate the selected NCA brain's sub-ticks (at x1) · W
save the current top genomes (--save-genomes path) to continue later · R reseed-from-hall
· ESC/Q quit.

The drawing is split from the loop (`draw` returns a Surface) so it can be
rendered headless (SDL dummy) for tests/screenshots without a display.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import time
from pathlib import Path

import numpy as np
import pygame

from vivarium.config import Config
from vivarium.genome_io import save_genomes
from vivarium.neat import Genome
from vivarium.world import World

_CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
# food type -> colour (green, amber, cyan, ...), so dietary niches read on the map
_FOOD_COLORS = [(60, 200, 90), (220, 170, 60), (70, 180, 220), (200, 90, 170)]


def _genome_hash(genome: Genome) -> int:
    return int(hashlib.md5(genome.key()).hexdigest(), 16)


def genome_color(genome: Genome) -> tuple[int, int, int]:
    """Colour on two axes so lineage AND structure both read at a glance:
      HUE        = the heritable lineage hue (families share a colour, drifts slowly);
      SATURATION = brain STRUCTURAL complexity (NCA: type diversity; NEAT: hidden
                   nodes), so a richer, more differentiated brain looks more VIVID
                   and a simple one looks pale — without breaking family visibility.
    (Brightness is added by the caller from energy. The brain fingerprint `key` is
    used for the NAME, not the colour — deliberately: tying colour to the hash would
    flip a family's colour on every mutation, hiding lineages.)"""
    nh, nc = genome.complexity()
    if getattr(genome, "kind", "neat") == "nca":
        frac = max(0.0, min(1.0, (nc - 1) / 3.0))       # type diversity: 1->pale, 4->vivid
    else:
        frac = max(0.0, min(1.0, nh / 6.0))             # hidden-node count
    sat = 0.45 + 0.5 * frac
    r, g, b = colorsys.hsv_to_rgb(genome.hue, sat, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _hash_suffix(genome: Genome, k: int = 3) -> str:
    """`k` consonants hashed from the full genome fingerprint — a weight-level
    identity tag that changes on ANY mutation (the '-XYZ' after the body plan)."""
    n = _genome_hash(genome)
    out = []
    for _ in range(k):
        out.append(_CONSONANTS[n % len(_CONSONANTS)])
        n //= len(_CONSONANTS)
    return "".join(out)


def genome_name(genome: Genome) -> str:
    """A structured GENOME CODE, not an opaque hash: the body plan read straight off
    the architecture, then a short weight-hash suffix. So the name is informative and
    the structural PREFIX is stable while only weights drift (you watch a body plan
    persist across a lineage), and it only changes when the architecture changes.
      NCA : '<cells>c<types>t<depth>d-XYZ'  (e.g. 9c3t4d-ZKQ)
      NEAT: '<hidden>h<conns>c-XYZ'         (e.g. 2h11c-BFK)"""
    nh, nc = genome.complexity()
    if getattr(genome, "kind", "neat") == "nca":
        plan = f"{nh}c{nc}t{genome.max_depth()}d{genome.max_width()}w"
    else:
        plan = f"{nh}h{nc}c"
    return f"{plan}-{_hash_suffix(genome)}"


def draw(world: World, cell_px: int, panel_w: int, font, small,
         selected: int | None = None, hud: dict | None = None,
         leader=None, hitboxes=None) -> pygame.Surface:
    """Render one frame to a Surface: the world on the left, HUD + (optional)
    inspected-agent panel on the right. `leader` (a Genome) pins a clicked
    leader in the inspector; `hitboxes` (a list) is filled with the clickable
    (rect, genome) of each leader row for the event loop."""
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
    # food halos (faint disc marking where each food type is spawning right now)
    for k, (hy, hx) in enumerate(world.halos):
        pygame.draw.circle(surf, tuple(c // 5 for c in _FOOD_COLORS[k % len(_FOOD_COLORS)]),
                           (int(hx) * cell_px, int(hy) * cell_px),
                           max(2, int(world.cfg.food_halo_sigma * world.w * cell_px)), 1)
    # food, coloured by type
    fy, fx = np.where(world.food > 0)
    fr = max(1, cell_px // 2)
    for y, x in zip(fy, fx):
        col = _FOOD_COLORS[(int(world.food[y, x]) - 1) % len(_FOOD_COLORS)]
        pygame.draw.circle(surf, col,
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

    _draw_panel(surf, world, W, panel_w, H, font, small, selected, hud, leader, hitboxes)
    return surf


def _draw_panel(surf, world, x0, panel_w, H, font, small, selected, hud=None,
                leader=None, hitboxes=None):
    surf.fill((20, 20, 26), (x0, 0, panel_w, H))
    # two columns: a narrow TEXT column (stats / leaders / populations) on the left
    # and a wide BRAIN column on the right where the inspected agent's CA lattice and
    # its perceptron rule-nets are drawn at native resolution (no scaling blur).
    text_w = min(250, panel_w)
    bx0, bw = x0 + text_w, panel_w - text_w
    if bw > 40:
        pygame.draw.line(surf, (40, 40, 50), (bx0, 8), (bx0, H - 8), 1)
    s = world.snapshot()
    lines = [f"tick {s['tick']}",
             f"epoch {s['epoch']}  +{s['epoch_age']}",
             f"pop  {s['population']}  sp {s['species']}  carn {s['carnivores'] * 100:.0f}%",
             f"food {s['food']}/{s['food_cap']}  rate {s['food_rate']:g}",
             f"energy {s['energy_bodies']:.0f}  food {s['energy_food']:.0f}",
             f"births {s['births']}", f"deaths {s['deaths']}",
             f"lifespan  mean {s['life_mean']:.0f}  med {s['life_median']:.0f}"]
    if hud is not None:
        think = "  THINK" if hud.get("think") else ""
        lines.append(f"fps {hud['fps']:.0f}  x{hud['speed']}  {hud['sps']:.0f}/s{think}")
    for i, t in enumerate(lines):
        surf.blit(font.render(t, True, (220, 220, 230)), (x0 + 12, 10 + i * 20))
    y = 10 + len(lines) * 20 + 10
    if hud is not None and hud.get("msg"):
        surf.blit(small.render(hud["msg"], True, (150, 230, 160)), (x0 + 12, y))
        y += 18

    # top-3 longest-lived lineages — clickable to inspect their genome
    surf.blit(small.render("leaders (click to inspect):", True, (150, 160, 180)),
              (x0 + 12, y)); y += 18
    if hitboxes is not None:
        hitboxes.clear()
    for age, g in world.hall[:3]:
        pygame.draw.rect(surf, genome_color(g), (x0 + 12, y + 2, 10, 10))
        surf.blit(small.render(f"{age:>4} {genome_name(g)}", True,
                               (210, 215, 230)), (x0 + 28, y))
        if hitboxes is not None:
            hitboxes.append((pygame.Rect(x0, y, text_w, 17), g))
        y += 17
    y += 6

    # per-POPULATION stats: one row per living colour-species (count, mean diet,
    # carnivore %, mean age) so you read EACH population, not just the aggregate
    groups = world.species_groups()
    surf.blit(small.render(f"populations ({len(groups)}):", True, (150, 160, 180)),
              (x0 + 12, y)); y += 18
    for gr in groups[:5]:
        r, g_, b = colorsys.hsv_to_rgb(gr["hue"], 0.85, 1.0)
        pygame.draw.rect(surf, (int(r * 255), int(g_ * 255), int(b * 255)),
                         (x0 + 12, y + 2, 10, 10))
        surf.blit(small.render(
            f"n{gr['n']:<3} diet{gr['diet']:.1f} carn{gr['carn'] * 100:.0f}% "
            f"age{gr['age']:.0f}", True, (205, 210, 225)), (x0 + 28, y))
        y += 16
    y += 8

    # inspector goes in the BRAIN column (its own space, full height), else — if the
    # panel is too narrow for a column — it falls back under the text.
    if bw > 120:
        ix, iy, iw = bx0, 10, bw
    else:
        ix, iy, iw = x0, y, panel_w
    # a clicked leader genome takes priority, else the selected agent
    if leader is not None:
        _draw_inspector(surf, leader, None, ix, iy, iw, H, font, small, title="LEADER")
        return
    a = world.agents.get(selected) if selected is not None else None
    if a is not None:
        _draw_inspector(surf, a.brain.genome, a, ix, iy, iw, H, font, small)
    elif bw > 120:
        surf.blit(small.render("click an agent to inspect its brain", True,
                               (110, 120, 140)), (bx0 + 12, 14))


def _draw_inspector(surf, g, agent, x0, y, panel_w, H, font, small, title=None):
    pygame.draw.rect(surf, genome_color(g), (x0 + 12, y + 3, 12, 12))
    label = genome_name(g) + (f"  ({title})" if title else "")
    surf.blit(font.render(label, True, (235, 235, 245)), (x0 + 30, y))
    y += 24
    nh, nc = g.complexity()
    info = ([f"agent #{agent.id}", f"energy {agent.energy:.1f}", f"age {agent.age}"]
            if agent is not None else [])
    info.append(f"cells {nh} types {nc} depth {g.max_depth()} width {g.max_width()}"
                if getattr(g, "kind", "neat") == "nca"
                else f"hidden {nh}  conns {nc}")
    info.append(f"diet {g.diet:.2f}  "
                + ("CARNIVORE" if g.carnivore >= 0.5 else f"herb {g.carnivore:.2f}"))
    for t in info:
        surf.blit(small.render(t, True, (200, 210, 230)), (x0 + 12, y)); y += 18
    box = (x0 + 12, y + 8, panel_w - 24, H - (y + 8) - 12)
    if getattr(g, "kind", "neat") == "nca":
        _draw_nca_brain(surf, g, agent.brain if agent is not None else None, *box)
    else:
        _draw_net(surf, g, *box)


# distinct colours per cell-TYPE, so you see what the brain is made of
_TYPE_COLORS = [(230, 120, 120), (120, 205, 140), (130, 160, 245), (232, 200, 120),
                (205, 130, 232), (120, 212, 212)]


def _type_color(k):
    return _TYPE_COLORS[k % len(_TYPE_COLORS)]


def _draw_nca_brain(surf, genome, phenotype, x, y, w, h):
    """Draw the agent's CA brain in two rows. Top: the G x G lattice (dead cells
    dark; alive cells filled by their first three state CHANNELS -> RGB, bordered by
    their cell-TYPE colour), flanked by the small "neurons" wired to it — input
    SENSES left, output MOTOR logits right (argmax highlighted). Bottom: each cell
    TYPE's update rule drawn as a compact layered net (channels -> hidden -> channel
    delta), so you see the diversity of nets the cells are made of."""
    state = None
    if phenotype is not None:
        state = phenotype.show_state if getattr(phenotype, "show_state", None) \
            is not None else phenotype.state
    inp = getattr(phenotype, "last_input", None) if phenotype is not None else None
    logits = getattr(phenotype, "last_logits", None) if phenotype is not None else None

    grid_h = int(h * 0.50)                                 # top row: the lattice
    col_w, pad = 6, 6
    gx0 = x + (col_w + pad if inp is not None else 0)
    gx1 = x + w - (col_w + pad if logits is not None else 0)
    G = genome.G
    # CROP to the alive bounding box (+1 cell margin) so the few live cells fill the
    # view — drawing the whole mostly-dead G x G lattice made cells tiny. Cells are
    # sized to fill the available box, so a small brain reads BIG.
    if genome.mask.any():
        ys_a, xs_a = np.where(genome.mask)
        ay0, ay1 = max(0, int(ys_a.min()) - 1), min(G - 1, int(ys_a.max()) + 1)
        ax0, ax1 = max(0, int(xs_a.min()) - 1), min(G - 1, int(xs_a.max()) + 1)
    else:
        ay0, ay1, ax0, ax1 = 0, G - 1, 0, G - 1
    rows, cols = ay1 - ay0 + 1, ax1 - ax0 + 1
    side = max(3, min(gx1 - gx0, grid_h) // max(rows, cols))
    xoff = gx0 + max(0, (gx1 - gx0 - cols * side) // 2)    # centre the cropped lattice

    def strip(vals, sx, hi=None, signed=False):
        n = len(vals)
        ch = max(1, min(10, grid_h // n))
        for i, v in enumerate(vals):
            if signed:                                     # motor logits: red<0, green>0
                t = float(np.tanh(v))
                col = (int(140 * max(0, -t)) + 30, int(140 * max(0, t)) + 30, 40)
            else:                                          # senses: green intensity
                gi = int(200 * min(1.0, abs(float(v))))
                col = (20, gi + 25, 30)
            rect = (sx, y + i * ch, col_w, ch - 1)
            surf.fill(col, rect)
            if hi is not None and i == hi:
                pygame.draw.rect(surf, (255, 255, 255), rect, 1)

    if inp is not None:
        strip(inp, x)                                      # input senses (left)
    for gy in range(ay0, ay1 + 1):                         # only the cropped bbox
        for gx in range(ax0, ax1 + 1):
            rect = (xoff + (gx - ax0) * side, y + (gy - ay0) * side,
                    side - 1, side - 1)
            if not genome.mask[gy, gx]:
                surf.fill((30, 30, 38), rect)              # dead cell (margin)
                continue
            if state is not None:
                ch = state[gy, gx]
                col = tuple(int(128 + 127 * np.tanh(ch[k] if k < len(ch) else 0.0))
                            for k in range(3))
            else:
                col = (150, 160, 200)                      # shape only (no live state)
            surf.fill(col, rect)
            if side >= 4:                                  # border = cell type
                pygame.draw.rect(surf, _type_color(int(genome.type_map[gy, gx])),
                                 rect, 1)
    if logits is not None:
        strip(logits, x + w - col_w, hi=int(np.argmax(logits)), signed=True)

    # middle band: the READOUT (pooled channels -> action logits) — the ONE place
    # weights change WITHIN a life, so learned edges are highlighted here.
    ry = y + grid_h + 4
    rh = int(h * 0.18)
    _draw_readout(surf, genome, phenotype, x, ry, w, rh)

    # bottom row: one compact rule-net per cell type (these weights are FIXED — genome)
    ny = ry + rh + 4
    nh = y + h - ny
    used = [k for k in range(genome.n_types)
            if (genome.mask & (genome.type_map == k)).any()] or [0]
    nw = w // len(used)
    for i, k in enumerate(used):
        _draw_rule_net(surf, genome, k, x + i * nw, ny, nw - 6, nh)


def _draw_readout(surf, genome, phenotype, x, y, w, h):
    """The motor READOUT (pooled channels -> action logits) = P_out, plus the LEARNED
    plastic delta dW. This is the only part that changes during a life, so it's drawn
    on its own: faint grey edges are the innate readout, and edges LIFE-LEARNING has
    moved glow green (strengthened) / red (weakened) — so you literally watch the
    weights change as the agent learns. Action nodes are labelled F/L/R/S."""
    C, n_out = genome.C, genome.n_out
    P = np.asarray(genome.P_out)
    dW = getattr(phenotype, "dW", None) if phenotype is not None else None
    plastic = phenotype is not None and getattr(phenotype, "plastic", False)
    surf.blit(pygame.font.Font(None, 15).render("readout", True, (150, 160, 180)),
              (x, y - 1))
    lx, rx = x + 16, x + w - 34
    top, span = y + 12, max(6, h - 14)
    cy = [top + int((i + 0.5) / C * span) for i in range(C)]
    ay = [top + int((j + 0.5) / n_out * span) for j in range(n_out)]
    bmax = float(np.abs(P).max()) + 1e-6
    dmax = (float(np.abs(dW).max()) + 1e-6) if dW is not None else 1.0
    for c in range(C):                                     # faint innate readout first
        for j in range(n_out):
            g = 35 + int(45 * abs(P[c, j]) / bmax)
            pygame.draw.line(surf, (g, g, g + 8), (lx, cy[c]), (rx, ay[j]), 1)
    if plastic and dW is not None:                         # overlay the learned changes
        for c in range(C):
            for j in range(n_out):
                d = float(dW[c, j])
                if abs(d) < 0.12 * dmax:
                    continue
                col = (90, 215, 120) if d > 0 else (235, 110, 90)
                pygame.draw.line(surf, col, (lx, cy[c]), (rx, ay[j]),
                                 1 + int(2 * abs(d) / dmax))
    for yy in cy:                                          # channel nodes (left)
        pygame.draw.circle(surf, (150, 160, 200), (lx, yy), 2)
    font = pygame.font.Font(None, 16)
    for j, yy in enumerate(ay):                            # action nodes + labels (right)
        pygame.draw.circle(surf, (240, 200, 120), (rx, yy), 3)
        surf.blit(font.render("FLRS"[j] if j < 4 else "?", True, (210, 210, 225)),
                  (rx + 8, yy - 7))
    if plastic and dW is not None:                         # how much has been learned
        mag = float(np.abs(dW).sum())
        surf.blit(font.render(f"|dW| {mag:.2f}", True, (140, 200, 150)),
                  (x + w - 62, y - 1))


def _draw_rule_net(surf, genome, k, x, y, w, h):
    """A cell TYPE's update rule as a small layered net whose DEPTH is what evolved:
    a column of nodes per layer (input channels on the left — the 9-neighbour weights
    averaged per channel — hidden layers in the middle, channel-delta output on the
    right), strongest edges only. The header dot is the type's colour; more columns =
    a deeper rule."""
    C, H = genome.C, genome.H
    rule = genome.rules[k]
    # per-layer weight matrices, with the input layer's 9-neighbour weights averaged
    # down to C so the leftmost column reads as "channels"
    mats = [rule[0]["W"].reshape(rule[0]["W"].shape[0], 9, C).mean(axis=1)]
    mats += [L["W"] for L in rule[1:]]                     # (out, in) per layer
    widths = [C] + [m.shape[0] for m in mats]             # node count per column
    top = max(6, y)
    n_col = len(widths)
    xs = [x + 8 + int(i / max(1, n_col - 1) * (w - 16)) for i in range(n_col)]
    def ys(n):
        return [top + int((j + 0.5) / max(1, n) * (h - 8)) for j in range(n)]
    ycols = [ys(nw) for nw in widths]
    pygame.draw.rect(surf, _type_color(k), (x, top - 4, 6, 6))
    for li, Wm in enumerate(mats):                        # edges between columns
        m = np.abs(Wm).max() + 1e-6
        thr = 0.6 * m
        ya, yb = ycols[li], ycols[li + 1]
        for a in range(Wm.shape[0]):
            for b in range(Wm.shape[1]):
                if abs(Wm[a, b]) < thr:
                    continue
                col = (90, 160, 255) if Wm[a, b] >= 0 else (255, 110, 90)
                pygame.draw.line(surf, col, (xs[li], ya[b]), (xs[li + 1], yb[a]),
                                 1 if abs(Wm[a, b]) < 0.85 * m else 2)
    for ci, ycol in enumerate(ycols):                     # nodes
        cc = (140, 220, 150) if ci == 0 else \
            (240, 200, 120) if ci == n_col - 1 else (210, 210, 230)
        for yy in ycol:
            pygame.draw.circle(surf, cc, (xs[ci], yy), 3 if ci in (0, n_col - 1) else 2)


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


_HELP_LINES = [
    "SPACE   pause / resume",
    "S       single step",
    "+  /  - speed up / down",
    "click   inspect an agent",
    "L       follow the oldest agent",
    "[  /  ] food rate  down / up",
    ",  /  . food ceiling  down / up",
    "T       animate NCA 'thinking'",
    "P       screenshot now",
    "W       save genome pool",
    "R       reseed from hall",
    "H       toggle this help",
    "ESC / Q quit",
]


def _shot_name(world) -> str:
    """Screenshot filename: the TICK first, zero-padded so files sort by tick in a
    plain name sort, then a few world params (epoch, population, species)."""
    s = world.snapshot()
    return (f"t{s['tick']:08d}_ep{s['epoch']}_pop{s['population']}"
            f"_sp{s['species']}.png")


def _save_shot(screen, shots_dir, world) -> str:
    """Save the current frame to shots_dir with a tick-sorted name; return the name."""
    Path(shots_dir).mkdir(parents=True, exist_ok=True)
    name = _shot_name(world)
    pygame.image.save(screen, str(Path(shots_dir) / name))
    return name


def _draw_help(surf, font, small):
    """A simple key-binding overlay, toggled with H, over the top-left of the world."""
    w, h = 330, len(_HELP_LINES) * 20 + 44
    box = pygame.Surface((w, h))
    box.set_alpha(236)
    box.fill((18, 18, 24))
    surf.blit(box, (18, 18))
    pygame.draw.rect(surf, (80, 90, 110), (18, 18, w, h), 1)
    surf.blit(font.render("keys", True, (210, 220, 240)), (34, 30))
    for i, line in enumerate(_HELP_LINES):
        surf.blit(small.render(line, True, (205, 210, 225)), (34, 58 + i * 20))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.app", description="watch a vivarium")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--cell", type=int, default=8, help="pixels per cell")
    p.add_argument("--brain", choices=("neat", "nca"), default="neat",
                   help="brain model: neat graph or nca cellular-automaton")
    p.add_argument("--load-genomes", default=None, dest="load_genomes",
                   help="seed the world from an evolved genome pool (.json)")
    p.add_argument("--save-genomes", default="vivarium/pools/session.json",
                   dest="save_genomes",
                   help="press W in the window to save the current top genomes here")
    p.add_argument("--shots-dir", default="out/vivarium", dest="shots_dir",
                   help="folder for periodic auto-screenshots")
    p.add_argument("--shot-every", type=float, default=1800.0, dest="shot_every",
                   help="seconds between auto-screenshots (default 1800 = 30 min; 0 = off)")
    a = p.parse_args(argv)

    cfg = Config(size=a.size, seed=a.seed, init_population=a.population,
                 brain=a.brain)
    pool = None
    if a.load_genomes:
        from vivarium.genome_io import load_genomes
        pool = load_genomes(a.load_genomes)
    world = World(cfg, genome_pool=pool)
    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    # Wide panel: a text column + a dedicated BRAIN column (CA lattice on top, its
    # perceptron rule-nets below), all drawn at native resolution — no scaling, so
    # text and lattice stay crisp. Make the world cells bigger (--cell) for a bigger
    # window, not a scaled-up (blurry) one.
    panel_w = 560
    screen = pygame.display.set_mode((world.w * a.cell + panel_w, world.h * a.cell))
    pygame.display.set_caption("vivarium")
    clock = pygame.time.Clock()
    last_shot = time.time()                          # periodic auto-screenshot timer

    # `speed` = simulation steps per RENDERED frame. Rendering is the bottleneck,
    # so running many steps per draw (and, at high speed, uncapping the frame
    # rate) lets the simulation outrun the display for fast-forward. The HUD shows
    # the measured FPS, the speed, and the resulting steps/sec.
    paused, speed, selected = False, 1, None
    show_help = False                                # H: key-binding overlay
    follow_oldest = False                            # L: auto-inspect the oldest agent
    leader = None                                    # a pinned leader genome to inspect
    leader_boxes = []                                # (rect, genome) hitboxes, filled by draw
    think = False                                    # subtick-animation mode (T), x1 only
    sub = 0                                          # which subtick frame we're showing
    recorded = None                                  # which agent is tracing its subticks
    msg, msg_frames = "press H for keys", 400        # transient HUD flash (startup hint)
    is_nca = cfg.brain == "nca"
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_h:            # toggle the key-binding overlay
                    show_help = not show_help
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
                elif e.key == pygame.K_RIGHTBRACKET:   # ] more food per tick
                    world.food_rate = round(world.food_rate
                                            + (1 if world.food_rate >= 1 else 0.1), 1)
                    msg, msg_frames = f"food rate {world.food_rate:g}", 60
                elif e.key == pygame.K_LEFTBRACKET:    # [ less food per tick (down to 0.1)
                    step = 1 if world.food_rate > 1 else 0.1
                    world.food_rate = max(0.1, round(world.food_rate - step, 1))
                    msg, msg_frames = f"food rate {world.food_rate:g}", 60
                elif e.key == pygame.K_PERIOD:         # . raise the standing-food ceiling
                    world.food_cap = round(min(0.6, world.food_cap + 0.01), 3)
                    msg, msg_frames = f"food cap {world.food_cap * 100:.0f}%", 60
                elif e.key == pygame.K_COMMA:          # , lower the standing-food ceiling
                    world.food_cap = round(max(0.0, world.food_cap - 0.01), 3)
                    msg, msg_frames = f"food cap {world.food_cap * 100:.0f}%", 60
                elif e.key == pygame.K_l:            # follow the longest-lived agent
                    follow_oldest = not follow_oldest
                    leader = None
                    msg, msg_frames = ("follow oldest ON" if follow_oldest
                                       else "follow oldest OFF"), 90
                elif e.key == pygame.K_p:            # take a screenshot now
                    name = _save_shot(screen, a.shots_dir, world)
                    msg, msg_frames = f"saved shot {name}", 120
                elif e.key == pygame.K_t:            # toggle subtick "thinking" view
                    think = not think
                    sub = 0
                elif e.key == pygame.K_w:            # write a rich pool: living DIVERSITY
                    # (top-3 per population) + the all-time hall CHAMPIONS, deduped
                    seen, genomes = set(), []
                    for g in world.population_genomes(3) + world.hall_genomes():
                        k = g.key()
                        if k not in seen:
                            seen.add(k)
                            genomes.append(g)
                    genomes = genomes or [ag.brain.genome.copy()
                                          for ag in world.agents.values()]
                    save_genomes(a.save_genomes, genomes)
                    msg, msg_frames = f"saved {len(genomes)} -> {a.save_genomes}", 150
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                hit = next((g for rect, g in leader_boxes if rect.collidepoint(mx, my)),
                           None)
                if hit is not None:                  # clicked a leader row -> pin it
                    leader = hit
                else:
                    gx, gy = mx // a.cell, my // a.cell
                    if 0 <= gy < world.h and 0 <= gx < world.w \
                            and world.occ[gy, gx] >= 0:
                        selected = int(world.occ[gy, gx])
                        leader = None                # a live agent unpins the leader
                        follow_oldest = False        # a manual pick stops auto-follow

        # auto-follow: keep the inspector on the longest-lived living agent
        if follow_oldest and world.agents:
            selected = max(world.agents.values(), key=lambda ag: ag.age).id
            leader = None

        # only the selected NCA agent records its subtick trace (for the T view)
        if is_nca and recorded != selected:
            if recorded in world.agents:
                world.agents[recorded].brain.record = False
                world.agents[recorded].brain.show_state = None
            recorded = selected
        anim = think and speed == 1 and is_nca and not paused and selected in world.agents
        if is_nca and selected in world.agents:
            world.agents[selected].brain.record = anim

        if not paused:
            if anim:                                 # 1 world tick per K frames; play it out
                if sub == 0:
                    world.step()
                if selected in world.agents:
                    tr = world.agents[selected].brain.trace
                    if tr:
                        world.agents[selected].brain.show_state = tr[min(sub, len(tr) - 1)]
                        sub = (sub + 1) % len(tr)
                    else:
                        sub = 0
                else:
                    sub = 0
            else:
                for _ in range(speed):
                    world.step()
                sub = 0
        if selected is not None and selected not in world.agents:
            selected = None
        fps = clock.get_fps()
        hud = {"fps": fps, "speed": speed, "sps": fps * (speed if not paused else 0),
               "think": anim, "msg": msg if msg_frames > 0 else ""}
        if msg_frames > 0:
            msg_frames -= 1
        screen.blit(draw(world, a.cell, panel_w, font, small, selected, hud,
                         leader=leader, hitboxes=leader_boxes), (0, 0))
        if show_help:
            _draw_help(screen, font, small)
        pygame.display.flip()
        if a.shot_every > 0 and time.time() - last_shot >= a.shot_every:
            name = _save_shot(screen, a.shots_dir, world)
            last_shot = time.time()
            msg, msg_frames = f"saved shot {name}", 120
        clock.tick(60 if (speed <= 2 or anim) else 0)   # capped when slow/animating
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
