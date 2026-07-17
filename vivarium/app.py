"""Real-time pygame view of a vivarium (DESIGN.md stage 2).

    python -m vivarium.app --size 96 --seed 1

Global camera over the whole world. Agents are coloured by GENOME (a stable hue
hash, so lineages/families share a colour and you watch them spread and compete);
brightness tracks energy; a short spoke shows heading. Food = green dots, corpses
= dim brown blocks, walls = grey. A side panel shows the ecology HUD and, when
you click an agent, that agent's brain (its MLP, edges coloured by weight) and
gene stats.

Controls: SPACE pause/resume · S single-step · +/- speed · click an agent to
inspect · T animate the selected NCA brain's sub-ticks (at x1) · W save the current
top genomes (--save-genomes path) to continue later · R reseed-from-hall · ESC/Q quit.

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


def genome_name(genome: Genome) -> str:
    """A 6-letter pronounceable-ish name for a genome, so lineages are nameable
    in the stats. Stable per genome, changes when the structure/weights change."""
    n = _genome_hash(genome)
    letters = []
    for _ in range(6):
        letters.append(_CONSONANTS[n % len(_CONSONANTS)])
        n //= len(_CONSONANTS)
    return "".join(letters)


def _brain_tag(genome) -> str:
    """Compact structure readout for a genome: NCA shows alive cells + type
    diversity (its 'neuron diversity'); NEAT shows hidden nodes + connections."""
    nh, nc = genome.complexity()
    return f"{nh}c {nc}t" if getattr(genome, "kind", "neat") == "nca" \
        else f"h{nh} c{nc}"


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
    s = world.snapshot()
    lines = [f"tick {s['tick']}", f"pop  {s['population']}  species {s['species']}",
             f"food {s['food']}  carn {s['carnivores'] * 100:.0f}%",
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
        surf.blit(small.render(f"{age:>4} {genome_name(g)}  {_brain_tag(g)}", True,
                               (210, 215, 230)), (x0 + 28, y))
        if hitboxes is not None:
            hitboxes.append((pygame.Rect(x0, y, panel_w, 17), g))
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

    # inspector: a clicked leader genome takes priority, else the selected agent
    if leader is not None:
        _draw_inspector(surf, leader, None, x0, y, panel_w, H, font, small,
                        title="LEADER")
        return
    a = world.agents.get(selected) if selected is not None else None
    if a is not None:
        _draw_inspector(surf, a.brain.genome, a, x0, y, panel_w, H, font, small)


def _draw_inspector(surf, g, agent, x0, y, panel_w, H, font, small, title=None):
    pygame.draw.rect(surf, genome_color(g), (x0 + 12, y + 3, 12, 12))
    label = genome_name(g) + (f"  ({title})" if title else "")
    surf.blit(font.render(label, True, (235, 235, 245)), (x0 + 30, y))
    y += 24
    nh, nc = g.complexity()
    info = ([f"agent #{agent.id}", f"energy {agent.energy:.1f}", f"age {agent.age}"]
            if agent is not None else [])
    info.append(f"cells {nh}  types {nc}  depth {g.max_depth()}"
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

    grid_h = int(h * 0.55)                                 # top row height
    col_w, pad = 6, 6
    gx0 = x + (col_w + pad if inp is not None else 0)
    gx1 = x + w - (col_w + pad if logits is not None else 0)
    G = genome.G
    side = max(1, min(gx1 - gx0, grid_h) // G)

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
    for gy in range(G):
        for gx in range(G):
            rect = (gx0 + gx * side, y + gy * side, side - 1, side - 1)
            if not genome.mask[gy, gx]:
                surf.fill((30, 30, 38), rect)              # dead cell
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

    # bottom row: one compact rule-net per cell type
    ny = y + grid_h + 4
    nh = y + h - ny
    used = [k for k in range(genome.n_types)
            if (genome.mask & (genome.type_map == k)).any()] or [0]
    nw = w // len(used)
    for i, k in enumerate(used):
        _draw_rule_net(surf, genome, k, x + i * nw, ny, nw - 6, nh)


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivarium.app", description="watch a vivarium")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--cell", type=int, default=7, help="pixels per cell")
    p.add_argument("--brain", choices=("neat", "nca"), default="neat",
                   help="brain model: neat graph or nca cellular-automaton")
    p.add_argument("--load-genomes", default=None, dest="load_genomes",
                   help="seed the world from an evolved genome pool (.json)")
    p.add_argument("--save-genomes", default="vivarium/pools/session.json",
                   dest="save_genomes",
                   help="press W in the window to save the current top genomes here")
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
    panel_w = 260
    screen = pygame.display.set_mode((world.w * a.cell + panel_w, world.h * a.cell))
    pygame.display.set_caption("vivarium")
    clock = pygame.time.Clock()

    # `speed` = simulation steps per RENDERED frame. Rendering is the bottleneck,
    # so running many steps per draw (and, at high speed, uncapping the frame
    # rate) lets the simulation outrun the display for fast-forward. The HUD shows
    # the measured FPS, the speed, and the resulting steps/sec.
    paused, speed, selected = False, 1, None
    leader = None                                    # a pinned leader genome to inspect
    leader_boxes = []                                # (rect, genome) hitboxes, filled by draw
    think = False                                    # subtick-animation mode (T), x1 only
    sub = 0                                          # which subtick frame we're showing
    recorded = None                                  # which agent is tracing its subticks
    msg, msg_frames = "", 0                          # transient HUD flash
    is_nca = cfg.brain == "nca"
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
                elif e.key == pygame.K_t:            # toggle subtick "thinking" view
                    think = not think
                    sub = 0
                elif e.key == pygame.K_w:            # write the current top genomes
                    genomes = world.hall_genomes() or \
                        [ag.brain.genome.copy() for ag in world.agents.values()]
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
        pygame.display.flip()
        clock.tick(60 if (speed <= 2 or anim) else 0)   # capped when slow/animating
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
