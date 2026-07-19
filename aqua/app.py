"""Real-time pygame view of the CONTINUOUS aqua world.

    python -m aqua.app --brain nca --size 48 --seed 1

The world (left) is drawn as SMOOTH vector graphics: agents are circles coloured
by lineage (brightness = energy) with a heading spoke, food are dots coloured by
type, walls/corpses are cells. Click an agent to inspect it — and its FIELD OF
VIEW is drawn as the vision-ray fan out to each ray's first hit, so you see
exactly how far and where it sees (the perception viz, now literally the brain's
ray vision). The right panel is reused from `vivarium.app`: the ecology HUD, the
top lineages, per-population stats, and the CA/NEAT brain inspector.

Controls: SPACE pause/resume · S single-step · +/- speed · click an agent to
inspect · L follow the longest-lived · [ / ] food rate · , / . food ceiling ·
T animate the selected NCA brain's sub-ticks (x1) · P screenshot · W save top
genomes · R reseed-from-hall · H toggle help · ESC/Q quit.

`draw()` returns a Surface so the view renders headless (SDL dummy) for tests.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pygame

from aqua.world import INTERO, MATE, SIGNAL, SMELL, THREAT, World
from vivarium.app import (_FOOD_COLORS, _draw_help, _draw_panel, _fmt_hms,
                          _save_shot, genome_color)
from vivarium.config import Config

# One colour per SENSOR TYPE, so the brain inspector's input strip reads by modality
# instead of a wall of green. Matches the layout in aqua.world._sense.
_SENSE_COL = {"food": (95, 220, 115), "obstacle": (175, 180, 195),
              "kin": (110, 160, 250), "nonkin": (235, 100, 95),
              "intero": (240, 200, 90), "smell": (215, 120, 235),
              "mate": (245, 150, 190), "signal": (240, 235, 120),
              "threat": (230, 70, 70)}
_SENSE_LEGEND = [("food", "food"), ("obstacle", "obstacle"), ("kin", "kin"),
                 ("non-kin", "nonkin"), ("body", "intero"), ("smell", "smell"),
                 ("mate", "mate"), ("signal", "signal"), ("threat", "threat")]

# organ MODALITY id (vivarium.nca / World._build_input_layout) -> colour, matching the
# input-strip legend, so a specialised sensory organ on the brain is coloured by WHAT it
# hears (a food organ green, a smell organ magenta...). 0 = ALL is left cyan by the drawer.
_MOD_COL = {1: _SENSE_COL["food"], 2: _SENSE_COL["obstacle"], 3: _SENSE_COL["kin"],
            4: _SENSE_COL["nonkin"], 5: _SENSE_COL["intero"], 6: _SENSE_COL["smell"],
            7: _SENSE_COL["mate"], 8: _SENSE_COL["signal"], 9: _SENSE_COL["threat"]}


def _sense_colors(world: World) -> list:
    """An RGB per brain-input slot, coloured by sensor TYPE (the layout in
    `aqua.world._sense`): each ray is food / obstacle / KIN / non-KIN, then the
    interoception block, then (if on) the smell pair."""
    ray = [_SENSE_COL["food"], _SENSE_COL["obstacle"],
           _SENSE_COL["kin"], _SENSE_COL["nonkin"]]
    cols = ray * world.cfg.n_rays_max
    cols += [_SENSE_COL["intero"]] * INTERO
    if world.smell:
        cols += [_SENSE_COL["smell"]] * SMELL
    cols += [_SENSE_COL["mate"]] * MATE
    cols += [_SENSE_COL["signal"]] * SIGNAL
    cols += [_SENSE_COL["threat"]] * THREAT
    return cols


def _draw_sense_legend(surf, small, x: int, y: int) -> None:
    """A compact key for the sensor-input colours (matches the brain's left strip)."""
    w, rows = 92, len(_SENSE_LEGEND)
    surf.fill((16, 16, 22), (x - 4, y - 3, w, rows * 14 + 20))     # dark backing
    surf.blit(small.render("senses", True, (150, 160, 180)), (x, y)); y += 15
    for label, key in _SENSE_LEGEND:
        surf.fill(_SENSE_COL[key], (x, y + 2, 9, 9))
        surf.blit(small.render(label, True, (205, 210, 225)), (x + 14, y))
        y += 14


def _ray_hit(world: World, a, ang: float) -> float:
    """Distance along a ray from agent `a` at world-angle `ang` to its first hit
    (wall / food / corpse / another agent), capped at the agent's own (heritable)
    ray reach — for drawing the cone exactly as far as this creature actually sees.
    Walls are analytic (shared `_ray_wall_dist`); food/corpses/agents are float circles."""
    rl = float(a.brain.genome.ray_len)
    dx, dy = math.cos(ang), math.sin(ang)
    d = world._ray_wall_dist(a.y, a.x, dx, dy, rl)
    half = 0.5 * world.cfg.fov / max(1, int(a.brain.genome.n_rays) - 1)
    for cy, cx in world.corpse_xy:                           # corpses (float obstacles)
        dc = math.hypot(cy - a.y, cx - a.x)
        if dc <= d and abs(((math.atan2(cy - a.y, cx - a.x) - ang + math.pi)
                            % (2 * math.pi)) - math.pi) <= half + world.corpse_radius / max(dc, 1e-9):
            d = dc
    for fy, fx in world.food_xy:                             # food morsels (float)
        df = math.hypot(fy - a.y, fx - a.x)
        if df <= d and abs(((math.atan2(fy - a.y, fx - a.x) - ang + math.pi)
                            % (2 * math.pi)) - math.pi) <= half + world.food_radius / max(df, 1e-9):
            d = df
    for b in world.agents.values():
        if b.id == a.id:
            continue
        db = math.hypot(b.y - a.y, b.x - a.x)
        if db <= d and abs(((math.atan2(b.y - a.y, b.x - a.x) - ang + math.pi)
                            % (2 * math.pi)) - math.pi) <= half + world.radius / max(db, 1e-9):
            d = db
    return d


def draw(world: World, cell: int, panel_w: int, font, small,
         selected=None, hud=None, leader=None, hitboxes=None) -> pygame.Surface:
    """One frame: the smooth continuous world on the left, the reused vivarium
    HUD + brain inspector on the right."""
    W, H = world.w * cell, world.h * cell
    surf = pygame.Surface((W + panel_w, H))
    surf.fill((12, 12, 16))

    for y0, y1, x0, x1 in world.wall_rects:                 # walls (float rectangles)
        surf.fill((60, 60, 68), ((x0 + 0.5) * cell, (y0 + 0.5) * cell,
                                 (x1 - x0) * cell, (y1 - y0) * cell))
    if world.signal.any():                                  # pheromone field (faint yellow glow)
        for sy, sx in np.argwhere(world.signal > 1e-3):
            t = min(1.0, float(world.signal[sy, sx]) / 2.0)
            surf.fill((int(70 * t) + 12, int(66 * t) + 12, 16),
                      (sx * cell, sy * cell, cell, cell))
    cr = max(1, int(round(world.corpse_radius * cell)))     # corpses (float obstacle circles)
    for y, x in world.corpse_xy:
        pygame.draw.circle(surf, (120, 74, 58), (int((x + 0.5) * cell), int((y + 0.5) * cell)), cr)
    for k, (hy, hx) in enumerate(world.halos):             # faint food-halo discs
        pygame.draw.circle(surf, tuple(c // 5 for c in _FOOD_COLORS[k % len(_FOOD_COLORS)]),
                           (int((hx + 0.5) * cell), int((hy + 0.5) * cell)),
                           max(2, int(world.cfg.food_halo_sigma * world.w * cell)), 1)
    fr = max(1, int(round(world.food_radius * cell)))       # drawn size == morsel radius
    for (y, x), t in zip(world.food_xy, world.food_type):   # food morsels (float), by type
        col = _FOOD_COLORS[(int(t) - 1) % len(_FOOD_COLORS)]
        pygame.draw.circle(surf, col, (int((x + 0.5) * cell), int((y + 0.5) * cell)), fr)

    sel = world.agents.get(selected) if selected is not None else None
    if sel is not None:                                    # the selected agent's FOV cone
        for rel in world.ray_angles(sel.brain.genome.n_rays):
            ang = sel.heading + rel
            d = _ray_hit(world, sel, ang)
            pygame.draw.line(surf, (90, 120, 160),
                             ((sel.x + 0.5) * cell, (sel.y + 0.5) * cell),
                             ((sel.x + math.cos(ang) * d + 0.5) * cell,
                              (sel.y + math.sin(ang) * d + 0.5) * cell), 1)

    ar = max(2, int(round(world.radius * cell)))           # circle = the body's real radius
    for a in world.agents.values():                        # agents as smooth circles
        col = genome_color(a.brain.genome)
        bright = 0.35 + 0.65 * min(1.0, a.energy / world.cfg.max_energy)
        col = tuple(int(ch * bright) for ch in col)
        px, py = (a.x + 0.5) * cell, (a.y + 0.5) * cell     # integer coord = cell CENTRE
        pygame.draw.circle(surf, col, (int(px), int(py)), ar)
        if a.id == selected:
            pygame.draw.circle(surf, (255, 255, 255), (int(px), int(py)), ar + 2, 1)
        pygame.draw.line(surf, (0, 0, 0), (px, py),
                         (px + math.cos(a.heading) * ar * 1.8,
                          py + math.sin(a.heading) * ar * 1.8), 1)

    _draw_panel(surf, world, W, panel_w, H, font, small, selected, hud, leader, hitboxes,
                sense_colors=_sense_colors(world), organ_colors=_MOD_COL)
    return surf


def _agent_at_px(world: World, mx: int, my: int, cell: int):
    """The agent whose circle is under the mouse (or None)."""
    wy, wx = my / cell - 0.5, mx / cell - 0.5              # invert the cell-centre mapping
    best, best_d = None, max(1.0, world.radius * 2)
    for a in world.agents.values():
        d = math.hypot(a.y - wy, a.x - wx)
        if d < best_d:
            best, best_d = a, d
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aqua.app", description="watch the continuous aqua world")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=18)
    p.add_argument("--cell", type=int, default=12, help="pixels per grid cell")
    p.add_argument("--brain", choices=("neat", "nca"), default="nca")
    p.add_argument("--rays", type=int, default=7, help="vision rays (wider, richer fan)")
    p.add_argument("--fov", type=float, default=1.8, help="total vision angle (radians)")
    p.add_argument("--no-smell", action="store_true", dest="no_smell",
                   help="disable the omnidirectional food-gradient sense")
    p.add_argument("--load-genomes", default=None, dest="load_genomes")
    p.add_argument("--save-genomes", default="aqua/pools/session.json", dest="save_genomes")
    p.add_argument("--shots-dir", default="out/aqua", dest="shots_dir")
    p.add_argument("--shot-every", type=float, default=1800.0, dest="shot_every",
                   help="seconds between auto-screenshots (0 = off)")
    a = p.parse_args(argv)

    cfg = Config(size=a.size, seed=a.seed, init_population=a.population, brain=a.brain,
                 n_rays=a.rays, fov=a.fov)
    pool = None
    if a.load_genomes:
        if Path(a.load_genomes).exists():
            from vivarium.genome_io import load_genomes
            pool = load_genomes(a.load_genomes)
            print(f"loaded {len(pool)} genomes from {a.load_genomes}")
        else:                                              # F5-friendly: no save yet -> fresh
            print(f"no saved pool at {a.load_genomes} — starting fresh (press W to save)")
    world = World(cfg, genome_pool=pool, smell=not a.no_smell)

    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    panel_w = 560
    screen = pygame.display.set_mode((world.w * a.cell + panel_w, world.h * a.cell))
    pygame.display.set_caption(f"aqua (continuous) — {a.brain} brain")
    clock = pygame.time.Clock()
    last_shot = time.time()
    t_start = time.time()
    epoch_t0 = t_start
    epoch_seen = world.epoch

    paused, speed, selected = False, 1, None
    show_help = False
    follow_oldest, leader = False, None
    leader_boxes: list = []
    think, sub, recorded = False, 0, None
    msg, msg_frames = "press H for keys", 400
    is_nca = cfg.brain == "nca"
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_h:
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
                    msg, msg_frames = "reseeded from hall", 90
                elif e.key == pygame.K_RIGHTBRACKET:       # ] more food per tick
                    world.food_rate = round(world.food_rate
                                            + (1 if world.food_rate >= 1 else 0.1), 1)
                    msg, msg_frames = f"food rate {world.food_rate:g}", 60
                elif e.key == pygame.K_LEFTBRACKET:        # [ less food per tick
                    step = 1 if world.food_rate > 1 else 0.1
                    world.food_rate = max(0.1, round(world.food_rate - step, 1))
                    msg, msg_frames = f"food rate {world.food_rate:g}", 60
                elif e.key == pygame.K_PERIOD:             # . raise the food ceiling
                    world.food_cap = round(min(0.6, world.food_cap + 0.01), 3)
                    msg, msg_frames = f"food cap {world.food_cap * 100:.0f}%", 60
                elif e.key == pygame.K_COMMA:              # , lower the food ceiling
                    world.food_cap = round(max(0.0, world.food_cap - 0.01), 3)
                    msg, msg_frames = f"food cap {world.food_cap * 100:.0f}%", 60
                elif e.key == pygame.K_l:
                    follow_oldest = not follow_oldest
                    leader = None
                    msg, msg_frames = ("follow oldest ON" if follow_oldest
                                       else "follow oldest OFF"), 90
                elif e.key == pygame.K_p:
                    name = _save_shot(screen, a.shots_dir, world)
                    msg, msg_frames = f"saved shot {name}", 120
                elif e.key == pygame.K_t:                  # toggle subtick "thinking" view
                    think = not think
                    sub = 0
                elif e.key == pygame.K_w:
                    from vivarium.genome_io import save_genomes
                    seen, genomes = set(), []
                    for g in world.population_genomes(3) + world.hall_genomes():
                        k = g.key()
                        if k not in seen:
                            seen.add(k)
                            genomes.append(g)
                    genomes = genomes or [ag.brain.genome.copy()
                                          for ag in world.agents.values()]
                    if genomes:
                        Path(a.save_genomes).parent.mkdir(parents=True, exist_ok=True)
                        save_genomes(a.save_genomes, genomes)
                        msg, msg_frames = f"saved {len(genomes)} -> {a.save_genomes}", 150
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                hit = next((g for rect, g in leader_boxes if rect.collidepoint(mx, my)),
                           None)
                if hit is not None:
                    leader = hit
                elif mx < world.w * a.cell:
                    picked = _agent_at_px(world, mx, my, a.cell)
                    if picked is not None:
                        selected, leader, follow_oldest = picked.id, None, False

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
            if anim:                                       # 1 world tick per K frames; play it
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
        if world.epoch != epoch_seen:                      # a reseed started a new epoch
            epoch_seen = world.epoch
            epoch_t0 = time.time()

        now = time.time()
        fps = clock.get_fps()
        hud = {"fps": fps, "speed": speed, "sps": fps * (speed if not paused else 0),
               "think": anim, "msg": msg if msg_frames > 0 else "",
               "real": _fmt_hms(now - t_start), "epoch_real": _fmt_hms(now - epoch_t0)}
        if msg_frames > 0:
            msg_frames -= 1
        screen.blit(draw(world, a.cell, panel_w, font, small, selected, hud,
                         leader=leader, hitboxes=leader_boxes), (0, 0))
        if show_help:
            _draw_help(screen, font, small)
            _draw_sense_legend(screen, small, 360, 24)     # the sensor-colour key, beside help
        pygame.display.flip()
        if a.shot_every > 0 and time.time() - last_shot >= a.shot_every:
            name = _save_shot(screen, a.shots_dir, world)
            last_shot = time.time()
            msg, msg_frames = f"saved shot {name}", 120
        clock.tick(30 if (speed <= 2 or anim) else 0)      # cap low-speed view at 30 fps
                                                            # (x1 = watchable real-time)
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
