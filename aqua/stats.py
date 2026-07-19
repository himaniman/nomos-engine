"""Live statistics dashboard for the aqua world (pyqtgraph, an external window).

    python -m aqua.stats --seed 1 --population 40 --speed 8

Runs the world headless and plots the key ecology metrics live, one stacked plot
per theme, with a shared time axis. Mouse: drag to PAN, wheel to ZOOM, right-drag
to box-zoom, 'A' (bottom-left button) to auto-range. SPACE pauses; +/- change how
many world steps run per refresh. A run is a pure function of (config, seed), so at
the same seed this shows the SAME trajectory as `aqua.app` — watch one, chart the
other. No files: the world runs in this process and the plots read snapshot()
directly (kept simple by request).
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from aqua.world import World
from vivarium.config import Config

_CAP = 20000    # rolling history length in points (older samples scroll off)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aqua.stats", description="live aqua statistics")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--speed", type=int, default=8, help="world steps per GUI refresh")
    p.add_argument("--interval", type=int, default=50, help="ms between refreshes")
    a = p.parse_args(argv)

    world = World(Config(size=a.size, seed=a.seed, init_population=a.population, brain="nca"))
    state = {"speed": max(1, a.speed), "paused": False}

    pg.setConfigOptions(antialias=True, background=(18, 18, 24), foreground=(205, 210, 220))
    pg.mkQApp("aqua stats")
    win = pg.GraphicsLayoutWidget(show=True, title=f"aqua — live statistics (seed {a.seed})")
    win.resize(1060, 880)

    ticks = deque(maxlen=_CAP)
    series: dict[str, deque] = {}
    curves: dict[str, object] = {}

    def new_plot(title):
        pl = win.addPlot(title=title)
        pl.addLegend(offset=(8, 6), labelTextSize="8pt")
        pl.showGrid(x=True, y=True, alpha=0.15)
        pl.setLabel("bottom", "tick")
        win.nextRow()
        return pl

    def add(plot, name, color):
        series[name] = deque(maxlen=_CAP)
        curves[name] = plot.plot(pen=pg.mkPen(color, width=2), name=name)

    p1 = new_plot("population & species")
    add(p1, "population", (240, 220, 90))
    add(p1, "species", (120, 200, 240))
    p2 = new_plot("trophic mix %")
    add(p2, "carnivore %", (235, 90, 90))
    add(p2, "scavenger %", (210, 120, 235))
    p3 = new_plot("lifespan (ticks)")
    add(p3, "life mean", (110, 220, 130))
    add(p3, "life median", (70, 155, 95))
    p4 = new_plot("energy & food")
    add(p4, "mean energy", (240, 200, 110))
    add(p4, "food", (200, 205, 215))
    p5 = new_plot("births / deaths per tick")
    add(p5, "births/t", (120, 210, 140))
    add(p5, "deaths/t", (235, 110, 110))
    for pl in (p2, p3, p4, p5):
        pl.setXLink(p1)

    prev = {"births": 0, "deaths": 0}

    def sample():
        s = world.snapshot()
        ticks.append(s["tick"])
        series["population"].append(s["population"])
        series["species"].append(s["species"])
        series["carnivore %"].append(s["carnivores"] * 100.0)
        series["scavenger %"].append(s["scavengers"] * 100.0)
        series["life mean"].append(s["life_mean"])
        series["life median"].append(s["life_median"])
        series["mean energy"].append(s["energy_mean"])
        series["food"].append(s["food"])
        db = max(0, s["births"] - prev["births"])
        dd = max(0, s["deaths"] - prev["deaths"])
        prev["births"], prev["deaths"] = s["births"], s["deaths"]
        series["births/t"].append(db / state["speed"])
        series["deaths/t"].append(dd / state["speed"])
        x = np.fromiter(ticks, dtype=float)
        for name, cv in curves.items():
            cv.setData(x, np.fromiter(series[name], dtype=float))
        win.setWindowTitle(f"aqua — seed {a.seed} · tick {s['tick']} · epoch {s['epoch']}"
                           f" · pop {s['population']} · x{state['speed']}"
                           + ("  [PAUSED]" if state["paused"] else ""))

    def refresh():
        if state["paused"]:
            return
        for _ in range(state["speed"]):
            world.step()
        sample()

    def on_key(ev):
        k = ev.key()
        if k == QtCore.Qt.Key_Space:
            state["paused"] = not state["paused"]
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            state["speed"] = min(256, state["speed"] * 2)
        elif k == QtCore.Qt.Key_Minus:
            state["speed"] = max(1, state["speed"] // 2)
    win.keyPressEvent = on_key

    sample()                                    # seed the first point so the plots aren't empty
    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(max(10, a.interval))
    pg.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
