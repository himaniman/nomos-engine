"""Live statistics for the aqua world (pyqtgraph).

Two ways to use it:
  * embedded — `aqua.app` opens a StatsPanel beside the game window by default and
    feeds it the SAME world each frame (true real-time sync); pass --no-stats to skip;
  * standalone — `python -m aqua.stats --seed 1 --speed 8` runs its own world headless
    and charts it (a run is pure in (config, seed), so the same seed = the same run
    as the game).

Stacked, time-linked plots per theme. Mouse: drag to PAN, wheel to ZOOM, right-drag
to box-zoom, the bottom-left 'A' button to auto-range. No file plumbing — the plots
read snapshot() directly.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

_CAP = 20000    # rolling history length in points (older samples scroll off)


class StatsPanel:
    """A pyqtgraph window that charts aqua ecology metrics. Call `push(snapshot_dict)`
    once per frame/tick to append a sample and refresh the curves."""

    def __init__(self, title: str = "aqua — live statistics", cap: int = _CAP):
        pg.setConfigOptions(antialias=True, background=(18, 18, 24), foreground=(205, 210, 220))
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(760, 900)
        self._ticks: deque = deque(maxlen=cap)
        self._series: dict[str, deque] = {}
        self._curves: dict[str, object] = {}
        self._prev = {"births": 0, "deaths": 0}
        self._cap = cap

        def new_plot(t):
            pl = self.win.addPlot(title=t)
            pl.addLegend(offset=(8, 6), labelTextSize="8pt")
            pl.showGrid(x=True, y=True, alpha=0.15)
            pl.setLabel("bottom", "tick")
            self.win.nextRow()
            return pl

        def add(plot, name, color):
            self._series[name] = deque(maxlen=cap)
            self._curves[name] = plot.plot(pen=pg.mkPen(color, width=2), name=name)

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
        self._p1 = p1

    def alive(self) -> bool:
        return self.win.isVisible()

    def push(self, s: dict, dt: int = 1) -> None:
        """Append one sample from a world snapshot and refresh the curves. `dt` = how many
        world ticks this sample spans (so the birth/death RATES are per-tick)."""
        self._ticks.append(s["tick"])
        self._series["population"].append(s["population"])
        self._series["species"].append(s["species"])
        self._series["carnivore %"].append(s["carnivores"] * 100.0)
        self._series["scavenger %"].append(s.get("scavengers", 0.0) * 100.0)
        self._series["life mean"].append(s["life_mean"])
        self._series["life median"].append(s["life_median"])
        self._series["mean energy"].append(s["energy_mean"])
        self._series["food"].append(s["food"])
        db = max(0, s["births"] - self._prev["births"])
        dd = max(0, s["deaths"] - self._prev["deaths"])
        self._prev["births"], self._prev["deaths"] = s["births"], s["deaths"]
        self._series["births/t"].append(db / max(1, dt))
        self._series["deaths/t"].append(dd / max(1, dt))
        x = np.fromiter(self._ticks, dtype=float)
        for name, cv in self._curves.items():
            cv.setData(x, np.fromiter(self._series[name], dtype=float))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aqua.stats", description="live aqua statistics")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--speed", type=int, default=8, help="world steps per GUI refresh")
    p.add_argument("--interval", type=int, default=50, help="ms between refreshes")
    a = p.parse_args(argv)

    from aqua.world import World
    from vivarium.config import Config
    world = World(Config(size=a.size, seed=a.seed, init_population=a.population, brain="nca"))
    state = {"speed": max(1, a.speed), "paused": False}

    pg.mkQApp("aqua stats")
    panel = StatsPanel(title=f"aqua — live statistics (seed {a.seed})")

    def refresh():
        if state["paused"]:
            return
        for _ in range(state["speed"]):
            world.step()
        panel.push(world.snapshot(), dt=state["speed"])
        panel.win.setWindowTitle(f"aqua — seed {a.seed} · tick {world.tick} · epoch {world.epoch}"
                                 f" · pop {world.population()} · x{state['speed']}"
                                 + ("  [PAUSED]" if state["paused"] else ""))

    def on_key(ev):
        k = ev.key()
        if k == QtCore.Qt.Key_Space:
            state["paused"] = not state["paused"]
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            state["speed"] = min(256, state["speed"] * 2)
        elif k == QtCore.Qt.Key_Minus:
            state["speed"] = max(1, state["speed"] // 2)
    panel.win.keyPressEvent = on_key

    panel.push(world.snapshot())
    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(max(10, a.interval))
    pg.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
