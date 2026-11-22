"""A ledger of exam 6: every record of a life in a land, one row, for a reader who was not here.

    python -m tissue.ledger [--out runs/tissue/world-ledger.csv]

Reads `runs/tissue/*/*.world*.json` (one file per genome, land and arm; inside, one result
per size of brain) and the yardsticks of the same land (`runs/tissue/world*.yardsticks*.json`),
and writes one row per (genome, land, switches, size). What a record's name says is parsed back
into columns: the land (side, walls, trees, ticks, shift), the arm, and the switches of thought
(budget, slack, imagery, mumble, curious, explore, sure, keep, fade), the seed of the lands. A column a
record does not have (older records lack `early`, `thoughts`, `sought`, `meal_ticks`) is left
empty.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

NUM = r"(\d+(?:\.\d+)?)"
DEFAULTS = {"walls": 0.3, "trees": 0, "ticks": 0, "shift": 0.0, "budget": 256, "slack": 0,
            "mumble": 0.0, "curious": 0.0, "explore": 0.0, "fade": 0.0, "wave": 0.0, "dreams": 0, "depth": 24,
            "seed": 90_000}


def parse(name: str) -> dict | None:
    """What a record's file name says. None: not a record of exam 6."""
    m = re.match(r"(?P<genome>.+?)\.world(?P<side>\d+)(?P<rest>\..*)\.json$", os.path.basename(name))
    if not m or ".yardsticks" in name:
        return None
    rest = m["rest"]
    arm = re.search(r"\.(thinks|numb)(?=\.|$)", rest)
    if not arm:
        return None
    row = {"genome": m["genome"], "side": int(m["side"]), "arm": arm[1]}
    ground, switches = rest[:arm.start()], rest[arm.end():]
    for key in ("walls", "trees", "ticks", "shift"):
        hit = re.search(rf"\.{key}{NUM}", ground)
        row[key] = float(hit[1]) if hit else DEFAULTS[key]
    for key in ("budget", "slack", "mumble", "curious", "explore", "fade", "wave", "dreams", "depth", "seed"):
        hit = re.search(rf"\.{key}{NUM}", switches)
        row[key] = float(hit[1]) if hit else DEFAULTS[key]
    row["reborn"] = ".reborn" in ground         # a look: the body born again when the land changed
    row["imagery"] = ".imagery" in switches
    row["keep"] = bool(re.search(r"\.keep(?=\.|$)", switches))
    row["sure"] = bool(re.search(r"\.sure(?=\.|$)", switches))
    row["cut"] = bool(re.search(r"\.cut(?=\.|$)", switches))
    row["again"] = bool(re.search(r"\.again(?=\.|$)", switches))
    row["ground"] = ground
    row["pilot"] = row["seed"] != DEFAULTS["seed"]
    return row


def yardsticks(side: int, ground: str, pilot: bool, seed: float) -> dict:
    path = f"runs/tissue/world{side}{ground}.yardsticks" + (f".seed{seed:g}" if pilot else "") + ".json"
    if not os.path.exists(path):
        return {}
    return {arm: res["meals"] for arm, res in json.load(open(path)).items()}


def rows(paths: list[str]) -> list[dict]:
    out = []
    for path in sorted(paths):
        head = parse(path)
        if head is None:
            continue
        sticks = yardsticks(head["side"], head["ground"], head["pilot"], head["seed"])
        for cells, res in json.load(open(path)).items():
            if not isinstance(res, dict) or "meals" not in res:
                continue
            late = res.get("late", {}).get("9+", {})
            every = res.get("trips", {}).get("9+", {})
            row = {k: v for k, v in head.items() if k not in ("ground",)}
            row.update({
                "cells": int(cells), "lives": len(res.get("per_life", [])), "meals": round(res["meals"], 3),
                "meals_third1": res["thirds"][0], "meals_third2": res["thirds"][1], "meals_third3": res["thirds"][2],
                "locked_lives": sum(x == 0 for x in res.get("late_per_life", [])) if "late_per_life" in res else "",
                "planned": round(res.get("planned", 0.0), 4),
                "thought_afresh": round(res["thoughts"], 4) if "thoughts" in res else "",
                "sought": round(res["sought"], 4) if "sought" in res else "",
                "hops_a_tick": round(res["hops"], 1) if "hops" in res else "",
                "early_ways_gone": round(res["early"], 4) if "early" in res else "",
                "far9_straight": round(every["straight"], 4) if every else "",
                "far9_straight_late": round(late["straight"], 4) if late else "",
                "map_right_gone": round(res["map"]["right_gone"], 4) if "map" in res else "",
                "perfect": sticks.get("perfect", ""), "seeker": sticks.get("seeker", ""),
                "explorer": sticks.get("explorer", ""),
                "oracle": sticks.get("oracle", ""), "blind": sticks.get("blind", ""),
                "has_meal_ticks": "meal_ticks" in res, "record": path.replace(os.sep, "/"),
            })
            out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="runs/tissue/world-ledger.csv")
    a = p.parse_args()
    table = rows(glob.glob("runs/tissue/*/*.world*.json"))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{a.out}: {len(table)} rows")


if __name__ == "__main__":
    main()
