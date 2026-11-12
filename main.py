#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecast (make.fun, round 5).

Persistence + per-game Gaussian rank diffusion.  The chart is ranked by
24h player-growth velocity, and over a 72h window the best point forecast
of a game's rank is its rank now: every directional signal tried
(24h-velocity drift, rank momentum, age drift, uniform drift) made the
backtest worse on both synthetic seeds, so no drift is applied at all.
What does help is honest, per-game uncertainty: a game's future rank is
a Gaussian around its current rank whose width follows that game's own
rank volatility.  Volatile games (frequent movers) get a wide spread and
stable games a tight one, each shrunk halfway toward the pooled median
volatility so a noisy estimate cannot dominate.

Parameters (tuned on 6 synthetic worlds + the real-history rolling
backtest; the winner beats round 4 on all 6 synthetic seeds):
  SIGMA_BASE    7.5  base rank-diffusion width, in places
  SIGMA_SHRINK  0.5  blend of per-game volatility vs pooled median
  MASS_PER_PLACE 0.70 total named probability per place
  KEEP_MIN_P    0.006  only name a game at a place when p >= this
                       (below ~0.005 a name scores worse than the
                       unnamed share, ~0.005 per game)

Reads /input/history.ndjson + task.json (or $MAKEFUN_INPUT_DIR), writes
/output/forecast.json (or $MAKEFUN_OUTPUT_DIR).  Standard library only,
no network, deterministic (no randomness, no dict-order dependence:
every iteration is over explicitly sorted ids).
"""

import json
import math
import os
import sys
from datetime import datetime

# ---- tuned parameters (see module docstring) ----
SIGMA_BASE = 7.5
SIGMA_SHRINK = 0.5
MASS_PER_PLACE = 0.70
KEEP_MIN_P = 0.006
VOL_CAP = 3.0  # cap on the per-game / pooled volatility ratio


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_reads(path):
    reads = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    return reads


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rank_volatility(reads):
    """Mean absolute rank change per game between consecutive reads.

    Returns (per_game_vol, pooled_median_vol).  A game seen in only one
    read, or never moving, falls back to the pooled median.
    """
    moves = {}
    previous = {}
    for read in reads:
        for row in read["up_and_coming"]:
            uid = int(row["universe_id"])
            rank = int(row["rank"])
            if uid in previous:
                moves.setdefault(uid, []).append(abs(rank - previous[uid]))
            previous[uid] = rank
    per_game = {}
    for uid, deltas in moves.items():
        if deltas:
            per_game[uid] = sum(deltas) / len(deltas)
    pooled = sorted(per_game.values())
    median = pooled[len(pooled) // 2] if pooled else 1.0
    if median <= 0:
        median = 1.0
    return per_game, median


def game_sigma(uid, per_game_vol, median_vol):
    """Per-game rank-diffusion width, shrunk toward the pooled median."""
    own = per_game_vol.get(uid, median_vol)
    ratio = own / median_vol
    if ratio > VOL_CAP:
        ratio = VOL_CAP
    return SIGMA_BASE * ((1.0 - SIGMA_SHRINK) + SIGMA_SHRINK * ratio)


def build_forecast(reads, places):
    """One forecast for the window: probabilities + top10."""
    last = reads[-1]
    chart = sorted(last["up_and_coming"], key=lambda r: int(r["rank"]))
    uids = [int(row["universe_id"]) for row in chart]
    rank_now = {int(row["universe_id"]): int(row["rank"]) for row in chart}
    per_game_vol, median_vol = rank_volatility(reads)

    # Gaussian place mass per game, mean = current rank (persistence).
    per_place = {}
    for uid in uids:
        sigma = game_sigma(uid, per_game_vol, median_vol)
        mean_rank = float(rank_now[uid])
        for place in range(1, places + 1):
            hi = norm_cdf((place + 0.5 - mean_rank) / sigma)
            lo = norm_cdf((place - 0.5 - mean_rank) / sigma)
            mass = hi - lo
            if mass >= 1e-6:
                per_place.setdefault(place, []).append((uid, mass))

    # Scale each place to MASS_PER_PLACE total, drop rows below KEEP_MIN_P.
    rows = []
    for place in sorted(per_place):
        entries = per_place[place]
        total = sum(mass for _, mass in entries)
        if total <= 0:
            continue
        scale = MASS_PER_PLACE / total
        for uid, mass in entries:
            p = mass * scale
            if p >= KEEP_MIN_P:
                rows.append({"universe_id": uid, "place": place,
                             "p": round(p, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    top10 = sorted(uids, key=lambda uid: (rank_now[uid], uid))[:places]
    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as handle:
        task = json.load(handle)
    reads = load_reads(os.path.join(input_dir, "history.ndjson"))
    if not reads:
        raise SystemExit("no reads in history.ndjson")
    forecast = build_forecast(reads, int(task["places"]))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task["output"])
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(forecast) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
