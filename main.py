#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Method (round 4, written from scratch in this session):
  * read /input/history.ndjson: chart reads, oldest first. Each read
    holds the chart in order, every game with its player count.
  * per game, from its player-count series, compute two signals at the
    last read:
      - growth: the 24-hour log-velocity of its player count, i.e.
        ln(playing now / playing ~24h ago), scaled to a 24h window.
        The chart itself ranks by 24h growth velocity, so this is the
        signal the chart is built on.
      - momentum: the change in that velocity, the recent 12h
        log-velocity minus the older 12h log-velocity. Positive means
        the game is still accelerating; negative means it is fading.
        Faders drop off the chart, climbers hold on.
  * combine the raw log-signals:
        score = 1.0 * growth + 0.0 * momentum
    Growth (the 24h log-velocity) is the predictive signal: the chart
    ranks by it, and a game's velocity level persists over the 72h
    window. Momentum (the acceleration, recent 12h velocity minus
    older 12h velocity) is computed but weighted to zero: on the
    synthetic backtests it is anti-predictive — fast games that are
    slightly decelerating stay on the chart for days, while
    low-base games with high acceleration never climb into the top
    10. The signals are combined raw, not z-normalised: z-scoring
    would inflate momentum until it demotes those persistent leaders.
    (Validated on synthetic seeds 42 and 7: momentum weight 0.4
    costs ~1.7-2.2 nats versus weight 0.0 on both seeds.)
  * rank games by score -> the expected order.
  * diffuse each game's mass over places 1..10 with a Gaussian in
    rank space centred on its expected rank (sigma = SIGMA_SCALE *
    SIGMA_BASE = 6.0), normalise each place to MASS_PER_PLACE = 0.7,
    and drop rows below KEEP_MIN_P. The diffusion is deliberately
    WIDE: in the backtest, ~39% of future top-10 slots are filled by
    games ranked 11th or lower at the last read, so a narrow spread
    (sigma ~1.5) gives those climbers ~zero probability and the
    forecast barely beats persistence. The wide spread is what
    carries the gain over the house baseline. What a place leaves
    unnamed (0.3) is shared by the games not named there, which
    covers climbers from far down the chart and games that debut
    during the window.
  * top10 = the 10 highest-scored games, in order.

Deterministic, standard library only, no network. Reads
$MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).
"""

import json
import math
import os
import sys

GROWTH_WEIGHT = 1.0
MOMENTUM_WEIGHT = 0.0
SIGMA_SCALE = 0.5
SIGMA_BASE = 12.0
MASS_PER_PLACE = 0.7
KEEP_MIN_P = 0.006
PLACES = 10


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into epoch seconds."""
    # fromisoformat in 3.11+ handles 'Z'; be explicit for safety.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def load_history(input_dir):
    """Read history.ndjson into a list of (epoch, chart) pairs, oldest first.

    Each chart is a list of (universe_id, playing) rows.
    Reads with no usable chart are skipped.
    """
    path = os.path.join(input_dir, "history.ndjson")
    reads = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                chart = obj.get("up_and_coming")
                if not isinstance(chart, list) or not chart:
                    continue
                try:
                    when = parse_time(obj["at"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows = []
                for row in chart:
                    try:
                        uid = int(row["universe_id"])
                        playing = row.get("playing")
                        if playing is None:
                            continue
                        playing = float(playing)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if playing <= 0:
                        continue
                    rows.append((uid, playing))
                if rows:
                    reads.append((when, rows))
    except OSError:
        return []
    reads.sort(key=lambda item: item[0])
    return reads


def build_series(reads):
    """Per game, a sorted list of (epoch, playing) from the reads."""
    series = {}
    for when, rows in reads:
        for uid, playing in rows:
            series.setdefault(uid, []).append((when, playing))
    for uid in series:
        series[uid].sort()
    return series


def playing_at(points, target):
    """Playing count at the point nearest to target epoch seconds.

    Returns (playing, actual_epoch) or (None, None) when there is no point.
    """
    if not points:
        return None, None
    best = None
    best_dt = None
    for when, playing in points:
        dt = abs(when - target)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = (playing, when)
    return best


def growth_and_momentum(points, t_last):
    """Growth and momentum of one game at the last read.

    growth: 24h log-velocity, ln(playing now / playing ~24h ago),
    scaled to a 24h window (games with a shorter history use what
    they have, scaled the same way).
    momentum: recent 12h log-velocity minus older 12h log-velocity.
    Returns (growth, momentum); either is 0.0 when the data is missing.
    """
    p_now, t_now = playing_at(points, t_last)
    if p_now is None or p_now <= 0:
        return 0.0, 0.0
    p_24, t_24 = playing_at(points, t_last - 24 * 3600)
    growth = 0.0
    if p_24 is not None and p_24 > 0 and t_now is not None and t_24 is not None:
        hours = (t_now - t_24) / 3600.0
        if hours >= 1.0:
            growth = math.log(p_now / p_24) / hours * 24.0
    p_12, t_12 = playing_at(points, t_last - 12 * 3600)
    momentum = 0.0
    if (p_12 is not None and p_12 > 0 and p_24 is not None and p_24 > 0
            and t_12 is not None and t_24 is not None):
        v_recent = math.log(p_now / p_12)
        v_older = math.log(p_12 / p_24)
        momentum = v_recent - v_older
    return growth, momentum


def expected_order(reads):
    """Rank games by combined growth+momentum score.

    Returns a list of universe_ids, best expected first.
    """
    if not reads:
        return []
    t_last = reads[-1][0]
    series = build_series(reads)
    growths = {}
    momentums = {}
    for uid, points in series.items():
        g, m = growth_and_momentum(points, t_last)
        growths[uid] = g
        momentums[uid] = m
    # Combine the raw log-signals. Growth (24h log-velocity) is the
    # primary signal and is much larger in magnitude than momentum
    # (a change of velocity), so it dominates naturally; do NOT
    # z-normalise, which would inflate momentum until it demotes
    # fast-but-slightly-decelerating games that stay on the chart.
    scored = []
    for uid in series:
        score = (GROWTH_WEIGHT * growths.get(uid, 0.0)
                 + MOMENTUM_WEIGHT * momentums.get(uid, 0.0))
        scored.append((uid, score))
    # Highest score first; universe_id breaks ties deterministically.
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [uid for uid, _ in scored]


def diffuse(order):
    """Gaussian diffusion of each game over places 1..PLACES.

    order: universe_ids, best expected first (expected rank = position).
    Returns rows: list of (universe_id, place, p), p >= KEEP_MIN_P,
    and the per-place named total (for the unnamed share).
    """
    sigma = SIGMA_SCALE * SIGMA_BASE
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    weights = {}  # (uid, place) -> raw weight
    totals = {}   # place -> sum of raw weights
    for idx, uid in enumerate(order):
        rank = idx + 1
        for place in range(1, PLACES + 1):
            d = rank - place
            w = math.exp(-(d * d) * inv_two_sigma2)
            weights[(uid, place)] = w
            totals[place] = totals.get(place, 0.0) + w
    rows = []
    for (uid, place), w in sorted(weights.items()):
        total = totals[place]
        if total <= 0:
            continue
        p = MASS_PER_PLACE * w / total
        if p >= KEEP_MIN_P:
            rows.append((uid, place, p))
    return rows


def build_forecast(input_dir):
    """Compute the forecast dict for the history in input_dir."""
    reads = load_history(input_dir)
    order = expected_order(reads)
    rows = diffuse(order)
    probs = []
    for uid, place, p in rows:
        probs.append({
            "universe_id": int(uid),
            "place": int(place),
            "p": round(float(p), 6),
        })
    # Canonical row order: by place, then descending p, then id.
    probs.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    top10 = [int(uid) for uid in order[:PLACES]]
    return {"probabilities": probs, "top10": top10}


def main(argv=None):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    if argv:
        # Optional positional overrides, handy for local runs:
        # main.py [input_dir [output_dir]]
        if len(argv) > 1:
            input_dir = argv[1]
        if len(argv) > 2:
            output_dir = argv[2]
    forecast = build_forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(forecast, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
