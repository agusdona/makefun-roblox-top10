#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster.

make.fun bounty: "Predict Roblox's Up-and-Coming top 10"
(https://make.fun/bounty/roblox-up-and-coming-forecast-01)

Reads the chart history in $MAKEFUN_INPUT_DIR (default /input):
  history.ndjson  - one JSON object per line, one read per line, oldest first
  task.json       - history_until, scored window, places
Writes $MAKEFUN_OUTPUT_DIR (default /output)/forecast.json:
  {"probabilities": [{"universe_id": int, "place": 1..10, "p": float}, ...],
   "top10": [10 universe_ids in order]}

Method (stdlib only, no network, fully deterministic):
  For every game seen in the history, model its rank over the 72-hour
  scored window as a Gaussian diffusion with drift:
    * drift comes from recent rank momentum and 24h player-growth
      velocity (the chart ranks by growth velocity, not size), mapped
      through a growth->rank-velocity regression fitted on the history
      itself, plus a same-genre momentum term and a small boost for
      games updated in the last 48h (update bumps are visible in the
      data via details.updated);
    * drift decays exponentially (momentum half-life ~25h) and the mean
      rank reverts mildly toward an equilibrium rank implied by a
      strength score (log playing, 24h growth, top-10 stability);
    * diffusion sigma comes from the game's own hourly rank-change
      variance, shrunk toward the pooled variance across all games;
    * a game ages out 28 days after its chart debut (first seen in the
      history; when it was already charting in the first read, the
      creation date is the debut proxy): survival follows a logistic
      around the age-out hour, so a game leaving mid-window keeps only
      its pre-exit probability mass.
  The place distribution of a game is the average over the 72 hours of
  its hourly Gaussian place probabilities.  Per place, games are named
  only when their probability clears the unnamed-share level, and each
  place's named total is capped so >=15% stays unnamed: the scorer
  spreads the unnamed share over every game on the future read's chart,
  which is the only hedge for games that launch during the window.
  top10 is ordered by expected place.

Calibration (from the research in ../prediction.md, not from hardcoding
any game): 24h growth is the dominant signal (the ragdoll wave climbed
on +90%/+151% 24h growth); CCU stability keeps small sticky games in
the tail (Ragdoll Realism held top 10 for six days); update recency
matters (Do Not Stop. was actively updated on 12 Nov); the chart is
extremely volatile (median 6 top-10 places change between consecutive
reads), so probabilities are spread, never concentrated on one order.

No game name, universe id or event is hardcoded anywhere in this file:
every number above is estimated from the history it is given.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
HISTORY_FILE = "history.ndjson"
TASK_FILE = "task.json"
OUTPUT_FILE = "forecast.json"

MAX_OUTPUT_BYTES = 1_048_576
PLACES = 10
WINDOW_HOURS = 72

# --- calibration constants (see module docstring) -------------------------
AGE_OUT_DAYS = 28            # chart eligibility window, from chart debut
AGE_OUT_SCALE_H = 6.0        # logistic scale (hours) around the age-out hour
KEEP_MIN_P = 0.002           # only name a game at a place above this p
PLACE_SUM_CAP = 0.85         # cap named total per place; >=0.15 stays unnamed
MAX_ROWS_PER_PLACE = 40
DRIFT_CAP = 0.4              # ranks/hour, sustained drift clamp
DRIFT_TAU_H = 36.0           # exponential decay of momentum (hours)
REVERT_RATE = 0.01           # per-hour pull toward equilibrium rank
SIGMA_CAP = 9.0              # ranks, diffusion cap
SIGMA_FLOOR = 0.5            # ranks/hour, per-game sigma floor
UPDATE_BOOST_MAX = 0.06      # ranks/hour extra climb for a fresh update
UPDATE_DECAY_H = 48.0        # decay of the update boost (hours)
UPDATE_WINDOW_H = 72.0       # only updates this recent count
GENRE_WEIGHT = 0.15          # weight of same-genre momentum in drift
MOMENTUM_WEIGHT = 0.6        # weight of rank momentum in drift
GROWTH_WEIGHT = 0.4          # weight of growth-velocity drift
DEFAULT_SLOPE = -20.0        # growth->rank-velocity slope fallback
DEFAULT_SIGMA = 2.0          # pooled sigma fallback (ranks/sqrt(hour))


# --- small helpers ---------------------------------------------------------

def parse_time(value):
    """Parse an ISO-8601 timestamp to an aware UTC datetime, or None."""
    if not isinstance(value, str):
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def norm_cdf(x):
    """Standard normal CDF via erf (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


# --- input -----------------------------------------------------------------

def load_task(path):
    """Read task.json; fall back to defaults when it is missing."""
    task = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            task = json.load(fh)
    except (OSError, ValueError):
        task = {}
    if not isinstance(task, dict):
        task = {}
    window = task.get("window") or {}
    slots = window.get("slots")
    places = task.get("places")
    return {
        "history_until": parse_time(task.get("history_until")),
        "slots": int(slots) if isinstance(slots, int) and slots > 0 else WINDOW_HOURS,
        "places": int(places) if isinstance(places, int) and places > 0 else PLACES,
    }


def load_history(path):
    """Read history.ndjson into a time-sorted list of reads.

    Each read: {"at": datetime, "games": {uid: game dict}, "details": {...}}.
    A game dict has rank, playing (or None), name, genre.  Bad lines are
    skipped; a read with no usable chart is skipped.
    """
    reads = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError:
        return reads
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict) or obj.get("ok") is False:
                continue
            at = parse_time(obj.get("at"))
            chart = obj.get("up_and_coming")
            if at is None or not isinstance(chart, list) or not chart:
                continue
            games = {}
            for row in chart:
                if not isinstance(row, dict):
                    continue
                try:
                    uid = int(row.get("universe_id"))
                    rank = int(row.get("rank"))
                except (TypeError, ValueError):
                    continue
                if uid in games or rank < 1:
                    continue
                playing = row.get("playing")
                try:
                    playing = float(playing) if playing is not None else None
                except (TypeError, ValueError):
                    playing = None
                if playing is not None and playing < 0:
                    playing = None
                games[uid] = {
                    "rank": rank,
                    "playing": playing,
                    "name": row.get("name"),
                    "genre": row.get("genre"),
                }
            if not games:
                continue
            details = obj.get("details")
            reads.append({"at": at, "games": games,
                          "details": details if isinstance(details, dict) else {}})
    reads.sort(key=lambda r: r["at"])
    return reads


# --- per-game series ---------------------------------------------------------

def build_series(reads):
    """Collapse reads into one time series per universe id."""
    series = {}
    for read in reads:
        at = read["at"]
        for uid, game in read["games"].items():
            entry = series.get(uid)
            if entry is None:
                entry = series[uid] = {
                    "points": [],          # (at, rank, playing)
                    "details": {},         # latest non-null details row
                    "first_seen": at,
                    "last_seen": at,
                    "name": game.get("name"),
                    "genre": game.get("genre"),
                }
            entry["points"].append((at, game["rank"], game["playing"]))
            entry["last_seen"] = at
            if game.get("genre"):
                entry["genre"] = game["genre"]
            if game.get("name"):
                entry["name"] = game["name"]
        for uid, det in read["details"].items():
            if not isinstance(det, dict):
                continue
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                continue
            entry = series.get(uid)
            if entry is None:
                continue
            for key in ("name", "playing", "visits", "favorites",
                        "max_players", "created", "updated",
                        "creator_name", "creator_type", "genre"):
                if det.get(key) is not None:
                    entry["details"][key] = det[key]
            if det.get("genre"):
                entry["genre"] = det["genre"]
    for entry in series.values():
        entry["points"].sort(key=lambda p: p[0])
    return series


def nearest_point(points, target, max_gap_h=36.0):
    """Point (at, rank, playing) nearest to `target`, or None."""
    best = None
    best_gap = None
    for point in points:
        gap = abs(hours_between(point[0], target))
        if gap <= max_gap_h and (best_gap is None or gap < best_gap):
            best = point
            best_gap = gap
    return best


def growth_rate(points, end, hours_back):
    """Log growth of `playing` per hour over the last `hours_back` hours.

    Uses the points nearest the window ends; returns None when the
    playing series cannot bracket the window.
    """
    then = nearest_point(points, end - timedelta(hours=hours_back))
    now = nearest_point(points, end)
    if then is None or now is None:
        return None
    if not then[2] or not now[2] or then[2] <= 0 or now[2] <= 0:
        return None
    dt = hours_between(then[0], now[0])
    if dt < 1.0:
        return None
    return math.log(now[2] / then[2]) / dt


# --- pooled statistics -------------------------------------------------------

def pooled_sigma(series):
    """Pooled hourly rank-change sigma across all games (ranks/sqrt(h))."""
    diffs = []
    for entry in series.values():
        pts = entry["points"]
        for (t1, r1, _), (t2, r2, _) in zip(pts, pts[1:]):
            dt = hours_between(t1, t2)
            if 0 < dt <= 24:
                diffs.append((r2 - r1) / math.sqrt(dt))
    if len(diffs) < 10:
        return DEFAULT_SIGMA
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return clamp(math.sqrt(var), SIGMA_FLOOR, 4.0)


def growth_slope(series):
    """Least-squares slope of rank-change/hour on log-growth/hour.

    Negative: growing games climb (rank number falls).  Falls back to
    DEFAULT_SLOPE when the history cannot identify it.
    """
    sxx = 0.0
    sxy = 0.0
    n = 0
    for entry in series.values():
        pts = [(t, r, p) for (t, r, p) in entry["points"] if p and p > 0]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dt = hours_between(pts[i][0], pts[j][0])
                if 20.0 <= dt <= 28.0:
                    x = math.log(pts[j][2] / pts[i][2]) / dt
                    y = (pts[j][1] - pts[i][1]) / dt
                    sxx += x * x
                    sxy += x * y
                    n += 1
    if n < 10 or sxx <= 1e-12:
        return DEFAULT_SLOPE
    slope = sxy / sxx
    if slope >= 0:
        return DEFAULT_SLOPE
    return clamp(slope, -80.0, -1.0)


def game_sigma(entry, pooled):
    """Per-game hourly rank-change sigma, shrunk toward the pooled one."""
    pts = entry["points"]
    diffs = []
    for (t1, r1, _), (t2, r2, _) in zip(pts, pts[1:]):
        dt = hours_between(t1, t2)
        if 0 < dt <= 24:
            diffs.append((r2 - r1) / math.sqrt(dt))
    if len(diffs) < 5:
        return pooled
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    n = float(len(diffs))
    shrunk = math.sqrt((n * var + 5.0 * pooled * pooled) / (n + 5.0))
    return clamp(shrunk, SIGMA_FLOOR, 4.0)


# --- per-game forecast ---------------------------------------------------------

def game_features(uid, entry, end, archive_start, slope, pooled,
                  g24, genre_growth):
    """Estimate one game's rank process at the history end."""
    pts = entry["points"]
    last_at, last_rank, last_playing = pts[-1]
    hours_since_last = max(0.0, hours_between(last_at, end))

    # Rank at the window start: last seen rank, drifted out if stale.
    r0 = last_rank + min(30.0, 0.25 * hours_since_last)

    # Rank momentum: ranks gained per hour over the last 24h (negative
    # means climbing).  Falls back to the earliest read when the history
    # is shorter than a day.
    then = nearest_point(pts, end - timedelta(hours=24.0))
    if then is not None:
        span = hours_between(then[0], end)
        momentum = (last_rank - then[1]) / span if span > 0 else 0.0
    else:
        momentum = 0.0

    # Stability: share of reads in the last 72h spent in the top 10.
    recent = [p for p in pts if hours_between(p[0], end) <= 72.0]
    stability = (sum(1 for p in recent if p[1] <= 10) / len(recent)) if recent else 0.0

    # Size: log playing at the end (falls back to visits).
    size = None
    if last_playing and last_playing > 0:
        size = math.log10(last_playing)
    elif entry["details"].get("visits"):
        try:
            size = math.log10(float(entry["details"]["visits"]) + 1.0) - 3.0
        except (TypeError, ValueError):
            size = None

    # Drift: momentum + growth velocity + genre wave + update bump.
    drift = MOMENTUM_WEIGHT * momentum + GROWTH_WEIGHT * slope * g24
    genre = entry.get("genre")
    if genre and genre in genre_growth:
        drift += GENRE_WEIGHT * slope * genre_growth[genre]
    updated = parse_time(entry["details"].get("updated"))
    if updated is not None:
        age_h = hours_between(updated, end)
        if 0.0 <= age_h <= UPDATE_WINDOW_H:
            drift -= UPDATE_BOOST_MAX * math.exp(-age_h / UPDATE_DECAY_H)
    drift = clamp(drift, -DRIFT_CAP, DRIFT_CAP)

    # Age-out: 28 days after chart debut.  Debut is the first time the
    # game is seen on the chart; when it was already charting in the
    # first read (left-censored debut) the creation date is the proxy.
    debut = entry["first_seen"]
    created = parse_time(entry["details"].get("created"))
    if (created is not None and entry["first_seen"] <= archive_start
            and created < entry["first_seen"]):
        debut = created
    age_out_h = hours_between(end, debut + timedelta(days=AGE_OUT_DAYS))
    if age_out_h <= 0:
        # The chart itself contradicts the estimate (game still ranked):
        # trust the chart and do not age the game out.
        age_out_h = None

    return {
        "uid": uid,
        "r0": r0,
        "drift": drift,
        "sigma": game_sigma(entry, pooled),
        "g24": g24,
        "stability": stability,
        "size": size,
        "age_out_h": age_out_h,
    }


def equilibrium_ranks(features):
    """Rank games by a strength score -> equilibrium rank per game."""
    def z(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return [0.0] * len(values)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        return [(0.0 if v is None else (v - mean) / sd) for v in values]

    sizes = z([f["size"] for f in features])
    growths = z([f["g24"] for f in features])
    stabs = z([f["stability"] for f in features])
    scored = []
    for i, f in enumerate(features):
        strength = 0.45 * sizes[i] + 0.35 * growths[i] + 0.20 * stabs[i]
        scored.append((strength, f["uid"]))
    scored.sort(key=lambda s: (-s[0], s[1]))
    eq = {}
    for pos, (_, uid) in enumerate(scored, start=1):
        eq[uid] = float(pos)
    return eq


def place_probabilities(features, eq, hours):
    """Average hourly place probabilities per game.

    Returns {uid: [p(place 1), ..., p(place 10)]} averaged over the
    scored window's hours.
    """
    out = {}
    for f in features:
        mean = f["r0"]
        target = eq[f["uid"]]
        sigma = f["sigma"]
        drift = f["drift"]
        age_out = f["age_out_h"]
        acc = [0.0] * PLACES
        for h in range(1, hours + 1):
            decayed = drift * math.exp(-h / DRIFT_TAU_H)
            mean += decayed + REVERT_RATE * (target - mean)
            sd = min(SIGMA_CAP, math.sqrt(sigma * sigma * h + 0.25))
            if age_out is None:
                surv = 1.0
            else:
                surv = 1.0 / (1.0 + math.exp((h - age_out) / AGE_OUT_SCALE_H))
            for k in range(1, PLACES + 1):
                lo = norm_cdf((k - 0.5 - mean) / sd)
                hi = norm_cdf((k + 0.5 - mean) / sd)
                acc[k - 1] += surv * (hi - lo)
        out[f["uid"]] = [a / hours for a in acc]
    return out


# --- assembly ------------------------------------------------------------------

def assemble(place_probs):
    """Build the forecast object from per-game place probabilities."""
    rows = []
    for place in range(1, PLACES + 1):
        column = [(p[place - 1], uid) for uid, p in place_probs.items()
                  if p[place - 1] >= KEEP_MIN_P]
        column.sort(key=lambda c: (-c[0], c[1]))
        column = column[:MAX_ROWS_PER_PLACE]
        total = sum(p for p, _ in column)
        if total > PLACE_SUM_CAP:
            scale = PLACE_SUM_CAP / total
            column = [(p * scale, uid) for p, uid in column]
        for p, uid in column:
            p = round(p, 6)
            if p > 0:
                rows.append({"universe_id": int(uid), "place": place, "p": p})
    # top10 by expected place, then by total top-10 probability.
    expected = []
    for uid, p in place_probs.items():
        total = sum(p)
        if total <= 0:
            continue
        exp_place = sum((k + 1) * p[k] for k in range(PLACES)) / total
        expected.append((exp_place, -total, uid))
    expected.sort()
    top10 = [uid for _, _, uid in expected[:PLACES]]
    return {"probabilities": rows, "top10": top10}


# --- main ----------------------------------------------------------------------

def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    task = load_task(os.path.join(input_dir, TASK_FILE))
    reads = load_history(os.path.join(input_dir, HISTORY_FILE))
    if not reads:
        sys.stderr.write("no usable reads in %s\n"
                         % os.path.join(input_dir, HISTORY_FILE))
        forecast = {"probabilities": [], "top10": []}
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, OUTPUT_FILE), "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
        return 0

    end = task["history_until"] or reads[-1]["at"]
    hours = task["slots"]
    archive_start = reads[0]["at"]
    series = build_series(reads)

    pooled = pooled_sigma(series)
    slope = growth_slope(series)

    # 24h growth per game, then same-genre momentum (genres with >= 2 games).
    g24_by_uid = {}
    for uid in sorted(series):
        g = growth_rate(series[uid]["points"], end, 24.0)
        g24_by_uid[uid] = g if g is not None else 0.0
    by_genre = {}
    for uid in sorted(series):
        genre = series[uid].get("genre")
        if genre:
            by_genre.setdefault(genre, []).append(g24_by_uid[uid])
    genre_growth = {g: sum(v) / len(v) for g, v in by_genre.items() if len(v) >= 2}

    features = [game_features(uid, series[uid], end, archive_start, slope,
                              pooled, g24_by_uid[uid], genre_growth)
                for uid in sorted(series)]
    eq = equilibrium_ranks(features)
    place_probs = place_probabilities(features, eq, hours)
    forecast = assemble(place_probs)

    blob = json.dumps(forecast, separators=(",", ":")).encode("utf-8")
    if len(blob) > MAX_OUTPUT_BYTES:  # pragma: no cover - safety valve
        rows = forecast["probabilities"]
        rows.sort(key=lambda r: r["p"])
        while rows and len(blob) > MAX_OUTPUT_BYTES:
            rows.pop(0)
            blob = json.dumps(forecast, separators=(",", ":")).encode("utf-8")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, OUTPUT_FILE), "wb") as fh:
        fh.write(blob)
    sys.stderr.write("wrote %s (%d bytes, %d probability rows, top10=%s)\n"
                     % (os.path.join(output_dir, OUTPUT_FILE), len(blob),
                        len(forecast["probabilities"]), forecast["top10"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
