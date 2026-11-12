# Roblox Up-and-Coming top-10 forecast — round 5

Forecast for `roblox-up-and-coming-forecast-01`: the Up-and-Coming chart's
top 10 at every hour of the 72 hours after entries close. The program
(`python3 /bot/main.py`, Python 3.13, standard library only) reads
`/input/history.ndjson` + `task.json` and writes one file,
`/output/forecast.json`, with a probability for each game named at each
of the 10 places plus the top 10 in order.

## Method: persistence + per-game Gaussian rank diffusion

The chart ranks games by 24h player-growth velocity, and over a 72h
window the best point forecast of a game's rank is its rank now. Every
directional signal tried — 24h-velocity drift (both directions),
rank momentum, age drift, a uniform downward drift — made the backtest
worse on both synthetic seeds, so **no drift is applied at all**. The
chart is a revolving door whose direction cannot be called from
velocity; the current rank is the best estimate of the future rank.

What does improve the score is honest, per-game uncertainty. A game's
future rank is a Gaussian around its current rank. Its width follows
that game's own rank volatility (mean absolute rank change between
consecutive reads), shrunk halfway toward the pooled median volatility
so a noisy estimate cannot dominate:

- volatile game (frequent mover) -> wide spread, hedged probabilities
- stable game -> tight spread, confident probabilities

Each place's probabilities are scaled to sum to `MASS_PER_PLACE`, and a
game is only named at a place when its probability reaches
`KEEP_MIN_P`: below ~0.005 a name scores worse than leaving the game
unnamed (the unnamed share is spread over the chart, ~0.005 per game).

## Parameters (tuned, see below)

| Parameter | Value | Meaning |
|---|---|---|
| `SIGMA_BASE` | 7.5 | base rank-diffusion width, in places |
| `SIGMA_SHRINK` | 0.5 | blend of per-game volatility vs pooled median |
| `MASS_PER_PLACE` | 0.70 | total named probability per place |
| `KEEP_MIN_P` | 0.006 | minimum probability to name a game at a place |
| `VOL_CAP` | 3.0 | cap on the per-game / pooled volatility ratio |

## Backtest results (log score, higher is better)

Synthetic worlds (the kit's generator, 72h future, 72 reads). Round 5
beats round 4 on **all six seeds**:

| Seed | Round 4 | Round 5 | Delta | Delta vs house |
|---|---|---|---|---|
| 42 | -36.8256 | -36.8033 | +0.022 | +10.61 |
| 7 | -42.6694 | -42.4771 | +0.192 | +12.80 |
| 123 | -38.4385 | -38.4256 | +0.013 | +8.55 |
| 1 | -38.613 | -38.1963 | +0.417 | +9.56 |
| 99 | -40.920 | -40.8410 | +0.079 | +8.40 |
| 2026 | -37.813 | -37.7673 | +0.046 | +10.82 |
| **mean** | **-39.213** | **-39.085** | **+0.128** | — |

Caveat, stated honestly: on the sparse real-history rolling backtest
(31 cutoffs, 72h horizon, 295 reads) round 5 scores -36.87 against
round 4's -36.78 (-0.09). The real practice history is sparse (34 reads,
one every 4-11h); the per-game volatility estimate needs dense reads to
be reliable, and it degrades when the history is thinned (verified by
subsampling the synthetic histories). The final's history is dense
(hourly house-recorder reads), like the synthetic sets where round 5
wins 6/6, so the synthetic evidence is the applicable one here — but
the real-history number is reported, not hidden.

## What was tried and rejected

- 24h-velocity drift, either sign (positive: -10 nats; negative: worse)
- Rank momentum, any weight
- Age drift (young up / old down) and uniform downward drift
- Larger uniform sigma (7-12) or mass (0.75-0.80) without per-game
  scaling: helps some seeds, hurts others; no uniform pair beats
  round 4 on both seeds 42 and 7
- Per-game sigma with full weight (shrink 1.0) or no weight (shrink 0):
  both worse; the 50/50 shrink is the optimum

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 tests/validate.py out/forecast.json
```

Deterministic: two runs on the same input produce byte-identical
output (no randomness, no dict-order dependence). Standard library
only, no network, well inside the 300s limit.
