# Roblox Up-and-Coming Top-10 Forecast — round 4

Entry for the make.fun competition `roblox-up-and-coming-forecast-01`
($500, closes 13 Nov 2026 13:00 WIB). The program reads the chart
history and forecasts which games will occupy the top 10 places of
Roblox's Up-and-Coming chart over the next 72 hours.

## Method

The chart itself ranks games by their 24-hour growth velocity, so the
forecast is built on that same signal, measured from the player-count
history:

1. **Growth.** For every game on the last chart, the 24-hour
   log-velocity of its player count: `ln(playing now / playing ~24h
   ago)`, scaled to a 24h window. Games with a shorter history use
   what they have, scaled the same way. This is the signal the chart
   moves on, and a game's velocity level persists well over the 72h
   window.
2. **Momentum.** The acceleration — recent 12h log-velocity minus
   older 12h log-velocity — is computed for every game, but its
   weight is **0.0**. On the synthetic backtests it is
   anti-predictive: fast games that are slightly decelerating stay
   on the chart for days, while low-base games with high
   acceleration never climb into the top 10. Weighting it at 0.4
   costs ~1.7–2.2 nats versus weight 0.0 on both seeds tested. The
   signals are combined raw, not z-normalised: z-scoring would
   inflate momentum until it demotes the persistent leaders.
3. **Expected order.** Games are ranked by `score = 1.0 * growth`.
   The top 10 of that order is the forecast's `top10`.
4. **Diffusion.** Each game's mass is spread over places 1–10 with
   a Gaussian in rank space centred on its expected rank, sigma
   6.0. The spread is deliberately **wide**: in the backtest, ~39%
   of future top-10 slots are filled by games ranked 11th or lower
   at the last read, so a narrow spread (sigma ~1.5) gives those
   climbers ~zero probability and the forecast barely beats
   persistence. The wide spread is what carries the gain over the
   house baseline. Each place is normalised to 0.7 named mass; rows
   below 0.006 are dropped. The remaining 0.3 per place is shared
   by the games not named there, which covers climbers from far down
   the chart and games that debut during the window.

## Running it

Python 3, standard library only, no network. Deterministic: two
runs on the same input produce byte-identical output.

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Or pass the directories as positional arguments:

```sh
python3 main.py <input_dir> <output_dir>
```

Reads `<input_dir>/history.ndjson`, writes
`<output_dir>/forecast.json` with exactly two keys:
`probabilities` (rows of `{universe_id, place, p}`) and `top10`
(10 distinct universe ids, best first).

## Backtest results (synthetic, kit `tests/score.py`)

| dataset | log score | house baseline | delta |
|---|---|---|---|
| seed 42 | **-36.825** | -47.411 | **+10.59 nats** |
| seed 7  | **-42.672** | -55.281 | **+12.61 nats** |

Round 3 scored -37.50 / +9.91 nats on seed 42; round 4 beats it
by ~0.68 nats. Output is deterministic (two runs byte-identical)
and passes `tests/validate.py` (179 rows, 10 places named,
per-place sums ~0.685). Also verified on the real kit input
(`VALID`) and on the `min_in` / `notask_in` / `empty_in` fixtures
(no crash; empty history yields an empty forecast).
