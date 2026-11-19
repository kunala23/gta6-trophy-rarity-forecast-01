# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kunala23/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kunala23/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate — the percent
of the players Exophase tracks who hold the trophy — at the case's target,
a week after launch (2026-11-26T00:00:00Z for the final case).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` case
file in the input directory and writes one `<id>.out` per case. Runs in
~0.06 s per case, deterministic byte-for-byte.

## How it works

An Exophase rate is a ratio, earners / players-tracked, and the two sides
move for different reasons, so the model splits them: players-tracked
N(t) grows as new players start the game, earners E_t(t) grow as tracked
players progress. Each is fitted in log time, x = ln(1 + hours since
release), by least squares over the case's *distinct* reads (consecutive
identical reads are one snapshot — Exophase only changes its counts when
it refreshes). The median forecast is 100 * E_t(target) / N(target), with
fitted slopes damped (x0.85) past the observed log-time range.

The flat history is the trap. On launch night the page sat unchanged for
hours (638 players, every rate flat) and then refreshed to 3,868 players
with every rate but the one at 100% lower: a refresh *dilutes* rates,
because the new players it adds have barely played. A history with no
refresh yet is a stale page, and a refresh is overdue — so for a stale
case the median is moved down by a dilution prior (tracked-player
multiplier 5x at the next refresh; newcomers earn a trophy at
sqrt(rate/100) of the old cohort's rate) instead of trusting the flat
rates, and a wide stale spread is added (1.2 logit units at a short
horizon, growing with the horizon).

Spread, in logit units, combines four honest sources of doubt in
quadrature: a random walk (0.5 per sqrt-day), the fit's residual error
extrapolated to the target, the size of the logit jumps seen at past
refreshes, and the stale component above — floored at 0.6, capped at 4.0.
Quantiles come from a normal on the logit scale at the contest's levels.

## Local practice check

Against proxy answers for the practice cases (built from the one verified
fact the rules give — the 04:30 refresh read 3,868 players, every rate but
the 100% one lower — with new joiners earning 0% to 30% of the old
cohort's rate), this program scores a mean pinball loss of about 0.37–0.39
on the logit scale, against the house persistence baseline's 0.78–1.69.
The true practice answers are the refreshed reads themselves, which no
one can fetch after the fact; the proxies are a lower and a middle
estimate of them, not the answers.
