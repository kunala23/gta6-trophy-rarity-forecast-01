"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, five quantiles of
its Exophase earn rate (percent of tracked players holding it) at the case's
target. Python 3.13, standard library only, offline, deterministic.

How it works
------------
An Exophase rate is a ratio: earners / players-tracked. Both move, and they
move for different reasons, so the model splits them:

* players-tracked N(t) grows as new players start the game (a launch surge
  that slows), and
* earners E_t(t) for trophy t grow as the tracked players progress.

Each is fitted in log time, x = ln(1 + hours since release), by least
squares over the case's *distinct* reads (consecutive identical reads are
one snapshot: Exophase only changes its counts when it refreshes, so a flat
stretch is one observation, not many). The median forecast is
100 * E_t(target) / N(target), with the fitted slopes damped (x0.85) because
the target sits past the end of the observed log-time range.

The flat history is the trap. On launch night the page sat unchanged for
hours (638 players, every rate flat) and then refreshed to 3,868 players
with every rate but the one at 100% lower: a refresh dilutes rates, because
the new players it adds have barely played. A history with no refresh yet
is a *stale* page, and a refresh is overdue. For a stale case the median is
moved down by a dilution prior instead of trusting the flat rates:

    r' = r * (1 + a*(m-1)) / m,  m = 5 (tracked-player multiplier at the
    next refresh), a = sqrt(r/100) (share of the newcomers who have already
    earned the trophy: ~0 for rare trophies, ~1 for the one everyone earns
    at the start).

and a wide stale spread (1.2 logit units at a short horizon, growing with
the horizon) is added. The same dilution
logic is blended in, with a weight that grows with the hours since the
last observed change, whenever the most recent snapshot is old.

Spread, in logit units, combines four honest sources of doubt, added in
quadrature: a random walk (0.5 per sqrt-day, the house baseline's scale),
the fit's residual error extrapolated to the target, the size of the
logit jumps seen at past refreshes, and the stale component above. The
total is floored at 0.6 and capped at 4.0 logit units, so a long horizon
with a flat history never gets a narrow band.

Quantiles come from a normal on the logit scale at the contest's levels
(0.05, 0.25, 0.5, 0.75, 0.95), z = -1.645, -0.674, 0, 0.674, 1.645,
rounded to 4 decimals half-up, clamped to [0.01, 99.99] before the logit
and to [0, 100] after. Output is one row per trophy, deterministic
byte-for-byte: pure arithmetic, no randomness, no network, no clock reads
other than the case's own timestamps.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread, logit units per sqrt-day (house baseline's scale).
SIGMA_RW = 0.5
# Damping applied to fitted log-time slopes when extrapolating past the data.
SLOPE_DAMP = 0.85
# Stale-page dilution prior: tracked-player multiplier at the next refresh.
STALE_MULTIPLIER = 5.0
# Extra spread (logit units) for a fully stale history at a short horizon;
# it grows with the horizon, since a stale page 4.5 days out can refresh
# much further than one 90 minutes out. Calibrated on the practice
# refresh (tracked 638 -> 3,868, every rate but the 100% one lower):
# 1.2 beats 1.6 and 2.0 on every dilution proxy of that refresh, including
# pure dilution, so the band is wide enough to cover the jump without
# paying for width it does not need.
SIGMA_STALE = 1.2
# Hours with no observed change before a refresh starts to be treated as
# overdue (blended in linearly up to STALE_FULL_HOURS).
STALE_START_HOURS = 6.0
STALE_FULL_HOURS = 18.0
# Spread floor / cap, logit units.
SIGMA_FLOOR = 0.6
SIGMA_CAP = 4.0
# Cap on how far the median may move from the last read, logit units.
MEDIAN_MOVE_CAP = 4.5


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least-squares line through (xs, ys); returns (intercept, slope, resid_sd).

    With fewer than 3 points the slope is 0 and the residual scale is a
    default that says "no trend information": 0.35 log units.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.35
    if n < 3:
        mean_y = sum(ys) / n
        return mean_y, 0.0, 0.35
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.35
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.35
    return inter, slope, resid


def distinct_snapshots(case):
    """Collapse history to one snapshot per observed change, oldest first.

    A snapshot is (time, players_tracked, rates tuple). Consecutive reads
    with identical players and rates carry no new information: Exophase
    had not refreshed between them.
    """
    snaps = []
    for read in case["history"]:
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        if snaps and snaps[-1][2] == key:
            continue
        snaps.append((parse_time(read["at"]), read["players_tracked"], key))
    return snaps


def refresh_jump_scale(snaps, n_trophies):
    """Pooled spread of per-trophy logit jumps at observed refreshes.

    At each snapshot change, every trophy's logit rate moves; the pooled
    standard deviation of those moves (over trophies and refreshes) is
    how big a refresh can swing a rate. No refresh seen -> 0 (the stale
    component below covers that case instead).
    """
    jumps = []
    for prev, cur in zip(snaps, snaps[1:]):
        for i in range(n_trophies):
            r0, r1 = prev[2][1][i], cur[2][1][i]
            if r0 is None or r1 is None:
                continue
            jumps.append(logit(r1) - logit(r0))
    if len(jumps) < 5:
        return 0.0
    mean = sum(jumps) / len(jumps)
    var = sum((j - mean) ** 2 for j in jumps) / (len(jumps) - 1)
    return min(1.0, max(0.0, math.sqrt(var)))


def dilution_rate(rate, multiplier):
    """Rate after a refresh multiplies the tracked base by `multiplier`.

    Earners stay put; the newcomers have earned the trophy at rate
    a = sqrt(rate/100) of the old cohort's rate (rare trophies: almost
    none of them; the trophy everyone earns at the start: all of them).
    """
    a = math.sqrt(clamp_pct(rate) / 100.0)
    return clamp_pct(rate) * (1.0 + a * (multiplier - 1.0)) / multiplier


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])

    # Hours since release; the release date is the case's own field, and
    # the first read is never before launch, so the floor at 0 is safe.
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    def hours_since_release(t):
        return max(0.0, (t - release).total_seconds() / 3600.0)
    def x_of(t):
        return math.log1p(hours_since_release(t))

    x_target = x_of(target)
    snaps = distinct_snapshots(case)
    xs = [x_of(s[0]) for s in snaps]
    x_last = xs[-1] if xs else x_of(at)

    # --- tracked players: log-time fit over distinct snapshots ----------
    n_tracked = [max(1, s[1]) for s in snaps]
    inter_n, slope_n, resid_n = ols(xs, [math.log(v) for v in n_tracked])
    log_n_target = inter_n + SLOPE_DAMP * slope_n * x_target

    # --- staleness: how long since the page last changed ---------------
    last_change = snaps[-1][0] if snaps else at
    stale_hours = max(0.0, (at - last_change).total_seconds() / 3600.0)
    if len(snaps) <= 1:
        stale_w = 1.0
    elif stale_hours <= STALE_START_HOURS:
        stale_w = 0.0
    else:
        stale_w = min(1.0, (stale_hours - STALE_START_HOURS)
                      / (STALE_FULL_HOURS - STALE_START_HOURS))

    sigma_jump = refresh_jump_scale(snaps, n)
    sigma_rw = SIGMA_RW * math.sqrt(horizon_days)
    extrap = max(0.0, x_target - x_last)

    rows = []
    for i, trophy in enumerate(trophies):
        # Last non-null rate for this trophy (the case's last read first).
        last_rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            last_rate = 0.01

        # Earners per snapshot: rate * tracked / 100, +0.5 so a 0.00%
        # read still has a finite log (half an earner).
        e_logs = []
        e_xs = []
        for t, tracked, (_n, rates) in snaps:
            r = rates[i]
            if r is None:
                continue
            e_logs.append(math.log(max(0.5, r / 100.0 * tracked + 0.5)))
            e_xs.append(x_of(t))
        inter_e, slope_e, resid_e = ols(e_xs, e_logs)

        # Trend weight: no distinct snapshots -> no trend at all; a
        # handful -> partial trust; many -> full trust.
        trend_w = len(snaps) / (len(snaps) + 3.0) if len(snaps) > 1 else 0.0
        slope_e_d = SLOPE_DAMP * slope_e * trend_w
        slope_n_d = SLOPE_DAMP * slope_n * trend_w

        # Median from the earners/tracked split, blended toward
        # persistence when the trend is untrusted.
        log_e_target = inter_e + slope_e_d * x_target
        log_e_last = inter_e + slope_e_d * x_last
        log_n_last = inter_n + slope_n_d * x_last
        logit_trend = (log_e_target - log_n_target) - (log_e_last - log_n_last)
        logit_trend = logit(last_rate) + logit_trend
        # Also blend the level itself toward the last rate when untrusted.
        logit_med = trend_w * logit_trend + (1.0 - trend_w) * logit(last_rate)

        # Stale-refresh dilution: shift the median down toward the rate
        # the next refresh would print, in proportion to staleness.
        if stale_w > 0.0:
            logit_dil = logit(dilution_rate(last_rate, STALE_MULTIPLIER))
            logit_med = (1.0 - stale_w) * logit_med + stale_w * logit_dil

        # Keep the median within a sane move of the last read.
        logit_med = min(logit(last_rate) + MEDIAN_MOVE_CAP,
                        max(logit(last_rate) - MEDIAN_MOVE_CAP, logit_med))

        # Spread: random walk + fit residuals extrapolated + refresh
        # jumps + staleness, in quadrature. The stale part grows with the
        # horizon: a page that has not refreshed in days can jump much
        # further over 4.5 days than over 90 minutes.
        sigma_trend = math.sqrt(resid_e ** 2 + resid_n ** 2) * (1.0 + 0.3 * extrap)
        sigma_stale = SIGMA_STALE * math.sqrt(1.0 + horizon_days / 2.0) * stale_w
        sigma = math.sqrt(sigma_rw ** 2 + sigma_trend ** 2
                          + sigma_jump ** 2 + sigma_stale ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

        qs = []
        prev = -1.0
        for z in Z:
            q = inv_logit(logit_med + z * sigma)
            q = min(100.0, max(0.0, q))
            q = max(prev, q)  # non-decreasing, as the format requires
            qs.append(round4(q))
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
