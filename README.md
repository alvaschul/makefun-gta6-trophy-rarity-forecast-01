# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alvaschul/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alvaschul/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## What it does

This program forecasts how common each Grand Theft Auto VI PlayStation 5 trophy will be at a future moment. For every case file it is given — a snapshot of the game's trophy list plus the recorded history of earn rates from Exophase — it writes a five-point quantile forecast (the 5th, 25th, 50th, 75th and 95th percentiles) of every trophy's earn rate at the case's target time. Forecasts are produced on the logit scale, because a rate moving from 1% to 2% is a very different event from one moving from 50% to 51%, and the scoring rewards getting that geometry right.

## The dilution-aware idea

An earn rate is not a free-standing number: it is a ratio, earners divided by players tracked. Around a big launch both sides of that ratio move fast and for different reasons. The player base can multiply several-fold in a single data refresh, which dilutes every rate even while thousands of people are earning trophies, while the earner count for a story trophy climbs steadily as players reach that point in the game and a grind trophy's count barely moves. A forecaster that only extrapolates the ratio itself will systematically overstate rates right after the tracked population jumps, and understate how fast early story trophies recover. This program therefore models the numerator and the denominator separately as well as the ratio, so a surge in tracked players is treated as dilution rather than as players un-earning trophies.

## How the forecast is built

The median blends two components equally. The first is a trend component: for each trophy, a weighted least-squares line of logit(rate) against log-time since release, with the slope shrunk toward zero for short histories, blended toward the mean slope of the trophy's tier group, and re-anchored so the line passes through the recent level before being read at the target time and pulled part-way back toward the last observed rate. The second is the counts component: players tracked are projected forward with their own shrunken log-growth trend, each trophy's earner count is projected from its recent growth (with relative growth shrunk toward its tier group's median), and the projected rate is the projected earners divided by the projected players, capped so earners can never exceed the player base.

The spread around that median is deliberately honest rather than tuned to look confident. Six sources of uncertainty are combined in quadrature on the logit scale: a base term that grows with the square root of the forecast horizon, the trend fit's residual scatter, a term proportional to how much slope uncertainty amplifies over the extrapolation distance, the observed step-to-step jitter in the trophy's own history, a regime term that is large when the history is short or contains few distinct states (exactly when a data-source refresh can rewrite the picture overnight), and a term for uncertainty in the player-count projection. Identical consecutive reads are deduplicated before fitting, flat histories fall back to the last observed rate, saturated trophies near 100% get a pulled-in median and a halved spread, and any unexpected failure in a case falls back to a simple persistence forecast so a valid output file is always written.

The program is fully offline and self-contained: a single Python file using only the standard library, with no network access and no randomness, so two runs on the same case produce byte-identical output.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

<sub>Entered on [make.fun](https://make.fun/?ref=1r237bkm&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
