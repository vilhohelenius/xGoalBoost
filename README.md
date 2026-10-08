# xGoalBoost

Expected-goals (xG) model for NHL shots: gradient boosting (XGBoost, CatBoost) and TabPFN-3.5, trained on
[MoneyPuck](https://moneypuck.com/data.htm) shot data for seasons 2023–2025 (+ the start of 2026).

This is the main repo: all modelling work lives here (MoneyPuck xG model, NHL play-by-play models, win probability). Its results are used in my NHL scores & stats app [Morning Hockey](https://github.com/vilhohelenius/morning-hockey/).

## Related repositories

| Repo | What it is | Relation to this repo |
|---|---|---|
| **xGoalBoost** (this repo) | Full working repo: models, experiments, analysis, `nhl_pbp/` and `winprob/` | Source of truth for all model code and results |
| [xGoalBoost-tabpfn](https://github.com/vilhohelenius/xGoalBoost-tabpfn) | Trimmed, runnable submission for the TabPFN-3.5 hackathon, plus the [live xG shot map demo](https://vilhohelenius.github.io/xGoalBoost-tabpfn/demo/) | Derived from this repo; separate so it stays small and runs standalone |
| [morning-hockey](https://github.com/vilhohelenius/morning-hockey) | NHL results, Finnish players' stats and xG analytics site (Cloudflare Pages Functions + D1) | Consumer: uses the `nhl_pbp/` models (`features.py` is copied verbatim, plus model JSONs) for player xG and goalie GSAx |

Changes to the models or feature code are made here first and then copied to the other repos.

## Results

Target: `goal` for every shot attempt (SHOT, MISS, GOAL; ~365k rows, 7.1 % goals). Blocked shots are not in the data.
No MoneyPuck model outputs, player IDs or team IDs are used as features.

| Model | Validation | AUC |
|---|---|---|
| MoneyPuck `xGoal` (reference) | all rows | 0.782 |
| XGBoost | leave-one-season-out, 2023–25 | **0.806** ± 0.004 |
| CatBoost | leave-one-season-out, 2023–25 | 0.805 ± 0.003 |
| XGBoost | holdout 2026 (n=3 537) | 0.788 |
| XGBoost, 50k train rows | train 2023–24, test 2025 sample | 0.789 |
| TabPFN-3.5, 50k train rows | same | 0.794 |
| XGBoost, 100k train rows | same | 0.797 |
| TabPFN-3.5, 100k train rows | same | **0.8005** |
| TabPFN-3.5 ensemble, 3 × 100k | same | **0.8021** |

TabPFN beats XGBoost on the same 50k and 100k-row samples and nearly closes the gap to XGBoost on all ~240k rows
(0.802 on the same test sample), but it cannot use the whole training set in one fit. Averaging three 100k-row TabPFN fits reaches 0.8021, on par with
full-data XGBoost (the 0.0004 gap is within test-sample noise). See `learning_curve.png`, `results.json` and
`scaling_experiment.py`.

![Learning curve](learning_curve.png)

## A data leak worth knowing about

`homePenalty1TimeLeft`, `awayPenalty1TimeLeft` and the `*Penalty1Length` columns are recorded **after** the shot:
a power-play goal ends the minor penalty, so the time left is 0 on goal rows. With a 5v4 power play and the defender's
penalty time at 0, the goal rate is **68 %** against ~5 % otherwise (`analysis/penalty_leakage.py`).
Leaving these columns in gives a fake AUC of ~0.84; removing them gives ~0.80. Other columns deliberately excluded:
`xGoal`, `x*` outputs, `shotWasOnGoal`, `shotPlayStopped`, `shotGoalieFroze`, `shotGeneratedRebound`,
`shotPlayContinued*`, `timeUntilNextEvent`, `homeTeamWon`, `event`, score columns.

## Run

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# download shots_2023.csv ... shots_2026.csv from https://moneypuck.com/data.htm into the repo root
.venv/bin/python train.py
TABPFN_TOKEN=... .venv/bin/python tabpfn_test.py 50000 30000   # token from https://ux.priorlabs.ai/account
```

Data files are not committed (third-party data, ~200 MB). XGBoost and `tabpfn-client` can segfault when TabPFN is imported
before XGBoost in the same process, so `tabpfn_test.py` runs XGBoost first.

## Demo: xG shot map

**Live: https://vilhohelenius.github.io/xGoalBoost-tabpfn/demo/** (`demo/index.html` is a self-contained interactive page). Pick a shot type, situation (5v5, power play, short-handed, 3v3, empty net), rebound and rush, then click the rink to see the goal probability of a shot from that spot. The heat map is precomputed from an XGBoost model trained on seasons 2023-25. Rebuild with `python demo/build_demo.py` (needs the CSVs).

## NHL play-by-play model (player xG and goalie GSAx)

`nhl_pbp/` builds a second model that uses only data available from the public NHL API (`api-web.nhle.com` play-by-play), so it can run in production apps. `download.py` caches regular-season games, `features.py` turns events into shot rows (normalised coordinates, strength, score, last event, rebound/rush), `train_pbp.py` trains and validates, `aggregate.py` writes per-game and per-season tables.

- Skater model: P(goal | unblocked attempt), leave-one-season-out AUC 0.789-0.799. Goalie model: P(goal | shot on goal, non-empty net), AUC 0.774-0.787.
- Player-season xG correlates 0.982 with MoneyPuck's xG (1,903 player-seasons, 2023-25).
- Goalie GSAx = expected goals against minus goals against. Held-out seasons are off by up to about 5% in total goals (league scoring varies by season), so compare players within a season.

## Pre-game win probability (`winprob/`)

MoneyPuck-style model: ability to win (points %, OT = tie), scoring chances (goal/xG/5v5 xG/shot differential, DZ giveaways), goaltending (rolling GSAx per 100 shots), home and back-to-back. Features are exponentially weighted (half-lives 10 and 40 games) from prior games only; logistic regression. Data: NHL play-by-play 2017-18 to 2025-26 (11 052 games), xG from `nhl_pbp/`.

Walk-forward backtest (each season predicted by a model trained on earlier seasons only):

| Season | Our log loss | Our favourite win % | MoneyPuck log loss | MoneyPuck fav. % |
|---|---|---|---|---|
| 2020-21 | 0.6561 | 61.2 | 0.6596 | 60.1 |
| 2021-22 | 0.6388 | 63.9 | 0.648 | 64.1 |
| 2022-23 | 0.6536 | 62.4 | 0.656 | 60.6 |
| 2023-24 | 0.6558 | 60.5 | 0.661 | 61.1 |
| 2024-25 | 0.6602 | 59.2 | 0.658 | 60.4 |
| 2025-26 | 0.6837 | 55.1 | n/a | n/a |

Baselines (mean): constant home rate 0.6906, Elo 0.6684, this model 0.6580 (AUC 0.642). 2025-26 was an unusually even season (points-% persistence r=0.43 vs 0.7 in 2021-25). Component weights in the final model: ability 16 %, chances 54 %, goalie 14 %, context 16 %.

Run: `python nhl_pbp/download.py 2017 ... 2025`, `python winprob/build_games.py`, `python winprob/train_wp.py` (writes `winprob/model_wp.json`).
