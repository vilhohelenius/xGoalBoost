# Handoff: player xG and goalie GSAx for morning-hockey

Audience: an agent implementing this feature in `vilhohelenius/morning-hockey`. The model work is finished and lives in a separate (private) repo, `vilhohelenius/xGoalBoost`, folder `nhl_pbp/`. Your job is the integration: sync job, D1 tables, pages. Ask the user the questions in section 9 before you build.

Caveat on what the author of this report knows: they read morning-hockey's README, directory layout and D1 table list, but NOT its source code (`nhl_api.py`, `d1_sync.py`, `web/functions/*`). Read those first and follow their conventions over anything suggested here.

## 1. What the feature is

- **Skater xG**: for each unblocked shot attempt (shot on goal, missed shot, goal), the probability it is a goal. Summed per player per game and per season ("ixG"), shown next to actual goals.
- **Goalie GSAx** (goals saved above expected): for each shot on goal at a non-empty net, the probability it is a goal. Per goalie, xGA = sum of those, GSAx = xGA - goals against. Positive means better than expected. Also GSAx per 100 shots.
- Per game and per season. Regular season only.

EDGE (`nhl-api-py`'s `client.edge.*`) has no xG, only save percentages and shot-location summaries. It is not needed for this feature, but could be used later for extra context on goalie pages.

## 2. What exists (xGoalBoost repo, `nhl_pbp/`)

| File | Purpose |
|---|---|
| `features.py` | `game_shots(play_by_play_json) -> list[dict]`: the exact feature extraction. `FEATURES`, `CATEGORICAL`, `NUMERIC` lists. **Copy this file verbatim into morning-hockey** (training and inference must use identical code). |
| `model_skater.json`, `model_goalie.json` | XGBoost models (about 1.5 MB each), trained on all three seasons 2023-24 to 2025-26. |
| `model_meta.json` | `{"features": [...], "categories": {"shotType": [...], "lastEvent": [...]}}`. Needed to build categorical columns. |
| `golden_2024020001.csv` | Per-shot predictions of the shipped models for game 2024020001; use it as a regression test (tolerance 1e-4). Expected sums: skater xG 5.715 over 90 shots; goalie xGA 3.786 over on-goal non-empty-net shots. |
| `download.py`, `train_pbp.py`, `aggregate.py` | Only needed to retrain or reproduce. Not needed in morning-hockey. |
| `data/*_game_xg.csv`, `data/*_season_xg.csv` | Backfill for seasons 2023-24, 2024-25, 2025-26 (out-of-fold predictions, so slightly different from what the shipped models give). Columns below. |

Versions used: Python 3.13, xgboost 3.4.1, pandas 2.3.3, numpy 2.5.3. Pin `xgboost` in morning-hockey's `pyproject.toml`; model JSON loading across major versions is not guaranteed.

Loading and predicting (this is verified to work):

```python
import json, pandas as pd
from xgboost import XGBClassifier
from features import game_shots, FEATURES, CATEGORICAL

meta = json.load(open("model_meta.json"))
rows = pd.DataFrame(game_shots(play_by_play_json))      # one row per shot attempt
X = rows[FEATURES].copy()
for c in CATEGORICAL:                                    # unseen category -> NaN, XGBoost handles it
    X[c] = pd.Categorical(X[c], categories=meta["categories"][c])
skater = XGBClassifier(); skater.load_model("model_skater.json")
goalie = XGBClassifier(); goalie.load_model("model_goalie.json")
rows["xg_skater"] = skater.predict_proba(X)[:, 1]
rows["xg_goalie"] = goalie.predict_proba(X)[:, 1]
# goalie model is only valid for: onGoal == 1 and emptyNet == 0 and goalieId notna
```

`game_shots` output columns: `gameId, season, date, eventId, teamId, oppId, isHome, shooterId, goalieId, shotType, x, absY, dist, angle, period, secInPeriod, gameSec, shooterSkaters, defenderSkaters, shooterGoalie, emptyNet, scoreDiff, sinceLast, distLast, speedLast, lastSame, lastX, lastAbsY, rebound, rush, sinceLastShotSame, sinceLastShotOpp, lastEvent, onGoal, goal`.

## 3. Data source

`GET https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play` (in `nhl-api-py`: `client.game_center.play_by_play(game_id)`). morning-hockey already calls the NHL API through its own `nhl_api.py`; reuse that client and its per-run cache instead of adding `nhl-api-py` as a dependency, unless the user prefers otherwise.

- Only process games with `gameState` in `OFF`/`FINAL`. Regular season only (`gameId` contains `02` at positions 5-6, e.g. `2025020123`).
- Player names are in `rosterSpots[]` (`playerId`, `firstName.default`, `lastName.default`, `teamId`); the shot rows only carry ids.
- About 280 events per game, 0.2 s per request in testing; the whole 3936-game backfill took about 3 minutes with 6 threads. No auth needed. Be polite (few threads, retries with backoff).

## 4. Semantics you must preserve (all encoded in `features.py`)

- Shots included: `shot-on-goal`, `missed-shot`, `goal`. Blocked shots are excluded (NHL coordinates for blocks are the blocker's position). Shootout (`periodType == "SO"`) and periods above 4 are skipped; playoff overtime is therefore not supported; keep it regular season only, or explicitly handle playoffs later.
- Coordinates are flipped so the attacked goal is always at x=+89, using `homeTeamDefendingSide` per play. Events without coordinates or `homeTeamDefendingSide` are skipped as context and, for shots, dropped.
- Strength comes from `situationCode` (digits: awayGoalie, awaySkaters, homeSkaters, homeGoalie). `emptyNet` means the defending goalie flag is 0. Goalie-model rows additionally require `goalieInNetId` to exist.
- Score difference is a running tally from goal events, from the shooter's perspective, before the shot.
- Shot attribution: shooter = `shootingPlayerId` or (for goals) `scoringPlayerId`.

## 5. Quality, so you can describe it honestly in the UI

Leave-one-season-out on 2023-24 to 2025-26:

| | AUC | Notes |
|---|---|---|
| Skater model | 0.789-0.799 | MoneyPuck-feature version scores 0.806 |
| Goalie model | 0.774-0.787 | |

- Season xG per player correlates 0.982 with MoneyPuck's xGoal (1,903 player-seasons, at least 50 shots).
- Sanity check, 2024-25 top GSAx (at least 1,000 shots): Vasilevskiy +22.6, Hellebuyck +21.9, Kuemper +20.3, Montembeault +19.0, Thompson +16.4.
- Calibration caveat: for a held-out season, total predicted goals can be off by up to about 5% (league scoring differs by season). Within a season, comparisons between players are fine. For the current season, consider recalibrating by scaling xG so the league total matches league goals; ask the user whether to do this (see section 9).
- Single-game xG is noisy. Prefer season totals and rolling windows in the UI.
- Backfilled CSV values come from out-of-fold predictions; values computed live with the shipped models for those same seasons are slightly in-sample (the models saw those seasons). The differences are small, but do not mix the two sources in one season if you can avoid it: either use the shipped models for everything (recompute the backfill by running the sync over all games, recommended) or the CSVs for past seasons only.

## 6. Suggested D1 schema (adapt to existing conventions, apply statements one at a time in the D1 Console)

```sql
CREATE TABLE IF NOT EXISTS skater_game_xg (
  game_id INTEGER NOT NULL, player_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
  season INTEGER NOT NULL, game_date TEXT NOT NULL,
  shots INTEGER NOT NULL, on_goal INTEGER NOT NULL, goals INTEGER NOT NULL, xg REAL NOT NULL,
  PRIMARY KEY (game_id, player_id));
CREATE INDEX IF NOT EXISTS idx_skater_game_xg_player ON skater_game_xg (player_id, season);
CREATE TABLE IF NOT EXISTS goalie_game_xg (
  game_id INTEGER NOT NULL, player_id INTEGER NOT NULL, opp_team_id INTEGER NOT NULL,
  season INTEGER NOT NULL, game_date TEXT NOT NULL,
  shots_against INTEGER NOT NULL, goals_against INTEGER NOT NULL, xga REAL NOT NULL,
  PRIMARY KEY (game_id, player_id));
CREATE INDEX IF NOT EXISTS idx_goalie_game_xg_player ON goalie_game_xg (player_id, season);
```

Season tables can be computed in queries (`SUM ... GROUP BY player_id, season`) so there is one source of truth. GSAx = `xga - goals_against`, GSAx/100 = that divided by shots_against times 100. Optionally keep a `xg_processed_games(game_id)` table to make the sync incremental, or derive "unprocessed" from `games LEFT JOIN skater_game_xg`.

Notes: `goalie_game_xg` excludes empty-net shots by construction. Goalie shots are attributed to the goalie in net (`goalieInNetId`) at the time of the shot, so a goalie change mid-game splits correctly. Player ids are NHL player ids, the same as `/pelaajat/<id>`.

## 7. Suggested implementation plan

1. Read morning-hockey's `nhl_api.py`, `d1_sync.py`, one `sync_*.py`, the `games` table usage, `web/functions/pelaajat/`, `maalivahtiporssi`, and the ottelu report code, plus the workflows and the pytest layout.
2. Add `src/morning_hockey/xg/` (or `xg.py`): `features.py` copied from xGoalBoost, `models/` with the three model files, `xg.py` with `compute_game(play_by_play_json) -> (skater_rows, goalie_rows)` as in the snippet above, names from `rosterSpots`.
3. Add `sync_xg.py` CLI: selects finished regular-season games without xG rows (and a `--backfill SEASON` option), fetches play-by-play, writes rows via `d1_sync.py`. Run it in the existing slow-tier or digest workflow, or as its own workflow triggered by the existing Cloudflare cron pattern (they use `workflow_dispatch` plus Worker cron because GitHub `schedule` is unreliable).
4. Tests: pytest for `compute_game` against `golden_2024020001.csv` (store a copy of the play-by-play JSON for that game as a fixture, about 250 kB, or trim it), plus a test that a shot at the slot scores far higher than one from the blue line, and that empty-net shots are excluded from goalie rows.
5. Backfill: run the sync for 2025-26 (and optionally earlier seasons) once.
6. Web (TypeScript Pages Functions): see section 8. Match existing layout helpers and the Finnish UI language (page names and labels in Finnish: "odotetut maalit (xG)", "torjutut maalit odottamaa vastaan (GSAx)").
7. Update README (sivut, D1-taulut, projektin rakenne) and `d1/schema.sql`.

## 8. Possible UI surfaces (user decides which first, see questions)

- Pelaajakortti `/pelaajat/<id>`: season row with goals, xG, goals minus xG; per-game log gets an xG column. Goalies: GSAx, xGA, GSAx/100 and per-game GSAx.
- Maalivahtipörssi `/maalivahtiporssi`: GSAx and GSAx/100 columns, sortable, minimum shots threshold (suggest 500 shots).
- Ottelun raportti `/ottelut/<id>`: team xG for the game (sum of skater xG per team) and the xG per player in the player tables; xGA/GSAx per goalie. Note that data exists only after the sync processed that game, so show it only when present.
- Suomipörssi `/suomiporssi`: add xG for Finnish skaters and GSAx for Finnish goalies.
- Analytiikka `/analytiikka`: a cumulative goals-vs-xG D3 line chart for the top scorers or any player.
- Optional: a shot map built from per-shot rows would need per-shot storage (about 120k rows per season); not part of the minimum scope.

## 9. Questions to ask the user before building

1. Which surfaces first (pelaajakortti, maalivahtipörssi, ottelun raportti, suomipörssi, analytiikka)?
2. Should past seasons be backfilled, and how many (2023-24 and 2024-25 are already modelled)? Backfilling means about 1,300 play-by-play requests per season.
3. Where should the sync run: inside the existing slow-tier (2 h), the digest (6 h), or a new workflow? Is a new Cloudflare Cron entry acceptable?
4. Should xG be recalibrated per season so the league total matches the league's actual goals (simple scaling factor, recomputed during the sync)? Default suggestion: no, show raw model output, and say so in a small info text.
5. Minimum shots for the goalie ranking, and whether playoffs are in scope (not supported by the current model; default: regular season only).
6. Naming and wording in Finnish (xG = "odotetut maalit", GSAx = "torjutut maalit odottamaa vastaan"; is "GSAx" fine as the column header?).
7. Is it acceptable to add `xgboost` (and numpy/pandas if not already present) to the Python dependencies of the sync workflows? Install time is roughly 30 seconds.
8. Where should the model files live: committed in morning-hockey (3 MB) or fetched from a release/URL? Default suggestion: committed.
9. Does the user want a short "Mitä xG tarkoittaa" info text and a note about model accuracy (AUC about 0.79) on the pages?

## 10. Retraining (later)

Run in xGoalBoost: `python nhl_pbp/download.py <seasons>`, `python nhl_pbp/features.py`, `python nhl_pbp/train_pbp.py`, `python nhl_pbp/aggregate.py`. Retrain after each season so the shipped models are not extrapolating; replace the three model files and regenerate the golden file, and bump a model version string stored with the rows if you add one.
