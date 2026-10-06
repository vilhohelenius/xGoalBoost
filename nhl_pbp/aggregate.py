"""Ottelu- ja kausikohtaiset pelaaja-xG:t ja maalivahtien GSAx (out-of-fold-ennusteista).

Tuottaa data/skater_game_xg.csv, data/goalie_game_xg.csv, data/skater_season_xg.csv, data/goalie_season_xg.csv
GSAx = odotetut päästetyt maalit (xGA) - päästetyt maalit; positiivinen = parempi kuin odotettu.
Ajo: .venv/bin/python nhl_pbp/aggregate.py
"""
import json
import pathlib

import pandas as pd

shots = pd.read_parquet("data/nhl_shots.parquet")
names = {}
for f in pathlib.Path("data/pbp").glob("*.json"):
    for r in json.loads(f.read_text()).get("rosterSpots", []):
        names[r["playerId"]] = f'{r["firstName"]["default"]} {r["lastName"]["default"]}'

sk = shots.merge(pd.read_parquet("data/oof_skater.parquet"), on=["gameId", "eventId"])
sg = sk[sk.shooterId.notna()].groupby(["season", "gameId", "date", "shooterId", "teamId"]).agg(
    shots=("goal", "size"), onGoal=("onGoal", "sum"), goals=("goal", "sum"), xG=("xg", "sum")).reset_index()
sg["name"] = sg.shooterId.map(names)
sg.round(3).to_csv("data/skater_game_xg.csv", index=False)
ss = sg.groupby(["season", "shooterId", "name"]).agg(games=("gameId", "nunique"), shots=("shots", "sum"), goals=("goals", "sum"), xG=("xG", "sum")).reset_index()
ss["goalsMinusXG"] = ss.goals - ss.xG
ss.round(3).to_csv("data/skater_season_xg.csv", index=False)

gl = shots[(shots.onGoal == 1) & (shots.emptyNet == 0) & shots.goalieId.notna()].merge(pd.read_parquet("data/oof_goalie.parquet"), on=["gameId", "eventId"])
gg = gl.groupby(["season", "gameId", "date", "goalieId", "oppId"]).agg(shotsAgainst=("goal", "size"), goalsAgainst=("goal", "sum"), xGA=("xg", "sum")).reset_index()
gg["GSAx"] = gg.xGA - gg.goalsAgainst
gg["name"] = gg.goalieId.map(names)
gg.round(3).to_csv("data/goalie_game_xg.csv", index=False)
gs = gg.groupby(["season", "goalieId", "name"]).agg(games=("gameId", "nunique"), shotsAgainst=("shotsAgainst", "sum"), goalsAgainst=("goalsAgainst", "sum"), xGA=("xGA", "sum"), GSAx=("GSAx", "sum")).reset_index()
gs["GSAx_per_100"] = gs.GSAx / gs.shotsAgainst * 100
gs.round(3).to_csv("data/goalie_season_xg.csv", index=False)

pd.set_option("display.width", 140)
print("Top 8 pelaajaa, kausi 2024-25: maalit vs xG")
print(ss[ss.season == 20242025].nlargest(8, "xG")[["name", "games", "shots", "goals", "xG", "goalsMinusXG"]].round(1).to_string(index=False))
print("\nTop 8 maalivahtia GSAx, kausi 2024-25 (>=1000 laukausta)")
print(gs[(gs.season == 20242025) & (gs.shotsAgainst >= 1000)].nlargest(8, "GSAx")[["name", "games", "shotsAgainst", "goalsAgainst", "xGA", "GSAx", "GSAx_per_100"]].round(1).to_string(index=False))
