"""Rakentaa ottelutason aineiston voittotodennäköisyysmallia varten kaikista ladatuista PBP-peleistä.

Tuottaa:
  data/wp_games.parquet    1 rivi / ottelu: koti/vieras, tulos, jatkoaika, päivä
  data/wp_team_game.parquet 1 rivi / joukkue / ottelu: maalit, xGF/xGA (kaikki + 5v5), laukaukset, DZ-giveawayt
  data/wp_goalie_game.parquet 1 rivi / maalivahti / ottelu: laukaukset, päästetyt, xGA, GSAx
xG: out-of-fold-ennuste kausille 2023-25 (data/oof_*.parquet), muille kausille valmiit mallit (ei nähty koulutuksessa).
Ajo: .venv/bin/python winprob/build_games.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "nhl_pbp")
from features import CATEGORICAL, FEATURES, game_shots

games, extra, shots = [], [], []
files = sorted(pathlib.Path("data/pbp").glob("*.json"))
for i, f in enumerate(files):
    d = json.loads(f.read_text())
    h, a = d["homeTeam"], d["awayTeam"]
    if h.get("score") is None or d.get("gameState") not in ("OFF", "FINAL"):
        continue
    games.append({"gameId": d["id"], "season": d["season"], "date": d["gameDate"], "start": d["startTimeUTC"],
                  "homeId": h["id"], "awayId": a["id"], "homeAbbrev": h["abbrev"], "awayAbbrev": a["abbrev"],
                  "homeGoals": h["score"], "awayGoals": a["score"],
                  "lastPeriod": d.get("gameOutcome", {}).get("lastPeriodType", "REG")})
    for p in d["plays"]:
        if p["typeDescKey"] == "giveaway" and p.get("details", {}).get("zoneCode") == "D":
            extra.append({"gameId": d["id"], "teamId": p["details"]["eventOwnerTeamId"], "dzGiveaway": 1})
    shots += game_shots(d)
    if i % 1000 == 0:
        print(i, len(files), flush=True)

games = pd.DataFrame(games).drop_duplicates("gameId")
sh = pd.DataFrame(shots)
sh["shooterId"] = sh.shooterId.astype("Int64"); sh["goalieId"] = sh.goalieId.astype("Int64")

meta = json.load(open("nhl_pbp/model_meta.json"))
X = sh[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = pd.Categorical(X[c], categories=meta["categories"][c])
sk, gl = XGBClassifier(), XGBClassifier()
sk.load_model("nhl_pbp/model_skater.json"); gl.load_model("nhl_pbp/model_goalie.json")
sh["xg"] = sk.predict_proba(X)[:, 1]
sh["xgg"] = gl.predict_proba(X)[:, 1]
for fn, col in (("oof_skater", "xg"), ("oof_goalie", "xgg")):     # ärsyttävä mutta rehellinen: OOF kausille 2023-25
    o = pd.read_parquet(f"data/{fn}.parquet").rename(columns={"xg": "_o"})
    sh = sh.merge(o, on=["gameId", "eventId"], how="left")
    sh[col] = sh["_o"].fillna(sh[col]); sh = sh.drop(columns="_o")

sh["is5v5"] = ((sh.shooterSkaters == 5) & (sh.defenderSkaters == 5) & (sh.emptyNet == 0)).astype(int)
f = sh.groupby(["gameId", "teamId"]).agg(xgf=("xg", "sum"), sog_f=("onGoal", "sum"), att_f=("goal", "size")).reset_index()
f5 = sh[sh.is5v5 == 1].groupby(["gameId", "teamId"]).xg.sum().rename("xgf5").reset_index()
tg = f.merge(f5, how="left", on=["gameId", "teamId"]).fillna({"xgf5": 0})
opp = tg.rename(columns={"teamId": "oppId", "xgf": "xga", "xgf5": "xga5", "sog_f": "sog_a", "att_f": "att_a"})
rows = []
for _, g in games.iterrows():
    for tid, oid, home, gf, ga in ((g.homeId, g.awayId, 1, g.homeGoals, g.awayGoals), (g.awayId, g.homeId, 0, g.awayGoals, g.homeGoals)):
        rows.append({"gameId": g.gameId, "season": g.season, "date": g.date, "teamId": tid, "oppId": oid, "isHome": home, "gf": gf, "ga": ga})
tgame = pd.DataFrame(rows).merge(tg, on=["gameId", "teamId"], how="left").merge(opp, on=["gameId", "oppId"], how="left", suffixes=("", "_o"))
tgame = tgame.drop(columns=[c for c in tgame.columns if c.endswith("_o")])
dz = pd.DataFrame(extra).groupby(["gameId", "teamId"]).dzGiveaway.sum().reset_index() if extra else pd.DataFrame(columns=["gameId", "teamId", "dzGiveaway"])
tgame = tgame.merge(dz, on=["gameId", "teamId"], how="left").fillna({"dzGiveaway": 0})

gs = sh[(sh.onGoal == 1) & (sh.emptyNet == 0) & sh.goalieId.notna()]
gg = gs.groupby(["gameId", "season", "date", "goalieId", "oppId"]).agg(
    shotsAgainst=("goal", "size"), goalsAgainst=("goal", "sum"), xGA=("xgg", "sum")).reset_index()
gg["GSAx"] = gg.xGA - gg.goalsAgainst
gg = gg.rename(columns={"oppId": "teamId"})        # laukauksen vastustaja = maalivahdin joukkue

games.to_parquet("data/wp_games.parquet"); tgame.to_parquet("data/wp_team_game.parquet"); gg.to_parquet("data/wp_goalie_game.parquet")
print(len(games), "ottelua;", len(tgame), "joukkuepeliä;", len(gg), "maalivahtipeliä")
print(games.groupby("season").size().to_string())
print("kotivoitto-%", ((games.homeGoals > games.awayGoals).mean()).round(4), "| jatkoaika/SO-%", (games.lastPeriod != "REG").mean().round(3))
