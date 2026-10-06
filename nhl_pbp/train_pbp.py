"""Kouluttaa NHL-PBP-pohjaiset xG-mallit ja validoi ne.

  skater  : P(maali | laukausyritys, ei torjuttu)      -> pelaajien ixG
  goalie  : P(maali | maalia kohti tullut laukaus)      -> maalivahdin GSAx

Leave-one-season-out CV, sitten lopulliset mallit kaikella datalla -> nhl_pbp/model_*.json.
Validointi: pelaajakohtaiset kausi-xG:t vs MoneyPuckin xGoal (shots_*.csv).
Ajo: .venv/bin/python nhl_pbp/train_pbp.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, "nhl_pbp")
from features import CATEGORICAL, FEATURES

df = pd.read_parquet("data/nhl_shots.parquet")
cats = {c: sorted(df[c].unique()) for c in CATEGORICAL}
json.dump({"features": FEATURES, "categories": cats}, open("nhl_pbp/model_meta.json", "w"), indent=1)


def X(d):
    x = d[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return x


def make():
    return XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7,
                         min_child_weight=20, reg_lambda=5, tree_method="hist", enable_categorical=True, n_jobs=-1)


def run(name, d):
    seasons = sorted(d.season.unique())
    d = d.reset_index(drop=True)
    oof = np.zeros(len(d))
    for s in seasons:
        tr, te = d.season != s, d.season == s
        m = make().fit(X(d[tr]), d.goal[tr])
        oof[te] = m.predict_proba(X(d[te]))[:, 1]
        print(f"  {name} {s}: AUC={roc_auc_score(d.goal[te], oof[te]):.4f} logloss={log_loss(d.goal[te], oof[te]):.4f}  sum xG={oof[te].sum():.0f} vs goals={d.goal[te].sum()}")
    m = make().fit(X(d), d.goal)
    m.save_model(f"nhl_pbp/model_{name}.json")
    d["xg"] = oof
    return d


sk = run("skater", df)
gl = run("goalie", df[(df.onGoal == 1) & (df.emptyNet == 0) & df.goalieId.notna()])
sk[["gameId", "eventId", "xg"]].to_parquet("data/oof_skater.parquet")
gl[["gameId", "eventId", "xg"]].to_parquet("data/oof_goalie.parquet")

# --- validointi MoneyPuckia vastaan (kausi 2023-25), pelaajakohtaiset kokonaisxG:t
mp = pd.concat([pd.read_csv(f"shots_{y}.csv", usecols=["season", "isPlayoffGame", "shooterPlayerId", "xGoal", "goal", "event"]) for y in (2023, 2024, 2025)])
mp = mp[mp.isPlayoffGame == 0]
mp_tot = mp.groupby(["season", "shooterPlayerId"]).agg(mp_xg=("xGoal", "sum"), mp_n=("goal", "size")).reset_index()
mine = sk.assign(season=sk.season // 10000).groupby(["season", "shooterId"]).agg(my_xg=("xg", "sum"), my_n=("goal", "size"), goals=("goal", "sum")).reset_index()
j = mp_tot.merge(mine, left_on=["season", "shooterPlayerId"], right_on=["season", "shooterId"])
j = j[j.mp_n >= 50]
print(f"\nPelaaja-kaudet (>=50 laukausta): {len(j)}  | r(oma xG, MoneyPuck xG)={np.corrcoef(j.my_xg, j.mp_xg)[0,1]:.3f}"
      f"  | r(oma xG, oikeat maalit)={np.corrcoef(j.my_xg, j.goals)[0,1]:.3f}  | r(MP xG, oikeat maalit)={np.corrcoef(j.mp_xg, j.goals)[0,1]:.3f}")
