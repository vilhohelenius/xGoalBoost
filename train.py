"""xG-malli: XGBoost vs CatBoost, leave-one-season-out CV + 2026 holdout.

Ajo: .venv/bin/python train.py
"""
import glob
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

TARGET = "goal"
CATEGORICAL = [
    "shotType", "lastEventCategory", "playerPositionThatDidEvent",
    "shooterLeftRight",
]
NUMERIC = [
    # geometria
    "arenaAdjustedShotDistance", "arenaAdjustedXCordABS", "arenaAdjustedYCord",
    "arenaAdjustedYCordAbs", "shotAngleAdjusted", "shotDistance", "offWing",
    # edellinen tapahtuma / rebound / rush
    "distanceFromLastEvent", "lastEventShotAngle", "lastEventShotDistance",
    "lastEventxCord_adjusted", "lastEventyCord_adjusted", "speedFromLastEvent",
    "timeSinceLastEvent", "shotRebound", "shotRush", "shotAnglePlusRebound",
    "shotAnglePlusReboundSpeed", "shotAngleReboundRoyalRoad",
    # tilanne
    "period", "time", "isHomeTeam", "isPlayoffGame", "timeSinceFaceoff",
    "timeDifferenceSinceChange", "homeSkatersOnIce", "awaySkatersOnIce",
    "homeEmptyNet", "awayEmptyNet", "shotOnEmptyNet",
    "averageRestDifference",
    # väsymys / vaihdot
    "shooterTimeOnIce", "shooterTimeOnIceSinceFaceoff",
    "shootingTeamAverageTimeOnIce", "shootingTeamMaxTimeOnIce",
    "shootingTeamMinTimeOnIce", "shootingTeamAverageTimeOnIceSinceFaceoff",
    "defendingTeamAverageTimeOnIce", "defendingTeamMaxTimeOnIce",
    "defendingTeamMinTimeOnIce", "defendingTeamAverageTimeOnIceSinceFaceoff",
    "shootingTeamForwardsOnIce", "shootingTeamDefencemenOnIce",
    "defendingTeamForwardsOnIce", "defendingTeamDefencemenOnIce",
]
# Tarkoituksella pois (vuoto): *Penalty1Length/TimeLeft (kirjattu maalin jälkeen: ylivoimamaali nollaa ajan), xGoal, x*-sarakkeet, shotWasOnGoal, shotPlayStopped,
# shotGoalieFroze, shotGeneratedRebound, shotPlayContinued*, timeUntilNextEvent,
# homeTeamWon, event, sekä joukkue-/pelaaja-id:t ja pistetilanne.


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("shots_*.csv"))], ignore_index=True)
    df["lastEventSameTeam"] = (df["lastEventTeam"] == df["team"]).astype(float)
    # lähtökohta ampujan joukkueen näkökulmaan
    home = df["isHomeTeam"] == 1
    df["shooterSkaters"] = np.where(home, df["homeSkatersOnIce"], df["awaySkatersOnIce"])
    df["defenderSkaters"] = np.where(home, df["awaySkatersOnIce"], df["homeSkatersOnIce"])
    df["emptyNetAgainst"] = np.where(home, df["awayEmptyNet"], df["homeEmptyNet"])
    for c in CATEGORICAL:
        df[c] = df[c].fillna("NA").astype(str)
    return df


EXTRA = ["lastEventSameTeam", "shooterSkaters", "defenderSkaters", "emptyNetAgainst"]


def make_xgb(cat_cols):
    return XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
        tree_method="hist", enable_categorical=True, eval_metric="auc", n_jobs=-1,
    )


def make_cat(cat_cols):
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        eval_metric="AUC", verbose=0, cat_features=cat_cols, thread_count=-1,
    )


def prep(df, feats, kind):
    X = df[feats].copy()
    if kind == "xgb":
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
    return X


def evaluate(df, feats, make, kind, name):
    seasons = sorted(df["season"].unique())
    cv_seasons = [s for s in seasons if (df["season"] == s).sum() > 50_000]
    aucs, lls = [], []
    for s in cv_seasons:
        tr, te = df[df.season != s], df[df.season == s]
        # 2026 on pieni: pidetään se erillisenä holdoutina, ei treenissä CV:n aikana
        tr = tr[tr.season.isin(cv_seasons)]
        m = make(CATEGORICAL)
        Xtr, Xte = prep(tr, feats, kind), prep(te, feats, kind)
        # XGBoost vaatii samat kategoriat treenissä ja testissä
        if kind == "xgb":
            for c in CATEGORICAL:
                cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
                Xtr[c] = Xtr[c].cat.set_categories(cats)
                Xte[c] = Xte[c].cat.set_categories(cats)
        m.fit(Xtr, tr[TARGET])
        p = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(te[TARGET], p))
        lls.append(log_loss(te[TARGET], p))
        print(f"  {name} testikausi {s}: AUC={aucs[-1]:.4f} logloss={lls[-1]:.4f}")
    print(f"{name} CV: AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  logloss={np.mean(lls):.4f}")
    return np.mean(aucs)


def final_holdout(df, feats, make, kind, name):
    tr, te = df[df.season.isin([2023, 2024, 2025])], df[df.season == 2026]
    m = make(CATEGORICAL)
    Xtr, Xte = prep(tr, feats, kind), prep(te, feats, kind)
    if kind == "xgb":
        for c in CATEGORICAL:
            cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
            Xtr[c] = Xtr[c].cat.set_categories(cats)
            Xte[c] = Xte[c].cat.set_categories(cats)
    m.fit(Xtr, tr[TARGET])
    p = m.predict_proba(Xte)[:, 1]
    print(f"{name} holdout 2026 (n={len(te)}): AUC={roc_auc_score(te[TARGET], p):.4f} "
          f"logloss={log_loss(te[TARGET], p):.4f}  keskiarvo p={p.mean():.4f} vs todellinen {te[TARGET].mean():.4f}")
    return m, Xte, p


def main():
    df = load()
    feats = NUMERIC + EXTRA + CATEGORICAL
    print(f"{len(df)} laukausta, {len(feats)} piirrettä, maaliprosentti {df[TARGET].mean():.3%}")
    print(f"Vertailu: MoneyPuck xGoal AUC = {roc_auc_score(df[TARGET], df['xGoal']):.4f}\n")

    results = {}
    for name, make, kind in [("XGBoost", make_xgb, "xgb"), ("CatBoost", make_cat, "cat")]:
        results[name] = evaluate(df, feats, make, kind, name)
    best = max(results, key=results.get)
    print(f"\nParas CV:ssä: {best}\n")

    for name, make, kind in [("XGBoost", make_xgb, "xgb"), ("CatBoost", make_cat, "cat")]:
        m, Xte, p = final_holdout(df, feats, make, kind, name)
        if name == best:
            imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
            print("\nTop 15 piirteen tärkeys:\n", imp.head(15).round(4).to_string())
            (m.save_model if name == "CatBoost" else m.save_model)(f"model_{name.lower()}.json" if name == "XGBoost" else "model_catboost.cbm")
            json.dump(feats, open("features.json", "w"))
            # kalibraatio: 10 desiiliä
            te = df[df.season == 2026].copy()
            te["p"] = p
            te["bin"] = pd.qcut(te["p"], 10, duplicates="drop")
            print("\nKalibraatio (2026):\n", te.groupby("bin", observed=True).agg(
                ennuste=("p", "mean"), toteuma=(TARGET, "mean"), n=(TARGET, "size")).round(4).to_string())


if __name__ == "__main__":
    main()
