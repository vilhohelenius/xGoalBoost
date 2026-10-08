"""Ennakkovoittotodennäköisyys (kotijoukkueen voitto) MoneyPuckin mallin hengessä.

Komponentit: kyky voittaa (voitto-%, jatkoaika = tasapeli), maalipaikat (maali-, xG-, 5v5-xG-, laukauseroa, DZ-giveawayt),
maalivahti (GSAx per 100 laukausta, rullaava), sekä koti + lepo. Kaikki piirteet lasketaan vain ennen ottelua pelatuista peleistä
(eksponentiaalinen painotus, uudemmat pelit painavat enemmän).

Takatesti: kukin testikausi ennustetaan vain sitä edeltävillä kausilla koulutetulla mallilla. Lopullinen malli koulutetaan kaikella datalla.
Ajo: .venv/bin/python winprob/train_wp.py
"""
import json
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HALF_LIVES = (10, 40)               # joukkueen rullaavat keskiarvot, pelejä
GOALIE_HALF_LIFE, GOALIE_PRIOR = 25, 800   # maalivahti: puoliintumisaika (esiintymiä), prioriksi 800 laukausta
OFFSEASON_KEEP = 0.6                # kausivaihteessa osuus, joka säilyy edelliseltä kaudelta
METRICS = ["win", "gd", "xgd", "xgd5", "sogd", "dz"]

games = pd.read_parquet("data/wp_games.parquet").sort_values(["start", "gameId"]).reset_index(drop=True)
tg = pd.read_parquet("data/wp_team_game.parquet").set_index(["gameId", "teamId"])
gg = pd.read_parquet("data/wp_goalie_game.parquet")
starter = gg.sort_values("shotsAgainst").groupby(["gameId", "teamId"]).tail(1).set_index(["gameId", "teamId"]).goalieId
gby = {k: v for k, v in gg.groupby("gameId")}

INIT = {m: 0.0 for m in METRICS}; INIT["win"] = 0.5; INIT["dz"] = float(tg.dzGiveaway.mean())


def metrics(r, home_goals, away_goals, is_home, last_period):
    gf, ga = r.gf, r.ga
    ot = last_period != "REG"
    win = 0.5 if ot else float(gf > ga)
    return {"win": win, "gd": gf - ga, "xgd": r.xgf - r.xga, "xgd5": r.xgf5 - r.xga5, "sogd": r.sog_f - r.sog_a, "dz": r.dzGiveaway}


team_state = defaultdict(lambda: {h: dict(INIT) for h in HALF_LIVES})
last_date, last_season = {}, {}
goalie_state = defaultdict(lambda: [0.0, 0.0])     # [decayed GSAx, decayed shots]
recent_starters = defaultdict(lambda: deque(maxlen=20))
lam = 0.5 ** (1 / GOALIE_HALF_LIFE)
elo, elo_season = defaultdict(lambda: 1500.0), {}
rows = []


def grating(gid):
    g, s = goalie_state[gid]
    return 100 * g / (s + GOALIE_PRIOR)


for g in games.itertuples():
    home, away = g.homeId, g.awayId
    # kausivaihde: pehmennetään tilat kohti lähtöarvoja ja elo kohti 1500
    for t in (home, away):
        if last_season.get(t) is not None and last_season[t] != g.season:
            for h in HALF_LIVES:
                for m in METRICS:
                    team_state[t][h][m] = OFFSEASON_KEEP * team_state[t][h][m] + (1 - OFFSEASON_KEEP) * INIT[m]
            elo[t] = 1500 + (elo[t] - 1500) * 2 / 3
    day = pd.Timestamp(g.date)
    row = {"gameId": g.gameId, "season": g.season, "date": g.date, "home": int(g.homeGoals > g.awayGoals)}
    for h in HALF_LIVES:
        for m in METRICS:
            row[f"{m}_{h}"] = team_state[home][h][m] - team_state[away][h][m]
    for side, t in (("h", home), ("a", away)):
        rest = (day - last_date[t]).days if t in last_date else 7
        row[f"b2b_{side}"] = int(rest <= 1)
        rs = list(recent_starters[t])
        row[f"gT_{side}"] = float(np.mean([grating(x) for x in rs])) if rs else 0.0
    # tämän ottelun todellinen aloittaja (tuotannossa vahvistettu aloittaja, muuten varianttiT)
    for side, t in (("h", home), ("a", away)):
        sid = starter.get((g.gameId, t))
        row[f"gS_{side}"] = grating(sid) if sid is not None and not pd.isna(sid) else row[f"gT_{side}"]
    row["gT"] = row["gT_h"] - row["gT_a"]; row["gS"] = row["gS_h"] - row["gS_a"]
    row["b2b_diff"] = row["b2b_h"] - row["b2b_a"]
    row["elo_p"] = 1 / (1 + 10 ** (-(elo[home] - elo[away] + 35) / 400))
    rows.append(row)
    # --- päivitykset ottelun jälkeen
    for t, is_home in ((home, 1), (away, 0)):
        m = metrics(tg.loc[(g.gameId, t)], g.homeGoals, g.awayGoals, is_home, g.lastPeriod)
        for h in HALF_LIVES:
            a = 1 - 0.5 ** (1 / h)
            for k in METRICS:
                team_state[t][h][k] += a * (m[k] - team_state[t][h][k])
        last_date[t], last_season[t] = day, g.season
        sid = starter.get((g.gameId, t))
        if sid is not None and not pd.isna(sid):
            recent_starters[t].append(sid)
    for r in gby.get(g.gameId, pd.DataFrame()).itertuples():
        st = goalie_state[r.goalieId]
        st[0] = st[0] * lam + r.GSAx; st[1] = st[1] * lam + r.shotsAgainst
    res = 0.5 if g.lastPeriod != "REG" else float(g.homeGoals > g.awayGoals)
    exp = row["elo_p"]
    d = 6 * (res - exp)
    elo[home] += d; elo[away] -= d

df = pd.DataFrame(rows)
df.to_parquet("data/wp_features.parquet")

GROUPS = {
    "ability": [f"win_{h}" for h in HALF_LIVES],
    "chances": [f"{m}_{h}" for m in ("gd", "xgd", "xgd5", "sogd", "dz") for h in HALF_LIVES],
    "goalie": None,   # gT eller gS
    "context": ["b2b_diff"],
}


def feats(goalie):
    return GROUPS["ability"] + GROUPS["chances"] + [goalie] + GROUPS["context"]


def fit(train, cols, C=0.05):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return m.fit(train[cols], train.home)


def score(y, p):
    return {"logloss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p),
            "fav_acc": float(((p > 0.5) == (y == 1)).mean())}


seasons = sorted(df.season.unique())
tests = [s for s in seasons if s >= 20202021]
out = []
for s in tests:
    tr, te = df[df.season < s], df[df.season == s]
    r = {"season": s, "n": len(te), "home_rate": float(te.home.mean())}
    res = {"constant": np.full(len(te), tr.home.mean()), "elo": te.elo_p.to_numpy()}
    for name, goalie in (("LR_goalieTeamAvg", "gT"), ("LR_goalieStarter", "gS")):
        res[name] = fit(tr, feats(goalie)).predict_proba(te[feats(goalie)])[:, 1]
    ng = GROUPS["ability"] + GROUPS["chances"] + GROUPS["context"]
    res["LR_noGoalie"] = fit(tr, ng).predict_proba(te[ng])[:, 1]
    for k, p in res.items():
        r[k] = score(te.home.to_numpy(), p)
    out.append(r)

pd.set_option("display.width", 160)
for k in ("constant", "elo", "LR_noGoalie", "LR_goalieTeamAvg", "LR_goalieStarter"):
    t = pd.DataFrame({r["season"]: r[k] for r in out}).T
    print(f"\n{k}\n", t.round(4).to_string())
summary = {k: pd.DataFrame({r["season"]: r[k] for r in out}).T.mean().round(4).to_dict() for k in ("constant", "elo", "LR_noGoalie", "LR_goalieTeamAvg", "LR_goalieStarter")}
print("\nKeskiarvo testikausien yli (2020-21 ... 2025-26):")
print(pd.DataFrame(summary).T.to_string())
json.dump(out, open("winprob/backtest.json", "w"), indent=1, default=float)

# --- lopullinen malli (kaikki kaudet) -> kertoimet JSON:iin
cols = feats("gS")
final = fit(df, cols)
sc, lr = final[0], final[1]
coef = {c: float(w / s) for c, w, s in zip(cols, lr.coef_[0], sc.scale_)}   # kertoimet raakapiirteille
intercept = float(lr.intercept_[0] - sum(w * m / s for w, m, s in zip(lr.coef_[0], sc.mean_, sc.scale_)))
groups = {g: [c for c in cs if c in cols] for g, cs in {**GROUPS, "goalie": ["gS"]}.items()}
json.dump({"features": cols, "coef": coef, "intercept": intercept, "groups": groups,
           "params": {"half_lives": HALF_LIVES, "goalie_half_life": GOALIE_HALF_LIFE, "goalie_prior_shots": GOALIE_PRIOR,
                      "offseason_keep": OFFSEASON_KEEP, "init": INIT},
           "trained_on": [int(seasons[0]), int(seasons[-1]), len(df)]},
          open("winprob/model_wp.json", "w"), indent=1)
imp = {g: float(np.abs(sc.transform(df[cols])[:, [cols.index(c) for c in cs]] @ np.array([lr.coef_[0][cols.index(c)] for c in cs])).mean()) for g, cs in groups.items()}
tot = sum(imp.values())
print("\nKomponenttien osuus (keskimääräinen |lineaarinen vaikutus|):", {k: f"{v / tot:.0%}" for k, v in imp.items()})
print("Vakio (kotietu) p =", round(1 / (1 + np.exp(-intercept)), 3))
