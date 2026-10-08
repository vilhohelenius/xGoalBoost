"""Sovittaa simulaattorin kutistusparametrin (horizon) walk-forward-backtestillä.

Kullekin kaudelle s (2020-21 ... 2025-26) malli koulutetaan vain kausilla < s. Kauden sisällä otetaan tilannekuvat
(0 %, 10 %, 25 %, 50 %, 75 % otteluista pelattu): joukkueiden tila jäädytetään, ja kaikki jäljellä olevat ottelut ennustetaan
kutistuksella z = intercept + lin / (1 + päiviä / H).

1) Ottelutaso: log loss yli H-ruudukon + optimi (bootstrap-luottamusväli), sekä päiväetäisyysluokittain vapaa kulmakerroin
   (kuinka paljon piirteisiin pitäisi luottaa d päivän päässä) verrattuna kaavaan 1/(1+d/H).
2) Kausitaso (2021-22 ... 2025-26, NHL:n nykyinen playoff-formaatti): simuloidut playoff-todennäköisyydet vs. toteutunut
   playoff-paikka (Brier, log loss), ja pisteiden RMSE.

Ajo: .venv/bin/python winprob/backtest_horizon.py [--sims 4000]
"""
import argparse
import json
import pathlib
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "winprob"))
from simcore import Ratings, lin_score, outcome_cum, record_tables, rest_days, shrink, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=4000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--sigmas", default="0,0.05,0.1,0.15,0.2,0.3")
ap.add_argument("--teamfx", action="store_true", help="aja lisäksi koe: joukkuekohtainen satunnainen vahvuusmuutos simulaatioissa")
args = ap.parse_args()

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75)
GAME_SEASONS = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
SEASON_LEVEL = GAME_SEASONS[1:]                    # 2020-21 oli lyhennetty ja eri formaatti
H_GRID = [10, 20, 40, 80, 160, 320, 640, np.inf]
DBINS = [0, 7, 21, 42, 70, 105, 150, 210, 400]

mdl = json.load(open(ROOT / "winprob/model_wp.json"))
FEATS, HL = mdl["features"], tuple(mdl["params"]["half_lives"])
games = pd.read_parquet(DATA / "wp_games.parquet").sort_values(["start", "gameId"]).reset_index(drop=True)
games = games[games.season <= GAME_SEASONS[-1]].reset_index(drop=True)
tg = pd.read_parquet(DATA / "wp_team_game.parquet").set_index(["gameId", "teamId"])
gg = pd.read_parquet(DATA / "wp_goalie_game.parquet")
feat = pd.read_parquet(DATA / "wp_features.parquet")


def fit_coefs(season):
    tr = feat[feat.season < season]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(tr[FEATS], tr.home)
    sc, lr = m[0], m[1]
    coef = {c: float(w / s) for c, w, s in zip(FEATS, lr.coef_[0], sc.scale_)}
    return coef, float(lr.intercept_[0] - sum(w * mu / s for w, mu, s in zip(lr.coef_[0], sc.mean_, sc.scale_)))


def final_standings(season):
    """Toteutunut loppusarjataulukko NHL:n API:sta (välimuistissa): playoff-paikka, divisioona, pisteet."""
    f = DATA / f"standings_final_{season}.json"
    if not f.exists():
        day = games[games.season == season].date.max()
        with urllib.request.urlopen(urllib.request.Request(f"https://api-web.nhle.com/v1/standings/{day}", headers={"User-Agent": "xgoalboost"}), timeout=30) as r:
            f.write_text(r.read().decode())
    return json.loads(f.read_text())["standings"]


# ---------- tilannekuvat
rows = []                                           # otteluriviä: lin, d, y, intercept, snapshot, peli
snaps = []                                          # kausitason tilannekuvat
R = Ratings(mdl["params"], tg, gg)
by_season = {s: g for s, g in games.groupby("season")}
for season in sorted(by_season):
    gs = by_season[season].reset_index(drop=True)
    if season in GAME_SEASONS:
        coef, icpt = fit_coefs(season)
        past = games[games.season < season]
        rec = past[past.season.isin(sorted(past.season.unique())[-3:])]
        ot_rate = float((rec.lastPeriod != "REG").mean())
        so_share = float((rec.lastPeriod == "SO").sum() / (rec.lastPeriod != "REG").sum())
        cal = defaultdict(list)
        for g in gs.itertuples():
            cal[g.homeId].append(g.date); cal[g.awayId].append(g.date)
        prev = rest_days(cal)
        cut = {int(f * len(gs)): f for f in FRACS}
        if season in SEASON_LEVEL:
            truth = final_standings(season)
            abbrevs = sorted(s["teamAbbrev"]["default"] for s in truth)
            idx = {a: i for i, a in enumerate(abbrevs)}
            tr_ = {s["teamAbbrev"]["default"]: s for s in truth}
            divs, confs = defaultdict(list), defaultdict(list)
            for a in abbrevs:
                divs[tr_[a]["divisionAbbrev"]].append(idx[a]); confs[tr_[a]["conferenceAbbrev"]].append(idx[a])
            actual = np.array([float(tr_[a]["divisionSequence"] <= 3 or tr_[a]["wildcardSequence"] in (1, 2)) for a in abbrevs])
            assert actual.sum() == 16, (season, actual.sum())
            final_pts = np.array([tr_[a]["points"] for a in abbrevs], float)
            assert set(gs.homeAbbrev) | set(gs.awayAbbrev) <= set(abbrevs), season
    for k, g in enumerate(gs.itertuples()):
        if season in GAME_SEASONS and k in cut:
            tids = sorted(set(gs.homeId) | set(gs.awayId))
            state, gt = R.snapshot(season, tids)
            rem = gs.iloc[k:]
            lin = np.array([lin_score(state, gt, r.homeId, r.awayId, int(prev[(r.homeId, r.date)] <= 1) - int(prev[(r.awayId, r.date)] <= 1),
                                      coef, FEATS, HL) for r in rem.itertuples()])
            d = (pd.to_datetime(rem.date) - pd.Timestamp(g.date)).dt.days.to_numpy()
            y = (rem.homeGoals > rem.awayGoals).to_numpy().astype(float)
            sid = len(rows) and rows[-1]["snap"] + 1 or 0
            rows.append({"season": season, "frac": cut[k], "snap": sid, "lin": lin, "d": d, "y": y, "icpt": icpt, "gid": rem.gameId.to_numpy()})
            if season in SEASON_LEVEL:
                base, gp = record_tables(gs.iloc[:k], idx)
                snaps.append({"season": season, "frac": cut[k], "base": base, "d": d, "lin": lin, "icpt": icpt, "ot": (ot_rate, so_share),
                              "hi": np.array([idx[a] for a in rem.homeAbbrev]), "ai": np.array([idx[a] for a in rem.awayAbbrev]),
                              "divs": divs, "confs": confs, "actual": actual, "final_pts": final_pts, "N": len(abbrevs)})
        R.update(g)
    print(season, "valmis", flush=True)

lin = np.concatenate([r["lin"] for r in rows]); d = np.concatenate([r["d"] for r in rows])
y = np.concatenate([r["y"] for r in rows]); ic = np.concatenate([np.full(len(r["y"]), r["icpt"]) for r in rows])
gid = np.concatenate([r["gid"] for r in rows]); snp = np.concatenate([np.full(len(r["y"]), r["snap"]) for r in rows])
frac = np.concatenate([np.full(len(r["y"]), r["frac"]) for r in rows])


def logloss(p, y, w=None):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def ll_h(H, w=None):
    return logloss(shrink(lin, d, ic, H), y, w)


# ---------- 1. otteluTaso
print(f"\n=== OTTELUTASO: {len(y):,} ennustetta, {len(set(gid)):,} ottelua, {len(rows)} tilannekuvaa ===")
print("H (pv)   log loss")
for H in H_GRID:
    print(f"{H:>6}   {ll_h(H):.5f}")
res = minimize_scalar(lambda lh: ll_h(np.exp(lh)), bounds=(np.log(5), np.log(5000)), method="bounded")
H_star = float(np.exp(res.x))
print(f"Optimi H* = {H_star:.0f} pv, log loss {ll_h(H_star):.5f} (vakio-kotietu: {logloss(np.full(len(y), 1 / (1 + np.exp(-ic.mean()))), y):.5f})")

rng = np.random.default_rng(args.seed)
uniq, inv = np.unique(gid, return_inverse=True)
boots = []
for _ in range(200):
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[inv].astype(float)
    r = minimize_scalar(lambda lh: ll_h(np.exp(lh), w), bounds=(np.log(5), np.log(5000)), method="bounded")
    boots.append(np.exp(r.x))
lo, hi = np.percentile(boots, [5, 95])
print(f"Bootstrap (200, otteluittain) 90 %: H* {lo:.0f} ... {hi:.0f} pv, mediaani {np.median(boots):.0f}")

print("\nPäiväetäisyysluokittain: vapaa kulmakerroin lin:lle (1 = täysi luottamus) vs. kaava 1/(1+d/H*) ja log loss")
tab = []
for lo_, hi_ in zip(DBINS[:-1], DBINS[1:]):
    m = (d >= lo_) & (d < hi_)
    if m.sum() < 500:
        continue
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lin[m, None], y[m])
    dm = float(d[m].mean())
    tab.append({"d_from": lo_, "d_to": hi_, "n": int(m.sum()), "d_mean": round(dm, 1), "slope": round(float(lr.coef_[0, 0]), 3),
                "formula_H*": round(1 / (1 + dm / H_star), 3), "ll_noshrink": round(logloss(shrink(lin[m], d[m], ic[m], np.inf), y[m]), 5),
                "ll_H120": round(logloss(shrink(lin[m], d[m], ic[m], 120), y[m]), 5), "ll_H*": round(logloss(shrink(lin[m], d[m], ic[m], H_star), y[m]), 5)})
pd.set_option("display.width", 200)
print(pd.DataFrame(tab).to_string(index=False))

print("\nTilannekuvan vaihe: optimi-H* ja log loss (H=120 / H* / ei kutistusta)")
for f in FRACS:
    m = frac == f
    r = minimize_scalar(lambda lh: logloss(shrink(lin[m], d[m], ic[m], np.exp(lh)), y[m]), bounds=(np.log(5), np.log(5000)), method="bounded")
    print(f"  {f:>4.0%} pelattu: H*={np.exp(r.x):6.0f}  ll {logloss(shrink(lin[m], d[m], ic[m], 120), y[m]):.5f} / {logloss(shrink(lin[m], d[m], ic[m], np.exp(r.x)), y[m]):.5f} / {logloss(shrink(lin[m], d[m], ic[m], np.inf), y[m]):.5f}")

# ---------- 2. kausitaso
print(f"\n=== KAUSITASO: {len(snaps)} tilannekuvaa, {args.sims:,} simulaatiota / H ===")
Hs = sorted({10, 20, 40, 80, 160, 320, np.inf, round(H_star)}, key=float)
out = []
for sn in snaps:
    cal_ = []
    for H in Hs:
        p = shrink(sn["lin"], sn["d"], sn["icpt"], H)
        r = simulate(sn["base"], sn["hi"], sn["ai"], outcome_cum(p, *sn["ot"]), sn["divs"], sn["confs"], args.sims, np.random.default_rng(args.seed), keep_pts=True)
        pp = np.clip(r["playoffs"], 1e-3, 1 - 1e-3)
        out.append({"season": sn["season"], "frac": sn["frac"], "H": float(H), "brier": float(np.mean((pp - sn["actual"]) ** 2)),
                    "logloss": logloss(pp, sn["actual"]), "pts_rmse": float(np.sqrt(np.mean((r["pts"].mean(0) - sn["final_pts"]) ** 2)))})
res_df = pd.DataFrame(out)
agg = res_df.groupby("H")[["brier", "logloss", "pts_rmse"]].mean().round(4)
print("Keskiarvo kaikkien tilannekuvien yli:\n", agg.to_string())
print("\nBrier tilannekuvan vaiheen mukaan:\n", res_df.pivot_table(index="H", columns="frac", values="brier").round(4).to_string())
print("\nPisteiden RMSE tilannekuvan vaiheen mukaan:\n", res_df.pivot_table(index="H", columns="frac", values="pts_rmse").round(2).to_string())
print("\nBrier kausittain (H*):\n", res_df[res_df.H == round(H_star)].pivot_table(index="season", columns="frac", values="brier").round(4).to_string())

H_best = float(agg.brier.idxmin())
print(f"\nKausitason paras H (pienin Brier) = {H_best:g} pv")
pv = res_df.pivot_table(index="season", columns="H", values="brier")
print("Brier kausittain, keskiarvo yli tilannekuvien (paras H vs. ei kutistusta):\n", pd.DataFrame({"H_best": pv[H_best], "ei_kutistusta": pv[np.inf], "ero": pv[H_best] - pv[np.inf]}).round(4).to_string())

# kalibrointi: ennustettu vs toteutunut playoff-%
def calibration(H):
    rows_ = []
    for sn in snaps:
        p = shrink(sn["lin"], sn["d"], sn["icpt"], H)
        r = simulate(sn["base"], sn["hi"], sn["ai"], outcome_cum(p, *sn["ot"]), sn["divs"], sn["confs"], args.sims, np.random.default_rng(args.seed))
        rows_ += [{"p": a, "y": b} for a, b in zip(r["playoffs"], sn["actual"])]
    c = pd.DataFrame(rows_)
    c["bin"] = pd.cut(c.p, [0, .05, .15, .3, .5, .7, .85, .95, 1], include_lowest=True)
    return c.groupby("bin", observed=True).agg(n=("y", "size"), ennuste=("p", "mean"), toteuma=("y", "mean")).round(3)


for H in (H_best, np.inf):
    print(f"\nKalibrointi, H={H:g} (kaikki tilannekuvat):\n", calibration(H).to_string())

if args.teamfx:
    print("\n=== JOUKKUEVAIKUTUS: satunnainen vahvuusmuutos (logit, sigma) per joukkue per simulaatio ===")
    SIG, HH = tuple(float(x) for x in args.sigmas.split(",")), (80, np.inf)
    fx = []
    for sn in snaps:
        for H in HH:
            z = sn["icpt"] + sn["lin"] / (1 + sn["d"] / H)
            for sg in SIG:
                r = simulate(sn["base"], sn["hi"], sn["ai"], None, sn["divs"], sn["confs"], args.sims, np.random.default_rng(args.seed), keep_pts=True,
                             team_sigma=sg, z=z, ot=sn["ot"]) if sg > 0 else simulate(
                    sn["base"], sn["hi"], sn["ai"], outcome_cum(1 / (1 + np.exp(-z)), *sn["ot"]), sn["divs"], sn["confs"], args.sims, np.random.default_rng(args.seed), keep_pts=True)
                pp = np.clip(r["playoffs"], 1e-3, 1 - 1e-3)
                fx.append({"season": sn["season"], "frac": sn["frac"], "H": float(H), "sigma": sg, "brier": float(np.mean((pp - sn["actual"]) ** 2)),
                           "logloss": logloss(pp, sn["actual"]), "pts_rmse": float(np.sqrt(np.mean((r["pts"].mean(0) - sn["final_pts"]) ** 2)))})
    fx_df = pd.DataFrame(fx)
    for m in ("brier", "logloss", "pts_rmse"):
        print(f"\n{m} (keskiarvo; rivit H, sarakkeet sigma):\n", fx_df.pivot_table(index="H", columns="sigma", values=m).round(4).to_string())
    print("\nBrier vaiheittain, H=inf:\n", fx_df[fx_df.H == np.inf].pivot_table(index="sigma", columns="frac", values="brier").round(4).to_string())
    print("\nBrier kausittain, H=inf:\n", fx_df[fx_df.H == np.inf].pivot_table(index="sigma", columns="season", values="brier").round(4).to_string())
    json.dump(fx_df.to_dict("records"), open(ROOT / "winprob/backtest_teamfx.json", "w"), indent=1, default=float)

json.dump({"H_star_days": H_star, "H_best_season_level": H_best, "bootstrap90": [float(lo), float(hi)], "game_level": {str(H): ll_h(H) for H in H_GRID}, "buckets": tab,
           "season_level": res_df.to_dict("records")}, open(ROOT / "winprob/backtest_horizon.json", "w"), indent=1, default=float)
