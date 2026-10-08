"""Runkosarjan simulointi: playoff-todennäköisyydet MoneyPuckin tyyliin.

Jokainen jäljellä oleva ottelu arvotaan ennakkomallin (winprob/model_wp.json) todennäköisyydellä, 100 000 kertaa.
Kauempana tulevaisuudessa olevien otteluiden ennuste kutistetaan kohti kotietua: z = intercept + (z - intercept) / (1 + päiviä / HORIZON).
Ottelu päättyy varsinaisella ajalla, jatkoajalla tai voittolaukauksilla (osuudet viimeisiltä kausilta), jotta pisteet ja
tasapisteiden ratkaisu (RW, ROW, voitot) menevät oikein. Playoff-paikat: 3 parasta / divisioona + 2 wild cardia / konferenssi.

API-kutsut (data/ on välimuisti, kaikki muu lasketaan paikallisesti):
  - standings/now: 1 kutsu / ajo, vain divisioonat ja konferenssit (välimuisti, päivittyy --refresh)
  - kausiaikataulu: 32 kutsua (club-schedule-season), välimuistissa 7 päivää tai --refresh
  - play-by-play: 1 kutsu / uusi pelattu ottelu (yleensä 5-16 / pelipäivä), ladatut ohitetaan
Ajo: .venv/bin/python winprob/simulate_season.py [--sims 100000] [--horizon 80] [--refresh] [--offline]
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "nhl_pbp"))
sys.path.insert(0, str(ROOT / "winprob"))
from simcore import Ratings, lin_score, outcome_cum, record_tables, rest_days, shrink, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=100_000)
ap.add_argument("--horizon", type=float, default=80, help="päivää; kutistus 1/(1+päiviä/horizon), sovitettu: winprob/backtest_horizon.py")
ap.add_argument("--refresh", action="store_true", help="hae standings ja aikataulu uudelleen")
ap.add_argument("--offline", action="store_true", help="älä käytä APIa lainkaan (vain välimuisti)")
ap.add_argument("--today", default=str(date.today()))
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
TODAY = pd.Timestamp(args.today)


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "xgoalboost"}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API-kutsu epäonnistui: {url}")


def stale(path, days):
    return not path.exists() or (time.time() - path.stat().st_mtime) > days * 86400


# ---------- 1. standings (divisiot) ja aikataulu
sp = DATA / "nhl_standings.json"
if not args.offline and (args.refresh or stale(sp, 30)):
    sp.write_text(json.dumps(get("https://api-web.nhle.com/v1/standings/now")))
standings = json.loads(sp.read_text())["standings"]
SEASON = int(standings[0]["seasonId"])
teams = {s["teamAbbrev"]["default"]: {"name": s["teamName"]["default"], "div": s["divisionAbbrev"], "conf": s["conferenceAbbrev"]}
         for s in standings}
abbrevs = sorted(teams)

schp = DATA / f"schedule_{SEASON}.json"
if not args.offline and (args.refresh or stale(schp, 7)):
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda a: get(f"https://api-web.nhle.com/v1/club-schedule-season/{a}/{SEASON}"), abbrevs))
    seen = {}
    for r in res:
        for g in r["games"]:
            if g["gameType"] == 2:
                seen[g["id"]] = {"id": g["id"], "date": g["gameDate"], "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"]}
    schp.write_text(json.dumps(sorted(seen.values(), key=lambda g: (g["date"], g["id"]))))
sched = pd.DataFrame(json.loads(schp.read_text()))
print(f"Kausi {SEASON}: {len(sched)} runkosarjan ottelua aikataulussa, {len(abbrevs)} joukkuetta")

# ---------- 2. lataa uudet pelatut ottelut ja päivitä ottelutaulut tarvittaessa
pbp_ids = {int(f.stem) for f in (DATA / "pbp").glob("*.json")}
if not args.offline:
    from download import fetch
    todo = [int(g) for g, d in zip(sched.id, sched.date) if pd.Timestamp(d) <= TODAY and int(g) not in pbp_ids]
    if todo:
        with ThreadPoolExecutor(6) as ex:
            stat = [s for _, s in ex.map(fetch, todo)]
        print(f"PBP-kutsuja {len(todo)}: " + ", ".join(f"{k} {stat.count(k)}" for k in sorted(set(stat))))
        pbp_ids = {int(f.stem) for f in (DATA / "pbp").glob("*.json")}
games = pd.read_parquet(DATA / "wp_games.parquet")
if any(int(g) in pbp_ids and g not in set(games.gameId) for g in sched.id):
    print("Uusia pelattuja otteluja -> build_games.py (paikallinen laskenta, ei API-kutsuja)")
    subprocess.run([sys.executable, str(ROOT / "winprob/build_games.py")], cwd=ROOT, check=True)
    games = pd.read_parquet(DATA / "wp_games.parquet")

games = games.sort_values(["start", "gameId"]).reset_index(drop=True)
tg = pd.read_parquet(DATA / "wp_team_game.parquet").set_index(["gameId", "teamId"])
gg = pd.read_parquet(DATA / "wp_goalie_game.parquet")
mdl = json.load(open(ROOT / "winprob/model_wp.json"))
HL = tuple(mdl["params"]["half_lives"])

# ---------- 3. toista pelatut ottelut -> joukkueiden ja maalivahtien nykytila
R = Ratings(mdl["params"], tg, gg)
ids = {}                                           # joukkueen NHL-id -> lyhenne
for g in games.itertuples():
    ids[g.homeId], ids[g.awayId] = g.homeAbbrev, g.awayAbbrev
    R.update(g)
id_of = {a: i for i, a in ids.items()}
st_id, gt_id = R.snapshot(SEASON, [id_of[a] for a in abbrevs])
state = {a: st_id[id_of[a]] for a in abbrevs}
gt = {a: gt_id[id_of[a]] for a in abbrevs}

# ---------- 4. tämän kauden tilanne ja jäljellä olevat ottelut
cur = games[games.season == SEASON]
rem = sched[~sched.id.isin(set(cur.gameId))].reset_index(drop=True)
idx = {a: i for i, a in enumerate(abbrevs)}
N = len(abbrevs)
base, gp = record_tables(cur, idx)

cal = defaultdict(list)                             # lepo: pelatut + jäljellä olevat samalta kaudelta
for g in cur.itertuples():
    cal[g.homeAbbrev].append(g.date); cal[g.awayAbbrev].append(g.date)
for g in rem.itertuples():
    cal[g.home].append(g.date); cal[g.away].append(g.date)
prev = rest_days(cal)

lin = np.array([lin_score(state, gt, g.home, g.away, int(prev[(g.home, g.date)] <= 1) - int(prev[(g.away, g.date)] <= 1),
                          mdl["coef"], mdl["features"], HL) for g in rem.itertuples()])
ahead = np.maximum((pd.to_datetime(rem.date) - TODAY).dt.days.to_numpy(), 0)
p = shrink(lin, ahead, mdl["intercept"], args.horizon)

# jatkoaika- ja voittolaukausosuudet viimeisiltä kolmelta kaudelta
rec = games[games.season.isin(sorted(games.season[games.season < SEASON].unique())[-3:])]
ot_rate = float((rec.lastPeriod != "REG").mean())
so_share = float((rec.lastPeriod == "SO").sum() / max((rec.lastPeriod != "REG").sum(), 1))
G = len(rem)

divs = defaultdict(list); confs = defaultdict(list)
for a in abbrevs:
    divs[teams[a]["div"]].append(idx[a]); confs[teams[a]["conf"]].append(idx[a])

# ---------- 5. simulointi
sim = simulate(base, np.array([idx[a] for a in rem.home]), np.array([idx[a] for a in rem.away]),
               outcome_cum(p, ot_rate, so_share), divs, confs, args.sims, np.random.default_rng(args.seed), keep_pts=True)
pts_all = sim["pts"]
out = pd.DataFrame({"team": abbrevs, "name": [teams[a]["name"] for a in abbrevs], "conf": [teams[a]["conf"] for a in abbrevs],
                    "div": [teams[a]["div"] for a in abbrevs], "gp": gp.astype(int), "pts": base[0].astype(int),
                    "exp_pts": pts_all.mean(0).astype(float).round(1), "pts_p10": np.percentile(pts_all, 10, axis=0), "pts_p90": np.percentile(pts_all, 90, axis=0),
                    "p_playoffs": sim["playoffs"], "p_division": sim["division"], "p_presidents": sim["presidents"]})
out = out.sort_values(["conf", "p_playoffs"], ascending=[True, False])
pd.set_option("display.width", 200)
print(f"\n{args.sims:,} simulaatiota, {G} jäljellä olevaa ottelua, jatkoaika-% {ot_rate:.3f}, SO-osuus {so_share:.2f}, horisontti {args.horizon:g} pv")
print(out.assign(**{c: (out[c] * 100).round(1) for c in ("p_playoffs", "p_division", "p_presidents")}).drop(columns="name").to_string(index=False))
json.dump({"season": SEASON, "asOf": str(TODAY.date()), "sims": args.sims, "games_remaining": G, "horizon_days": args.horizon,
           "teams": out.round(4).to_dict("records")}, open(ROOT / "winprob/season_sim.json", "w"), indent=1, default=float)
