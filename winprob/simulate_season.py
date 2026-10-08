"""Runkosarjan simulointi: playoff-todennäköisyydet MoneyPuckin tyyliin.

Jokainen jäljellä oleva ottelu arvotaan ennakkomallin (winprob/model_wp.json) todennäköisyydellä, 100 000 kertaa.
Kauempana tulevaisuudessa olevien otteluiden ennuste kutistetaan kohti kotietua: z = intercept + (z - intercept) / (1 + päiviä / HORIZON).
Ottelu päättyy varsinaisella ajalla, jatkoajalla tai voittolaukauksilla (osuudet viimeisiltä kausilta), jotta pisteet ja
tasapisteiden ratkaisu (RW, ROW, voitot) menevät oikein. Playoff-paikat: 3 parasta / divisioona + 2 wild cardia / konferenssi.

API-kutsut (data/ on välimuisti, kaikki muu lasketaan paikallisesti):
  - standings/now: 1 kutsu / ajo, vain divisioonat ja konferenssit (välimuisti, päivittyy --refresh)
  - kausiaikataulu: 32 kutsua (club-schedule-season), välimuistissa 7 päivää tai --refresh
  - play-by-play: 1 kutsu / uusi pelattu ottelu (yleensä 5-16 / pelipäivä), ladatut ohitetaan
Ajo: .venv/bin/python winprob/simulate_season.py [--sims 100000] [--horizon 120] [--refresh] [--offline]
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "nhl_pbp"))

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=100_000)
ap.add_argument("--horizon", type=float, default=120, help="päivää, jolla ennusteen ja kotietutason etäisyys puolittuu (kutistus)")
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
starter = gg.sort_values("shotsAgainst").groupby(["gameId", "teamId"]).tail(1).set_index(["gameId", "teamId"]).goalieId
gby = {k: v for k, v in gg.groupby("gameId")}
mdl = json.load(open(ROOT / "winprob/model_wp.json"))
P = mdl["params"]
HL, METRICS = tuple(P["half_lives"]), ["win", "gd", "xgd", "xgd5", "sogd", "dz"]
INIT = P["init"]

# ---------- 3. toista pelatut ottelut (sama päivityslogiikka kuin train_wp.py) -> joukkueiden ja maalivahtien nykytila
team_state = defaultdict(lambda: {h: dict(INIT) for h in HL})
goalie_state = defaultdict(lambda: [0.0, 0.0])
recent = defaultdict(lambda: deque(maxlen=20))
last_date, last_season = {}, {}
lam = 0.5 ** (1 / P["goalie_half_life"])
ids = {}                                           # joukkueen NHL-id -> lyhenne
for g in games.itertuples():
    ids[g.homeId], ids[g.awayId] = g.homeAbbrev, g.awayAbbrev
    day = pd.Timestamp(g.date)
    for t, is_home in ((g.homeId, 1), (g.awayId, 0)):
        if last_season.get(t) not in (None, g.season):                          # kausivaihde
            for h in HL:
                for m in METRICS:
                    team_state[t][h][m] = P["offseason_keep"] * team_state[t][h][m] + (1 - P["offseason_keep"]) * INIT[m]
        r = tg.loc[(g.gameId, t)]
        win = 0.5 if g.lastPeriod != "REG" else float(r.gf > r.ga)
        m = {"win": win, "gd": r.gf - r.ga, "xgd": r.xgf - r.xga, "xgd5": r.xgf5 - r.xga5, "sogd": r.sog_f - r.sog_a, "dz": r.dzGiveaway}
        for h in HL:
            a = 1 - 0.5 ** (1 / h)
            for k in METRICS:
                team_state[t][h][k] += a * (m[k] - team_state[t][h][k])
        last_date[t], last_season[t] = day, g.season
        sid = starter.get((g.gameId, t))
        if sid is not None and not pd.isna(sid):
            recent[t].append(sid)
    for r in gby.get(g.gameId, pd.DataFrame()).itertuples():
        st = goalie_state[r.goalieId]
        st[0] = st[0] * lam + r.GSAx; st[1] = st[1] * lam + r.shotsAgainst
id_of = {a: i for i, a in ids.items()}


def current(t):
    """Joukkueen tila nyt; jos kauden ensimmäinen peli on vielä pelaamatta, kausivaihteen pehmennys."""
    s = {h: dict(team_state[t][h]) for h in HL}
    if last_season.get(t) not in (None, SEASON):
        for h in HL:
            for m in METRICS:
                s[h][m] = P["offseason_keep"] * s[h][m] + (1 - P["offseason_keep"]) * INIT[m]
    return s


def goalie_t(t):
    rs = list(recent[t])
    return float(np.mean([100 * goalie_state[x][0] / (goalie_state[x][1] + P["goalie_prior_shots"]) for x in rs])) if rs else 0.0


state = {a: current(id_of[a]) for a in abbrevs}
gt = {a: goalie_t(id_of[a]) for a in abbrevs}

# ---------- 4. tämän kauden tilanne ja jäljellä olevat ottelut
cur = games[games.season == SEASON]
done = set(cur.gameId)
rem = sched[~sched.id.isin(done)].reset_index(drop=True)
idx = {a: i for i, a in enumerate(abbrevs)}
N = len(abbrevs)
base = np.zeros((4, N))                             # pisteet, RW, ROW, voitot
gp = np.zeros(N)
for g in cur.itertuples():
    for a, own, opp in ((g.homeAbbrev, g.homeGoals, g.awayGoals), (g.awayAbbrev, g.awayGoals, g.homeGoals)):
        if a not in idx:
            continue
        i, won, per = idx[a], own > opp, g.lastPeriod
        gp[i] += 1
        base[0, i] += 2 if won else (1 if per != "REG" else 0)
        base[1, i] += won and per == "REG"
        base[2, i] += won and per != "SO"
        base[3, i] += won

# lepo: edellisen pelin päivä (pelatut + jäljellä olevat samalta kaudelta)
cal = defaultdict(list)
for g in cur.itertuples():
    cal[g.homeAbbrev].append(g.date); cal[g.awayAbbrev].append(g.date)
for g in rem.itertuples():
    cal[g.home].append(g.date); cal[g.away].append(g.date)
prev = {}
for a, ds in cal.items():
    ds = sorted(ds)
    for i, d in enumerate(ds):
        prev[(a, d)] = (pd.Timestamp(d) - pd.Timestamp(ds[i - 1])).days if i else 7

coef, icpt = mdl["coef"], mdl["intercept"]
z = np.zeros(len(rem))
for k, g in enumerate(rem.itertuples()):
    x = {f"{m}_{h}": state[g.home][h][m] - state[g.away][h][m] for h in HL for m in METRICS}
    x["gS"] = gt[g.home] - gt[g.away]
    x["b2b_diff"] = int(prev[(g.home, g.date)] <= 1) - int(prev[(g.away, g.date)] <= 1)
    ahead = max((pd.Timestamp(g.date) - TODAY).days, 0)
    z[k] = icpt + sum(coef[f] * x[f] for f in mdl["features"]) / (1 + ahead / args.horizon)
p = 1 / (1 + np.exp(-z))

# jatkoaika- ja voittolaukausosuudet viimeisiltä kolmelta kaudelta
rec = games[games.season.isin(sorted(games.season[games.season < SEASON].unique())[-3:])]
ot_rate = float((rec.lastPeriod != "REG").mean())
so_share = float((rec.lastPeriod == "SO").sum() / max((rec.lastPeriod != "REG").sum(), 1))
q = 0.5 + 0.5 * (p - 0.5)                           # kotijoukkueen voitto-osuus jatkoajasta/SO:sta
pr = np.stack([p - ot_rate * q,                                 # koti, varsinainen aika
               ot_rate * q * (1 - so_share), ot_rate * q * so_share,
               ot_rate * (1 - q) * so_share, ot_rate * (1 - q) * (1 - so_share),
               1 - p - ot_rate * (1 - q)], 1)
cum = np.cumsum(pr, 1)[:, :5]

HP = np.array([2, 2, 2, 1, 1, 0]); AP = np.array([0, 1, 1, 2, 2, 2])
HRW = np.array([1, 0, 0, 0, 0, 0]); ARW = np.array([0, 0, 0, 0, 0, 1])
HROW = np.array([1, 1, 0, 0, 0, 0]); AROW = np.array([0, 0, 0, 0, 1, 1])
HW = np.array([1, 1, 1, 0, 0, 0]); AW = np.array([0, 0, 0, 1, 1, 1])
G = len(rem)
Hm, Am = np.zeros((G, N), np.float32), np.zeros((G, N), np.float32)
Hm[np.arange(G), [idx[a] for a in rem.home]] = 1
Am[np.arange(G), [idx[a] for a in rem.away]] = 1

divs = defaultdict(list); confs = defaultdict(list)
for a in abbrevs:
    divs[teams[a]["div"]].append(idx[a]); confs[teams[a]["conf"]].append(idx[a])

# ---------- 5. simulointi
rng = np.random.default_rng(args.seed)
CH = 5000
playoffs, divwin, presidents = np.zeros(N), np.zeros(N), np.zeros(N)
pts_all = np.zeros((args.sims, N), np.float32)
done_sims = 0
while done_sims < args.sims:
    S = min(CH, args.sims - done_sims)
    o = (rng.random((S, G), dtype=np.float32)[:, :, None] > cum[None].astype(np.float32)).sum(2) if G else np.zeros((S, 0), int)
    tot = []
    for b, (h, a) in zip(base, ((HP, AP), (HRW, ARW), (HROW, AROW), (HW, AW))):
        tot.append(b + h[o].astype(np.float32) @ Hm + a[o].astype(np.float32) @ Am)
    pts = tot[0]
    key = pts * 1e6 + tot[1] * 1e4 + tot[2] * 1e2 + tot[3] + rng.random((S, N))     # pisteet, RW, ROW, voitot, arpa
    q_ = np.zeros((S, N), bool)
    for m in divs.values():
        m = np.array(m); rk = np.argsort(-key[:, m], 1)
        np.put_along_axis(q_, m[rk[:, :3]], True, 1)
        divwin[m] += np.bincount(m[rk[:, 0]].ravel(), minlength=N)[m]
    for m in confs.values():
        m = np.array(m); sub = np.where(q_[:, m], -np.inf, key[:, m]); rk = np.argsort(-sub, 1)
        np.put_along_axis(q_, m[rk[:, :2]], True, 1)
    playoffs += q_.sum(0)
    presidents += np.bincount(key.argmax(1), minlength=N)
    pts_all[done_sims:done_sims + S] = pts
    done_sims += S

out = pd.DataFrame({"team": abbrevs, "name": [teams[a]["name"] for a in abbrevs], "conf": [teams[a]["conf"] for a in abbrevs],
                    "div": [teams[a]["div"] for a in abbrevs], "gp": gp.astype(int), "pts": base[0].astype(int),
                    "exp_pts": pts_all.mean(0).astype(float).round(1), "pts_p10": np.percentile(pts_all, 10, axis=0), "pts_p90": np.percentile(pts_all, 90, axis=0),
                    "p_playoffs": playoffs / args.sims, "p_division": divwin / args.sims, "p_presidents": presidents / args.sims})
out = out.sort_values(["conf", "p_playoffs"], ascending=[True, False])
pd.set_option("display.width", 200)
print(f"\n{args.sims:,} simulaatiota, {G} jäljellä olevaa ottelua, jatkoaika-% {ot_rate:.3f}, SO-osuus {so_share:.2f}, horisontti {args.horizon:g} pv")
print(out.assign(**{c: (out[c] * 100).round(1) for c in ("p_playoffs", "p_division", "p_presidents")}).drop(columns="name").to_string(index=False))
json.dump({"season": SEASON, "asOf": str(TODAY.date()), "sims": args.sims, "games_remaining": G, "horizon_days": args.horizon,
           "teams": out.round(4).to_dict("records")}, open(ROOT / "winprob/season_sim.json", "w"), indent=1, default=float)
