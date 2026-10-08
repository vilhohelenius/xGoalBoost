"""Yhteinen ydin simulate_season.py:lle ja backtest_horizon.py:lle: joukkuetilojen toisto, ennusteet ja kausisimulaatio."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

METRICS = ["win", "gd", "xgd", "xgd5", "sogd", "dz"]
# lopputulokset: koti reg, koti JA, koti VL, vieras VL, vieras JA, vieras reg
HP = np.array([2, 2, 2, 1, 1, 0]); AP = np.array([0, 1, 1, 2, 2, 2])
HRW = np.array([1, 0, 0, 0, 0, 0]); ARW = np.array([0, 0, 0, 0, 0, 1])
HROW = np.array([1, 1, 0, 0, 0, 0]); AROW = np.array([0, 0, 0, 0, 1, 1])
HW = np.array([1, 1, 1, 0, 0, 0]); AW = np.array([0, 0, 0, 1, 1, 1])


class Ratings:
    """Toistaa pelatut ottelut ja pitää yllä joukkueiden ja maalivahtien tilaa (sama logiikka kuin train_wp.py)."""

    def __init__(self, params, tg, gg):
        self.P = params
        self.HL = tuple(params["half_lives"])
        self.INIT = params["init"]
        self.tg = tg
        self.starter = gg.sort_values("shotsAgainst").groupby(["gameId", "teamId"]).tail(1).set_index(["gameId", "teamId"]).goalieId
        self.gby = {k: v for k, v in gg.groupby("gameId")}
        self.team = defaultdict(lambda: {h: dict(self.INIT) for h in self.HL})
        self.goalie = defaultdict(lambda: [0.0, 0.0])
        self.recent = defaultdict(lambda: deque(maxlen=20))
        self.last_season = {}
        self.lam = 0.5 ** (1 / params["goalie_half_life"])

    def _offseason(self, s):
        k = self.P["offseason_keep"]
        for h in self.HL:
            for m in METRICS:
                s[h][m] = k * s[h][m] + (1 - k) * self.INIT[m]

    def update(self, g):
        for t in (g.homeId, g.awayId):
            if self.last_season.get(t) not in (None, g.season):
                self._offseason(self.team[t])
            r = self.tg.loc[(g.gameId, t)]
            win = 0.5 if g.lastPeriod != "REG" else float(r.gf > r.ga)
            m = {"win": win, "gd": r.gf - r.ga, "xgd": r.xgf - r.xga, "xgd5": r.xgf5 - r.xga5, "sogd": r.sog_f - r.sog_a, "dz": r.dzGiveaway}
            for h in self.HL:
                a = 1 - 0.5 ** (1 / h)
                for k in METRICS:
                    self.team[t][h][k] += a * (m[k] - self.team[t][h][k])
            self.last_season[t] = g.season
            sid = self.starter.get((g.gameId, t))
            if sid is not None and not pd.isna(sid):
                self.recent[t].append(sid)
        for r in self.gby.get(g.gameId, pd.DataFrame()).itertuples():
            st = self.goalie[r.goalieId]
            st[0] = st[0] * self.lam + r.GSAx; st[1] = st[1] * self.lam + r.shotsAgainst

    def snapshot(self, season, team_ids):
        """Joukkueiden tila (dict team_id -> tila) ja maalivahtiarvio; jos joukkue ei ole vielä pelannut kautta, kausivaihteen pehmennys."""
        state, gt = {}, {}
        for t in team_ids:
            s = {h: dict(self.team[t][h]) for h in self.HL}
            if self.last_season.get(t) not in (None, season):
                self._offseason(s)
            state[t] = s
            rs = list(self.recent[t])
            gt[t] = float(np.mean([100 * self.goalie[x][0] / (self.goalie[x][1] + self.P["goalie_prior_shots"]) for x in rs])) if rs else 0.0
        return state, gt


def lin_score(state, gt, home, away, b2b_diff, coef, features, HL):
    """Mallin lineaarinen osa ilman vakiotermiä (piirteet nykytilasta)."""
    x = {f"{m}_{h}": state[home][h][m] - state[away][h][m] for h in HL for m in METRICS}
    x["gS"] = gt[home] - gt[away]
    x["b2b_diff"] = b2b_diff
    return sum(coef[f] * x[f] for f in features)


def rest_days(dates_by_team):
    """(joukkue, päivä) -> päiviä edelliseen peliin (7 = kauden eka)."""
    prev = {}
    for a, ds in dates_by_team.items():
        ds = sorted(ds)
        for i, d in enumerate(ds):
            prev[(a, d)] = (pd.Timestamp(d) - pd.Timestamp(ds[i - 1])).days if i else 7
    return prev


def shrink(lin, days_ahead, intercept, horizon):
    """Kutistaa ennusteen piirteiden osuuden: lin / (1 + päiviä / horizon). horizon=inf -> ei kutistusta."""
    return 1 / (1 + np.exp(-(intercept + lin / (1 + np.asarray(days_ahead, float) / horizon))))


def outcome_cum(p, ot_rate, so_share):
    """Kotivoittotodennäköisyydestä kumulatiiviset lopputulosten todennäköisyydet (5 ensimmäistä)."""
    q = 0.5 + 0.5 * (p - 0.5)                       # kotijoukkueen voitto-osuus jatkoajasta/VL:stä
    pr = np.stack([p - ot_rate * q, ot_rate * q * (1 - so_share), ot_rate * q * so_share,
                   ot_rate * (1 - q) * so_share, ot_rate * (1 - q) * (1 - so_share), 1 - p - ot_rate * (1 - q)], -1)
    return np.cumsum(pr, -1)[..., :5]


def simulate(base, home_idx, away_idx, cum, divs, confs, sims, rng, chunk=5000, keep_pts=False, team_sigma=0.0, z=None, ot=None):
    """base: (4, N) nykyiset pisteet, RW, ROW, voitot. Palauttaa osuudet: playoffs, division, presidents (+ pisteet).

    team_sigma > 0: jokaiselle simulaatiolle arvotaan joukkuekohtainen vahvuusmuutos ~ N(0, sigma^2) (logit-asteikko), joka
    pysyy samana koko kauden. Silloin tarvitaan ottelujen logit z (G,) ja ot=(ot_rate, so_share); cum ohitetaan."""
    N, G = base.shape[1], len(home_idx)
    Hm, Am = np.zeros((G, N), np.float32), np.zeros((G, N), np.float32)
    Hm[np.arange(G), home_idx] = 1; Am[np.arange(G), away_idx] = 1
    cum32 = None if team_sigma > 0 else cum[None].astype(np.float32)
    if team_sigma > 0:
        chunk = min(chunk, 1000)
    playoffs, divwin, presidents = np.zeros(N), np.zeros(N), np.zeros(N)
    pts_all = np.zeros((sims, N), np.float32) if keep_pts else None
    done = 0
    while done < sims:
        S = min(chunk, sims - done)
        if team_sigma > 0:
            A = rng.normal(0, team_sigma, (S, N))
            c = outcome_cum(1 / (1 + np.exp(-(z[None] + A[:, home_idx] - A[:, away_idx]))), *ot).astype(np.float32)
            o = (rng.random((S, G), dtype=np.float32)[:, :, None] > c).sum(2)
        else:
            o = (rng.random((S, G), dtype=np.float32)[:, :, None] > cum32).sum(2) if G else np.zeros((S, 0), int)
        tot = [b + h[o].astype(np.float32) @ Hm + a[o].astype(np.float32) @ Am
               for b, (h, a) in zip(base, ((HP, AP), (HRW, ARW), (HROW, AROW), (HW, AW)))]
        key = tot[0] * 1e6 + tot[1] * 1e4 + tot[2] * 1e2 + tot[3] + rng.random((S, N))     # pisteet, RW, ROW, voitot, arpa
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
        if keep_pts:
            pts_all[done:done + S] = tot[0]
        done += S
    res = {"playoffs": playoffs / sims, "division": divwin / sims, "presidents": presidents / sims}
    if keep_pts:
        res["pts"] = pts_all
    return res


def record_tables(games, idx):
    """Pelatuista otteluista (DataFrame) taulukko (4, N): pisteet, RW, ROW, voitot, sekä pelatut ottelut."""
    base, gp = np.zeros((4, len(idx))), np.zeros(len(idx))
    for g in games.itertuples():
        for a, own, opp in ((g.homeAbbrev, g.homeGoals, g.awayGoals), (g.awayAbbrev, g.awayGoals, g.homeGoals)):
            if a not in idx:
                continue
            i, won, per = idx[a], own > opp, g.lastPeriod
            gp[i] += 1
            base[0, i] += 2 if won else (1 if per != "REG" else 0)
            base[1, i] += won and per == "REG"
            base[2, i] += won and per != "SO"
            base[3, i] += won
    return base, gp
