"""Rakentaa demon: ennustaa xG:n ruudukkoon jokaiselle skenaariolle ja upottaa sen demo/index.html:ään.

Skenaario = laukaustyyppi x tilanne (5v5, ylivoima, alivoima, 3v3, tyhjä maali) x rebound x rush.
Muut piirteet pidetään skenaarion tyypillisissä arvoissa (mediaani / yleisin luokka vastaavista oikeista laukauksista).
Paikka (x, y) ylikirjoitetaan, ja ennuste lasketaan symmetrisesti (y ja -y keskiarvo).

Ajo (repon juuresta): .venv/bin/python demo/build_demo.py
Tuottaa demo/index.html (täysi sivu) ja demo/fragment.html (vain sisältö, esim. Artifact-julkaisuun).
"""
import base64
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from train import CATEGORICAL, EXTRA, NUMERIC, TARGET, load, make_xgb, prep

HERE = pathlib.Path(__file__).parent
TYPES = ["WRIST", "SNAP", "SLAP", "BACK", "TIP", "DEFL", "WRAP"]
STRENGTHS = {  # nimi -> (ampujat, puolustajat) tai None = tyhjä maali
    "5v5": (5, 5), "5v4": (5, 4), "4v5": (4, 5), "3v3": (3, 3), "empty": None,
}
XS = np.arange(25, 100, 2)          # x-koordinaatti: sinisestä viivasta päätylaitaan (maali x=89)
YS = np.arange(0, 43, 2)            # |y|, peilataan
GOAL_X = 89

df = load()
feats = NUMERIC + EXTRA + CATEGORICAL
train = df[df.season.isin([2023, 2024, 2025])]
print("koulutetaan", len(train), "laukausta")
model = make_xgb(CATEGORICAL)
Xtr = prep(train, feats, "xgb")
model.fit(Xtr, train[TARGET])
cats = {c: Xtr[c].cat.categories for c in CATEGORICAL}


def subset(t, s, rebound, rush):
    d = df
    d = d[d.shotOnEmptyNet == 1] if STRENGTHS[s] is None else d[(d.shotOnEmptyNet == 0) & (d.shooterSkaters == STRENGTHS[s][0]) & (d.defenderSkaters == STRENGTHS[s][1])]
    for keep in ("full", "no_type", "no_type_rush", "no_type_rush_rebound"):
        m = d
        if keep in ("full",):
            m = m[m.shotType == t]
        if keep in ("full", "no_type"):
            m = m[m.shotRush == rush]
        if keep != "no_type_rush_rebound":
            m = m[m.shotRebound == rebound]
        if len(m) >= 40:
            return m, keep
    return d, "strength_only"


rows, index = [], []
for t in TYPES:
    for s in STRENGTHS:
        for rebound in (0, 1):
            for rush in (0, 1):
                sub, how = subset(t, s, rebound, rush)
                tmpl = sub[feats].median(numeric_only=True)
                base = {f: tmpl[f] for f in tmpl.index}
                for c in CATEGORICAL:
                    base[c] = sub[c].mode().iloc[0]
                base.update(shotType=t, shotRebound=rebound, shotRush=rush)
                if STRENGTHS[s] is None:
                    base.update(shotOnEmptyNet=1, emptyNetAgainst=1)
                else:
                    base.update(shooterSkaters=STRENGTHS[s][0], defenderSkaters=STRENGTHS[s][1], shotOnEmptyNet=0, emptyNetAgainst=0)
                    base.update(homeSkatersOnIce=STRENGTHS[s][0], awaySkatersOnIce=STRENGTHS[s][1], isHomeTeam=1)
                index.append((t, s, rebound, rush, how, len(sub)))
                for x in XS:
                    for y in YS:
                        for sign in ((1, -1) if y else (1,)):
                            dist = float(np.hypot(GOAL_X - x, y))
                            r = dict(base)
                            r.update(arenaAdjustedXCordABS=x, arenaAdjustedYCord=sign * y, arenaAdjustedYCordAbs=y,
                                     shotDistance=dist, arenaAdjustedShotDistance=round(dist),
                                     shotAngleAdjusted=min(90.0, float(np.degrees(np.arctan2(y, max(GOAL_X - x, 0.5))))))
                            rows.append(r)

X = pd.DataFrame(rows)[feats]
for c in CATEGORICAL:
    X[c] = pd.Categorical(X[c], categories=cats[c])
p = model.predict_proba(X)[:, 1]

# keskiarvo y / -y, järjestys [scenario][x][y]
per_scn = len(XS) * (2 * len(YS) - 1)
out = np.zeros((len(index), len(XS), len(YS)))
k = 0
for si in range(len(index)):
    for xi in range(len(XS)):
        for yi, y in enumerate(YS):
            if y:
                out[si, xi, yi] = (p[k] + p[k + 1]) / 2; k += 2
            else:
                out[si, xi, yi] = p[k]; k += 1
assert k == len(p)
q = np.round(out * 1000).astype("<u2")

def get(t, s, r, u):
    return next(i for i, e in enumerate(index) if e[:4] == (t, s, r, u))
print("Tarkistus (xG):")
for t, s, r, u, x, y in [("WRIST", "5v5", 0, 0, 85, 0), ("WRIST", "5v5", 0, 0, 55, 20), ("WRIST", "5v5", 0, 0, 40, 0), ("TIP", "5v5", 0, 0, 85, 0),
                         ("WRIST", "5v5", 1, 0, 85, 0), ("WRIST", "5v4", 0, 0, 79, 10), ("WRIST", "empty", 0, 0, 60, 0), ("SLAP", "5v5", 0, 0, 60, 10)]:
    print(f"  {t} {s} reb={r} rush={u} x={x} y={y}: {out[get(t, s, r, u), list(XS).index(x) if x in XS else int((x - 25) // 2), y // 2]:.3f}")

weak = [e for e in index if e[4] != "full"]
print(f"{len(weak)}/{len(index)} skenaariota käyttää väljempää pohjaa (liian vähän vastaavia laukauksia)")
meta = {
    "types": TYPES, "strengths": list(STRENGTHS), "xs": XS.tolist(), "ys": YS.tolist(), "goalX": GOAL_X,
    "data": base64.b64encode(q.tobytes()).decode(),
    "base": float(train[TARGET].mean()), "shots": int(len(train)),
    "thin": [[e[0], e[1], e[2], e[3]] for e in weak],
    "cv_auc": 0.806,
}
payload = json.dumps(meta, separators=(",", ":"))
tmpl = (HERE / "template.html").read_text()
fragment = tmpl.replace("/*__DATA__*/null", payload)
head, body = fragment.split("<!--BODY-->")
(HERE / "fragment.html").write_text(fragment.replace("<!--BODY-->", ""))
(HERE / "index.html").write_text(
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
    + head + "</head><body>" + body + "</body></html>\n")
print("kirjoitettu", HERE / "index.html", f"({len(payload) / 1024:.0f} kB data)")
