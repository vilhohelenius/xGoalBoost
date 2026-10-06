"""Oppimiskäyrä + TabPFN-ensemble.

Train: kaudet 2023-24, testi: sama 30k-otos kaudelta 2025 kuin compare_tabpfn.py.
  1. XGBoost ja TabPFN-3.5 kasvavilla treenikoilla (10k..100k) + XGBoost koko datalla
  2. TabPFN-ensemble: K erillistä 100k-otosta, ennusteiden keskiarvo
Tulokset -> results.json ja learning_curve.png

Ajo: .venv/bin/python scaling_experiment.py   (TABPFN_TOKEN tai tabpfn_client-kirjautuminen)
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from train import CATEGORICAL, EXTRA, NUMERIC, TARGET, load, make_xgb, prep

SIZES = [10_000, 25_000, 50_000, 100_000]
N_TEST, K_ENSEMBLE, ENS_SIZE = 30_000, 3, 100_000

df = load()
feats = NUMERIC + EXTRA + CATEGORICAL
train_pool = df[df.season.isin([2023, 2024])]
te = df[df.season == 2025].sample(N_TEST, random_state=0)
y = te[TARGET].to_numpy()


def score(p):
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}


def xgb_pred(trd):
    x = make_xgb(CATEGORICAL)
    A, B = prep(trd, feats, "xgb"), prep(te, feats, "xgb")
    for c in CATEGORICAL:
        cats = pd.api.types.union_categoricals([A[c], B[c]]).categories
        A[c] = A[c].cat.set_categories(cats)
        B[c] = B[c].cat.set_categories(cats)
    x.fit(A, trd[TARGET])
    return x.predict_proba(B)[:, 1]


results = {"xgboost": {}, "tabpfn": {}}
# --- XGBoost ensin: se kaatuu segfaultiin, jos tabpfn_client on jo tuotu (OpenMP-ristiriita)
for n in SIZES:
    results["xgboost"][n] = score(xgb_pred(train_pool.sample(n, random_state=0)))
    print("XGB", n, results["xgboost"][n], flush=True)
results["xgboost"]["full"] = score(xgb_pred(train_pool))
print("XGB full", results["xgboost"]["full"], flush=True)

# --- TabPFN (pilvi)
import tabpfn_client
from tabpfn_client import TabPFNClassifier

if os.environ.get("TABPFN_TOKEN"):
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])

cat_idx = [feats.index(c) for c in CATEGORICAL]
allc = pd.concat([train_pool, te])


def enc(d):
    X = d[feats].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=sorted(allc[c].unique())).codes
    return X.astype(float)


Xte = enc(te)


def tabpfn_pred(trd):
    m = TabPFNClassifier(categorical_features_indices=cat_idx)
    m.fit(enc(trd), trd[TARGET])
    return np.concatenate([m.predict_proba(Xte.iloc[i:i + 10_000])[:, 1] for i in range(0, len(Xte), 10_000)])


for n in SIZES:
    results["tabpfn"][n] = score(tabpfn_pred(train_pool.sample(n, random_state=0)))
    print("TabPFN", n, results["tabpfn"][n], flush=True)

preds = []
for k in range(K_ENSEMBLE):
    preds.append(tabpfn_pred(train_pool.sample(ENS_SIZE, random_state=100 + k)))
    print(f"TabPFN ensemble member {k}", score(preds[-1]), "| ensemble so far", score(np.mean(preds, axis=0)), flush=True)
results["tabpfn_ensemble"] = {"members": K_ENSEMBLE, "size_each": ENS_SIZE, **score(np.mean(preds, axis=0))}

json.dump(results, open("results.json", "w"), indent=2)

# --- kuvaaja
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = SIZES
ax.plot(x, [results["xgboost"][n]["auc"] for n in x], color=BLUE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label="XGBoost")
ax.plot(x, [results["tabpfn"][n]["auc"] for n in x], color=ORANGE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label="TabPFN-3.5")
full = results["xgboost"]["full"]["auc"]
ax.axhline(full, color=BLUE, lw=1.2, ls=(0, (4, 3)))
ax.text(x[0], full + 0.0006, f"XGBoost, all ~240k rows: {full:.4f}", color=MUTED, fontsize=8.5, va="bottom")
ens = results["tabpfn_ensemble"]["auc"]
ax.scatter([ENS_SIZE], [ens], s=70, color=ORANGE, edgecolor=INK, linewidth=1.5, zorder=5, marker="D")
ax.annotate(f"TabPFN ensemble\n({K_ENSEMBLE}×100k): {ens:.4f}", (ENS_SIZE, ens), xytext=(-12, 14), textcoords="offset points",
            ha="right", color=INK, fontsize=8.5)
ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{n // 1000}k" for n in x])
ax.minorticks_off()
ax.set_xlabel("Training rows (seasons 2023-24)", color=MUTED, fontsize=9)
ax.set_ylabel("AUC on 30k shots from 2025", color=MUTED, fontsize=9)
ax.set_title("NHL xG: TabPFN-3.5 vs XGBoost as training data grows", loc="left", color=INK, fontsize=11, fontweight="bold")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
fig.tight_layout(); fig.savefig("learning_curve.png")
print("saved results.json, learning_curve.png")
