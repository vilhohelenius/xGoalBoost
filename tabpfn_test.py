"""TabPFN vs XGBoost samalla treeniotoksella. Train 2023-24 (otos), testi 2025 (otos)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import os
from train import load, NUMERIC, EXTRA, CATEGORICAL, TARGET, make_xgb, prep

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 15_000
df = load()
feats = NUMERIC + EXTRA + CATEGORICAL
tr = df[df.season.isin([2023, 2024])].sample(N_TRAIN, random_state=0)
te = df[df.season == 2025].sample(N_TEST, random_state=0)
# TabPFN: kategoriat numeerisiksi koodeiksi
allc = pd.concat([tr, te])
def enc(d):
    X = d[feats].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=sorted(allc[c].unique())).codes
    return X.astype(float)
Xtr, Xte = enc(tr), enc(te)

# XGBoost samalla otoksella ja koko datalla
for label, trd in [(f"XGB n_train={N_TRAIN}", tr), ("XGB full 2023-24", df[df.season.isin([2023, 2024])])]:
    x = make_xgb(CATEGORICAL)
    A, B = prep(trd, feats, "xgb"), prep(te, feats, "xgb")
    for c in CATEGORICAL:
        cats = pd.api.types.union_categoricals([A[c], B[c]]).categories
        A[c] = A[c].cat.set_categories(cats); B[c] = B[c].cat.set_categories(cats)
    x.fit(A, trd[TARGET]); q = x.predict_proba(B)[:, 1]
    print(f"{label}: AUC={roc_auc_score(te[TARGET],q):.4f} logloss={log_loss(te[TARGET],q):.4f}")

# TabPFN (pilvi) ajetaan XGB:n jälkeen: kirjastojen OpenMP-ristiriita aiheuttaa muuten segfaultin
import tabpfn_client
from tabpfn_client import TabPFNClassifier
tabpfn_client.set_access_token(os.environ['TABPFN_TOKEN'])
t = time.time()
m = TabPFNClassifier(categorical_features_indices=[feats.index(c) for c in CATEGORICAL])
m.fit(Xtr, tr[TARGET])
p = np.concatenate([m.predict_proba(Xte.iloc[i:i+10000])[:, 1] for i in range(0, len(Xte), 10000)])
print(f"TabPFN n_train={N_TRAIN}: AUC={roc_auc_score(te[TARGET],p):.4f} logloss={log_loss(te[TARGET],p):.4f} ({time.time()-t:.0f}s)")

