"""One (dataset, resample) unit of the prespecified protocol (PROTOCOL.md).
Writes results/<ds>_r<r>_models.csv, _shap.csv, _dec.npz."""
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from audit.data import LOADERS

warnings.filterwarnings('ignore')
OUT = Path(__file__).resolve().parents[1] / 'results'
SEEDS = range(5)
EPS_SHAP = 0.02
N_SHAP, N_BG = 200, 50


def model_pool():
    pool = []
    for C in (0.01, 0.1, 1, 10):
        pool.append(('LR', f'C={C}', 0, lambda s, C=C: LogisticRegression(C=C, max_iter=2000)))
    for md in (8, None):
        for msl in (1, 5):
            for s in SEEDS:
                pool.append(('RF', f'depth={md},leaf={msl}', s,
                             lambda s, md=md, msl=msl: RandomForestClassifier(
                                 200, max_depth=md, min_samples_leaf=msl, n_jobs=1, random_state=s)))
    for md in (3, 6):
        for eta in (0.05, 0.1):
            for s in SEEDS:
                pool.append(('XGB', f'depth={md},eta={eta}', s,
                             lambda s, md=md, eta=eta: XGBClassifier(
                                 n_estimators=300, max_depth=md, learning_rate=eta, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=1, random_state=s, verbosity=0)))
    for h in ((64,), (64, 32)):
        for s in SEEDS:
            pool.append(('MLP', f'h={h}', s,
                         lambda s, h=h: MLPClassifier(h, alpha=1e-3, early_stopping=True,
                                                      max_iter=300, random_state=s)))
    return pool


def fairness(dec, y, A):
    """dec, y: 1 = favourable; A: 1 = unprivileged."""
    pu, pp = dec[A == 1].mean(), dec[A == 0].mean()
    dir_ = pu / pp if pp > 0 else np.nan
    tu = dec[(A == 1) & (y == 1)].mean()
    tp = dec[(A == 0) & (y == 1)].mean()
    return dir_, abs(tu - tp)


def run(ds, r):
    t0 = time.time()
    X, y, A, cat, num, attr = LOADERS[ds]()
    cols = list(X.columns)
    Xc = X.copy()
    for c in cat:                                  # integer codes (lookup only, no statistics)
        Xc[c] = pd.Categorical(Xc[c]).codes.astype(float)
    Xc = Xc.to_numpy(float)
    cat_i = [cols.index(c) for c in cat]
    num_i = [cols.index(c) for c in num]
    idx = np.arange(len(y))
    seed = 1000 + r
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, au = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    rng = np.random.default_rng(seed)
    shap_pts = np.sort(rng.choice(au, min(N_SHAP, len(au)), replace=False))
    bg_pts = np.sort(rng.choice(tr, N_BG, replace=False))
    p_train = y[tr].mean()

    rows, decs, fitted = [], {}, []
    for j, (fam, cfg, s, make) in enumerate(model_pool()):
        pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_i),
                                 ('num', StandardScaler(), num_i)])
        m = Pipeline([('pre', pre), ('clf', make(s))]).fit(Xc[tr], y[tr])
        sv = m.predict_proba(Xc[va])[:, 1]
        sa = m.predict_proba(Xc[au])[:, 1]
        t_prev = np.quantile(sv, 1 - p_train)
        row = dict(dataset=ds, resample=r, model=j, family=fam, config=cfg, seed=s,
                   val_auc=roc_auc_score(y[va], sv), audit_auc=roc_auc_score(y[au], sa),
                   t_prev=t_prev)
        for tag, t in (('prev', t_prev), ('half', 0.5)):
            d = (sa >= t).astype(int)
            dir_, eo = fairness(d, y[au], A[au])
            row.update({f'dir_{tag}': dir_, f'eo_{tag}': eo, f'acc_{tag}': (d == y[au]).mean(),
                        f'favrate_{tag}': d.mean()})
            decs[f'{tag}_{j}'] = d.astype(np.int8)
        rows.append(row)
        fitted.append(m)
    res = pd.DataFrame(rows)
    best = res.val_auc.max()
    res['in_shap_set'] = res.val_auc >= best - EPS_SHAP - 1e-12

    from audit.shapley import shapley
    shap_rows = []
    best_j = int(res.val_auc.idxmax())
    jobs = [(j, 0) for j in res.index[res.in_shap_set]] + [(best_j, 1)]   # (model, explainer seed)
    for j, es in jobs:
        f = lambda Z, m=fitted[j]: m.predict_proba(Z)[:, 1]
        sv = shapley(f, Xc[shap_pts], Xc[bg_pts], n_perm=4, seed=es)
        imp = np.abs(sv).mean(0)
        shap_rows.append(dict(dataset=ds, resample=r, model=j, explainer_seed=es,
                              **{f'imp::{c}': v for c, v in zip(cols, imp)}))
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / f'{ds}_r{r}_models.csv', index=False)
    pd.DataFrame(shap_rows).to_csv(OUT / f'{ds}_r{r}_shap.csv', index=False)
    np.savez_compressed(OUT / f'{ds}_r{r}_dec.npz', y=y[au], A=A[au], **decs)
    return ds, r, time.time() - t0, int(res.in_shap_set.sum())


if __name__ == '__main__':
    print(run(sys.argv[1], int(sys.argv[2])))
