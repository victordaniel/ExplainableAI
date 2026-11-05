"""Prespecified analyses (PROTOCOL.md). Reads results/, writes tables/*.csv and tables/*.tex."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / 'results', ROOT / 'tables'
ATTR = {'adult': 'sex', 'acs_income': 'SEX', 'compas': 'race', 'german': 'age'}
NAMES = {'adult': 'Adult', 'acs_income': 'ACS Income', 'compas': 'COMPAS', 'german': 'German Credit'}
EPS = (0.005, 0.01, 0.02)
B = 2000
RNG = np.random.default_rng(7)


def load():
    M = pd.concat([pd.read_csv(f) for f in sorted(RES.glob('*_models.csv'))], ignore_index=True)
    S = pd.concat([pd.read_csv(f) for f in sorted(RES.glob('*_shap.csv'))], ignore_index=True)
    return M, S


def shap_stats(S):
    """Per (dataset, resample, model, explainer_seed): rank of A, its share, top-5 set."""
    out = []
    for _, row in S.iterrows():
        imp = {k[5:]: v for k, v in row.items() if k.startswith('imp::') and pd.notna(v)}
        order = sorted(imp, key=lambda k: -imp[k])
        a = ATTR[row['dataset']]
        out.append(dict(dataset=row['dataset'], resample=row['resample'], model=row['model'],
                        explainer_seed=row['explainer_seed'], attr_rank=order.index(a) + 1,
                        attr_share=imp[a] / sum(imp.values()), top5=tuple(sorted(order[:5])),
                        top3=tuple(sorted(order[:3]))))
    out = pd.DataFrame(out)
    for c in ('resample', 'model', 'explainer_seed', 'attr_rank'):
        out[c] = out[c].astype(int)
    return out


def add_checks(D, tag='prev', eo_cut=0.05, topk=5):
    D = D.copy()
    D['C1'] = (D[f'dir_{tag}'] < 0.8).astype(int)
    D['C2'] = (D[f'eo_{tag}'] > eo_cut).astype(int)
    D['C3'] = (D['attr_rank'] <= topk).astype(int)
    D['ALL'] = ((D.C1 + D.C2 + D.C3) > 0).astype(int)
    return D


def majority_agreement(v, m, draws=2000):
    """P(majority of m models drawn without replacement == full-set majority). Ties = disagreement."""
    n, p = len(v), v.mean()
    if p == 0.5:
        return np.nan
    full = int(p > 0.5)
    if m >= n:
        return 1.0
    k = np.array([v[RNG.choice(n, m, replace=False)].sum() for _ in range(draws)])
    maj = np.where(2 * k > m, 1, np.where(2 * k < m, 0, -1))
    return float((maj == full).mean())


def boot_mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = RNG.choice(x, (B, len(x)), replace=True).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def nested_ss(D, val):
    """Exact nested decomposition: resample > family > config > seed (residual)."""
    D = D[['resample', 'family', 'config', val]].dropna()
    g = D[val].mean()
    tot = ((D[val] - g) ** 2).sum()
    m_r = D.groupby('resample')[val].transform('mean')
    m_f = D.groupby(['resample', 'family'])[val].transform('mean')
    m_c = D.groupby(['resample', 'family', 'config'])[val].transform('mean')
    parts = {'resample': ((m_r - g) ** 2).sum(), 'family': ((m_f - m_r) ** 2).sum(),
             'config': ((m_c - m_f) ** 2).sum(), 'seed': ((D[val] - m_c) ** 2).sum()}
    return {k: v / tot if tot > 0 else np.nan for k, v in parts.items()}


def main():
    TAB.mkdir(exist_ok=True)
    M, S = load()
    SS = shap_stats(S)
    main_shap = SS[SS.explainer_seed == 0].drop(columns='explainer_seed')
    M['best'] = M.groupby(['dataset', 'resample']).val_auc.transform('max')
    MS = M.merge(main_shap, on=['dataset', 'resample', 'model'], how='left')

    # ---------- primary + sensitivity: verdict instability
    rows, curves = [], []
    settings = [('primary', 0.01, 'prev', 0.05, 5)] + \
               [(f'eps={e}', e, 'prev', 0.05, 5) for e in (0.005, 0.02)] + \
               [('threshold=0.5', 0.01, 'half', 0.05, 5), ('EO cut=0.10', 0.01, 'prev', 0.10, 5),
                ('top-3', 0.01, 'prev', 0.05, 3)]
    for name, eps, tag, cut, k in settings:
        R = MS[MS.val_auc >= MS.best - eps - 1e-12]
        R = add_checks(R, tag, cut, k)
        for ds in NAMES:
            for chk in ('C1', 'C2', 'C3', 'ALL'):
                per = []
                for r, g in R[R.dataset == ds].groupby('resample'):
                    v = g[chk].to_numpy()
                    p = v.mean()
                    per.append(dict(r=r, n=len(v), p=p, mixed=float(0 < p < 1),
                                    rel=max(p, 1 - p)))
                    if name == 'primary':
                        for m in range(1, 26):
                            curves.append(dict(dataset=ds, check=chk, resample=r, m=m,
                                               agree=majority_agreement(v, m)))
                per = pd.DataFrame(per)
                mx, ml, mh = boot_mean(per.mixed)
                rl, rll, rlh = boot_mean(per.rel)
                rows.append(dict(setting=name, dataset=ds, check=chk, n_resamples=len(per),
                                 rashomon_size=per.n.mean(), fail_share=per.p.mean(),
                                 mixed=mx, mixed_lo=ml, mixed_hi=mh,
                                 reliability=rl, rel_lo=rll, rel_hi=rlh))
    inst = pd.DataFrame(rows)
    inst.to_csv(TAB / 'instability.csv', index=False)

    C = pd.DataFrame(curves)
    curve = C.groupby(['dataset', 'check', 'm']).agree.mean().reset_index()
    curve.to_csv(TAB / 'agreement_curve.csv', index=False)
    mstar = []
    for (ds, chk), g in curve.groupby(['dataset', 'check']):
        ok = g[g.agree >= 0.95].m
        # divisions entering m*: those whose full set has a majority verdict (exact ties excluded)
        first = C[(C.dataset == ds) & (C.check == chk) & (C.m == 1)]
        mstar.append(dict(dataset=ds, check=chk, m_star=int(ok.min()) if len(ok) else np.nan,
                          agree_m1=float(g[g.m == 1].agree.iloc[0]),
                          n_div=int(first.agree.notna().sum()), n_tied=int(first.agree.isna().sum())))
    pd.DataFrame(mstar).to_csv(TAB / 'auditor_m.csv', index=False)

    # ---------- Rashomon set description
    R01 = MS[MS.val_auc >= MS.best - 0.01 - 1e-12]
    desc = []
    for ds in NAMES:
        g = R01[R01.dataset == ds]
        comp = g.groupby('resample').family.value_counts().unstack(fill_value=0).mean()
        desc.append(dict(dataset=ds, n=len(pd.read_csv(next(RES.glob(f'{ds}_r*_models.csv'))))
                         if False else None,
                         best_val_auc=M[M.dataset == ds].groupby('resample').val_auc.max().mean(),
                         size=g.groupby('resample').size().mean(),
                         **{f'n_{f}': comp.get(f, 0.0) for f in ('LR', 'RF', 'XGB', 'MLP')},
                         audit_auc_min=g.groupby('resample').audit_auc.min().mean(),
                         audit_auc_max=g.groupby('resample').audit_auc.max().mean(),
                         dir_min=g.groupby('resample').dir_prev.min().mean(),
                         dir_max=g.groupby('resample').dir_prev.max().mean(),
                         eo_min=g.groupby('resample').eo_prev.min().mean(),
                         eo_max=g.groupby('resample').eo_prev.max().mean(),
                         rank_min=g.groupby('resample').attr_rank.min().mean(),
                         rank_max=g.groupby('resample').attr_rank.max().mean()))
    pd.DataFrame(desc).drop(columns='n').to_csv(TAB / 'rashomon_desc.csv', index=False)

    # ---------- decision ambiguity and explanation disagreement
    amb = []
    for ds in NAMES:
        for r, g in R01[R01.dataset == ds].groupby('resample'):
            Z = np.load(RES / f'{ds}_r{r}_dec.npz')
            dec = np.stack([Z[f'prev_{j}'] for j in g.model])
            ambig = (dec.min(0) != dec.max(0)).mean()
            sets = list(g.top5)
            jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(sets, 2)]
            amb.append(dict(dataset=ds, resample=r, ambiguity=ambig,
                            jaccard_top5=np.mean(jac) if jac else np.nan))
    amb = pd.DataFrame(amb)
    amb_s = amb.groupby('dataset').agg(['mean', 'min', 'max'])
    amb_s.columns = ['_'.join(c) for c in amb_s.columns]
    amb_s.drop(columns=[c for c in amb_s.columns if c.startswith('resample')]).to_csv(TAB / 'ambiguity.csv')

    # ---------- SHAP estimation-noise control
    ctrl = []
    for (ds, r, j), g in SS.groupby(['dataset', 'resample', 'model']):
        if set(g.explainer_seed) >= {0, 1}:
            a, b = g[g.explainer_seed == 0].iloc[0], g[g.explainer_seed == 1].iloc[0]
            ctrl.append(dict(dataset=ds, resample=r, rank_diff=abs(a.attr_rank - b.attr_rank),
                             c3_agree=int((a.attr_rank <= 5) == (b.attr_rank <= 5)),
                             jaccard=len(set(a.top5) & set(b.top5)) / len(set(a.top5) | set(b.top5))))
    pd.DataFrame(ctrl).groupby('dataset').mean(numeric_only=True).drop(columns='resample') \
        .to_csv(TAB / 'shap_noise_control.csv')

    # ---------- variance decomposition
    vd = []
    for ds in NAMES:
        full = M[M.dataset == ds]
        for v in ('dir_prev', 'eo_prev', 'audit_auc'):
            vd.append(dict(dataset=ds, pool='full', stat=v, **nested_ss(full, v)))
        R = R01[R01.dataset == ds]
        for v in ('dir_prev', 'eo_prev', 'attr_rank', 'attr_share'):
            vd.append(dict(dataset=ds, pool='rashomon', stat=v, **nested_ss(R, v)))
    pd.DataFrame(vd).to_csv(TAB / 'variance_decomp.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
