"""Builds paper/generated/*.tex tables and paper/figures/*.pdf from tables/*.csv (made by analyze.py)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB, GEN, FIG = ROOT / 'tables', ROOT / 'paper' / 'generated', ROOT / 'paper' / 'figures'
NAMES = {'adult': 'Adult', 'acs_income': 'ACS Income', 'compas': 'COMPAS', 'german': 'German Credit'}
CHK = {'C1': 'C1 disparate impact', 'C2': 'C2 equal opportunity', 'C3': 'C3 attribution', 'ALL': 'Overall'}
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # validated categorical slots 1-4
MARK = ['o', 's', '^', 'D']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],   # match the paper's Times
                     'mathtext.fontset': 'stix', 'font.size': 8, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK,             # black label text
                     'pdf.fonttype': 42,                                           # TrueType, not Type 3 (IEEE PDF eXpress)
                     'axes.spines.top': False, 'axes.spines.right': False})


def f(x, d=2):
    return '--' if pd.isna(x) else f'{x:.{d}f}'


def table_instability():
    I = pd.read_csv(TAB / 'instability.csv')
    A = pd.read_csv(TAB / 'auditor_m.csv')
    P = I[I.setting == 'primary']
    lines = [r'\begin{tabular}{llrrcc r}', r'\toprule',
             r'Data set & Test & $|\mathcal{R}|$ & Fail & Mixed (95\% CI) & Reliability (95\% CI) & $m^\ast$\\',
             r'\midrule']
    for ds, nm in NAMES.items():
        for i, (c, cn) in enumerate(CHK.items()):
            r = P[(P.dataset == ds) & (P.check == c)].iloc[0]
            a = A[(A.dataset == ds) & (A.check == c)].iloc[0]
            m = a.m_star
            ms = '$>$25' if pd.isna(m) else str(int(m))
            if a.n_tied > 0:                                   # some divisions tied: m* over fewer divisions
                ms += f'$^{{\\dagger}}$'
            lines.append(f"{nm if i == 0 else ''} & {cn.split(' ')[0]} & {f(r.rashomon_size, 1) if i == 0 else ''} & "
                         f"{f(r.fail_share)} & {f(r.mixed)} ({f(r.mixed_lo)}--{f(r.mixed_hi)}) & "
                         f"{f(r.reliability)} ({f(r.rel_lo)}--{f(r.rel_hi)}) & "
                         f"{ms}\\\\")
        lines.append(r'\addlinespace')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (GEN / 'instability.tex').write_text('\n'.join(lines))


def table_sensitivity():
    I = pd.read_csv(TAB / 'instability.csv')
    I = I[I.check == 'ALL']
    sets = list(dict.fromkeys(I.setting))
    lines = [r'\begin{tabular}{l' + 'c' * len(NAMES) + '}', r'\toprule',
             'Setting & ' + ' & '.join(NAMES.values()) + r'\\', r'\midrule']
    for s in sets:
        cells = []
        for ds in NAMES:
            r = I[(I.setting == s) & (I.dataset == ds)].iloc[0]
            cells.append(f'{f(r.mixed)} / {f(r.reliability)}')
        lab = {'primary': 'Principal', 'eps=0.005': r'$\varepsilon=0.005$', 'eps=0.02': r'$\varepsilon=0.02$',
               'threshold=0.5': 'Threshold 0.5', 'EO cut=0.10': r'$\Delta_{\mathrm{EO}}$ limit 0.10',
               'top-3': 'C3: top 3'}[s]
        lines.append(f'{lab} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'sensitivity.tex').write_text('\n'.join(lines))


def table_desc():
    D = pd.read_csv(TAB / 'rashomon_desc.csv').set_index('dataset')
    Amb = pd.read_csv(TAB / 'ambiguity.csv').set_index('dataset')
    N = pd.read_csv(TAB / 'shap_noise_control.csv').set_index('dataset')
    lines = [r'\begin{tabular}{l' + 'c' * len(NAMES) + '}', r'\toprule',
             ' & ' + ' & '.join(NAMES.values()) + r'\\', r'\midrule']
    def row(label, fn):
        lines.append(label + ' & ' + ' & '.join(fn(ds) for ds in NAMES) + r'\\')
    row('Best validation AUROC', lambda d: f(D.loc[d, 'best_val_auc'], 3))
    row('Rashomon set size', lambda d: f(D.loc[d, 'size'], 1))
    row('\\quad LR / RF / XGB / MLP', lambda d: '/'.join(f(D.loc[d, f'n_{k}'], 1) for k in ('LR', 'RF', 'XGB', 'MLP')))
    row('Audit AUROC, range', lambda d: f"{f(D.loc[d, 'audit_auc_min'], 3)}--{f(D.loc[d, 'audit_auc_max'], 3)}")
    row('DIR, range', lambda d: f"{f(D.loc[d, 'dir_min'])}--{f(D.loc[d, 'dir_max'])}")
    row('$\\Delta_{\\mathrm{EO}}$, range', lambda d: f"{f(D.loc[d, 'eo_min'], 3)}--{f(D.loc[d, 'eo_max'], 3)}")
    row('Rank of $A$, range', lambda d: f"{f(D.loc[d, 'rank_min'], 1)}--{f(D.loc[d, 'rank_max'], 1)}")
    row('Ambiguity', lambda d: f(Amb.loc[d, 'ambiguity_mean'], 3))
    row('Top-5 Jaccard (models)', lambda d: f(Amb.loc[d, 'jaccard_top5_mean']))
    row('Top-5 Jaccard (estimator)', lambda d: f(N.loc[d, 'jaccard']))
    row('C3 agreement (estimator)', lambda d: f(N.loc[d, 'c3_agree']))
    row('$|\\Delta$rank$|$ of $A$ (estimator)', lambda d: f(N.loc[d, 'rank_diff']))
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'description.tex').write_text('\n'.join(lines))


def fig_agreement():
    C = pd.read_csv(TAB / 'agreement_curve.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, (ds, nm) in zip(axes.flat, NAMES.items()):
        ax.axhline(0.95, color=INK2, lw=0.8, ls=(0, (3, 3)))
        for i, (c, cn) in enumerate(CHK.items()):
            g = C[(C.dataset == ds) & (C.check == c) & (C.m % 2 == 1)].sort_values('m')   # odd m: no ties
            ax.plot(g.m, g.agree, color=SERIES[i], lw=1.6, marker=MARK[i], ms=4, markevery=2,
                    markeredgecolor=SURF, markeredgewidth=0.8, label=cn)
        ax.set_title(nm, fontsize=9, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_ylim(0.45, 1.02)
    for ax in axes[1]:
        ax.set_xlabel('Number of models audited, $m$')
    for ax in axes[:, 0]:
        ax.set_ylabel('P(majority verdict = set verdict)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / 'agreement.pdf')
    plt.close(fig)


def fig_variance():
    V = pd.read_csv(TAB / 'variance_decomp.csv')
    V = V[V.pool == 'rashomon']
    stat_names = {'dir_prev': 'DIR', 'eo_prev': r'$\Delta_{\mathrm{EO}}$', 'attr_rank': 'Rank of $A$',
                  'attr_share': 'Share of $A$'}
    comps = ['resample', 'family', 'config', 'seed']
    comp_lab = ['Data resample', 'Algorithm family', 'Hyperparameters', 'Random seed']
    labels, rows = [], []
    for ds, nm in NAMES.items():
        for s, sn in stat_names.items():
            r = V[(V.dataset == ds) & (V.stat == s)]
            if len(r):
                labels.append(f'{nm}: {sn}')
                rows.append([r.iloc[0][c] for c in comps])
    R = np.nan_to_num(np.array(rows))
    fig, ax = plt.subplots(figsize=(3.36, 0.22 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for i, c in enumerate(comps):
        ax.barh(y, R[:, i], left=left, height=0.7, color=SERIES[i], edgecolor=SURF, linewidth=1.0,
                label=comp_lab[i])
        left += R[:, i]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Share of total sum of squares', fontsize=8)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8,
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / 'variance.pdf')
    plt.close(fig)


def table_variance():
    V = pd.read_csv(TAB / 'variance_decomp.csv')
    lines = [r'\begin{tabular}{lllcccc}', r'\toprule',
             r'Data set & Pool & Statistic & Resample & Family & Config. & Seed\\', r'\midrule']
    sn = {'dir_prev': 'DIR', 'eo_prev': r'$\Delta_{\mathrm{EO}}$', 'audit_auc': 'Audit AUROC',
          'attr_rank': 'Rank of $A$', 'attr_share': 'Share of $A$'}
    for ds, nm in NAMES.items():
        first = True
        for _, r in V[V.dataset == ds].iterrows():
            lines.append(f"{nm if first else ''} & {r.pool} & {sn[r.stat]} & {f(r['resample'])} & {f(r['family'])} & "
                         f"{f(r['config'])} & {f(r['seed'])}\\\\")
            first = False
        lines.append(r'\addlinespace')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (GEN / 'variance.tex').write_text('\n'.join(lines))


if __name__ == '__main__':
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table_instability(); table_sensitivity(); table_desc(); table_variance()
    fig_agreement(); fig_variance()
    print('report done')
