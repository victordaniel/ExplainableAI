# Audit verdict instability across equally accurate models

Code, prespecified protocol and per-model results for:

> T. Jhansi Rani, G. Victor Daniel, Vinnakota Saran Chaitanya
> "One Model, One Verdict? Instability of Fairness and Explanation Audits Across Equally
> Accurate Models" (manuscript submitted to *IEEE Access*).

The study asks how often a standard fairness-and-explanation audit (four-fifths rule,
equal-opportunity gap, SHAP reliance on the protected attribute) changes its verdict across
models of equal accuracy (the Rashomon set), how many models an auditor must examine, and which
development choices drive the instability. Four public data sets, 54 candidate models per
division, ten divisions.

The analysis plan was fixed before any experiment: see [`PROTOCOL.md`](PROTOCOL.md)
(one amendment, logged there, concerning only the Shapley-value computation).

## Data (not redistributed)

Place these files in `data/`:

| Data set | Source | File(s) |
|---|---|---|
| Adult | UCI ML Repository, doi:10.24432/C5XW20 | `adult.data`, `adult.test` from <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/> |
| German Credit | UCI ML Repository, doi:10.24432/C5NC77 | `german.data` from <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/> |
| COMPAS | ProPublica | `compas-scores-two-years.csv` from <https://github.com/propublica/compas-analysis> |
| ACS Income (California, 2018) | US Census via `folktables` | downloaded automatically on first run |

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py          # 4 data sets x 10 divisions x 54 models (~90 min on 4 CPU cores)
python -m audit.analyze    # prespecified analyses -> tables/*.csv
python -m audit.report     # LaTeX tables and figures -> paper/generated, paper/figures
```

The stored outputs of `run_all.py` are included in `results/`, so `audit.analyze` and
`audit.report` reproduce every table and figure of the paper without rerunning the models.

## Contents

- `audit/data.py` dataset loaders; `audit/run.py` one (data set, division) unit;
  `audit/shapley.py` vectorised permutation-sampling estimator of interventional Shapley values;
  `audit/analyze.py` prespecified analyses; `audit/report.py` tables and figures.
- `results/` per model: metrics and audit statistics (`*_models.csv`), mean |SHAP| per variable
  (`*_shap.csv`), and audit-part decisions with outcome and group labels (`*_dec.npz`).
- `tables/` analysis outputs used in the paper.

Random seeds: divisions use seed `1000 + r` (r = 0..9); model seeds 0-4; ACS subsample seed 2026;
SHAP estimator seeds 0 (main) and 1 (repeat check); bootstrap seed 7.

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the paper and this archive (see `CITATION.cff`).
The data sets remain under their original terms.
