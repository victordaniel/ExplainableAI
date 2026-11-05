# Prespecified protocol — Audit verdict instability under model multiplicity

Version 1.0 — written 2026-11-04, before any experiment was run.
Changes after this point are recorded under "Amendments" with their reason and time.

## Question
Among models that are equally accurate on validation data (the Rashomon set), how often does
the *verdict* of a standard fairness-and-explanation audit change? How many models must an
auditor examine before the verdict is stable? Which source of variation (data resample,
algorithm family, hyperparameters, random seed) drives the changes?

## Datasets, protected attribute (A), favourable outcome
| Dataset | n | Outcome | Favourable | A (unprivileged vs privileged) |
|---|---|---|---|---|
| Adult (UCI; train+test, rows with "?" removed) | ~45k | income > 50K | > 50K | Female vs Male |
| ACS Income, California 2018 (folktables `ACSIncome`) | 20,000 stratified subsample (seed 2026) | PINCP > 50K | > 50K | Female vs Male |
| COMPAS (ProPublica two-year; standard ProPublica filters; African-American and Caucasian only) | ~5.3k | two_year_recid | no recidivism | African-American vs Caucasian |
| German Credit (UCI Statlog) | 1000 | credit risk | good | age <= 25 vs > 25 |

Adult: fnlwgt (survey weight) and education (duplicate of education-num) are dropped.
COMPAS inputs: age, sex, race, c_charge_degree, priors_count, juv_fel_count, juv_misd_count, juv_other_count.
The protected attribute is included as a model input (the audit asks whether the model relies on it).
ACS preprocessing: OCCP coarsened to its leading digit (major group), POBP to US-born (<100) vs not;
other categorical codes one-hot encoded (COW, MAR, RELP, RAC1P); AGEP, SCHL, WKHP numeric.

## Splits
R outer resamples (R = 10; may be reduced for computing time only, see Amendments). Each resample:
stratified 60% training / 20% validation / 20% audit set, seed = 1000 + r. Preprocessing (imputation,
scaling, encoding) is fitted on the training part only.

## Model pool (identical for every resample)
- Logistic regression, C in {0.01, 0.1, 1, 10} (deterministic; 4 models)
- Random forest, 200 trees, (max_depth, min_samples_leaf) in {8, None} x {1, 5}, seeds 0-4 (20)
- XGBoost, 300 rounds, subsample = colsample = 0.8, (max_depth, eta) in {3, 6} x {0.05, 0.1}, seeds 0-4 (20)
- MLP, hidden layers in {(64), (64, 32)}, alpha = 1e-3, early stopping, seeds 0-4 (10)
54 models per resample. No other tuning.

## Rashomon set
Models with validation AUROC >= (best validation AUROC in the resample) - eps.
Primary eps = 0.01; sensitivity eps in {0.005, 0.02}.

## Decision threshold
Primary: per model, the threshold on validation scores that makes the validation favourable-decision
rate equal to the favourable rate in the training labels. Sensitivity: fixed 0.5.

## Audit checks (computed on the audit set)
- C1 Disparate impact: DIR = P(fav decision | unpriv) / P(fav decision | priv). FAIL if DIR < 0.8.
- C2 Equal opportunity: |TPR_unpriv - TPR_priv|, TPR of the favourable decision among truly favourable.
  FAIL if > 0.05 (sensitivity: 0.10).
- C3 Explanation reliance: mean |SHAP| per original variable (one-hot columns summed). FAIL if A ranks
  in the top 5 variables (sensitivity: top 3).
- Overall verdict: PASS only if C1, C2 and C3 all pass.
SHAP: the same model-agnostic Permutation SHAP explainer for every family, on original variables,
200 audit points and 50 training background points fixed per resample.
Estimation-noise control: for the best validation model of every resample SHAP is recomputed with a
different explainer seed; C3 verdict agreement and the A-rank difference between the two runs are reported,
so that explanation disagreement between models can be separated from SHAP sampling noise.
SHAP is computed only for models in the eps = 0.02 set (which contains the other two).

## Outcomes
Primary: for each dataset and check, the verdict disagreement within the Rashomon set, averaged over
resamples: (i) share of resamples in which the Rashomon set contains both PASS and FAIL models;
(ii) single-model reliability = probability that one randomly chosen Rashomon model gives the
majority verdict of the set.
Secondary:
- auditor sample size: smallest m such that the majority verdict of m randomly drawn Rashomon models
  (without replacement; ties counted as disagreement) equals the full-set majority with probability >= 0.95;
- nested variance decomposition of DIR, EO gap and the SHAP rank of A into resample / family /
  configuration within family / seed (exact nested sums of squares), in the full pool and in the
  Rashomon set;
- decision ambiguity: share of audit individuals whose decision differs across Rashomon models;
- explanation disagreement: mean pairwise Jaccard similarity of top-5 SHAP variables.
Uncertainty: 95% percentile bootstrap over resamples (2000 draws).

## Amendments

1. 2026-11-04, before the main experiments (only timings of one pilot resample were inspected;
   pilot outputs deleted). For computing time, SHAP is computed with a vectorised permutation-sampling
   estimator of interventional Shapley values (`audit/shapley.py`; 4 permutations per point as two
   antithetic pairs; same 200 audit points and 50 background points) instead of shap.PermutationExplainer.
   Validation on COMPAS (LR, RF, XGB; 200 points): efficiency error < 1e-6, correlation of global
   importances with shap.ExactExplainer >= 0.9999, identical variable rankings.
