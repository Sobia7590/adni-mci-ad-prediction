"""
04_improved_analysis.py

Follow-up to 03_modeling.ipynb. Addresses the gaps flagged in the paper-readiness
review of Direction A:

  FIX 1  Censoring-aware target definition (was: "ever converts", no follow-up floor)
  FIX 2  Imputation method made consistent with the proposal deck (KNN, not median)
  FIX 3  Repeated Stratified CV + nested hyperparameter search (was: single 80/20 split, default params)
  FIX 4  Bootstrap 95% CI on AUC (was: point estimate only)
  FIX 5  APOE4 x feature interaction tested in one model (was: post-hoc subgroup split, n=65/70)
  FIX 6  Feature-group ablation, to justify the "multimodal" claim
  FIX 7  Calibration curve + Brier score (was: AUC only, no clinical-risk calibration check)
  FIX 8  Cox proportional-hazards template (optional, fully removes censoring bias)

Run as Jupyter cells (the "# %%" markers work in VS Code / Spyder / Jupytext) or
top to bottom as a plain script. Uses the same DATA path and column names as your
existing notebooks -- no changes needed there.

Needs: pip install lifelines statsmodels   (sklearn/xgboost/lightgbm you already have)
"""

# %% Setup
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, GridSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

DATA = r'D:\DATA-adni\ADNIMERGE_CSVs'
mci = pd.read_csv(os.path.join(DATA, 'mci_with_labels.csv'))
print(f"Loaded: {mci.shape}")


# %% ------------------------------------------------------------------------
# FIX 1: Censoring-aware target definition
# -----------------------------------------------------------------------------
# Problem: the original CONVERTER column labels a subject "0" (non-converter)
# even if they were only followed for 6 months. A subject followed 6 months
# and a subject followed 10 years with no conversion get the same label, but
# the second is much stronger evidence of actually not converting. This biases
# the negative class toward under-observed subjects (right-censoring bias).
#
# Fix: pick a fixed prediction horizon (24mo, matching your existing CONV_24M
# column) and only keep subjects whose label is trustworthy at that horizon:
#   - converters who converted BY the horizon -> label 1
#   - non-converters who were FOLLOWED AT LEAST the horizon -> label 0
#   - everyone else (converted after horizon, or dropped out before horizon
#     with no conversion) is genuinely ambiguous -> excluded, not guessed at

HORIZON = 24  # months; rerun with 36 as a robustness check

def censoring_safe_subset(df, horizon):
    converters = df[(df['CONVERTER'] == 1) & (df['CONV_MONTH'] <= horizon)]
    non_converters = df[(df['CONVERTER'] == 0) & (df['LAST_VISIT_MONTH'] >= horizon)]
    out = pd.concat([converters, non_converters]).copy()
    out['CONV_LABEL'] = out['RID'].isin(converters['RID']).astype(int)
    return out

mci_24 = censoring_safe_subset(mci, HORIZON)
print(f"Original:        {len(mci)} subjects, {mci['CONVERTER'].mean()*100:.1f}% converters")
print(f"Censoring-safe:  {len(mci_24)} subjects, {mci_24['CONV_LABEL'].mean()*100:.1f}% converters "
      f"(dropped {len(mci) - len(mci_24)} with ambiguous follow-up)")


# %% Feature matrix (same feature set as 03_modeling.ipynb, corrected label)
COGNITIVE = ['MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTAL13']
GENETIC   = ['APOE4', 'APOE4_count']
MRI       = ['ST29SV', 'ST88SV', 'ST40TS', 'ST99TS', 'ST101SV', 'HIPPO_TOTAL', 'HIPPO_ICV']
DEMOG     = ['PTEDUCAT']

mci_24['SEX'] = (mci_24['PTGENDER'] == 'Male').astype(float) if 'PTGENDER' in mci_24.columns else np.nan
DEMOG += ['SEX']

ALL_FEATURES = COGNITIVE + GENETIC + MRI + DEMOG
available = [f for f in ALL_FEATURES if f in mci_24.columns]

X = mci_24[available].values
y = mci_24['CONV_LABEL'].values
print(f"X: {X.shape}, positive rate: {y.mean()*100:.1f}%")


# %% ------------------------------------------------------------------------
# FIX 2: Imputation consistency
# -----------------------------------------------------------------------------
# Your proposal deck states "KNN imputation" but 03_modeling.ipynb actually
# ran SimpleImputer(strategy='median'). Made the code match the claimed
# method here. If you'd rather keep median (simpler, defensible, and what you
# actually validated), swap KNNImputer(n_neighbors=5) back to
# SimpleImputer(strategy='median') below AND update the deck to say "median" --
# either is fine, they just need to match.

def make_pipe(model):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])


# %% ------------------------------------------------------------------------
# FIX 3: Repeated Stratified CV + nested hyperparameter search
# -----------------------------------------------------------------------------
# Problem: a single 80/20 split gives one noisy AUC number, and all models
# used default hyperparameters (C=1.0, n_estimators=200, ...) with no tuning.
# Fix: nested CV -- inner loop tunes hyperparameters, outer loop (5x5 repeated
# stratified folds = 25 AUC estimates) gives an honest, low-variance estimate
# of generalization performance.

outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grids = {
    'Logistic Regression': (
        LogisticRegression(max_iter=2000, random_state=42),
        {'model__C': [0.01, 0.1, 1, 10]}),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'model__n_estimators': [200, 400], 'model__max_depth': [3, 5, None]}),
    'XGBoost': (
        xgb.XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
        {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5],
         'model__learning_rate': [0.01, 0.05, 0.1]}),
}

results = {}
print("\nNested CV results (this will take a few minutes):")
for name, (est, grid) in param_grids.items():
    pipe = make_pipe(est)
    search = GridSearchCV(pipe, grid, scoring='roc_auc', cv=inner_cv, n_jobs=-1)
    scores = cross_val_score(search, X, y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"  {name:22s} AUC = {scores.mean():.3f} +/- {scores.std():.3f}  (n={len(scores)} folds)")


# %% ------------------------------------------------------------------------
# FIX 4: Bootstrap 95% CI on test-set AUC
# -----------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

best_pipe = make_pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=42))
best_pipe.fit(X_train, y_train)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

point_auc = roc_auc_score(y_test, y_prob)
_, lo, hi = bootstrap_auc_ci(y_test, y_prob)
print(f"\nTest AUC: {point_auc:.3f}  [95% CI {lo:.3f}-{hi:.3f}]  (bootstrap, n=2000)")
# Report every headline AUC in the paper this way, not as a bare point estimate.


# %% ------------------------------------------------------------------------
# FIX 5 (v2): APOE4 interaction effect, multi-candidate + multiple-comparison
# correction, on the FULL imputed cohort (not complete-case dropna)
# -----------------------------------------------------------------------------
# v1 of this test used dropna(), which cut the usable sample from 473 to 365
# and lost power. It also only tried one interaction partner (CDRSB), which
# turned out to have ~zero main effect of its own (p=0.985) -- interacting
# APOE4 with a feature that isn't predictive by itself is not a fair test.
#
# v2 fixes both: impute with the same KNNImputer used for the ML pipeline (so
# all 473 subjects are used, not just the 365 with zero missingness), and
# test APOE4's interaction against each of the 5 strongest clinical predictors
# in turn, then apply a Benjamini-Hochberg correction across the 5 tests so
# you're not just reporting whichever one happened to look best (p-hacking).

imputer_sm = KNNImputer(n_neighbors=5)
X_imp = imputer_sm.fit_transform(mci_24[available])
df_sm = pd.DataFrame(X_imp, columns=available)
df_sm['CONV_LABEL'] = y
print(f"\nImputed sample for interaction testing: {len(df_sm)} (full cohort, no rows dropped)")

candidates = ['CDRSB', 'FAQTOTAL', 'TOTAL13', 'MMSCORE', 'HIPPO_ICV']
raw_pvals, coefs, partner_names = [], [], []

print("\nAPOE4 x feature interactions (each fit as its own model, same control set):")
for partner in candidates:
    other_controls = [c for c in candidates if c != partner] + ['PTEDUCAT']
    formula = f"CONV_LABEL ~ APOE4 * {partner} + " + " + ".join(other_controls)
    m = smf.logit(formula, data=df_sm).fit(disp=0)
    key = f'APOE4:{partner}'
    p = m.pvalues.get(key, float('nan'))
    coef = m.params.get(key, float('nan'))
    raw_pvals.append(p); coefs.append(coef); partner_names.append(partner)
    print(f"  APOE4 x {partner:10s}  coef={coef:+.4f}  raw p={p:.4f}")

reject, p_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
print("\nBenjamini-Hochberg corrected (alpha=0.05, 5 tests):")
for partner, p_raw, p_c, sig in zip(partner_names, raw_pvals, p_corrected, reject):
    flag = "SIGNIFICANT" if sig else "not significant"
    print(f"  APOE4 x {partner:10s}  raw p={p_raw:.4f}  corrected p={p_c:.4f}  -> {flag}")
# Whatever survives correction (if anything) is your defensible headline
# interaction claim. If none survive, that's a valid and reportable result too:
# "APOE4 shows a significant main effect / ablation contribution but no
# significant interaction with cognitive decline severity in this cohort" --
# say that plainly rather than reaching for the subgroup-split framing.


# %% ------------------------------------------------------------------------
# FIX 6: Feature-group ablation
# -----------------------------------------------------------------------------
# Justifies the "multimodal" claim in your title/abstract by showing combined
# features actually beat any single modality, rather than just asserting it.

groups = {
    'Cognitive only':   COGNITIVE,
    'Genetic only':      GENETIC,
    'MRI only':          MRI,
    'Demographic only':  DEMOG,
    'All combined':      available,
}

print("\nFeature-group ablation (5-fold CV AUC, Logistic Regression):")
for gname, feats in groups.items():
    feats = [f for f in feats if f in mci_24.columns]
    Xg = mci_24[feats].values
    pipe = make_pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=42))
    scores = cross_val_score(
        pipe, Xg, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1)
    print(f"  {gname:20s} AUC = {scores.mean():.3f} +/- {scores.std():.3f}  (k={len(feats)} features)")


# %% ------------------------------------------------------------------------
# FIX 7: Calibration (is predicted risk = observed risk?)
# -----------------------------------------------------------------------------
# AUC only measures ranking, not whether "70% predicted risk" corresponds to
# roughly 70% observed conversion -- which is what a clinician actually needs.

brier = brier_score_loss(y_test, y_prob)
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
print(f"\nBrier score: {brier:.3f}  (0 = perfect, 0.25 = uninformative)")
print("Calibration curve points (mean predicted, fraction positive):")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"  predicted={mp:.2f}  observed={fp:.2f}")
# Plot mean_pred (x) vs frac_pos (y) against the y=x diagonal as a figure for the paper.


# %% ------------------------------------------------------------------------
# FIX 8 (optional, most rigorous): Cox proportional hazards
# -----------------------------------------------------------------------------
# Fully sidesteps the censoring problem in FIX 1 by modeling time-to-conversion
# directly instead of binarizing it at a horizon. This is the standard
# approach in the ADNI conversion-prediction literature, so it also puts your
# results on directly comparable footing with published benchmarks.
#
# Uncomment to run (pip install lifelines):
#
# from lifelines import CoxPHFitter
# cox_df = mci[available + ['LAST_VISIT_MONTH', 'CONVERTER']].dropna()
# cox_df = cox_df.rename(columns={'LAST_VISIT_MONTH': 'duration', 'CONVERTER': 'event'})
# cph = CoxPHFitter()
# cph.fit(cox_df, duration_col='duration', event_col='event')
# cph.print_summary()
# print(f"Concordance index (AUC-equivalent for survival models): {cph.concordance_index_:.3f}")


# %% ------------------------------------------------------------------------
# NOTE: Amyloid PET (not coded -- schema-dependent)
# -----------------------------------------------------------------------------
# Fixmissing.py already extracts UCBERKELEYAV45/AMY PET files to CSV but they
# were never merged into mci_with_labels.csv. Amyloid status is a core AT(N)
# biomarker and would strengthen both novelty and performance. To add it:
#   1. Load the extracted PET CSV, inspect columns for a SUMMARY_SUVR or
#      AMYLOID_STATUS-type field and a VISCODE2/RID key.
#   2. Merge onto mci_24 on RID using the baseline visit PET value.
#   3. Check coverage (%), since PET is often missing for a large subset --
#      report an "amyloid-available subset" sensitivity analysis if coverage
#      is under ~60-70%, rather than imputing a biomarker that's structurally
#      missing (not at random).

print("\nDone. Use CONV_LABEL results (FIX 1-4) and the interaction model (FIX 5) "
      "as your headline numbers instead of the original CONVERTER-based results.")