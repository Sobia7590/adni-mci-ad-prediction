
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import shap
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'figure.dpi': 500, 'savefig.dpi': 500,
                      'font.family': 'DejaVu Sans'})

DATA = r'D:\DATA-adni\ADNIMERGE_CSVs'
OUT  = r'D:\DATA-adni\figures'
os.makedirs(OUT, exist_ok=True)

mci = pd.read_csv(os.path.join(DATA, 'mci_with_labels.csv'))
print(f"Loaded: {mci.shape}")

HORIZON = 24
def censoring_safe_subset(df, horizon):
    converters = df[(df['CONVERTER'] == 1) & (df['CONV_MONTH'] <= horizon)]
    non_converters = df[(df['CONVERTER'] == 0) & (df['LAST_VISIT_MONTH'] >= horizon)]
    out = pd.concat([converters, non_converters]).copy()
    out['CONV_LABEL'] = out['RID'].isin(converters['RID']).astype(int)
    return out

mci_24 = censoring_safe_subset(mci, HORIZON)

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

def make_pipe(model):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])


# %% ------------------------------------------------------------------------
# FIGURE 1: Patient selection flow (censoring-safe cohort)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 8))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def box(xy, w, h, text, fc='#eaf2fb'):
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                           fc=fc, ec='black', lw=1.2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=9.5)

def varrow(x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                 arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black'))

n_total = len(mci)
n_dropped = n_total - len(mci_24)
n_safe = len(mci_24)
n_conv = int(y.sum())
n_nonconv = n_safe - n_conv

box((0.10, 0.86), 0.80, 0.09, f"ADNI MCI baseline cohort\nn = {n_total}")
varrow(0.50, 0.86, 0.79)
box((0.10, 0.62), 0.80, 0.11,
    f"Excluded: ambiguous follow-up status\n(< {HORIZON} months, no observed conversion)\nn = {n_dropped}",
    fc='#fbeaea')
varrow(0.50, 0.62, 0.55)
box((0.10, 0.38), 0.80, 0.09, f"Censoring-safe analysis cohort\nn = {n_safe}")
varrow(0.28, 0.38, 0.28)
varrow(0.72, 0.38, 0.28)
box((0.02, 0.10), 0.44, 0.14,
    f"Converters\n(within {HORIZON} months)\nn = {n_conv} ({n_conv/n_safe*100:.1f}%)",
    fc='#eafbea')
box((0.54, 0.10), 0.44, 0.14,
    f"Non-converters\n(>= {HORIZON} months follow-up)\nn = {n_nonconv} ({n_nonconv/n_safe*100:.1f}%)",
    fc='#eafbea')

plt.title("Figure 1. Patient Selection Flow", fontsize=12, weight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_patient_flow.png'), bbox_inches='tight')
plt.close()
print("Saved fig1_patient_flow.png")


# %% ------------------------------------------------------------------------
# FIGURE 2: Model comparison (AUC, tuned + repeated CV)
# -----------------------------------------------------------------------------
# Numbers below are copied from your 04_improved_analysis.py FIX 3 output.
# If you rerun FIX 3 and get different numbers, update this dict to match.
model_results = {
    'Logistic\nRegression': (0.899, 0.032),
    'Random\nForest':       (0.895, 0.031),
    'XGBoost':               (0.895, 0.028),
}

names = list(model_results.keys())
means = [model_results[n][0] for n in names]
stds  = [model_results[n][1] for n in names]
colors = ['#e74c3c' if m == max(means) else '#3498db' for m in means]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, edgecolor='black', linewidth=0.8)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015, f'{m:.3f}',
             ha='center', fontsize=10, weight='bold')
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('AUC-ROC (5x5 repeated stratified CV)')
ax.set_title('Figure 2. Model Comparison', fontsize=12, weight='bold')
ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_model_comparison.png'), bbox_inches='tight')
plt.close()
print("Saved fig2_model_comparison.png")


# %% ------------------------------------------------------------------------
# FIGURE 3: ROC curve with bootstrap 95% CI band
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

best_pipe = make_pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=42))
best_pipe.fit(X_train, y_train)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
point_auc = roc_auc_score(y_test, y_prob)

# Bootstrap band: resample test set, interpolate TPR at fixed FPR grid
rng = np.random.RandomState(42)
fpr_grid = np.linspace(0, 1, 100)
tpr_boot = []
aucs_boot = []
n = len(y_test)
for _ in range(500):
    idx = rng.randint(0, n, n)
    yt, yp = np.asarray(y_test)[idx], y_prob[idx]
    if len(np.unique(yt)) < 2:
        continue
    f, t, _ = roc_curve(yt, yp)
    tpr_boot.append(np.interp(fpr_grid, f, t))
    aucs_boot.append(roc_auc_score(yt, yp))
tpr_boot = np.array(tpr_boot)
lo_band = np.percentile(tpr_boot, 2.5, axis=0)
hi_band = np.percentile(tpr_boot, 97.5, axis=0)
auc_lo, auc_hi = np.percentile(aucs_boot, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(6, 6))
ax.fill_between(fpr_grid, lo_band, hi_band, color='#e74c3c', alpha=0.15, label='95% CI')
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2,
        label=f'Logistic Regression (AUC={point_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}])')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Figure 3. ROC Curve (Test Set)', fontsize=12, weight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_roc_curve.png'), bbox_inches='tight')
plt.close()
print("Saved fig3_roc_curve.png")


# %% ------------------------------------------------------------------------
# FIGURE 4: Feature-group ablation
# -----------------------------------------------------------------------------
groups = {
    'Cognitive\nonly':   COGNITIVE,
    'Genetic\nonly':      GENETIC,
    'MRI\nonly':          MRI,
    'Demographic\nonly':  DEMOG,
    'All\ncombined':      available,
}

ablation_results = {}
for gname, feats in groups.items():
    feats = [f for f in feats if f in mci_24.columns]
    Xg = mci_24[feats].values
    pipe = make_pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=42))
    scores = cross_val_score(
        pipe, Xg, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1)
    ablation_results[gname] = (scores.mean(), scores.std())

names = list(ablation_results.keys())
means = [ablation_results[n][0] for n in names]
stds  = [ablation_results[n][1] for n in names]
colors = ['#e74c3c' if n == 'All\ncombined' else '#3498db' for n in names]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f'{m:.3f}',
             ha='center', fontsize=9.5, weight='bold')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='chance')
ax.set_ylim(0.4, 1.0)
ax.set_ylabel('AUC-ROC (5-fold CV)')
ax.set_title('Figure 4. Feature-Group Ablation', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_ablation.png'), bbox_inches='tight')
plt.close()
print("Saved fig4_ablation.png")


# %% ------------------------------------------------------------------------
# FIGURE 5: Calibration / reliability diagram
# -----------------------------------------------------------------------------
brier = brier_score_loss(y_test, y_prob)
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=8)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Perfect calibration')
ax.plot(mean_pred, frac_pos, marker='o', color='#e74c3c', linewidth=2,
        label=f'Logistic Regression (Brier={brier:.3f})')
ax.set_xlabel('Mean predicted risk')
ax.set_ylabel('Observed conversion rate')
ax.set_title('Figure 5. Calibration Curve', fontsize=12, weight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_calibration.png'), bbox_inches='tight')
plt.close()
print("Saved fig5_calibration.png")


# %% ------------------------------------------------------------------------
# FIGURE 6: APOE4 interaction forest plot (all 5 tests, with 95% CIs)
# -----------------------------------------------------------------------------
imputer_sm = KNNImputer(n_neighbors=5)
X_imp = imputer_sm.fit_transform(mci_24[available])
df_sm = pd.DataFrame(X_imp, columns=available)
df_sm['CONV_LABEL'] = y

# NOTE: a raw (uncorrected) 95% CI can fail to cross zero for a test that is
# still non-significant after multiple-comparison correction (e.g. APOE4 x
# MMSCORE below: raw p=0.025, but corrected p=0.127). Plotting only the raw
# CI next to a caption about "corrected" results would visually contradict
# the caption -- so each point is annotated with its own corrected p-value
# directly, making the plot and the correction agree.

candidates = ['CDRSB', 'FAQTOTAL', 'TOTAL13', 'MMSCORE', 'HIPPO_ICV']
forest_rows = []
raw_pvals = []
for partner in candidates:
    other_controls = [c for c in candidates if c != partner] + ['PTEDUCAT']
    formula = f"CONV_LABEL ~ APOE4 * {partner} + " + " + ".join(other_controls)
    m = smf.logit(formula, data=df_sm).fit(disp=0)
    key = f'APOE4:{partner}'
    coef = m.params.get(key, np.nan)
    ci = m.conf_int().loc[key] if key in m.params.index else (np.nan, np.nan)
    p = m.pvalues.get(key, np.nan)
    forest_rows.append((f'APOE4 x {partner}', coef, ci[0], ci[1]))
    raw_pvals.append(p)

reject, p_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
any_significant = bool(np.any(reject))
title_suffix = ("at least one survives Benjamini-Hochberg correction -- see annotations"
                 if any_significant else
                 "none significant after Benjamini-Hochberg correction")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ylabels = [r[0] for r in forest_rows]
coefs   = [r[1] for r in forest_rows]
los     = [r[1] - r[2] for r in forest_rows]
his     = [r[3] - r[1] for r in forest_rows]

ax.errorbar(coefs, range(len(coefs)), xerr=[los, his], fmt='o', color='#3498db',
            ecolor='#3498db', elinewidth=2, capsize=4, markersize=7)

x_span = max(r[3] for r in forest_rows) - min(r[2] for r in forest_rows)
for i, (row, p_c) in enumerate(zip(forest_rows, p_corrected)):
    label = f'p_corr={p_c:.3f}' + ('' if p_c >= 0.05 else '  *')
    ax.text(row[3] + x_span * 0.03, i, label, va='center', fontsize=9, color='#555555')

ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels)
ax.set_xlim(min(r[2] for r in forest_rows) - x_span * 0.1,
            max(r[3] for r in forest_rows) + x_span * 0.35)
ax.set_xlabel('Interaction coefficient (95% CI)')
ax.set_title(f'Figure 6. APOE4 x Feature Interaction Tests\n({title_suffix})',
              fontsize=11.5, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_interaction_forest.png'), bbox_inches='tight')
plt.close()
print("Saved fig6_interaction_forest.png")


# %% ------------------------------------------------------------------------
# FIGURE 7: Converter label overview (corrected, censoring-safe cohort)
# -----------------------------------------------------------------------------
# Replaces 02_converter_labels.html, which was built on the old "ever
# converts, no follow-up floor" label (404/271 split, 62.7% APOE4 rate in
# converters). Those numbers don't match the corrected cohort used everywhere
# else in this script and shouldn't be reported alongside the 0.895 AUC
# headline number -- this figure uses the same mci_24 / CONV_LABEL cohort as
# everything else, so the paper is internally consistent.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

counts = mci_24['CONV_LABEL'].value_counts().reindex([0, 1])
axes[0].bar(['Non-Converter', 'Converter'], counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + max(counts.values) * 0.02, str(int(v)), ha='center', fontsize=11, weight='bold')
axes[0].set_title(f'Converter vs Non-Converter\n(n={len(mci_24)}, {HORIZON}mo horizon)')
axes[0].set_ylim(0, counts.max() * 1.2)

conv_times = mci_24.loc[mci_24['CONV_LABEL'] == 1, 'CONV_MONTH'].dropna()
axes[1].hist(conv_times, bins=12, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].set_title(f'Conversion Timing\n(months; all <= {HORIZON} by construction)')
axes[1].set_xlabel('Months to conversion')

apoe_rates = mci_24.groupby('CONV_LABEL')['APOE4'].apply(lambda s: (s > 0).mean() * 100)
vals = [apoe_rates.get(0, 0), apoe_rates.get(1, 0)]
axes[2].bar(['Non-Converter', 'Converter'], vals, color=['#2ecc71', '#e74c3c'], edgecolor='black')
for i, v in enumerate(vals):
    axes[2].text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=11, weight='bold')
axes[2].set_title('APOE4+ Rate by Conversion Status')
axes[2].set_ylim(0, 100)

plt.suptitle('Figure 7. Corrected Converter Labels (Censoring-Safe Cohort)', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_converter_labels.png'), bbox_inches='tight')
plt.close()
print("Saved fig7_converter_labels.png")


# %% ------------------------------------------------------------------------
# FIGURE 8: SHAP feature importance (corrected cohort + tuned model)
# -----------------------------------------------------------------------------
# The original SHAP panel was computed on the old uncorrected model. Rerun
# here on best_pipe (fit on the censoring-safe cohort above, in the ROC
# section), so this figure and your headline 0.895 AUC come from the same
# model instead of two different ones.

imputer_fit = best_pipe.named_steps['imputer']
scaler_fit  = best_pipe.named_steps['scaler']
lr_fit      = best_pipe.named_steps['model']

X_train_t = scaler_fit.transform(imputer_fit.transform(X_train))
X_test_t  = scaler_fit.transform(imputer_fit.transform(X_test))

explainer = shap.LinearExplainer(lr_fit, X_train_t)
shap_values = explainer.shap_values(X_test_t)

shap_df = pd.DataFrame({
    'Feature': available,
    'SHAP_mean': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP_mean', ascending=True)

fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#2ecc71' if f in GENETIC else '#3498db' for f in shap_df['Feature']]
ax.barh(shap_df['Feature'], shap_df['SHAP_mean'], color=colors, edgecolor='black')
for i, v in enumerate(shap_df['SHAP_mean']):
    ax.text(v + shap_df['SHAP_mean'].max() * 0.01, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Figure 8. SHAP Feature Importance\n(corrected cohort, tuned Logistic Regression)',
             fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig8_shap_importance.png'), bbox_inches='tight')
plt.close()
print("Saved fig8_shap_importance.png")


print(f"\nAll figures saved to: {OUT}")