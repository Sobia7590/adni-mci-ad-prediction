
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

DATA = r'D:\DATA-adni'
CSV_DATA = os.path.join(DATA, 'ADNIMERGE_CSVs')
NEUROBAT_PATH = os.path.join(DATA, 'Neuropsychological', 'NEUROBAT_28May2026.csv')
CSF_PATH = os.path.join(DATA, 'Biospecimen_Results', 'UPENNBIOMK_ROCHE_ELECSYS_28May2026.csv')

mci = pd.read_csv(os.path.join(CSV_DATA, 'mci_with_labels.csv'))
print(f"Loaded: {mci.shape}")

HORIZON = 24
def censoring_safe_subset(df, horizon):
    converters = df[(df['CONVERTER'] == 1) & (df['CONV_MONTH'] <= horizon)]
    non_converters = df[(df['CONVERTER'] == 0) & (df['LAST_VISIT_MONTH'] >= horizon)]
    out = pd.concat([converters, non_converters]).copy()
    out['CONV_LABEL'] = out['RID'].isin(converters['RID']).astype(int)
    return out

mci_24 = censoring_safe_subset(mci, HORIZON)
print(f"Censoring-safe cohort: {len(mci_24)}")

COGNITIVE = ['MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTAL13']
GENETIC   = ['APOE4', 'APOE4_count']
MRI       = ['ST29SV', 'ST88SV', 'ST40TS', 'ST99TS', 'ST101SV', 'HIPPO_TOTAL', 'HIPPO_ICV']
DEMOG     = ['PTEDUCAT']
mci_24['SEX'] = (mci_24['PTGENDER'] == 'Male').astype(float) if 'PTGENDER' in mci_24.columns else np.nan
DEMOG += ['SEX']
ALL_FEATURES = COGNITIVE + GENETIC + MRI + DEMOG
available = [f for f in ALL_FEATURES if f in mci_24.columns]

def make_pipe(model):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

cv5 = StratifiedKFold(5, shuffle=True, random_state=42)

def eval_features(df, feats, y):
    X = df[feats].values
    pipe = make_pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=42))
    scores = cross_val_score(pipe, X, y, cv=cv5, scoring='roc_auc', n_jobs=-1)
    return scores.mean(), scores.std()

y_full = mci_24['CONV_LABEL'].values

base_mean, base_std = eval_features(mci_24, available, y_full)
print(f"\nBaseline (16 features, full n={len(mci_24)}): AUC = {base_mean:.3f} +/- {base_std:.3f}")


# %% ------------------------------------------------------------------------
# EXT 1: NEUROBAT LDELTOTAL (Logical Memory Delayed Recall)
# -----------------------------------------------------------------------------
nb = pd.read_csv(NEUROBAT_PATH, low_memory=False)
nb_bl = nb[nb['VISCODE2'].isin(['bl', 'sc'])].drop_duplicates(subset='RID', keep='first')
nb_bl = nb_bl[['RID', 'LDELTOTAL']].copy()
nb_bl['LDELTOTAL'] = pd.to_numeric(nb_bl['LDELTOTAL'], errors='coerce')
nb_bl.loc[nb_bl['LDELTOTAL'] < 0, 'LDELTOTAL'] = np.nan  # ADNI negative sentinel codes = missing

mci_ext = mci_24.merge(nb_bl, on='RID', how='left')
print(f"\nLDELTOTAL coverage: {mci_ext['LDELTOTAL'].notna().sum()} / {len(mci_ext)}")

available_plus_ld = available + ['LDELTOTAL']
cog_plus_ld = COGNITIVE + ['LDELTOTAL']

ld_cog_mean, ld_cog_std = eval_features(mci_ext, cog_plus_ld, y_full)
ld_all_mean, ld_all_std = eval_features(mci_ext, available_plus_ld, y_full)

print(f"Cognitive-only + LDELTOTAL:  AUC = {ld_cog_mean:.3f} +/- {ld_cog_std:.3f}  (was 0.882 without it)")
print(f"All combined  + LDELTOTAL:   AUC = {ld_all_mean:.3f} +/- {ld_all_std:.3f}  (was {base_mean:.3f} without it)")


# %% ------------------------------------------------------------------------
# EXT 2: CSF biomarkers (ABETA42, PTAU) -- fair subset sensitivity analysis
# -----------------------------------------------------------------------------
csf = pd.read_csv(CSF_PATH, low_memory=False)
csf_bl = csf[csf['VISCODE2'].isin(['bl', 'sc'])].drop_duplicates(subset='RID', keep='first')
csf_bl = csf_bl[['RID', 'ABETA42', 'PTAU']].copy()
for col in ['ABETA42', 'PTAU']:
    csf_bl[col] = pd.to_numeric(csf_bl[col], errors='coerce')

mci_csf = mci_ext.merge(csf_bl, on='RID', how='inner')
mci_csf = mci_csf.dropna(subset=['ABETA42', 'PTAU'])
print(f"\nCSF subset: n = {len(mci_csf)}")

y_csf = mci_csf['CONV_LABEL'].values

csf_without_mean, csf_without_std = eval_features(mci_csf, available, y_csf)
csf_with_mean, csf_with_std = eval_features(mci_csf, available + ['ABETA42', 'PTAU'], y_csf)

print(f"Same {len(mci_csf)}-subject subset, WITHOUT CSF:  AUC = {csf_without_mean:.3f} +/- {csf_without_std:.3f}")
print(f"Same {len(mci_csf)}-subject subset, WITH CSF:     AUC = {csf_with_mean:.3f} +/- {csf_with_std:.3f}")
print(f"Delta from adding CSF: {csf_with_mean - csf_without_mean:+.3f}")

print("\nDone. Report EXT 1 as a main-cohort result (n=473, same as headline).")
print(f"Report EXT 2 as a subset sensitivity analysis (n={len(mci_csf)}), explicitly")
print("labeled as smaller and not directly comparable to the n=473 headline AUC.")
