# 🧠 ADNI-MCI-AD-Prediction

**Rigorous Multimodal MCI-to-Alzheimer's Disease Conversion Prediction using the ADNI Dataset**
> Companion code for: *"Rigorous Multimodal Prediction of MCI-to-Alzheimer's Conversion on ADNI: Addressing Censoring Bias and Testing APOE4 Interaction Effects"*
> Author: Sobia Arshad | Inje University, South Korea | GKS Scholar

---

## 📋 Project Overview

Alzheimer's Disease (AD) affects over 55 million people worldwide. A critical clinical challenge is identifying patients at the **Mild Cognitive Impairment (MCI)** stage who will progress to AD — enabling earlier intervention before significant neurodegeneration occurs.

This project builds and validates a **censoring-safe, multimodal machine learning pipeline** for MCI-to-AD conversion prediction using the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset (cohorts ADNI1–4), covering:

- Censoring-safe cohort construction with an explicit 24-month follow-up floor
- Feature engineering from cognitive, genetic, and MRI modalities
- Supervised ML model training with repeated, nested cross-validation and bootstrap confidence intervals
- SHAP explainability analysis and feature-group ablation
- Formal, multiplicity-corrected APOE4 × feature interaction testing (not a naive subgroup split)

This repository addresses three methodological gaps common in the MCI-to-AD prediction literature: (1) "non-converter" labels assigned without a minimum follow-up requirement, (2) APOE4 subgroup claims based on descriptive test-set splits without correction for multiple comparisons, and (3) multimodal fusion asserted rather than demonstrated via controlled ablation. See the full writeup for details.

---

## 🏆 Key Results

| Metric | Value |
| --- | --- |
| Dataset | 473 ADNI MCI subjects (censoring-safe subset of 675) |
| Follow-up rule | Converters: diagnosis change within 24 months. Non-converters: ≥24 months follow-up with no conversion. Subjects with shorter, inconclusive follow-up excluded (n=202). |
| Features | 16 multimodal (cognitive + genetic + MRI + demographic) |
| Best Model | Logistic Regression (repeated CV AUC 0.899 ± 0.032) |
| Test AUC | **0.895** (95% CI 0.829–0.952) |
| Brier score | 0.132 |

### 📊 Feature-Group Ablation

| Feature group | AUC (5-fold CV) |
| --- | --- |
| Demographic only | 0.492 |
| Genetic only | 0.657 |
| MRI only | 0.750 |
| Cognitive only | 0.882 |
| **All combined** | **0.901** |

Multimodal fusion outperforms any single modality, though the margin over cognitive features alone is modest.

### 🔑 SHAP Feature Importance (Top 5)

| Rank | Feature | Mean \|SHAP\| | Category |
| --- | --- | --- | --- |
| 1 | TOTAL13 (ADAS-Cog13) | 0.893 | Cognitive |
| 2 | **APOE4** | **0.567** | **Genetic** |
| 3 | FAQTOTAL | 0.546 | Cognitive |
| 4 | HIPPO_ICV | 0.526 | MRI |
| 5 | ST40TS | 0.217 | MRI |

APOE4 is the strongest standalone genetic predictor and the 2nd most important feature overall, ahead of MRI volumetric measures.

### ⚠️ APOE4 Interaction Testing — A Corrected Null Result

An earlier iteration of this analysis reported a descriptive APOE4 subgroup split (APOE4-positive AUC 0.916 vs. APOE4-negative AUC 0.801, n=65/70) as a headline finding, in line with how this kind of result is commonly reported in the literature.

Under the corrected methodology, this effect **does not replicate**. Rather than splitting the test set by APOE4 status, we fit five formal APOE4 × feature interaction models (CDR-SB, FAQ, ADAS-Cog13, MMSE, hippocampal volume/ICV) and applied Benjamini-Hochberg correction across all five:

| Interaction term | Coef. | Raw p | Corrected p |
| --- | --- | --- | --- |
| APOE4 × CDR-SB | 0.145 | 0.632 | 0.632 |
| APOE4 × FAQTOTAL | 0.098 | 0.186 | 0.310 |
| APOE4 × TOTAL13 | -0.080 | 0.125 | 0.310 |
| APOE4 × MMSE | 0.367 | 0.025 | 0.127 |
| APOE4 × HIPPO_ICV | -0.006 | 0.287 | 0.359 |

**None of the five interaction terms are statistically significant after correction.** APOE4 remains a genuinely strong *standalone* predictor (genetic features alone reach 0.657 AUC; APOE4 ranks 2nd by SHAP), but the stronger claim — that APOE4 status modulates the predictive strength of other clinical features — is not supported once properly tested. This contrast is reported explicitly as a case study in how uncorrected subgroup splits on small test sets can suggest effects that don't survive a properly powered, corrected test.

---

## 🗂️ Repository Structure

```
adni-mci-ad-prediction/
│
├── 01_explore.ipynb           # Data loading, EDA, baseline statistics
├── 02_converter_labels.ipynb  # Raw MCI-to-Dementia conversion labels (any/24mo/36mo)
├── 03_modeling.ipynb          # Early exploratory models (superseded, see note below)
├── 04_improved_analysis.py    # Censoring-safe cohort, nested CV, bootstrap CI, ablation,
│                               #   calibration, BH-corrected APOE4 interaction tests
├── 05_generate_figures.py     # Generates all reported figures from the corrected pipeline
├── 06_biomarker_extension.py  # Sensitivity analysis: adds LDELTOTAL and CSF biomarkers
│
├── requirements.txt          # Python dependencies
└── README.md
```

**Note on 03_modeling.ipynb:** this notebook reflects an early exploratory pass (single 80/20 split, default hyperparameters, no censoring-safe cohort) and is kept for transparency, but the numbers it prints (AUC 0.869 on the full 675-subject cohort) are **not** the reported results. `04_improved_analysis.py` is the pipeline actually used for the results below; `03_modeling.ipynb` should not be read as the current methodology.

---

## 📁 Dataset

This project uses the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset.

- **Access:** Requires approved application at [ida.loni.usc.edu](https://ida.loni.usc.edu)
- **Raw data is NOT included** in this repository (ADNI Data Use Agreement prohibits public sharing)
- **Cohorts used:** ADNI1, ADNI-GO, ADNI2, ADNI3, ADNI4
- **Files used:** DXSUM.csv, REGISTRY.csv, MMSE.csv, ADAS.csv, CDR.csv, FAQ.csv, APOERES.csv, UCSFFSX7.csv, UCBERKELEYFDG_8mm.csv, UCBERKELEY_AMY_6MM.csv, UPENNBIOMK_MASTER.csv

---

## 🔬 Notebooks & Scripts

### 01_explore.ipynb — Data Exploration
- Loads DXSUM, REGISTRY, PTDEMOG, cognitive scores
- Builds baseline master table (CN / MCI / Dementia)
- Merges APOE4 genotype and FreeSurfer MRI volumes

### 02_converter_labels.ipynb — Conversion Label Construction
- Builds the longitudinal diagnosis timeline per subject and flags MCI subjects who ever convert to Dementia
- Computes conversion timing and 24-/36-month conversion flags
- Saves `mci_with_labels.csv`; the censoring-safe filtering (dropping subjects with ambiguous, sub-horizon follow-up) is applied downstream in `04_improved_analysis.py`, not in this notebook

### 03_modeling.ipynb — Early Exploratory Models *(superseded)*
- First-pass models on the raw (non-censoring-safe) 675-subject cohort, single 80/20 split, default hyperparameters, median imputation
- Kept for transparency only; produced the earlier AUC 0.869 result and the naive APOE4 subgroup split, both superseded by the corrected pipeline below

### 04_improved_analysis.py — Corrected Pipeline (reported methodology)
- Builds the censoring-safe cohort (24-month follow-up floor; excludes 202 of 675 subjects with ambiguous follow-up → n=473)
- KNN imputation → standardization → classifier, inside a leakage-safe pipeline
- Hyperparameters tuned via nested grid search within 5×5 repeated stratified cross-validation (Logistic Regression, Random Forest, XGBoost)
- Bootstrap (n=2000) 95% CI on test-set AUC
- Feature-group ablation (cognitive/genetic/MRI/demographic/combined)
- Calibration via Brier score and reliability curve
- Formal, multiplicity-corrected APOE4 × feature interaction testing (Benjamini-Hochberg, 5 candidate interactions: CDR-SB, FAQ, ADAS-Cog13, MMSE, hippocampal volume/ICV)

### 05_generate_figures.py — Figure Generation
- Regenerates every reported figure (patient selection flow, model comparison, ROC curve with bootstrap CI band, feature-group ablation, calibration curve, APOE4 interaction forest plot, converter-label breakdown, SHAP importance) directly from the corrected pipeline in `04_improved_analysis.py`, so figures and headline numbers come from the same model run

### 06_biomarker_extension.py — Biomarker Sensitivity Analysis *(exploratory, not yet in the paper)*
- Tests whether adding NEUROBAT LDELTOTAL (logical memory delayed recall) to the main n=473 cohort changes AUC
- Tests whether adding CSF biomarkers (ABETA42, PTAU) changes AUC, on the smaller subset of subjects with CSF data available, reported separately as a sensitivity analysis rather than mixed into the headline n=473 result

---

## ⚙️ Installation & Setup

```
# Clone the repository
git clone https://github.com/Sobia7590/adni-mci-ad-prediction.git
cd adni-mci-ad-prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.45
plotly>=5.0
pyreadr>=0.5
missingno>=0.5
statsmodels>=0.14
jupyter
```

---

## 🔗 Connection to PhD Research (Direction B)

This preliminary work (Direction A) directly motivates the proposed PhD project, but the connection is grounded in APOE4's standalone predictive signal, not an unproven interaction effect:

| Direction A (This Repository) | Direction B (Proposed PhD) |
| --- | --- |
| Binary classification: will MCI convert? | Generative: what will the brain look like at +12M/+24M? |
| APOE4 as a predictive feature (2nd by SHAP, 16 features) | APOE4 as a generative conditioning signal |
| Output: risk probability | Output: synthesized 3D brain MRI at future timepoint |
| Finding: APOE4 is a strong standalone predictor; naive subgroup AUC gap does not survive formal interaction testing | Hypothesis: explicit genotype conditioning improves individualization of generative progression models, independent of whether a population-level interaction effect exists |
| Method: Logistic Regression + SHAP + corrected interaction testing | Method: 4D Diffusion Transformer, genotype-conditioned |

---

## 👩‍💻 Author

**Sobia Arshad**
M.Sc. AI in Healthcare — Inje University, South Korea
Korean Government Scholar (GKS)
📧 <sobiaarshad392@gmail.com>
🔗 [github.com/Sobia7590](https://github.com/Sobia7590)
🔗 ORCID: 0009-0003-4791-9620

---

## ⚖️ Data Use Notice

This project uses ADNI data. ADNI is a public-private partnership. Raw data files are **not included** in this repository and must be obtained independently through the LONI Image and Data Archive ([ida.loni.usc.edu](https://ida.loni.usc.edu)) with approved access.

Principal Investigator: Michael W. Weiner, MD
Data Use Agreement: [ADNI DUA](https://adni.loni.usc.edu/data-samples/access-data/)
