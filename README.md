# 🧠 ADNI-MCI-AD-Prediction

**Multimodal MCI-to-Alzheimer's Disease Conversion Prediction using the ADNI Dataset**

> | 
> Author: Sobia Arshad | Inje University, South Korea | 

---

## 📋 Project Overview

Alzheimer's Disease (AD) affects over 55 million people worldwide. A critical clinical challenge is identifying patients at the **Mild Cognitive Impairment (MCI)** stage who will progress to AD — enabling earlier intervention before significant neurodegeneration occurs.

This project builds and validates a **complete multimodal machine learning pipeline** for MCI-to-AD conversion prediction using the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset (cohorts ADNI1–4), covering:

- Data access, extraction, and preprocessing
- Feature engineering from cognitive, genetic, and MRI modalities
- Supervised ML model training and cross-validation
- SHAP explainability analysis
- APOE4 genotype subgroup analysis

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Dataset | 675 MCI subjects — ADNI1–4 |
| Features | 16 multimodal (cognitive + genetic + MRI) |
| Best Model | Logistic Regression |
| CV AUC (5-fold) | **0.808 ± 0.024** |
| Test AUC | **0.869** |
| Accuracy | 79% |
| Precision | 0.80 |

### 🔑 Key Finding — APOE4 Subgroup Effect

| Subgroup | N | Conversion Rate | Model AUC |
|----------|---|-----------------|-----------|
| APOE4 Negative (ε4−) | 70 | 31.4% | 0.801 |
| **APOE4 Positive (ε4+)** | **65** | **49.2%** | **0.916** |
| Overall test set | 135 | 40.0% | 0.869 |

> **APOE4+ subjects convert at 49.2% vs 31.4%** — a 23 percentage-point gap. Model AUC rises to **0.916** in the APOE4+ subgroup. This demonstrates that APOE4 genotype carries strong residual predictive signal beyond cognitive and MRI features — directly motivating the proposed PhD work (Direction B: 4D Diffusion Transformer).

---

## 📊 SHAP Feature Importance (Top 5)

| Rank | Feature | SHAP Value | Category |
|------|---------|------------|----------|
| 1 | TOTAL13 (ADAS-Cog 13) | 0.507 | Cognitive |
| 2 | HIPPO_ICV (Hippocampal vol / ICV) | 0.451 | MRI |
| 3 | FAQTOTAL (Functional Activities) | 0.379 | Cognitive |
| 4 | **APOE4 (ε4 carrier status)** | **0.309** | **Genetic** |
| 5 | ST101SV (ICV) | 0.213 | MRI |

---

## 🗂️ Repository Structure

```
adni-mci-ad-prediction/
│
├── 01_explore.ipynb          # Data loading, EDA, baseline statistics
├── 02_converter_labels.ipynb # MCI converter label definition
├── 03_modeling.ipynb         # ML models, CV, SHAP analysis
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📁 Dataset

This project uses the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset.

- **Access:** Requires approved application at [ida.loni.usc.edu](https://ida.loni.usc.edu)
- **Raw data is NOT included** in this repository (ADNI Data Use Agreement prohibits public sharing)
- **Cohorts used:** ADNI1, ADNI-GO, ADNI2, ADNI3, ADNI4
- **Files used:** DXSUM.csv, REGISTRY.csv, MMSE.csv, ADAS.csv, CDR.csv, FAQ.csv, APOERES.csv, UCSFFSX7.csv, UCBERKELEYFDG_8mm.csv, UCBERKELEY_AMY_6MM.csv, UPENNBIOMK_MASTER.csv

---

## 🔬 Notebooks

### 01_explore.ipynb — Data Exploration
- Loads DXSUM, REGISTRY, PTDEMOG, cognitive scores
- Builds baseline master table (CN / MCI / Dementia)
- Merges APOE4 genotype and FreeSurfer MRI volumes
- Interactive Plotly visualisations: subject counts, APOE4 rates, hippocampal volumes, MMSE/CDR distributions

### 02_converter_labels.ipynb — Converter Definition
- Defines MCI-to-Dementia conversion labels from longitudinal diagnosis records
- Time-windowed analysis: 24-month and 36-month conversion windows
- APOE4 conversion rate analysis (49.2% vs 31.4%)
- Saves `mci_with_labels.csv` — ML-ready dataset

### 03_modeling.ipynb — Machine Learning
- Trains 5 classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- 5-fold stratified cross-validation + 20% held-out test set
- KNN imputation + StandardScaler inside Pipeline (no data leakage)
- SHAP LinearExplainer for feature importance
- APOE4 subgroup evaluation
- Plotly interactive results visualisation

---

## ⚙️ Installation & Setup

```bash
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
lightgbm>=4.0
shap>=0.45
plotly>=5.0
pyreadr>=0.5
missingno>=0.5
jupyter
```

---

## 🔗 Connection to PhD Research (Direction B)

This preliminary work (Direction A) directly motivates the proposed PhD project:

| Direction A (This Repository) | Direction B (Proposed PhD) |
|-------------------------------|---------------------------|
| Binary classification: will MCI convert? | Generative: what will the brain look like at +12M/+24M? |
| APOE4 as a predictive feature | APOE4 as a generative conditioning signal (FiLM) |
| Output: risk probability | Output: synthesised 3D brain MRI at future timepoint |
| Finding: APOE4 modulates AUC by 11.5 points | Hypothesis: APOE4 modulates spatiotemporal atrophy trajectory |
| Method: Logistic Regression + SHAP | Method: 4D Diffusion Transformer in 3D VAE latent space |

---

## 👩‍💻 Author

**Sobia Arshad**  
M.Sc. AI in Healthcare — Inje University, South Korea  
Korean Government Scholar (GKS)  
📧 sobiaarshad392@gmail.com  
🔗 [github.com/Sobia7590](https://github.com/Sobia7590)

---

## ⚖️ Data Use Notice

This project uses ADNI data. ADNI is a public-private partnership. Raw data files are **not included** in this repository and must be obtained independently through the LONI Image and Data Archive ([ida.loni.usc.edu](https://ida.loni.usc.edu)) with approved access.

Principal Investigator: Michael W. Weiner, MD  
Data Use Agreement: [ADNI DUA](https://adni.loni.usc.edu/data-samples/access-data/)
