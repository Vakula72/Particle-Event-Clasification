# Explainable AI for High-Energy Particle Event Classification
## CERN Openlab Summer Student Portfolio Project

---

## Overview

This project applies **Explainable AI (XAI)** to the classification of particle collision events, using simulated ATLAS detector data (H→ττ signal vs Z/tt̄/W background). It combines three ML models with SHAP and LIME interpretability methods, explicitly connecting model decisions to known physics.

**Why this matters at CERN:** The High-Luminosity LHC upgrade requires real-time trigger systems that physicists can *trust*. XAI bridges the gap between high-performing black-box models and interpretable, physics-validated decisions.

---

## Results Summary

| Model | AUC-ROC | AMS Score | Signal F1 |
|-------|---------|-----------|-----------|
| **XGBoost** | **0.9683** | 121.6 | 0.818 |
| Deep MLP | 0.9640 | 137.2 | 0.825 |
| Random Forest | 0.9564 | 112.5 | 0.783 |

**Key finding:** SHAP top features (MMC mass, visible mass, Δη jet-jet) align with known physics discriminants — validating that the model has learned genuine particle physics, not data artifacts.

---

## Project Structure

```
cern_xai_project/
├── generate_data.py      # ATLAS-like dataset generator (100k events, 28 features)
├── eda.py                # EDA, feature distributions, correlation, preprocessing
├── train_models.py       # XGBoost + Random Forest + Deep MLP training & evaluation
├── xai_explainability.py # SHAP (TreeSHAP + KernelSHAP) + LIME + physics validation
├── run_pipeline.py       # Master script — runs everything end-to-end
├── data/                 # Generated dataset CSV
├── models/               # Saved model pkl files + scaler
├── plots/                # All 16 output figures
└── results/              # model_metrics.json
```

---

## How to Run

```bash
# Install dependencies
pip install xgboost scikit-learn shap lime optuna matplotlib seaborn pandas numpy

# Run the full pipeline
python run_pipeline.py
```

---

## Pipeline Steps

### Step 1 — Data Generation
Simulates 100,000 events (25k signal, 75k background) matching the ATLAS Higgs Challenge structure:
- 28 features: kinematic (PRI_*) and derived (DER_*) variables
- Realistic signal/background distributions based on known physics
- Event weights for luminosity normalization

**To use the real CERN dataset:** Replace `data/atlas_higgs_simulated.csv` with the file from:
- https://opendata.cern.ch/record/328
- https://www.kaggle.com/c/higgs-boson/data

### Step 2 — EDA & Preprocessing
- Feature distributions: signal vs background for 12 key observables
- Pearson correlation heatmap (28×28)
- Log-transform of skewed pT/mass features
- StandardScaler normalization
- 70/15/15 train/val/test stratified split

### Step 3 — Model Training
Three models trained with physics-aware evaluation:
- **XGBoost**: 400 trees, depth 6, scale_pos_weight=3 for class imbalance
- **Random Forest**: 200 trees, balanced class weights
- **Deep MLP**: 4-layer (256→128→64→32), early stopping, L2 regularization

**Metrics**: AUC-ROC (discrimination) + AMS (physics significance, CERN's official metric)

### Step 4 — XAI Explanations

#### SHAP (primary method)
- **TreeSHAP** on XGBoost: exact, O(TLD²) computation
  - Beeswarm plot: global feature importance with directionality
  - Waterfall plot: single event explanation
  - Bar chart: mean |SHAP| ranking
- **KernelSHAP** on MLP: model-agnostic approximation (n=200 events)

#### LIME (secondary method)
- Per-event local linear surrogates for 3 example events:
  - True signal (high confidence)
  - True background (high confidence)
  - Borderline/uncertain event

#### Physics Validation (key differentiator)
- Top SHAP features compared against physicist expectations:
  - `DER_mass_MMC` → Higgs mass estimator, peaks ~125 GeV ✓
  - `DER_mass_vis` → separates H→ττ from Z→ττ (~91 GeV) ✓
  - `DER_deltaeta_jet_jet` → VBF topology signature ✓
  - `PRI_met` → neutrino missing energy ✓
  - `DER_sum_pt` → global event activity ✓

---

## Output Plots

| File | Description |
|------|-------------|
| `00_project_summary.png` | 4-panel project overview |
| `1_feature_distributions.png` | Signal vs background for 12 features |
| `2_correlation_heatmap.png` | Feature correlation matrix |
| `3_class_balance.png` | Class imbalance visualization |
| `4_roc_curves.png` | ROC curves for all 3 models |
| `5_confusion_matrices.png` | Test set confusion matrices |
| `6_score_distributions.png` | Classifier score distributions |
| `7_shap_beeswarm_xgboost.png` | SHAP beeswarm (XGBoost) |
| `8_shap_bar_xgboost.png` | SHAP bar importance |
| `9_shap_waterfall.png` | Single-event SHAP waterfall |
| `10_shap_bar_mlp.png` | SHAP bar importance (MLP) |
| `11_lime_*.png` | LIME per-event explanations (×3) |
| `12_physics_validation.png` | SHAP vs physics expectation |
| `13_shap_model_comparison.png` | XGBoost vs MLP agreement |

---

## Datasets Referenced
- [ATLAS Higgs ML Challenge 2014 (CERN record 328)](https://opendata.cern.ch/record/328)
- [HIGGS dataset, UCI ML Repository](https://archive.ics.uci.edu/dataset/280/higgs)
- [JetClass dataset](https://www.emergentmind.com/topics/jetclass-dataset)

## Key Papers
- Baldi et al. (2014), "Searching for exotic particles in HEP with deep learning", *Nature Comms*
- E-PCN: "Jet Tagging with Explainable Particle Chebyshev Networks" (2025)
- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions" (SHAP)

---

## CERN Internship Alignment
This project directly targets the **CERN openlab Summer Student Programme** (Data Analytics & AI track):
- Uses official CERN open data format and physics conventions
- References ATLAS experiment and HL-LHC upgrade context
- Demonstrates both ML performance *and* physics interpretability
- Codebase is clean, documented, and reproducible

*Next steps to strengthen the portfolio:*
- Implement ParticleNet / GNN on JetClass dataset
- Add Optuna hyperparameter tuning with AMS as objective
- Train on the full 11M HIGGS UCI dataset
- Write a 2-page technical report in IEEE format
