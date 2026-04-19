"""
MASTER PIPELINE — CERN XAI Particle Event Classification
=========================================================
Run this file to execute the full pipeline end-to-end:
  1. Generate ATLAS-like dataset
  2. EDA + Preprocessing
  3. Train XGBoost / Random Forest / Deep MLP
  4. XAI: SHAP (TreeSHAP + KernelSHAP) + LIME
  5. Save all results and plots

Usage:
  python run_pipeline.py
"""

import os, sys, time, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*65)
print("  CERN XAI PROJECT — Particle Event Classification Pipeline")
print("  'Explainable AI for High-Energy Particle Physics'")
print("="*65)

# ─── STEP 1: DATA ───────────────────────────────────────────────
print("\n[STEP 1/5]  Generating ATLAS-like dataset...")
from generate_data import generate_dataset
t0 = time.time()
df = generate_dataset(n_signal=25000, n_background=75000)
print(f"  Done in {time.time()-t0:.1f}s")

# ─── STEP 2: EDA ────────────────────────────────────────────────
print("\n[STEP 2/5]  EDA + Preprocessing...")
from eda import load_and_split, preprocess, plot_class_balance, \
                plot_feature_distributions, plot_correlation_heatmap, ams_score
X, y, w, feature_cols = load_and_split()
plot_class_balance(y)
plot_feature_distributions(X, y, feature_cols)
plot_correlation_heatmap(X)
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 w_train, w_val, w_test, scaler) = preprocess(X, y, w, feature_cols)
with open("models/scaler.pkl","wb") as f: pickle.dump(scaler, f)

# ─── STEP 3: TRAIN ──────────────────────────────────────────────
print("\n[STEP 3/5]  Training 3 models...")
from train_models import (train_xgboost, train_random_forest, train_mlp,
                           plot_roc_curves, plot_confusion_matrices,
                           plot_score_distributions, evaluate_all)

xgb_m = train_xgboost(X_train, y_train, w_train, X_val, y_val)
rf_m  = train_random_forest(X_train, y_train, w_train, X_val, y_val)
mlp_m = train_mlp(X_train, y_train, X_val, y_val)
models = {"XGBoost": xgb_m, "Random Forest": rf_m, "Deep MLP": mlp_m}

results = evaluate_all(models, X_test, y_test, w_test)
plot_roc_curves(models, X_test, y_test)
plot_confusion_matrices(models, X_test, y_test)
plot_score_distributions(models, X_test, y_test)

# Save results JSON
with open("results/model_metrics.json","w") as f:
    json.dump({k: {m: float(f"{v:.4f}") for m,v in v2.items()}
               for k,v2 in results.items()}, f, indent=2)

# ─── STEP 4: XAI ────────────────────────────────────────────────
print("\n[STEP 4/5]  XAI: SHAP + LIME explanations...")
from xai_explainability import (run_shap_xgboost, run_shap_mlp, run_lime,
                                  plot_physics_validation, plot_shap_comparison)

xgb_shap, _ = run_shap_xgboost(xgb_m, X_test, feature_cols, n_samples=800)
mlp_shap     = run_shap_mlp(mlp_m, X_test, feature_cols, n_samples=200)
run_lime(xgb_m, X_train, X_test, y_test, feature_cols, n_events=3)
plot_physics_validation(xgb_shap, feature_cols)
plot_shap_comparison(xgb_shap, mlp_shap, feature_cols)

# ─── STEP 5: SUMMARY FIGURE ─────────────────────────────────────
print("\n[STEP 5/5]  Generating project summary figure...")

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

SIG_COLOR = "#1D9E75"
BKG_COLOR = "#D85A30"
MLP_COLOR = "#534AB7"
RF_COLOR  = "#BA7517"

# Panel A — ROC summary
ax_roc = fig.add_subplot(gs[0, :2])
from sklearn.metrics import roc_curve, roc_auc_score
colors_ = {"XGBoost": SIG_COLOR, "Random Forest": RF_COLOR, "Deep MLP": MLP_COLOR}
styles_ = {"XGBoost": "-", "Random Forest": "--", "Deep MLP": "-."}
for name, model in models.items():
    proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=colors_[name], lw=2, linestyle=styles_[name])
ax_roc.plot([0,1],[0,1],"k--",lw=1,alpha=0.35)
ax_roc.set_xlabel("FPR", fontsize=9); ax_roc.set_ylabel("TPR", fontsize=9)
ax_roc.set_title("A — ROC Curves", fontsize=10, fontweight="bold")
ax_roc.legend(fontsize=8); ax_roc.set_facecolor("#FAFAFA"); ax_roc.grid(True, alpha=0.3)

# Panel B — Metrics bar chart
ax_met = fig.add_subplot(gs[0, 2:])
metric_names = list(results.keys())
aucs = [results[m]["AUC"] for m in metric_names]
amss = [results[m]["AMS"] for m in metric_names]
x = np.arange(len(metric_names))
w_ = 0.35
ax_met.bar(x - w_/2, aucs, w_, label="AUC-ROC",
           color=[colors_[m] for m in metric_names], alpha=0.85)
ax_met2 = ax_met.twinx()
ax_met2.bar(x + w_/2, amss, w_, label="AMS score",
            color=[colors_[m] for m in metric_names], alpha=0.45, hatch="//")
ax_met.set_xticks(x); ax_met.set_xticklabels(metric_names, fontsize=9)
ax_met.set_ylabel("AUC-ROC", fontsize=9); ax_met2.set_ylabel("AMS", fontsize=9)
ax_met.set_ylim(0.5, 1.05); ax_met.set_title("B — Model Comparison", fontsize=10, fontweight="bold")
ax_met.set_facecolor("#FAFAFA"); ax_met.grid(axis="y", alpha=0.3)

# Panel C — SHAP top features
ax_shap = fig.add_subplot(gs[1, :2])
from xai_explainability import FEATURE_SHORT, PHYSICS_NOTES
mean_shap  = np.mean(np.abs(xgb_shap), axis=0)
order      = np.argsort(mean_shap)[::-1][:10]
top_vals   = mean_shap[order]
top_labels = [FEATURE_SHORT.get(feature_cols[i], feature_cols[i]) for i in order]
bar_colors = ["#1D9E75" if feature_cols[i] in PHYSICS_NOTES else "#888780" for i in order]
ax_shap.barh(range(10), top_vals[::-1], color=bar_colors[::-1], height=0.65)
ax_shap.set_yticks(range(10))
ax_shap.set_yticklabels(top_labels[::-1], fontsize=8)
ax_shap.set_xlabel("Mean |SHAP|", fontsize=9)
ax_shap.set_title("C — SHAP Feature Importance (XGBoost)", fontsize=10, fontweight="bold")
ax_shap.set_facecolor("#FAFAFA"); ax_shap.grid(axis="x", alpha=0.3)

# Panel D — Score distributions (XGBoost)
ax_dist = fig.add_subplot(gs[1, 2:])
proba_xgb = xgb_m.predict_proba(X_test)[:,1]
bins = np.linspace(0, 1, 35)
ax_dist.hist(proba_xgb[y_test==1], bins=bins, density=True, alpha=0.65,
             color=SIG_COLOR, label="Signal (H→ττ)", histtype="stepfilled")
ax_dist.hist(proba_xgb[y_test==0], bins=bins, density=True, alpha=0.55,
             color=BKG_COLOR, label="Background", histtype="stepfilled")
ax_dist.set_xlabel("XGBoost Score", fontsize=9)
ax_dist.set_ylabel("Density", fontsize=9)
ax_dist.set_title("D — Score Distribution (XGBoost)", fontsize=10, fontweight="bold")
ax_dist.legend(fontsize=9); ax_dist.set_facecolor("#FAFAFA"); ax_dist.grid(True, alpha=0.3)

fig.suptitle("Explainable AI for High-Energy Particle Event Classification\n"
             "ATLAS H→ττ Simulated Dataset  |  XGBoost + SHAP + LIME",
             fontsize=13, fontweight="bold", y=1.01)

path = "plots/00_project_summary.png"
plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved: {path}")

# ─── FINAL SUMMARY ──────────────────────────────────────────────
print("\n" + "="*65)
print("  PIPELINE COMPLETE")
print("="*65)
best_model = max(results, key=lambda k: results[k]["AUC"])
print(f"  Best model  : {best_model}")
print(f"  Best AUC    : {results[best_model]['AUC']:.4f}")
print(f"  Best AMS    : {results[best_model]['AMS']:.4f}")
print(f"\n  Plots saved : plots/  ({len(os.listdir('plots'))} files)")
print(f"  Models saved: models/")
print(f"  Metrics     : results/model_metrics.json")
print("="*65)
