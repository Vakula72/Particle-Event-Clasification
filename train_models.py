"""
Step 3: Train 3 Models
  1. XGBoost (primary — fast, interpretable)
  2. Random Forest (baseline comparison)
  3. Deep MLP Neural Network (high-capacity)
Evaluation: AUC-ROC, AMS score, classification report
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import xgboost as xgb
import os, pickle, warnings, time
warnings.filterwarnings("ignore")

PLOT_DIR = "plots"
MODEL_DIR = "models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SIG_COLOR  = "#1D9E75"
BKG_COLOR  = "#D85A30"
MLP_COLOR  = "#534AB7"
RF_COLOR   = "#BA7517"

def ams_score(y_true, y_pred_proba, weights, threshold=0.5):
    mask = y_pred_proba >= threshold
    s = np.sum(weights[mask] * (y_true[mask] == 1))
    b = np.sum(weights[mask] * (y_true[mask] == 0))
    b_reg = 10.0
    if b + b_reg <= 0:
        return 0.0
    return float(np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s)))

def train_xgboost(X_train, y_train, w_train, X_val, y_val):
    print("\n--- Training XGBoost ---")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=1.0,
        scale_pos_weight=3.0,  # handle class imbalance
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              sample_weight=w_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    elapsed = time.time() - t0
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    print(f"  AUC: {auc:.4f}  |  Time: {elapsed:.1f}s")
    with open(f"{MODEL_DIR}/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def train_random_forest(X_train, y_train, w_train, X_val, y_val):
    print("\n--- Training Random Forest ---")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    elapsed = time.time() - t0
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    print(f"  AUC: {auc:.4f}  |  Time: {elapsed:.1f}s")
    with open(f"{MODEL_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def train_mlp(X_train, y_train, X_val, y_val):
    print("\n--- Training Deep MLP ---")
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,           # L2 regularization
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    print(f"  AUC: {auc:.4f}  |  Time: {elapsed:.1f}s  |  Iters: {model.n_iter_}")
    with open(f"{MODEL_DIR}/mlp_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

def plot_roc_curves(models_dict, X_test, y_test):
    """Overlay ROC curves for all models"""
    print("\nPlotting ROC curves...")
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"XGBoost": SIG_COLOR, "Random Forest": RF_COLOR, "Deep MLP": MLP_COLOR}
    styles = {"XGBoost": "-", "Random Forest": "--", "Deep MLP": "-."}

    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})",
                color=colors[name], lw=2, linestyle=styles[name])

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random classifier")
    ax.fill_between([0,1],[0,1],[0,1], alpha=0.04, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Particle Event Classification\n(Signal: H→ττ  |  Background: Z, tt̄, W)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/4_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def plot_confusion_matrices(models_dict, X_test, y_test):
    """Side-by-side confusion matrices"""
    print("Plotting confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    class_names = ["Background", "Signal"]

    for ax, (name, model) in zip(axes, models_dict.items()):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/5_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def plot_score_distributions(models_dict, X_test, y_test):
    """Classifier output score distributions — signal vs background"""
    print("Plotting score distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, model) in zip(axes, models_dict.items()):
        proba = model.predict_proba(X_test)[:,1]
        bins = np.linspace(0, 1, 40)
        ax.hist(proba[y_test==1], bins=bins, density=True, alpha=0.65,
                color=SIG_COLOR, label="Signal", histtype="stepfilled")
        ax.hist(proba[y_test==0], bins=bins, density=True, alpha=0.55,
                color=BKG_COLOR, label="Background", histtype="stepfilled")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Classifier Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=9)
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Score Distributions: Signal vs Background",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{PLOT_DIR}/6_score_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def evaluate_all(models_dict, X_test, y_test, w_test):
    """Print a clean results summary"""
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    results = {}
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:,1]
        preds = model.predict(X_test)
        auc   = roc_auc_score(y_test, proba)
        ams   = ams_score(y_test, proba, w_test, threshold=0.5)
        rep   = classification_report(y_test, preds, target_names=["Background","Signal"],
                                      output_dict=True)
        results[name] = {"AUC": auc, "AMS": ams,
                         "Signal_F1": rep["Signal"]["f1-score"],
                         "Background_F1": rep["Background"]["f1-score"]}
        print(f"\n{name}:")
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  AMS     : {ams:.4f}")
        print(f"  Signal F1     : {rep['Signal']['f1-score']:.4f}")
        print(f"  Background F1 : {rep['Background']['f1-score']:.4f}")
    print("="*60)
    return results

if __name__ == "__main__":
    # Quick self-test
    from eda import load_and_split, preprocess
    X, y, w, feature_cols = load_and_split()
    X_train,X_val,X_test,y_train,y_val,y_test,w_train,w_val,w_test,_ = preprocess(X,y,w,feature_cols)
    xgb_m = train_xgboost(X_train, y_train, w_train, X_val, y_val)
    rf_m  = train_random_forest(X_train, y_train, w_train, X_val, y_val)
    mlp_m = train_mlp(X_train, y_train, X_val, y_val)
    models = {"XGBoost": xgb_m, "Random Forest": rf_m, "Deep MLP": mlp_m}
    evaluate_all(models, X_test, y_test, w_test)
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_score_distributions(models, X_test, y_test)
