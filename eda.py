"""
Step 2: Exploratory Data Analysis & Preprocessing
- Feature distributions (signal vs background)
- Correlation heatmap
- Log-transform skewed features
- StandardScaler normalization
- Train/val/test split
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, warnings
warnings.filterwarnings("ignore")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Physics-motivated color palette
SIG_COLOR = "#1D9E75"   # teal  = signal (Higgs)
BKG_COLOR = "#D85A30"   # coral = background
GRID_COLOR = "#E8E8E8"

def load_and_split(csv_path="data/atlas_higgs_simulated.csv"):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ["Label","Weight"]]
    X = df[feature_cols].copy()
    y = df["Label"].values
    w = df["Weight"].values
    return X, y, w, feature_cols

def plot_feature_distributions(X, y, feature_cols):
    """Signal vs background for top 12 most discriminating features"""
    print("Plotting feature distributions...")
    key_features = [
        "DER_mass_MMC", "DER_mass_vis", "DER_pt_h", "DER_sum_pt",
        "PRI_tau_pt", "PRI_lep_pt", "PRI_met", "DER_mass_jet_jet",
        "DER_deltaeta_jet_jet", "DER_deltar_tau_lep", "PRI_jet_num", "DER_pt_tot"
    ]
    key_features = [f for f in key_features if f in feature_cols]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        ax = axes[i]
        sig_vals = X[feat][y == 1]
        bkg_vals = X[feat][y == 0]
        lo = np.percentile(np.concatenate([sig_vals, bkg_vals]), 1)
        hi = np.percentile(np.concatenate([sig_vals, bkg_vals]), 99)
        bins = np.linspace(lo, hi, 40)
        ax.hist(sig_vals, bins=bins, density=True, alpha=0.65,
                color=SIG_COLOR, label="Signal (H→ττ)", histtype="stepfilled")
        ax.hist(bkg_vals, bins=bins, density=True, alpha=0.55,
                color=BKG_COLOR, label="Background", histtype="stepfilled")
        ax.set_title(feat.replace("_"," "), fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.7)

    plt.suptitle("Feature Distributions: Signal vs Background\n(ATLAS H→ττ Simulated Dataset)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{PLOT_DIR}/1_feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def plot_correlation_heatmap(X):
    """Pearson correlation heatmap"""
    print("Plotting correlation heatmap...")
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8, center=0,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.7},
                annot=False, ax=ax)
    ax.set_title("Feature Correlation Matrix\n(ATLAS Simulated Dataset)",
                 fontsize=13, fontweight="bold", pad=10)
    short_names = [f.replace("DER_","D:").replace("PRI_","P:") for f in X.columns]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, rotation=0, fontsize=7)
    plt.tight_layout()
    path = f"{PLOT_DIR}/2_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def preprocess(X, y, w, feature_cols):
    """Log-transform, scale, and split"""
    print("Preprocessing: log-transform + standardize...")
    X = X.copy()

    # Log-transform skewed positive features (pT, mass, MET)
    log_features = [f for f in feature_cols if any(
        k in f for k in ["pt","mass","met","sumet","jet_jet"])]
    for feat in log_features:
        if feat in X.columns:
            X[feat] = np.log1p(np.abs(X[feat]))

    # Train / val / test  70 / 15 / 15
    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        X.values, y, w, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test, scaler

def plot_class_balance(y):
    """Class imbalance bar chart"""
    unique, counts = np.unique(y, return_counts=True)
    labels = ["Background (Z,tt,W)", "Signal (H→ττ)"]
    colors = [BKG_COLOR, SIG_COLOR]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.4, edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Events", fontsize=10)
    ax.set_title("Class Distribution\n(Note: 3:1 imbalance — use AMS metric)", fontsize=11)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = f"{PLOT_DIR}/3_class_balance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def ams_score(y_true, y_pred_proba, weights, threshold=0.5):
    """
    Approximate Median Significance (AMS) — CERN's metric for the Higgs challenge.
    AMS = sqrt(2*((s+b+b_reg)*log(1 + s/(b+b_reg)) - s))
    """
    mask = y_pred_proba >= threshold
    s = np.sum(weights[mask] * (y_true[mask] == 1))
    b = np.sum(weights[mask] * (y_true[mask] == 0))
    b_reg = 10.0  # regularization
    if b + b_reg <= 0:
        return 0.0
    ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s))
    return float(ams)

if __name__ == "__main__":
    X, y, w, feature_cols = load_and_split()
    plot_class_balance(y)
    plot_feature_distributions(X, y, feature_cols)
    plot_correlation_heatmap(X)
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test, scaler = preprocess(X, y, w, feature_cols)
    print("EDA complete.")
