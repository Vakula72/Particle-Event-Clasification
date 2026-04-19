"""
Step 4: Explainability Layer
  - TreeSHAP on XGBoost (global beeswarm + local waterfall)
  - KernelSHAP on MLP (global summary)
  - LIME per-event explanations (3 example events)
  - Physics interpretation: link top features to known HEP observables
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import lime
import lime.lime_tabular
import warnings, os
warnings.filterwarnings("ignore")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_SHORT = {
    "DER_mass_MMC":              "MMC mass (Higgs candidate)",
    "DER_mass_transverse_met_lep":"Transverse mass (MET+lep)",
    "DER_mass_vis":              "Visible mass (τ+lep)",
    "DER_pt_h":                  "Higgs pT",
    "DER_deltaeta_jet_jet":      "Δη (jet-jet)",
    "DER_mass_jet_jet":          "Invariant mass (jet-jet)",
    "DER_prodeta_jet_jet":       "η product (jet-jet)",
    "DER_deltar_tau_lep":        "ΔR (τ-lep)",
    "DER_pt_tot":                "Total pT",
    "DER_sum_pt":                "Scalar sum pT",
    "DER_pt_ratio_lep_tau":      "pT ratio (lep/τ)",
    "DER_met_phi_centrality":    "MET φ centrality",
    "DER_lep_eta_centrality":    "Lepton η centrality",
    "PRI_tau_pt":                "τ pT",
    "PRI_tau_eta":               "τ η",
    "PRI_tau_phi":               "τ φ",
    "PRI_lep_pt":                "Lepton pT",
    "PRI_lep_eta":               "Lepton η",
    "PRI_lep_phi":               "Lepton φ",
    "PRI_met":                   "Missing ET",
    "PRI_met_phi":               "MET φ",
    "PRI_met_sumet":             "Scalar ET sum",
    "PRI_jet_num":               "Number of jets",
    "PRI_jet_leading_pt":        "Leading jet pT",
    "PRI_jet_leading_eta":       "Leading jet η",
    "PRI_jet_leading_phi":       "Leading jet φ",
    "PRI_jet_subleading_pt":     "Subleading jet pT",
    "PRI_jet_subleading_eta":    "Subleading jet η",
}

PHYSICS_NOTES = {
    "DER_mass_MMC":          "★ Primary Higgs mass estimator — peaks ~125 GeV for signal",
    "DER_mass_vis":          "★ Visible tau mass — separates H→ττ from Z→ττ (~91 GeV)",
    "DER_deltaeta_jet_jet":  "★ VBF topology: large Δη indicates vector-boson fusion",
    "DER_mass_jet_jet":      "★ Di-jet invariant mass — VBF signature > 500 GeV",
    "PRI_met":               "★ Missing ET from neutrinos in tau decay",
    "DER_sum_pt":            "★ Global event activity — higher for signal",
    "DER_deltar_tau_lep":    "★ Angular separation — boosted Higgs → collimated decay",
    "PRI_tau_pt":            "★ Tau pT — signal taus are harder (higher pT)",
}

# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def run_shap_xgboost(model, X_test, feature_names, n_samples=800):
    """TreeSHAP — fast exact computation for XGBoost"""
    print("\n--- SHAP Analysis: XGBoost (TreeSHAP) ---")
    explainer   = shap.TreeExplainer(model)
    X_sample    = X_test[:n_samples]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    short_names = [FEATURE_SHORT.get(f, f) for f in feature_names]

    # 1. Beeswarm summary plot
    print("  Plotting SHAP beeswarm...")
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=short_names,
                      plot_type="beeswarm",
                      max_display=15,
                      show=False, color_bar=True)
    plt.title("SHAP Feature Importance — XGBoost\n(Beeswarm: red=high value, blue=low value)",
              fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/7_shap_beeswarm_xgboost.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    # 2. Bar importance plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=short_names,
                      plot_type="bar",
                      max_display=15,
                      show=False, color_bar=False)
    plt.title("Mean |SHAP| Feature Importance — XGBoost",
              fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/8_shap_bar_xgboost.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    # 3. Waterfall for single signal event
    print("  Plotting SHAP waterfall (single event)...")
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
        data=X_sample[0],
        feature_names=short_names,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title("SHAP Waterfall — Single Event Explanation\n(why this event is classified as signal)",
              fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/9_shap_waterfall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    return shap_values, short_names

def run_shap_mlp(model, X_test, feature_names, n_samples=200):
    """KernelSHAP — model-agnostic, slower, for MLP"""
    print("\n--- SHAP Analysis: MLP (KernelSHAP, n=200) ---")
    background  = shap.kmeans(X_test, 30)
    explainer   = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:,1], background)
    X_sample    = X_test[:n_samples]
    shap_values = explainer.shap_values(X_sample, nsamples=100, silent=True)

    short_names = [FEATURE_SHORT.get(f, f) for f in feature_names]

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=short_names,
                      plot_type="bar",
                      max_display=15,
                      show=False, color_bar=False)
    plt.title("Mean |SHAP| Feature Importance — Deep MLP (KernelSHAP)",
              fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    path = f"{PLOT_DIR}/10_shap_bar_mlp.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")
    return shap_values

# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────

def run_lime(model, X_train, X_test, y_test, feature_names, n_events=3):
    """LIME per-event local explanations"""
    print(f"\n--- LIME Analysis: {n_events} example events ---")
    short_names = [FEATURE_SHORT.get(f, f) for f in feature_names]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=short_names,
        class_names=["Background","Signal"],
        discretize_continuous=True,
        random_state=42,
    )

    # Find 1 true-positive signal, 1 true-negative background, 1 borderline
    proba = model.predict_proba(X_test)[:,1]
    tp_idx = np.where((y_test==1) & (proba > 0.80))[0]
    tn_idx = np.where((y_test==0) & (proba < 0.20))[0]
    bd_idx = np.argsort(np.abs(proba - 0.5))[:5]

    event_indices = []
    if len(tp_idx) > 0: event_indices.append(("True Signal (high confidence)", tp_idx[0]))
    if len(tn_idx) > 0: event_indices.append(("True Background (high confidence)", tn_idx[0]))
    if len(bd_idx) > 0: event_indices.append(("Borderline event (uncertain)", bd_idx[0]))

    for label, idx in event_indices[:n_events]:
        exp = explainer.explain_instance(
            X_test[idx], model.predict_proba,
            num_features=10, num_samples=500)
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(8, 5)
        score = proba[idx]
        true_lbl = "Signal" if y_test[idx]==1 else "Background"
        fig.suptitle(f"LIME — {label}\nTrue: {true_lbl}  |  Model score: {score:.3f}",
                     fontsize=10, fontweight="bold", y=1.02)
        plt.tight_layout()
        fname = label.lower().replace(" ","_").replace("(","").replace(")","")
        path = f"{PLOT_DIR}/11_lime_{fname}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

# ─────────────────────────────────────────────
# Physics validation summary
# ─────────────────────────────────────────────

def plot_physics_validation(shap_values, feature_names):
    """
    Key CERN differentiator: compare SHAP rankings to physicist expectations.
    Plot which top features have physics justification.
    """
    print("\n--- Physics Validation Plot ---")
    short_names = [FEATURE_SHORT.get(f, f) for f in feature_names]

    mean_shap = np.mean(np.abs(shap_values), axis=0)
    order     = np.argsort(mean_shap)[::-1][:15]
    top_names = [short_names[i] for i in order]
    top_shap  = [mean_shap[i] for i in order]
    orig_names= [feature_names[i] for i in order]

    colors = []
    for name in orig_names:
        if name in PHYSICS_NOTES:
            colors.append("#1D9E75")   # teal  = physics-motivated
        else:
            colors.append("#888780")   # gray  = additional feature

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_names)), top_shap[::-1], color=colors[::-1], height=0.65)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=10)
    ax.set_title("SHAP Rankings vs Physics Expectation\n"
                 "(teal = known physics discriminant | gray = additional)",
                 fontsize=11, fontweight="bold")
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", alpha=0.35)

    legend_handles = [
        mpatches.Patch(color="#1D9E75", label="Physics-motivated feature (matches HEP theory)"),
        mpatches.Patch(color="#888780", label="Additional learned feature"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    # Annotate physics notes
    for i, (orig, val) in enumerate(zip(orig_names[::-1], top_shap[::-1])):
        if orig in PHYSICS_NOTES:
            note = PHYSICS_NOTES[orig].replace("★ ","")
            ax.text(val + max(top_shap)*0.01, i, note,
                    fontsize=7, va="center", color="#0F6E56", style="italic")

    plt.tight_layout()
    path = f"{PLOT_DIR}/12_physics_validation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

def plot_shap_comparison(xgb_shap, mlp_shap, feature_names):
    """Side-by-side: do XGBoost and MLP agree on important features?"""
    print("Plotting SHAP model comparison...")
    short_names = [FEATURE_SHORT.get(f, f) for f in feature_names]
    n = min(len(xgb_shap[0]), len(mlp_shap[0]), len(feature_names))

    xgb_imp = np.mean(np.abs(xgb_shap), axis=0)[:n]
    mlp_imp = np.mean(np.abs(mlp_shap), axis=0)[:n]

    order = np.argsort(xgb_imp)[::-1][:12]
    names = [short_names[i] for i in order]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, xgb_imp[order], width, label="XGBoost (TreeSHAP)", color="#1D9E75", alpha=0.85)
    ax.bar(x + width/2, mlp_imp[order], width, label="MLP (KernelSHAP)",   color="#534AB7", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Feature Importance Agreement: XGBoost vs MLP\n"
                 "(High agreement → robust physics insight)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    path = f"{PLOT_DIR}/13_shap_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

if __name__ == "__main__":
    import pickle
    from eda import load_and_split, preprocess
    X, y, w, feature_cols = load_and_split()
    X_train,X_val,X_test,y_train,y_val,y_test,w_train,w_val,w_test,_ = preprocess(X,y,w,feature_cols)
    with open("models/xgboost_model.pkl","rb") as f: xgb_m = pickle.load(f)
    with open("models/mlp_model.pkl","rb") as f:     mlp_m = pickle.load(f)
    xgb_shap, _ = run_shap_xgboost(xgb_m, X_test, feature_cols)
    mlp_shap     = run_shap_mlp(mlp_m, X_test, feature_cols)
    run_lime(xgb_m, X_train, X_test, y_test, feature_cols)
    plot_physics_validation(xgb_shap, feature_cols)
    plot_shap_comparison(xgb_shap, mlp_shap, feature_cols)
    print("\nXAI analysis complete.")
