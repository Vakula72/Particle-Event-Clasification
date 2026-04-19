"""
ATLAS-like Particle Event Data Generator
Simulates the CERN ATLAS Higgs ML Challenge dataset structure.
Features match the real dataset: 28 kinematic + derived features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

FEATURE_NAMES = [
    "DER_mass_MMC",         # Higgs boson mass estimate (MMC)
    "DER_mass_transverse_met_lep",  # transverse mass of MET+lepton
    "DER_mass_vis",         # visible mass of tau+lepton
    "DER_pt_h",             # pT of Higgs candidate
    "DER_deltaeta_jet_jet", # delta-eta between jets
    "DER_mass_jet_jet",     # invariant mass of jet pair
    "DER_prodeta_jet_jet",  # product of jet pseudorapidities
    "DER_deltar_tau_lep",   # delta-R between tau and lepton
    "DER_pt_tot",           # total pT of all objects
    "DER_sum_pt",           # scalar sum of pT
    "DER_pt_ratio_lep_tau", # pT ratio lepton/tau
    "DER_met_phi_centrality",# MET phi centrality
    "DER_lep_eta_centrality",# lepton eta centrality
    "PRI_tau_pt",           # tau pT (primary)
    "PRI_tau_eta",          # tau pseudorapidity
    "PRI_tau_phi",          # tau azimuthal angle
    "PRI_lep_pt",           # lepton pT
    "PRI_lep_eta",          # lepton pseudorapidity
    "PRI_lep_phi",          # lepton azimuthal angle
    "PRI_met",              # missing transverse energy
    "PRI_met_phi",          # MET azimuthal angle
    "PRI_met_sumet",        # scalar sum of all ET
    "PRI_jet_num",          # number of jets
    "PRI_jet_leading_pt",   # leading jet pT
    "PRI_jet_leading_eta",  # leading jet eta
    "PRI_jet_leading_phi",  # leading jet phi
    "PRI_jet_subleading_pt",# subleading jet pT
    "PRI_jet_subleading_eta",# subleading jet eta
]

def generate_signal_events(n):
    """Simulate H→ττ signal events"""
    # Higgs mass ~125 GeV — MMC peaks near 125
    DER_mass_MMC          = np.random.normal(125, 25, n).clip(0)
    DER_mass_transverse   = np.random.exponential(50, n)
    DER_mass_vis          = np.random.normal(80, 20, n).clip(0)
    DER_pt_h              = np.random.exponential(40, n)
    DER_deltaeta_jet_jet  = np.random.normal(3.5, 1.2, n)
    DER_mass_jet_jet      = np.random.exponential(200, n)
    DER_prodeta_jet_jet   = np.random.normal(-2, 3, n)
    DER_deltar_tau_lep    = np.random.uniform(0.4, 4.0, n)
    DER_pt_tot            = np.random.exponential(30, n)
    DER_sum_pt            = np.random.normal(120, 50, n).clip(0)
    DER_pt_ratio          = np.random.normal(1.1, 0.4, n).clip(0.1)
    DER_met_phi           = np.random.uniform(-1, 1, n)
    DER_lep_eta           = np.random.uniform(-1, 1, n)
    PRI_tau_pt            = np.random.exponential(45, n)
    PRI_tau_eta           = np.random.normal(0, 1.5, n)
    PRI_tau_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_lep_pt            = np.random.exponential(55, n)
    PRI_lep_eta           = np.random.normal(0, 1.5, n)
    PRI_lep_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_met               = np.random.exponential(45, n)
    PRI_met_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_met_sumet         = np.random.normal(200, 80, n).clip(0)
    PRI_jet_num           = np.random.choice([0,1,2,3], n, p=[0.1,0.2,0.4,0.3])
    PRI_jet_lead_pt       = np.random.exponential(60, n)
    PRI_jet_lead_eta      = np.random.normal(0, 2, n)
    PRI_jet_lead_phi      = np.random.uniform(-np.pi, np.pi, n)
    PRI_jet_sub_pt        = np.random.exponential(40, n)
    PRI_jet_sub_eta       = np.random.normal(0, 2, n)

    return np.column_stack([
        DER_mass_MMC, DER_mass_transverse, DER_mass_vis, DER_pt_h,
        DER_deltaeta_jet_jet, DER_mass_jet_jet, DER_prodeta_jet_jet,
        DER_deltar_tau_lep, DER_pt_tot, DER_sum_pt, DER_pt_ratio,
        DER_met_phi, DER_lep_eta, PRI_tau_pt, PRI_tau_eta, PRI_tau_phi,
        PRI_lep_pt, PRI_lep_eta, PRI_lep_phi, PRI_met, PRI_met_phi,
        PRI_met_sumet, PRI_jet_num, PRI_jet_lead_pt, PRI_jet_lead_eta,
        PRI_jet_lead_phi, PRI_jet_sub_pt, PRI_jet_sub_eta
    ])

def generate_background_events(n):
    """Simulate Z→ττ + tt̄ + W background events"""
    # Z boson ~91 GeV — MMC smeared lower
    DER_mass_MMC          = np.random.normal(91, 30, n).clip(0)
    DER_mass_transverse   = np.random.exponential(35, n)
    DER_mass_vis          = np.random.normal(65, 22, n).clip(0)
    DER_pt_h              = np.random.exponential(30, n)
    DER_deltaeta_jet_jet  = np.random.normal(2.0, 1.5, n)
    DER_mass_jet_jet      = np.random.exponential(120, n)
    DER_prodeta_jet_jet   = np.random.normal(0, 3, n)
    DER_deltar_tau_lep    = np.random.uniform(0.4, 5.0, n)
    DER_pt_tot            = np.random.exponential(20, n)
    DER_sum_pt            = np.random.normal(90, 45, n).clip(0)
    DER_pt_ratio          = np.random.normal(0.9, 0.5, n).clip(0.1)
    DER_met_phi           = np.random.uniform(-1, 1, n)
    DER_lep_eta           = np.random.uniform(-1, 1, n)
    PRI_tau_pt            = np.random.exponential(35, n)
    PRI_tau_eta           = np.random.normal(0, 2.0, n)
    PRI_tau_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_lep_pt            = np.random.exponential(45, n)
    PRI_lep_eta           = np.random.normal(0, 2.0, n)
    PRI_lep_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_met               = np.random.exponential(30, n)
    PRI_met_phi           = np.random.uniform(-np.pi, np.pi, n)
    PRI_met_sumet         = np.random.normal(160, 70, n).clip(0)
    PRI_jet_num           = np.random.choice([0,1,2,3], n, p=[0.3,0.3,0.3,0.1])
    PRI_jet_lead_pt       = np.random.exponential(45, n)
    PRI_jet_lead_eta      = np.random.normal(0, 2.5, n)
    PRI_jet_lead_phi      = np.random.uniform(-np.pi, np.pi, n)
    PRI_jet_sub_pt        = np.random.exponential(30, n)
    PRI_jet_sub_eta       = np.random.normal(0, 2.5, n)

    return np.column_stack([
        DER_mass_MMC, DER_mass_transverse, DER_mass_vis, DER_pt_h,
        DER_deltaeta_jet_jet, DER_mass_jet_jet, DER_prodeta_jet_jet,
        DER_deltar_tau_lep, DER_pt_tot, DER_sum_pt, DER_pt_ratio,
        DER_met_phi, DER_lep_eta, PRI_tau_pt, PRI_tau_eta, PRI_tau_phi,
        PRI_lep_pt, PRI_lep_eta, PRI_lep_phi, PRI_met, PRI_met_phi,
        PRI_met_sumet, PRI_jet_num, PRI_jet_lead_pt, PRI_jet_lead_eta,
        PRI_jet_lead_phi, PRI_jet_sub_pt, PRI_jet_sub_eta
    ])

def generate_dataset(n_signal=25000, n_background=75000):
    """Generate full dataset with signal/background and event weights"""
    print(f"Generating {n_signal} signal + {n_background} background events...")

    sig = generate_signal_events(n_signal)
    bkg = generate_background_events(n_background)

    X = np.vstack([sig, bkg])
    y = np.array([1]*n_signal + [0]*n_background)

    # Realistic event weights (luminosity normalization)
    w_sig = np.random.uniform(1.0, 3.0, n_signal)
    w_bkg = np.random.uniform(0.5, 2.0, n_background)
    weights = np.concatenate([w_sig, w_bkg])

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Label"] = y
    df["Weight"] = weights

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/atlas_higgs_simulated.csv", index=False)
    print(f"Saved: data/atlas_higgs_simulated.csv  ({len(df)} rows, {len(FEATURE_NAMES)} features)")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(3))
    print(f"\nClass balance: {df['Label'].value_counts().to_dict()}")
