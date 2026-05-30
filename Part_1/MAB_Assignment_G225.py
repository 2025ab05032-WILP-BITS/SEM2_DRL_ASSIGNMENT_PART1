# =============================================================
# Adaptive Treatment Recommendation System using Multi-Armed Bandit Learning
# Group Number: G = 225
# Assignment: DRL Semester 2 – Phase #1 (MAB)
# Author: Group 225
# =============================================================

# ─────────────────────────────────────────────────────────────────
# CELL 0 – Virtual Lab Header
# Prints execution timestamp and VM identifier as required by the
# assignment submission guidelines.
# ─────────────────────────────────────────────────────────────────

import datetime
import platform
import socket

print("=" * 60)
print("VIRTUAL LAB EXECUTION HEADER")
print("=" * 60)
print(f"Timestamp     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Hostname (VM) : {socket.gethostname()}")
print(f"Platform      : {platform.system()} {platform.release()}")
print(f"Python Version: {platform.python_version()}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# CELL 1 – Library Imports
# All libraries imported once at the top so every subsequent cell
# can reference them directly without re-importing.
# ─────────────────────────────────────────────────────────────────

import os
import json
import random          # Python built-in RNG (seeded for reproducibility)
import numpy as np     # Numerical operations and stochastic sampling
import pandas as pd    # DataFrame construction and manipulation
import matplotlib.pyplot as plt  # Plotting (used in Task 5)

print("Libraries loaded successfully.")

# ─────────────────────────────────────────────────────────────────
# CELL 2 – Configuration & Reproducibility Seeds
#
# All environment parameters are defined at module (global) scope
# so that every task cell can reference TRUE_P, K, and G directly
# without passing them as arguments or re-declaring them.
# ─────────────────────────────────────────────────────────────────

# ── Group identifier ──────────────────────────────────────────────
G = 225

# ── Strict reproducibility: seed BOTH random engines ─────────────
# This guarantees that every Bernoulli trial produces the same
# sequence of outcomes across all runs and machines.
random.seed(G)
np.random.seed(G)

# ── Number of medicines (MAB arms) ───────────────────────────────
# Formula: K = (G mod 3) + 5
# G=225 → 225 % 3 = 0 → K = 0 + 5 = 5
K = (G % 3) + 5

# ── Hidden success probabilities for each medicine ───────────────
# Formula: P_i = 0.4 + ((G + i) mod 6) × 0.07
# Computed for i ∈ {0, 1, 2, 3, 4}:
#   P_0 = 0.4 + (225 % 6) * 0.07 = 0.4 + 3*0.07 = 0.61
#   P_1 = 0.4 + (226 % 6) * 0.07 = 0.4 + 4*0.07 = 0.68
#   P_2 = 0.4 + (227 % 6) * 0.07 = 0.4 + 5*0.07 = 0.75  ← optimal
#   P_3 = 0.4 + (228 % 6) * 0.07 = 0.4 + 0*0.07 = 0.40
#   P_4 = 0.4 + (229 % 6) * 0.07 = 0.4 + 1*0.07 = 0.47
TRUE_P = [round(0.4 + ((G + i) % 6) * 0.07, 10) for i in range(K)]

print("Configuration locked.")
print(f"  G (Group Number)              : {G}")
print(f"  K (Number of Medicines/Arms)  : {K}")
print(f"  TRUE_P (Hidden Probabilities) : {TRUE_P}")

# ─────────────────────────────────────────────────────────────────
# CELL 3 – Base Dataset Construction (df_base)
#
# Creates the static patient registry with 1000 sequential records.
# The three dynamic columns (assigned_medicine, clinical_outcome,
# utility_score) are pre-allocated as NaN; each bandit algorithm
# will work on a COPY of this DataFrame so df_base always stays
# clean and can be reused across all tasks.
# ─────────────────────────────────────────────────────────────────

# ── Static columns ────────────────────────────────────────────────

# patient_id: sequential integer index 0 → 999
patient_ids = np.arange(1000)

# severity_score: deterministic cycle using (patient_id % 5) + 1
# Produces repeating pattern: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 ...
severity_scores = (patient_ids % 5) + 1

# ── Assemble the DataFrame ────────────────────────────────────────
df_base = pd.DataFrame({
    "patient_id"        : patient_ids,
    "severity_score"    : severity_scores,
    # Dynamic columns – populated by each bandit algorithm at runtime
    "assigned_medicine" : np.nan,
    "clinical_outcome"  : np.nan,
    "utility_score"     : np.nan,
})

# ── Set patient_id as the index for O(1) row access during sim ────
df_base.set_index("patient_id", inplace=True)

print(f"df_base created: {len(df_base)} rows × {len(df_base.columns)} columns")
print(f"Columns: {list(df_base.columns)}")

# ─────────────────────────────────────────────────────────────────
# CELL 4 – Environmental Feedback Engine: simulate_treatment()
#
# This function models the hospital's feedback loop when a medicine
# is administered to a patient. It is the single source of truth
# for all stochastic outcomes and is called by every bandit
# algorithm in Tasks 2, 3, and 4.
#
# Parameters
# ----------
# medicine_idx  : int  – index of the selected medicine (0 to K-1)
# severity_score: int  – patient's disease severity (1 to 5)
#
# Returns
# -------
# (clinical_outcome, utility_score) : tuple(int, float)
#   clinical_outcome – 1 if patient recovered, 0 otherwise
#   utility_score    – severity-penalised reward signal
# ─────────────────────────────────────────────────────────────────

def simulate_treatment(medicine_idx: int, severity_score: int) -> tuple:
    """
    Simulate administering a medicine to a patient and return the
    clinical outcome and the corresponding utility (reward) score.

    Clinical Outcome
    ----------------
    Drawn as a single Bernoulli trial with success probability
    TRUE_P[medicine_idx]. np.random.binomial(1, p) returns 1 with
    probability p and 0 with probability (1-p).

    Utility Score
    -------------
    Penalises recoveries for high-severity patients to reflect
    the reduced clinical benefit of any treatment in critical cases:

        UtilityScore = clinical_outcome × (1 − severity / 10)

    Examples:
      Recovered,  severity=1 → 1 × 0.9 = 0.9
      Recovered,  severity=5 → 1 × 0.5 = 0.5
      Not recovered          → 0 × anything = 0.0
    """

    # ── Step 1: Stochastic Bernoulli trial ────────────────────────
    # Retrieve the hidden success probability for the chosen medicine
    success_prob = TRUE_P[medicine_idx]

    # Draw a single binary sample: 1 (recovered) or 0 (not recovered)
    clinical_outcome = int(np.random.binomial(1, success_prob))

    # ── Step 2: Severity-penalised utility calculation ────────────
    # The reward diminishes linearly as disease severity increases,
    # so the bandit must balance both arm quality and patient context.
    utility_score = clinical_outcome * (1 - severity_score / 10)

    return (clinical_outcome, utility_score)


# ── Quick sanity check (does not consume the main simulation seed) ─
print("simulate_treatment() defined.")
print("Quick type-check (medicine=2, severity=3):")
_out = simulate_treatment(2, 3)  # P=0.75; expected utility: 0.7 or 0.0
print(f"  clinical_outcome = {_out[0]}  |  utility_score = {_out[1]}")
print("  (Result is stochastic; 0.7 expected ~75% of the time)")

# ─────────────────────────────────────────────────────────────────
# CELL 5 – Task 1 Output Verification
#
# Prints a formatted environment summary and the first 10 rows of
# df_base to confirm that:
#   1. All group parameters are correctly computed.
#   2. severity_score cycles cleanly: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5
#   3. Dynamic columns are initialised as NaN (ready for algorithms).
# ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("TASK 1 – ENVIRONMENT SUMMARY")
print("=" * 60)
print(f"  Group Number  (G)  : {G}")
print(f"  No. of Medicines (K): {K}")
print()
print("  Hidden Success Probabilities (TRUE_P):")
for idx, p in enumerate(TRUE_P):
    marker = "  ← OPTIMAL" if p == max(TRUE_P) else ""
    print(f"    Medicine {idx} : P = {p:.2f}{marker}")
print()

print("-" * 60)
print("First 10 rows of df_base (patient registry):")
print("-" * 60)

# Reset index for display so patient_id appears as a visible column
display_df = df_base.reset_index()
print(display_df.head(10).to_string(index=False))

print()
print(f"Total records : {len(df_base)}")
print(f"severity_score unique values : {sorted(df_base['severity_score'].unique())}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# CELL 6 – Persist df_base to Parquet
#
# Saves the base patient registry (with NaN dynamic columns) to a
# Parquet file so that Task 2, 3, and 4 notebooks can each load a
# clean copy without re-running the construction logic.
#
# Also saves a lightweight JSON sidecar with the group configuration
# (G, K, TRUE_P) so downstream notebooks can restore constants
# without re-deriving them.
# ─────────────────────────────────────────────────────────────────

# ── Output directory (co-located with this script) ───────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(DATA_DIR, "df_base.parquet")
CONFIG_PATH  = os.path.join(DATA_DIR, "group_config.json")

# ── Save df_base (reset index so patient_id becomes a column) ─────
df_base.reset_index().to_parquet(PARQUET_PATH, index=False)

# ── Save group configuration ──────────────────────────────────────
group_config = {"G": G, "K": K, "TRUE_P": TRUE_P}
with open(CONFIG_PATH, "w") as f:
    json.dump(group_config, f, indent=2)

print(f"df_base saved  → {PARQUET_PATH}")
print(f"Config saved   → {CONFIG_PATH}")
print(f"Parquet shape  : {df_base.reset_index().shape}")
