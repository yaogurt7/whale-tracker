"""
FROZEN -- Do not modify this file.
Handles chronological data partitioning, feature insulation, and BSS evaluation.
Standard: scikit-learn, polars, pandas.
"""
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
DATA_DIR = "data"
MASTER_FILE = os.path.join(DATA_DIR, "master_trades.parquet")
POOL_FILE = os.path.join(DATA_DIR, "train_val_pool.parquet")
TEST_FILE = os.path.join(DATA_DIR, "test_trades.parquet")
RESULTS_FILE = "results.tsv"

# ── Data Partitioning & Leakage Protection ──────────────────
def initialize_files_if_needed():
    """Perform the 80/20 Chronological Split and save permanent files."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(POOL_FILE) or not os.path.exists(TEST_FILE):
        print("🛠️ Initializing data splits (Chronological 80/20)...")
        df = pl.read_parquet(MASTER_FILE).sort("date")
        
        # Partition: Most recent 20% becomes the 'Future' Test Set
        split_idx = int(len(df) * 0.8)
        train_val_pool = df.slice(0, split_idx)
        test_set = df.slice(split_idx)
        
        train_val_pool.write_parquet(POOL_FILE)
        test_set.write_parquet(TEST_FILE)
        print(f"✅ Created {POOL_FILE} and {TEST_FILE}")

def load_data():
    """Load the pool data and perform the secondary 80/20 Train/Val split."""
    initialize_files_if_needed()
    
    df = pl.read_parquet(POOL_FILE)
    
    # --- FEATURE FIREWALL ---
    # target = outcome (0 or 1). We strip IDs and Time information to prevent leakage.
    target_col = "outcome"
    forbidden = [target_col, "market_id", "date", "timestamp", "final_price", "settlement"]
    feature_cols = [c for c in df.columns if c not in forbidden]
    
    df_clean = df.drop_nulls(subset=feature_cols + [target_col])
    
    X = df_clean.select(feature_cols).to_pandas()
    y = df_clean[target_col].to_pandas()

    # SECONDARY SPLIT: 80% Train, 20% Val from the Pool
    # shuffle=False is critical to maintain chronological order
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, shuffle=False
    )
    
    return X_train, y_train, X_val, y_val, feature_cols

# ── Evaluation (Brier Skill Score) ──────────────────────────
def evaluate(model, X_val, y_val):
    """
    Computes BSS. Higher is better.
    Benchmark: p_close (The market price at trade time).
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    
    bs_model = float(brier_score_loss(y_val, y_prob))
    bs_market = float(brier_score_loss(y_val, X_val["p_close"]))
    
    # Calculate Skill: How much better are we than the price?
    bss = 1.0 - (bs_model / bs_market) if bs_market > 0 else 0.0
    
    return bss, bs_model

# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_bss, val_bs, status, description):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_bss", "val_bs", "status", "description"])
        writer.writerow([experiment_id, f"{val_bss:.6f}", f"{val_bs:.6f}", status, description])

# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    if not os.path.exists(RESULTS_FILE):
        return

    bss_list, bs_list, statuses, descriptions = [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            bss_list.append(float(row["val_bss"]))
            bs_list.append(float(row["val_bs"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top: BSS
    ax1.scatter(range(len(bss_list)), bss_list, c=colors, s=80, zorder=3, edgecolors="white")
    ax1.plot(range(len(bss_list)), bss_list, "k--", alpha=0.2)
    
    best_bss = []
    curr_max = -float("inf")
    for b in bss_list:
        curr_max = max(curr_max, b)
        best_bss.append(curr_max)
    ax1.plot(range(len(bss_list)), best_bss, color="#2ecc71", linewidth=2, label="Best so far")
    
    ax1.set_ylabel("Brier Skill Score (Higher is Better)")
    ax1.set_title("AutoResearch: Whale Flow Predictive Edge", fontweight="bold")
    ax1.axhline(0, color='black', alpha=0.5)

    # Bottom: BS
    ax2.scatter(range(len(bs_list)), bs_list, c=colors, s=80, zorder=3, edgecolors="white")
    ax2.plot(range(len(bs_list)), bs_list, "k--", alpha=0.2)
    ax2.set_ylabel("Brier Score (Lower is Better)")
    ax2.set_xlabel("Experiment #")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Performance chart saved to {save_path}")

if __name__ == "__main__":
    plot_results()