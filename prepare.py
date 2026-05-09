import polars as pl
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import datetime 
import csv

plt.switch_backend('Agg')

WHALE_BASE = [
    "whale_flow_top10p_quote", # Core: Aggressive whale positioning
    "hhi",                     # Core: Market concentration
    "top10_share",             # Core: Dominance of top wallets
    "gini",                    # Added: Distribution inequality (context)
    "vol_quote"                # Added: Absolute volume to scale the signals
]

DATA_PATH = "data/master_trades.parquet"
RESULTS_FILE = "results.tsv"
PLOT_FILE = "performance.png"
RANDOM_SEED = 42

def load_data():
    df = pl.read_parquet(DATA_PATH)
    target_col = "outcome"
    # Filter features strictly from WHALE_BASE
    df_clean = df.drop_nulls(subset=[target_col])
    numeric_features = [c for c in df_clean.columns if c in WHALE_BASE]
    
    X = df_clean.select(numeric_features).fill_null(0).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, shuffle=False
    )
    return X_train, y_train, X_val, y_val, numeric_features

def evaluate(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    bs = brier_score_loss(y_val, probs)
    bs_baseline = brier_score_loss(y_val, np.full_like(y_val, 0.5))
    bss = 1 - (bs / bs_baseline)
    return {"bss": bss, "bs": bs}

def log_result(commit, bss, bs, status, description):
    """
    Logs research results to results.tsv using the csv.writer infrastructure.
    This prevents formatting errors and handles tab-delimitation cleanly.
    """
    file_exists = os.path.exists(RESULTS_FILE)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Still a good idea to strip tabs from description just in case,
    # though csv.writer is generally better at handling this.
    clean_desc = str(description).replace("\t", " ").replace("\n", " ").strip()

    with open(RESULTS_FILE, "a", newline="") as f:
        # We use delimiter="\t" to maintain your TSV format
        writer = csv.writer(f, delimiter="\t")
        
        # If the file is brand new, write the header first
        if not file_exists:
            writer.writerow(["timestamp", "commit", "bss", "bs", "status", "description"])
        
        # Write the data row
        writer.writerow([
            ts, 
            commit, 
            f"{bss:.6f}", 
            f"{bs:.6f}", 
            status, 
            clean_desc
        ])
    
    generate_plot()

def generate_plot(save_path="performance.png"):
    """
    Generates a two-panel industry-standard chart tracking 
    Brier Skill Score (BSS) and raw Brier Score (BS).
    """
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run experiments first.")
        return

    # Using pandas for compatibility with the new infrastructure logic
    # We use engine='python' and on_bad_lines to keep it robust against TSV errors
    df = pd.read_csv(RESULTS_FILE, sep="\t", on_bad_lines='skip')
    
    if len(df) == 0:
        return

    plt.close('all') # Prevent memory leaks

    # Colors: green=keep, red=discard, blue=baseline
    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(str(s).strip(), "#95a5a6") for s in df["status"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Top Panel: Brier Skill Score (BSS) ---
    # BSS: Higher is better
    ax1.scatter(range(len(df)), df["bss"], c=colors, s=120, zorder=3, edgecolors="white", linewidth=1.5)
    ax1.plot(range(len(df)), df["bss"], "k--", alpha=0.15, zorder=2)
    
    # "Best-so-far" envelope for BSS (Cumulative Max)
    best_bss = df["bss"].cummax()
    ax1.plot(range(len(df)), best_bss, color="#2ecc71", linewidth=2.5, label="Best BSS (Alpha)", alpha=0.8)
    
    ax1.set_ylabel("Brier Skill Score", fontsize=12, fontweight="bold")
    ax1.set_title("Whale-Tracker: Optimization Progress", fontsize=16, fontweight="bold", loc='left', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')

    # --- Bottom Panel: Brier Score (BS) ---
    # BS: LOWER is better
    ax2.scatter(range(len(df)), df["bs"], c=colors, s=120, zorder=3, edgecolors="white", linewidth=1.5)
    ax2.plot(range(len(df)), df["bs"], "k--", alpha=0.15, zorder=2)
    
    # "Best-so-far" envelope for BS (Cumulative MINIMUM since lower is better)
    best_bs = df["bs"].cummin()
    ax2.plot(range(len(df)), best_bs, color="#3498db", linewidth=2.5, label="Best (Lowest) BS", alpha=0.8)
    
    ax2.set_ylabel("Raw Brier Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Experiment Iteration", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')

    # --- X-Axis Formatting (Descriptions) ---
    # Shorten descriptions for the x-axis so they don't overlap
    short_labels = [str(d)[:18] + ".." if len(str(d)) > 20 else str(d) for d in df["description"]]
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"DONE: Plot updated at {save_path}")
