import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

plt.switch_backend('Agg')

# --- CONFIGURATION ---
WHALE_BASE = ["whale_flow_top10p_quote", "hhi", "top10_share"]
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
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # CRITICAL: Strip tabs/newlines to prevent TSV corruption
    clean_desc = str(description).replace("\t", " ").replace("\n", " ").strip()
    
    line = f"{ts}\t{commit}\t{bss:.6f}\t{bs:.6f}\t{status}\t{clean_desc}\n"
    
    # Ensure header exists if file is new
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("timestamp\tcommit\tbss\tbs\tstatus\tdescription\n")
            
    with open(RESULTS_FILE, "a") as f:
        f.write(line)
    
    generate_plot()

def generate_plot():
    if not os.path.exists(RESULTS_FILE): return
    plt.close('all') 

    # Robust reading: ignore malformed lines if the agent messed up
    df = pl.read_csv(RESULTS_FILE, separator="\t", truncate_ragged_lines=True, ignore_errors=True)
    if len(df) == 0: return

    plt.figure(figsize=(10, 5), dpi=120)
    colors = {"baseline": "#4A90E2", "keep": "#27AE60", "discard": "#E74C3C"}
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    for i in range(len(df)):
        try:
            val = float(df[i, "bss"])
            status = str(df[i, "status"]).strip()
            plt.scatter(i, val, color=colors.get(status, "gray"), s=100, edgecolors='white', zorder=5)
        except: continue

    plt.title("Thesis Progress: Brier Skill Score Tracking", loc='left', fontweight='bold')
    plt.xlabel("Iteration")
    plt.ylabel("BSS")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"✅ Created plot with {len(df)} iterations.")