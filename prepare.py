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
POOL_PATH = "data/train_val_pool.parquet"
TEST_PATH = "data/test_trades.parquet"
RESULTS_FILE = "results.tsv"
PLOT_FILE = "performance.png"
RANDOM_SEED = 42

def load_data():
    """
    1. Loads the master parquet file.
    2. Creates a chronological 80/20 split if it doesn't exist.
    3. Handles nulls safely (fills numeric with 0, drops missing targets).
    4. Returns clean data ready for Scikit-Learn.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Ensure your data is in the data/ folder.")

    df = pl.read_parquet(DATA_PATH)
    
    # Partition data (80% Pool for training/val, 20% Held-out for final thesis test)
    if not os.path.exists(POOL_PATH) or not os.path.exists(TEST_PATH):
        print("Initializing chronological data partitions...")
        split_idx = int(len(df) * 0.8)
        df_pool = df.slice(0, split_idx)
        df_test = df.slice(split_idx)
        df_pool.write_parquet(POOL_PATH)
        df_test.write_parquet(TEST_PATH)
    else:
        df_pool = pl.read_parquet(POOL_PATH)

    target_col = "outcome"
    # Exclude non-feature columns from the model
    exclude = [target_col, "p_close", "trade_id", "timestamp", "date"] 
    feature_cols = [c for c in df_pool.columns if c not in exclude]

    # Data Cleaning Phase
    # We only drop rows if we don't have the "answer" (outcome)
    df_clean = df_pool.drop_nulls(subset=[target_col])
    
    # We find which columns are numeric so we don't crash when filling nulls with 0
    numeric_features = [
        name for name, dtype in df_clean.schema.items() 
        if name in feature_cols and name in WHALE_BASE and dtype.is_numeric()
    ]
    
    # Fill missing whale signals with 0 (meaning 'No Activity')
    df_clean = df_clean.with_columns([
        pl.col(c).fill_null(0) for c in numeric_features
    ])
    
    if len(df_clean) == 0:
        raise ValueError("CRITICAL: Resulting dataset is empty. Check your 'outcome' column.")

    # Convert to Numpy for Scikit-Learn
    X = df_clean.select(numeric_features).to_numpy()
    y = df_clean.select(target_col).to_numpy().ravel()
    
    # Chronological validation split for the active experiment
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, shuffle=False
    )
    
    print(f"Data Ready: {len(X_train)} train rows, {len(X_val)} val rows.")
    return X_train, y_train, X_val, y_val, numeric_features

def evaluate(model, X_val, y_val):
    """Calculates Brier Score and Brier Skill Score."""
    probs = model.predict_proba(X_val)[:, 1]
    bs = brier_score_loss(y_val, probs)
    
    # Baseline: A 50/50 coin flip prediction
    bs_baseline = brier_score_loss(y_val, np.full_like(y_val, 0.5, dtype=float))
    
    # BSS: Positive = better than a coin flip; Negative = worse
    bss = 1 - (bs / bs_baseline)
    return {
        "bss": bss,
        "bs": bs,
        "bs_baseline": bs_baseline
    }

def log_result(commit, bss, bs, status, description):
    """Saves the experiment record to a TSV file."""
    import datetime
    header = "timestamp\tcommit\tbss\tbs\tstatus\tdescription\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
            
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}\t{commit}\t{bss:.6f}\t{bs:.6f}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)

def generate_plot():
    """Generates a professional, industry-standard BSS tracking chart."""
    if not os.path.exists(RESULTS_FILE):
        return

    # Clear all existing figures to prevent "ghosting" or memory accumulation
    plt.close('all') 

    # Load data
    df = pl.read_csv(RESULTS_FILE, separator="\t")
    if len(df) == 0:
        return

    # --- Styling Setup ---
    plt.figure(figsize=(10, 5), dpi=120)
    ax = plt.gca()
    
    # Professional Palette: Slate Blue, Emerald Green, Soft Red
    colors = {"baseline": "#4A90E2", "keep": "#27AE60", "discard": "#E74C3C"}
    
    # 1. Plot the "No Skill" Baseline
    plt.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8, alpha=0.5, label="No Skill (0.5 Bias)")
    
    # 2. Plot the Experiments
    for i in range(len(df)):
        val = df[i, "bss"]
        status = df[i, "status"]
        desc = df[i, "description"]
        
        color = colors.get(status, "#7F8C8D")
        
        # Draw point with a subtle white border for pop
        plt.scatter(i, val, color=color, s=120, edgecolors='white', linewidth=1.5, zorder=4)
        
        # Minimalist Annotation: Offset slightly to the right/up
        plt.annotate(
            f" {desc}", 
            (i, val), 
            xytext=(6, 4), 
            textcoords='offset points',
            fontsize=9, 
            fontweight='medium',
            color='#2C3E50',
            alpha=0.9
        )

    # 3. Clean Up Axes (Industry Standard "Despining")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # 4. Refined Grid & Labels
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)
    plt.title("Brier Skill Score (BSS) Optimization", loc='left', fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    plt.xlabel("Iteration Number", fontsize=10, color='#7F8C8D', labelpad=10)
    plt.ylabel("BSS (Improvement over 0.5)", fontsize=10, color='#7F8C8D', labelpad=10)
    
    # Ensure X-axis only shows integers for experiment numbers
    plt.xticks(range(len(df)))
    
    # Adjust margins to fit annotations
    plt.tight_layout()
    
    plt.savefig(PLOT_FILE)
    print(f"📊 Industry-standard chart saved: {PLOT_FILE}")

if __name__ == "__main__":
    generate_plot()
