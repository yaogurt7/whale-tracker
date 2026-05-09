import sys
import time
import subprocess
import os
import polars as pl
from prepare import load_data, evaluate, log_result

def get_git_hash():
    """
    Retrieves the short version of the current Git commit hash.
    Explicitly captures stdout while silencing stderr to avoid conflicts.
    """
    try:
        # We replace capture_output=True with explicit stdout/stderr mapping
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,     # This captures the hash string
            stderr=subprocess.DEVNULL,   # This silences the "not a git repo" errors
            text=True,                  # Returns a string instead of bytes
            check=True                  # Raises an error if git command fails
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback for non-git environments or missing git installations
        return "no-git"

def main():
    # --- 1. Argument Parsing (New Structure Style) ---
    args = sys.argv[1:]
    status = "keep"
    # Cleaner way to grab description without explicitly naming flags
    desc = " ".join([a for a in args if not a.startswith("--")]) or "experiment"
    
    if "--baseline" in args: status = "baseline"
    if "--discard" in args: status = "discard"

    # --- 2. Load Data ---
    # We use _ for feature_names to match the new structure's style
    X_train, y_train, X_val, y_val, _ = load_data()

    # --- 3. Build and Train (New Structure Style with Duration) ---
    from model import build_model
    model = build_model()
    
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    # --- 4. Evaluate ---
    metrics = evaluate(model, X_val, y_val)
    bss, bs = metrics['bss'], metrics['bs']
    
    # Adopting the new structure's print style
    print(f"Result: BSS={bss:.4f}, BS={bs:.4f}, Time={duration:.2f}s")

    # --- 5. Auto-Discard Logic (Preserved and Optimized) ---
    # We use pandas here for consistency since generate_plot now uses it
    best_bss = -1.0
    if os.path.exists("results.tsv") and status == "keep":
        try:
            history = pd.read_csv("results.tsv", sep="\t", on_bad_lines='skip')
            # Only compare against successful previous runs
            valid_history = history[history["status"].isin(["baseline", "keep"])]
            if not valid_history.empty:
                best_bss = valid_history["bss"].max()
        except Exception as e:
            print(f"Note: Could not read history for comparison ({e})")

    if status == "keep" and bss < best_bss:
        print(f"AUTO-DISCARD: Current BSS {bss:.4f} < Best {best_bss:.4f}")
        status = "discard"

    # --- 6. Log Result ---
    commit = get_git_hash()
    # Passing the variables required by your refactored log_result
    log_result(commit, bss, bs, status, desc)
    print(f"DONE: Run Complete. Status: {status}")

if __name__ == "__main__":
    main()