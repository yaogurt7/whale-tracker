"""
Run one experiment: build model, train, evaluate, log result.
Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard
"""
import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result


def get_git_hash():
    """Returns the current git hash to link code version to results."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load data (Uses the chronological 80/20 pool)
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"--- Experiment: {description} ---")
    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {len(feature_names)} features")

    # 2. Build model (from model.py)
    from model import build_model
    model = build_model()
    print(f"Model Architecture: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate (Metric: Brier Skill Score vs Market Price)
    val_bss, val_bs = evaluate(model, X_val, y_val)
    print(f"Validation BSS: {val_bss:.6f} (Higher is better)")
    print(f"Validation BS:  {val_bs:.6f}  (Lower is better)")

    # 5. Log Result
    commit = get_git_hash()
    log_result(commit, val_bss, val_bs, status, description)
    print(f"Result logged to results.tsv (status={status}, commit={commit})")
    print("-" * 40)


if __name__ == "__main__":
    main()