import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from prepare import load_data

# Use headless backend for Windows stability
plt.switch_backend('Agg')

def check_importance():
    # 1. Load the exact data used in your run
    X_train, y_train, X_val, y_val, feature_names = load_data()

    # 2. Fit the Baseline Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 3. Extract Coefficients 
    # For LogReg, the absolute value of the coefficient represents its 'pull'
    importances = model.coef_[0]
    abs_importances = np.abs(importances)

    # 4. Create a DataFrame for easy sorting
    feat_df = pl.DataFrame({
        "feature": feature_names,
        "importance": abs_importances,
        "raw_coefficient": importances
    }).sort("importance", descending=True)

    print("\n--- Top 10 Most Influential Features ---")
    print(feat_df.head(10))

    # 5. Professional Visualization
    plt.figure(figsize=(10, 8))
    
    # Take top 15 features for the plot
    plot_df = feat_df.head(15).sort("importance", descending=False)
    
    plt.barh(plot_df["feature"], plot_df["importance"], color='#4A90E2')
    plt.title("Feature Importance (Absolute Coefficients)", loc='left', fontsize=14, fontweight='bold')
    plt.xlabel("Absolute Weight in Model")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("feature_importance.png")
    print("\n✅ Importance chart saved as: feature_importance.png")

    # 6. Leakage Warning
    top_feat = feat_df[0, "feature"]
    top_val = feat_df[0, "importance"]
    if top_val > 5.0:
        print(f"\nWARNING: Potential Data Leak detected!")
        print(f"The feature '{top_feat}' has a massive importance score ({top_val:.2f}).")
        print("This usually means this column contains information from the future.")

if __name__ == "__main__":
    check_importance()