Maximize BSS on master_trades.parquet.

## Rules

1. You may ONLY modify model.py
2. prepare.py and run.py are FROZEN — do not touch them
3. build_model() must return an sklearn-compatible estimator (Pipeline preferred)
4. Training + evaluation must complete in under 90 seconds on CPU
5. No additional data sources or external downloads
6. Do NOT remove core whale features: whale_flow_top10p_quote, hhi, top10_share

## Workflow

1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check BSS in output
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:      git checkout model.py   (revert)
8. Repeat from step 1

## Ideas to explore

- Different classifiers: LogisticRegression, RidgeClassifier, SVC
- Ensemble methods: RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
- Feature engineering: PolynomialFeatures (for interactions), adding context features (vol_quote, gini, etc.)
- Preprocessing: StandardScaler, RobustScaler, QuantileTransformer
- Target calibration: CalibratedClassifierCV
- Hyperparameter tuning within the pipeline (max_depth, n_estimators, learning_rate)

## What NOT to do

- Do not modify prepare.py (data split, BSS metric calculation)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of build_model()
