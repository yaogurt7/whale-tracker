## Objective

Maximize BSS on master_trades.parquet.

## Rules

1. You may ONLY modify model.py
2. prepare.py and run.py are FROZEN — do not touch them
3. build_model() must return an sklearn-compatible estimator (Pipeline preferred)
4. After completing the iterations, update results.tsv and performance.png with the results
5. Training + evaluation must complete in under 240 seconds on CPU
6. No additional data sources or external downloads
7. Do NOT remove core whale features: whale_flow_top10p_quote, hhi, top10_share
8. Do not use future data
9. No additional data sources or external downloads

## Workflow

1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change". If there are code issues, adjust code to ensure no errors.
5. Check BSS in output
6. If improved BSS:  git add model.py && git commit -m "feat: <description>"
7. If worse BSS: git checkout model.py   (revert)
8. Log results in performance.png and results.tsv through the established channels.
9. Repeat from step 1

## Ideas to explore

- Different classifiers: LogisticRegression, RidgeClassifier, SVC with different tuning methods
- Ensemble methods: RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier with different tuning methods
- Feature engineering: PolynomialFeatures (for interactions), adding context features (vol_quote, gini, etc.)
- Preprocessing: StandardScaler, RobustScaler, QuantileTransformer
- Target calibration: CalibratedClassifierCV
- Hyperparameter tuning within the pipeline (max_depth, n_estimators, learning_rate)

## What NOT to do

- Do not modify prepare.py (data split, BSS metric calculation)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of build_model()
- Do not touch the test_trades file
