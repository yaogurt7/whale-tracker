# Test Performance

Test protocol: selected model was fixed from train/validation results before test evaluation. The model was trained on `data/train_val_pool.parquet` and evaluated once on `data/test_trades.parquet`.

Test model: `rf_temporal_fe`

Pipeline:

1. Compact row-local engineered features.
2. Lag-only rolling context over windows 5 and 20.
3. `RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=100, max_features=None, n_jobs=-1, random_state=42)`.

Test result:

```tsv
metric	value
train_path	data/train_val_pool.parquet
test_path	data/test_trades.parquet
features	vol_quote,whale_flow_top10p_quote,hhi,top10_share,gini
train_rows	75873
test_rows	18969
test_positive_rate	0.518214
test_bs_baseline_0p5	0.250000
test_bss	0.888794
test_bs	0.027801
fit_seconds	7.48
```

No test labels were used for model selection.

## Experiment Log

# Whale Tracker Experiment Log

Objective: maximize validation Brier Skill Score (BSS) on `data/master_trades.parquet` using only the frozen feature matrix exposed by `prepare.py`, while keeping the three required whale features in the model path.

Validation protocol: all scores below come from the project runner against the fixed chronological train/validation split in `prepare.py`. Higher BSS is better; lower Brier Score (BS) is better.

## Previous Experiments

The prior run sequence established a strong baseline ladder:

| Status | BSS | BS | Description |
| --- | ---: | ---: | --- |
| baseline | 0.521131 | 0.245127 | Whale-base Logistic Regression baseline. |
| keep | 0.522837 | 0.244254 | Degree-2 polynomial features with Logistic Regression `C=0.1`; only a tiny gain over baseline. |
| keep | 0.727548 | 0.139465 | Robust-scaled Logistic Regression `C=1.0`; large calibration gain from scaling. |
| keep | 0.841753 | 0.081005 | Normal `QuantileTransformer` with Logistic Regression; best linear-family result. |
| keep | 0.841767 | 0.080998 | HistGradientBoosting with 200 iterations, learning rate 0.03, 31 leaves, L2 0.1; marginally beat quantile logistic. |
| keep | 0.841773 | 0.080995 | Shallow RandomForest with 200 trees, max depth 5, min leaf 100, all features; best previous model. |
| discard | 0.519326 | 0.246051 | Class-balanced Logistic Regression `C=0.3`; class weighting hurt calibration. |
| discard | 0.841729 | 0.081017 | Shallower HistGradientBoosting depth/leaf variation; slightly worse than best. |
| discard | 0.841498 | 0.081135 | GradientBoosting with 150 estimators, depth 2, subsample 0.8; close but worse. |
| discard | 0.518605 | 0.246420 | ExtraTrees depth 5/leaf 100; very poor calibration compared with RandomForest. |
| discard | 0.812537 | 0.095960 | Quantile-transformed GaussianNB; respectable but below tree/logistic best. |

Interpretation: most model-only changes saturated around BSS 0.8417. The tree models and quantile logistic were very close, suggesting the raw five whale/context fields already carried a strong low-dimensional signal but left little room without engineered context.

## Feature-Engineering Pass

I changed only `model.py` and added sklearn-compatible transformers inside the model pipeline:

- `WhaleFeatureEngineer`: row-local features derived from the five allowed inputs: signed log whale flow, log volume, concentration/inequality blends, flow-per-volume, flow-per-concentration, pressure terms, products, squares, crosses, square roots, and bounded tanh ratios.
- `CausalRollingFeatures`: lag-only rolling means and deviations for each raw input over previous rows. This uses only rows before the current row in the observed chronological order. Validation transformation starts from the validation block itself, so it does not peek at future validation rows or pull labels.
- `WHALE_EXPERIMENT`: environment switch used only to run reproducible variants through frozen `run.py`. The default `build_model()` now returns the best current model.

New trials:

| Status | BSS | BS | Description | Notes |
| --- | ---: | ---: | --- | --- |
| keep | 0.842011 | 0.080873 | Compact ratios/interactions plus shallow RandomForest | Small but real gain over the previous forest. The most useful row-local additions appear to be signed log flow, concentration blends, and pressure ratios. |
| discard | 0.841964 | 0.080897 | Full squares/crosses plus shallow RandomForest | Extra polynomial detail did not improve over compact features. |
| discard | 0.841966 | 0.080896 | Dense transforms plus shallow RandomForest | Square roots and bounded ratios did not add useful signal beyond compact/full sets. |
| discard | 0.841914 | 0.080923 | Full engineered features with RandomForest depth 4, leaf 50 | More reactive leaves and shallower trees were slightly worse. |
| discard | 0.841911 | 0.080924 | Full engineered features with RandomForest depth 6, leaf 150 | Deeper trees with heavier smoothing were also slightly worse. |
| keep | 0.898122 | 0.052150 | Causal rolling lag features plus compact engineered RandomForest | Large improvement. Lag-only rolling context captures regime/state information that row-local features miss. |
| discard | 0.896935 | 0.052758 | Causal rolling lag features with RandomForest depth 4, leaf 75 | Very strong, but worse than the default depth-5/leaf-100 forest. |
| discard | 0.841412 | 0.081179 | Full engineered features with HistGradientBoosting | HGB did not benefit from the expanded row-local feature set. |
| discard | 0.841717 | 0.081023 | Dense engineered features with regularized HistGradientBoosting | Close to old best but below the forest. |
| discard | 0.832408 | 0.085788 | Full engineered features with ExtraTrees | ExtraTrees remained poorly calibrated. |
| discard | 0.840979 | 0.081401 | Full engineered features with GradientBoosting | Below the RandomForest variants. |
| discard | 0.841758 | 0.081002 | Compact engineered features with Quantile LogisticRegression | Linear model could not exploit the engineered set enough to beat the forest. |
| discard | 0.512925 | 0.249328 | Compact engineered features with polynomial robust LogisticRegression | Polynomial expansion was badly miscalibrated/overfit. |

## Current Best

Best model: `rf_temporal_fe`

Pipeline:

1. Add compact row-local engineered features.
2. Add lag-only rolling context over windows 5 and 20 for the five raw inputs.
3. Fit `RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=100, max_features=None, n_jobs=-1, random_state=42)`.

Result: BSS `0.898122`, BS `0.052150`, verification runtime about 12 seconds on CPU after replacing repeated rolling slice means with equivalent cumulative-sum lag means.

The main lesson from this pass is that feature engineering helped most when it encoded chronological market state. Row-local algebraic transforms gave a small lift, but lagged rolling means/deviations produced the first major jump beyond the 0.841 plateau.

## Implementation Notes

`run.py` currently prints `Note: Could not read history for comparison (name 'pd' is not defined)` for keep runs because it references `pd.read_csv` without importing pandas. I did not modify `run.py` because the rules freeze it. To avoid incorrect keep/discard labeling, I pre-scored candidates and passed `--discard` for known non-improvements.

## Results TSV

```tsv
timestamp	commit	bss	bs	status	description
2026-05-08 23:05:16	75c0a48	0.521131	0.245127	baseline	Baseline: Whale-Base Logistic Regression
2026-05-08 23:06:18	75c0a48	0.521131	0.245127	baseline	experiment
2026-05-08 23:28:27	75c0a48	0.522837	0.244254	keep	PolynomialFeatures degree 2 plus LogisticRegression C=0.1
2026-05-08 23:28:44	8cb8170	0.727548	0.139465	keep	RobustScaler plus LogisticRegression C=1.0
2026-05-08 23:28:57	9b755d8	0.841753	0.081005	keep	QuantileTransformer normal plus LogisticRegression C=1.0
2026-05-08 23:29:27	8076449	0.841767	0.080998	keep	HistGradientBoosting max_iter 200 lr 0.03 leaf 31 l2 0.1
2026-05-08 23:29:45	5e853dd	0.841773	0.080995	keep	RandomForest 200 trees depth 5 min leaf 100 all features
2026-05-08 23:30:01	0fa7540	0.519326	0.246051	discard	Class-balanced LogisticRegression C=0.3
2026-05-08 23:30:20	0fa7540	0.841729	0.081017	discard	HistGradientBoosting depth 3 leaf 15 lr 0.04
2026-05-08 23:30:42	0fa7540	0.841498	0.081135	discard	GradientBoosting 150 estimators depth 2 subsample 0.8
2026-05-08 23:30:56	0fa7540	0.518605	0.246420	discard	ExtraTrees 250 trees depth 5 min leaf 100 all features
2026-05-08 23:31:13	0fa7540	0.812537	0.095960	discard	QuantileTransformer normal plus GaussianNB
2026-05-11 13:13:16	855bcd8	0.842011	0.080873	keep	Feature engineering compact ratios interactions plus shallow RandomForest
2026-05-11 13:13:28	855bcd8	0.841964	0.080897	discard	Feature engineering full squares crosses plus shallow RandomForest
2026-05-11 13:13:41	855bcd8	0.841966	0.080896	discard	Feature engineering dense transforms plus shallow RandomForest
2026-05-11 13:13:52	855bcd8	0.841914	0.080923	discard	Full engineered features with RandomForest depth 4 leaf 50
2026-05-11 13:14:04	855bcd8	0.841911	0.080924	discard	Full engineered features with RandomForest depth 6 leaf 150
2026-05-11 13:14:29	855bcd8	0.898122	0.052150	keep	Causal rolling lag features plus compact engineered RandomForest
2026-05-11 13:14:52	855bcd8	0.896935	0.052758	discard	Causal rolling lag features RandomForest depth 4 leaf 75
2026-05-11 13:15:03	855bcd8	0.841412	0.081179	discard	Full engineered features with HistGradientBoosting
2026-05-11 13:15:12	855bcd8	0.841717	0.081023	discard	Dense engineered features with regularized HistGradientBoosting
2026-05-11 13:15:22	855bcd8	0.832408	0.085788	discard	Full engineered features with ExtraTrees
2026-05-11 13:15:51	855bcd8	0.840979	0.081401	discard	Full engineered features with GradientBoosting
2026-05-11 13:15:59	855bcd8	0.841758	0.081002	discard	Compact engineered features with Quantile LogisticRegression
2026-05-11 13:16:08	855bcd8	0.512925	0.249328	discard	Compact engineered features with polynomial robust LogisticRegression
```
