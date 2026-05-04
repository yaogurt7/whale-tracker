### Experiment 1 (Round 2): RandomForestClassifier Baseline

**Date:** 2026-05-04

**Changes to `model.py`:**
- Replaced `LogisticRegression` with `RandomForestClassifier`.
- Removed `PolynomialFeatures` for this initial test with RandomForest.

**Rationale:** Exploring different classifier types as suggested in `program.md`. Starting with RandomForestClassifier to see its baseline performance without complex feature engineering.

---

### Experiment 2: GradientBoostingClassifier

**Date:** 2026-05-04

**Changes to `model.py`:**
- Replaced `RandomForestClassifier` with `GradientBoostingClassifier`.
- Kept `StandardScaler` for preprocessing.

**Rationale:** Exploring another ensemble method, `GradientBoostingClassifier`, as suggested in `program.md`. This often provides strong performance and could be an improvement over `RandomForestClassifier`.

---

### Experiment 3: HistGradientBoostingClassifier

**Date:** 2026-05-04

**Changes to `model.py`:**
- Replaced `GradientBoostingClassifier` with `HistGradientBoostingClassifier`.
- Kept `StandardScaler` for preprocessing.

**Rationale:** Exploring `HistGradientBoostingClassifier` for potentially better performance and scalability compared to `GradientBoostingClassifier`, as suggested in `program.md`.
### Baseline: Logistic Regression

**Date:** 2026-05-04

**Changes to `model.py`:**
- Initialized `model.py` to use `LogisticRegression`.
- Standard `StandardScaler` for preprocessing.

**Rationale:** Establishing a baseline performance using a simple, well-understood linear model as per instructions.
