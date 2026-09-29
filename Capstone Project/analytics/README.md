# Module 2 — Analytics Pipeline

## What this does
An end-to-end pipeline on the Titanic dataset: profile it, clean it, tell a visual data story about survival, then build and evaluate a full predictive-modeling pipeline (classification + a regression side-task) on the same cleaned data.

## Structure
- `data_prep.py` — shared module: loads the raw dataset once (`sns.load_dataset('titanic')`), saves the offline-fallback `titanic.csv`, and applies the missing-value cleaning strategy. Imported by both scripts below so cleaning logic is identical and never duplicated.
- `01_eda.py` — profiling, missing-value handling, univariate/bivariate/multivariate analysis, correlation heatmap, standardization check. Produces all chart PNGs in `charts/`.
- `02_modeling.py` — reads the already-committed `titanic.csv` (does **not** call `sns.load_dataset` again), cleans it via the same `data_prep.py` function, then runs the full modeling pipeline: stratified split, preprocessing, 3 classifiers, imbalance comparison, hyperparameter tuning, regression side-task, final comparison, and saves the best pipeline.
- `titanic.csv` — the one committed offline fallback, saved immediately after the single raw load.
- `charts/` — all supporting chart images.
- `best_pipeline.joblib` — the full fitted pipeline (preprocessing + best classifier), reloadable and usable on raw input.
- `eda_output.txt`, `modeling_output.txt` — captured run logs.

## How to run
```
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
python 01_eda.py
python 02_modeling.py
```
`01_eda.py` must be run first since it produces `titanic.csv`. After that, `02_modeling.py` never touches the network — it reads the committed CSV.

## Part A — Profiling, cleaning, and the data story

### Missing values
| Column | % Missing | Strategy |
|---|---|---|
| deck | 77.22% | Dropped the column — missing rate is far above the 30% threshold, and imputing three-quarters of a categorical field would be unreliable. |
| age | 19.87% | Imputed with the median (28.00) — falls in the 5–30% band. |
| embarked | 0.22% | Dropped the affected rows — below the 5% threshold. |
| embark_town | 0.22% | Dropped the affected rows — below the 5% threshold. |

### Univariate analysis (age, fare)
- **Age**: 65 IQR outliers (bounds: 2.50–54.50).
- **Fare**: 114 IQR outliers (bounds: −26.76 to 65.66).
- **Fare distribution**: mean = 32.10, median = 14.45, mode = 8.05. Since mean > median > mode, fare is **right-skewed** — a small number of very high-fare (mostly first-class) passengers pull the mean well above the typical passenger's fare.

### Bivariate analysis (survival rates)
- **By sex**: male 18.9%, female 74.0%.
- **By class**: 1st 62.6%, 2nd 47.3%, 3rd 24.2%.
- **By sex and class**: 1st-class women 96.7%, 2nd-class women 92.1%, 3rd-class women 50.0%, 1st-class men 36.9%, 2nd-class men 15.7%, 3rd-class men 13.5%.

### Correlation matrix (6 columns: survived, pclass, age, sibsp, parch, fare)
Two strongest correlations by absolute value:
1. **pclass vs fare: −0.548** — higher-numbered (lower) classes paid substantially less, confirming fare is largely a proxy for class.
2. **sibsp vs parch: 0.415** — passengers traveling with more siblings/spouses also tended to travel with more parents/children, i.e. family size clusters together rather than these being independent.

### Multivariate data story (4 charts, `charts/story_*.png`)
1. **Survival by class and sex** — women survived at a far higher rate than men in every class, and the gap holds even in third class. First-class women had the highest survival rate of any group.
2. **Age by survival and sex** — surviving men skew slightly younger than non-surviving men; among women, age barely matters since women were prioritized regardless of age.
3. **Age vs fare, colored by survival** — higher-fare passengers survived at a visibly higher rate; age alone shows no clear separation, so fare/class carried more predictive signal than age.
4. **Survival by embarkation port and class** — Cherbourg passengers survived more often, mostly because Cherbourg had a higher share of first-class passengers — an indirect class effect, not a direct effect of the port itself.

### Standardization check (exploratory only)
Before: age mean ≈ 29.32, std ≈ 12.98; fare mean ≈ 32.10, std ≈ 49.70.
After z-scoring: both columns have mean ≈ 0 and std ≈ 1, confirming the transform worked as expected. This check does not feed into the modeling pipeline below — that pipeline fits its own scaler on the training split only.

## Part B — Predictive modeling

### Train/test split
Stratified split (80/20) on `survived`. Justification: the class balance is ~62% not-survived / 38% survived — not even. A stratified split preserves this ratio in both folds, so the test set's metrics aren't distorted by a randomly skewed sample.

### Preprocessing
`ColumnTransformer` inside a `Pipeline`: numeric features (`age`, `fare`, `sibsp`, `parch`, `pclass`) get median imputation + `StandardScaler`; categorical features (`sex`, `embarked`) get most-frequent imputation + one-hot encoding. All steps are fit on the training split only and applied transform-only to the test split.

### Classifier comparison
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.770 | 0.690 | 0.721 | 0.705 | 0.754 |
| Random Forest | 0.820 | 0.781 | 0.735 | 0.758 | 0.821 |

Decision tree visualization: `charts/decision_tree.png`. ROC curves for all three: `charts/roc_curves.png`.

### Imbalance handling comparison
Training class balance: 61.7% not-survived / 38.3% survived (Random Forest used for this comparison).

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.781 | 0.735 | 0.758 |
| class_weight='balanced' | 0.761 | 0.750 | 0.756 |
| SMOTE (train fold only) | 0.791 | 0.779 | 0.785 |

**Conclusion**: SMOTE gave the best F1 (0.785), improving both precision and recall over the baseline. `class_weight='balanced'` nudged recall up slightly but cost some precision. Since the imbalance here is moderate (not severe), SMOTE's synthetic oversampling gave the model more balanced exposure to the minority class during training without distorting the decision boundary as much as reweighting did.

### Hyperparameter tuning (Random Forest, GridSearchCV)
Best params: `n_estimators=100`, `max_depth=None`, `max_features='sqrt'`.
OOB score: **0.795**.

### Regression side-task (predicting fare)
| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 21.139 | 41.747 | 0.347 | 0.324 |

Residual plot: `charts/regression_residuals.png`. Residual spread grows sharply from std ≈ 13.25 at low predicted fares to std ≈ 57.25 at high predicted fares — a clear sign of **heteroscedasticity**: the model's errors are not uniform across the prediction range, and it's noticeably less reliable when predicting high fares.

## Final model comparison and recommendation

**Classifiers:**
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.770 | 0.690 | 0.721 | 0.705 | 0.754 |
| Random Forest | 0.820 | 0.781 | 0.735 | 0.758 | 0.821 |

**Regression:**
| Model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (fare) | 21.139 | 41.747 | 0.347 | 0.324 |

**Recommendation**: Random Forest is the classifier I'd deploy. It has the best F1 (0.758) and accuracy (0.820) of the three, and while Logistic Regression edges it out on AUC (0.861 vs 0.821), Random Forest gives a stronger balance of precision (0.781) and recall (0.735) at the default threshold, which matters more than raw AUC for a system that needs to act on individual predictions. It also generalizes better than the single Decision Tree, which overfits more readily. For a production setting prioritizing balanced, dependable classification over pure interpretability, Random Forest is the safer choice.

## Saved pipeline
The best-performing full pipeline (`ColumnTransformer` preprocessing + Random Forest classifier, fit together) is saved to `best_pipeline.joblib` via `joblib.dump`. It was reloaded with `joblib.load` and confirmed to predict correctly on raw, unpreprocessed test rows (see `modeling_output.txt`, Step 10).
