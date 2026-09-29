import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

from data_prep import load_raw_from_csv, clean_data

CHARTS_DIR = "charts"

NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch", "pclass"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "survived"


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def evaluate_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
    }


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    fitted_pipelines = {}
    results = []

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        result = evaluate_classifier(name, pipeline, X_test, y_test)
        results.append(result)

    plt.figure(figsize=(7, 6))
    for result in results:
        plt.plot(result["fpr"], result["tpr"], label=f"{result['name']} (AUC={result['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Classifiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/roc_curves.png")
    plt.close()

    tree_model = fitted_pipelines["Decision Tree"].named_steps["classifier"]
    encoded_feature_names = (
        NUMERIC_FEATURES
        + list(fitted_pipelines["Decision Tree"].named_steps["preprocessor"]
               .named_transformers_["cat"].named_steps["encoder"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=encoded_feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree (max_depth=3 shown for readability)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/decision_tree.png")
    plt.close()

    comparison_df = pd.DataFrame([
        {
            "Model": r["name"],
            "Accuracy": round(r["accuracy"], 3),
            "Precision": round(r["precision"], 3),
            "Recall": round(r["recall"], 3),
            "F1": round(r["f1"], 3),
            "AUC": round(r["auc"], 3),
        }
        for r in results
    ])
    print("\n=== Classifier Comparison Table ===")
    print(comparison_df.to_string(index=False))

    return fitted_pipelines, results, comparison_df


def imbalance_comparison(X_train, X_test, y_train, y_test, preprocessor):
    print("\n=== Class Balance (training set) ===")
    balance = y_train.value_counts(normalize=True)
    print(balance)

    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_result = evaluate_classifier("Baseline (no handling)", baseline_pipeline, X_test, y_test)

    balanced_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    balanced_pipeline.fit(X_train, y_train)
    balanced_result = evaluate_classifier("class_weight='balanced'", balanced_pipeline, X_test, y_test)

    preprocessor_for_smote = build_preprocessor()
    X_train_transformed = preprocessor_for_smote.fit_transform(X_train, y_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

    smote_classifier = RandomForestClassifier(random_state=42)
    smote_classifier.fit(X_train_resampled, y_train_resampled)
    X_test_transformed = preprocessor_for_smote.transform(X_test)
    y_pred_smote = smote_classifier.predict(X_test_transformed)
    y_proba_smote = smote_classifier.predict_proba(X_test_transformed)[:, 1]

    smote_precision = precision_score(y_test, y_pred_smote)
    smote_recall = recall_score(y_test, y_pred_smote)
    smote_f1 = f1_score(y_test, y_pred_smote)
    smote_auc = roc_auc_score(y_test, y_proba_smote)
    print(f"\n=== SMOTE (train fold only) ===")
    print(f"Precision: {smote_precision:.3f}")
    print(f"Recall: {smote_recall:.3f}")
    print(f"F1: {smote_f1:.3f}")
    print(f"AUC: {smote_auc:.3f}")

    imbalance_df = pd.DataFrame([
        {"Strategy": "Baseline", "Precision": round(baseline_result["precision"], 3),
         "Recall": round(baseline_result["recall"], 3), "F1": round(baseline_result["f1"], 3)},
        {"Strategy": "class_weight=balanced", "Precision": round(balanced_result["precision"], 3),
         "Recall": round(balanced_result["recall"], 3), "F1": round(balanced_result["f1"], 3)},
        {"Strategy": "SMOTE", "Precision": round(smote_precision, 3),
         "Recall": round(smote_recall, 3), "F1": round(smote_f1, 3)},
    ])
    print("\n=== Imbalance Strategy Comparison ===")
    print(imbalance_df.to_string(index=False))

    best_strategy = imbalance_df.loc[imbalance_df["F1"].idxmax(), "Strategy"]
    print(f"\nConclusion: '{best_strategy}' produced the best F1 balance between precision and recall "
          f"on this split, suggesting it handles the moderate class imbalance most effectively here "
          f"without overcorrecting recall at the expense of precision.")

    return imbalance_df


def tune_random_forest(X_train, y_train, preprocessor):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    oob_score = best_pipeline.named_steps["classifier"].oob_score_

    print("\n=== GridSearchCV — Random Forest Tuning ===")
    print(f"Best params: {grid_search.best_params_}")
    print(f"OOB score: {oob_score:.3f}")

    return best_pipeline, grid_search.best_params_, oob_score


def regression_side_task(df):
    reg_features = ["pclass", "age", "sibsp", "parch", "sex", "embarked"]
    X = df[reg_features]
    y = df["fare"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_reg_features = ["pclass", "age", "sibsp", "parch"]
    categorical_reg_features = ["sex", "embarked"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    reg_preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_reg_features),
        ("cat", categorical_pipeline, categorical_reg_features),
    ])

    reg_pipeline = Pipeline(steps=[
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression()),
    ])
    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print("\n=== Regression Side-Task: Predicting fare ===")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Adjusted R2: {adjusted_r2:.3f}")

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    residuals = y_test - y_pred_series

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred_series, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residuals")
    plt.title("Residual Plot — Fare Regression")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/regression_residuals.png")
    plt.close()

    median_pred = y_pred_series.median()
    residual_spread_low = residuals[y_pred_series < median_pred].std()
    residual_spread_high = residuals[y_pred_series >= median_pred].std()
    heteroscedasticity_conclusion = (
        "The residual spread grows noticeably as predicted fare increases "
        f"(std={residual_spread_low:.2f} for low predictions vs std={residual_spread_high:.2f} for high predictions), "
        "indicating heteroscedasticity: the model's errors are not uniformly distributed across the prediction range."
        if residual_spread_high > residual_spread_low * 1.3
        else "The residual spread is roughly stable across the prediction range, suggesting no strong heteroscedasticity."
    )
    print(heteroscedasticity_conclusion)

    return {
        "mae": mae, "rmse": rmse, "r2": r2, "adjusted_r2": adjusted_r2,
        "heteroscedasticity_conclusion": heteroscedasticity_conclusion,
    }, reg_pipeline


def final_comparison_and_recommendation(classifier_comparison_df, regression_metrics):
    print("\n=== Final Model Comparison ===")
    print("\nClassifiers:")
    print(classifier_comparison_df.to_string(index=False))

    regression_df = pd.DataFrame([{
        "Model": "Linear Regression (fare)",
        "MAE": round(regression_metrics["mae"], 3),
        "RMSE": round(regression_metrics["rmse"], 3),
        "R2": round(regression_metrics["r2"], 3),
        "Adjusted_R2": round(regression_metrics["adjusted_r2"], 3),
    }])
    print("\nRegression:")
    print(regression_df.to_string(index=False))

    best_row = classifier_comparison_df.loc[classifier_comparison_df["F1"].idxmax()]
    recommendation = (
        f"Based on the comparison table, {best_row['Model']} is the recommended classifier for deployment, "
        f"with the strongest balance of F1 ({best_row['F1']}) and AUC ({best_row['AUC']}) among the three models tested. "
        f"It achieved {best_row['Accuracy']} accuracy, {best_row['Precision']} precision, and {best_row['Recall']} recall, "
        f"indicating it identifies survivors reliably without excessive false positives. "
        f"Random Forest-style ensembling also tends to generalize better than a single decision tree, "
        f"reducing overfitting risk on unseen passenger data. "
        f"For a production system prioritizing balanced performance over pure interpretability, this model is the safer choice."
    )
    print(f"\nFinal recommendation:\n{recommendation}")

    return regression_df, recommendation


def run():
    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("Step 1: Loading data from committed titanic.csv (no re-fetch from network) ...")
    raw_df = load_raw_from_csv()

    print("\nStep 2: Cleaning data (same strategy as EDA stage) ...")
    cleaned_df, _ = clean_data(raw_df, verbose=False)

    print("\nStep 3: Stratified train/test split ...")
    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(
        "Justification: survived has a ~62/38 class split (not evenly balanced). "
        "A stratified split preserves this ratio in both train and test sets, "
        "preventing the test set from randomly ending up with a skewed survivor ratio "
        "that would make evaluation metrics unreliable."
    )

    preprocessor = build_preprocessor()

    print("\nStep 4: Training and evaluating three classifiers ...")
    fitted_pipelines, results, comparison_df = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test, preprocessor
    )

    print("\nStep 5: Imbalance handling comparison ...")
    imbalance_comparison(X_train, X_test, y_train, y_test, preprocessor)

    print("\nStep 6: Hyperparameter tuning (Random Forest, GridSearchCV) ...")
    best_rf_pipeline, best_params, oob_score = tune_random_forest(X_train, y_train, preprocessor)
    tuned_result = evaluate_classifier("Tuned Random Forest (GridSearchCV)", best_rf_pipeline, X_test, y_test)

    print("\nStep 7: Regression side-task (predicting fare) ...")
    regression_metrics, reg_pipeline = regression_side_task(cleaned_df)

    print("\nStep 8: Final comparison table and recommendation ...")
    regression_df, recommendation = final_comparison_and_recommendation(comparison_df, regression_metrics)

    print("\nStep 9: Saving best full pipeline (preprocessing + estimator) ...")
    best_model_name = comparison_df.loc[comparison_df["F1"].idxmax(), "Model"]
    best_full_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(best_full_pipeline, "best_pipeline.joblib")
    print(f"Saved '{best_model_name}' pipeline to best_pipeline.joblib")

    print("\nStep 10: Reloading and verifying pipeline on raw input ...")
    reloaded_pipeline = joblib.load("best_pipeline.joblib")
    sample_raw = X_test.iloc[:5]
    sample_predictions = reloaded_pipeline.predict(sample_raw)
    print(f"Sample predictions on raw test rows: {sample_predictions}")
    print(f"Actual values: {y_test.iloc[:5].values}")

    print("\nModeling pipeline complete.")


if __name__ == "__main__":
    run()
