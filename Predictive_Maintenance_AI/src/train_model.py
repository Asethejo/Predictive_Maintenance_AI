
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. INPUT FEATURES
# ==========================================

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

CAT = ["Type"]
NUM = [c for c in FEATURES if c not in CAT]


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================

def train():

    # 1. Load dataset
    print("Loading AI4I 2020 dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # 2. Validate required columns
    missing = [
        c for c in FEATURES + [TARGET]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 3. Select required columns
    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("\nTarget distribution:")
    print(y.value_counts())

    # 4. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # ==========================================
    # 4. DATA PREPROCESSING
    # ==========================================

    preprocessor = ColumnTransformer([

        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUM),

        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT)

    ])

    # ==========================================
    # 5. DEFINE ML ALGORITHMS
    # ==========================================

    candidates = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    }

    results = []
    fitted_models = {}

    # ==========================================
    # 6. TRAIN AND EVALUATE MODELS
    # ==========================================

    print("\nTraining machine learning models...\n")

    for name, estimator in candidates.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {

            "model": name,

            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),

            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),

            "pr_auc": float(
                average_precision_score(
                    y_test,
                    probabilities
                )
            )

        }

        results.append(metrics)
        fitted_models[name] = pipeline

        print(f"{name} completed!")

    # ==========================================
    # 7. SELECT BEST MODEL USING F1
    # ==========================================

    best_result = max(
        results,
        key=lambda x: x["f1"]
    )

    best_name = best_result["model"]

    best_model = fitted_models[best_name]

    # ==========================================
    # 8. EVALUATE SELECTED MODEL
    # ==========================================

    best_predictions = best_model.predict(X_test)

    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    report = classification_report(
        y_test,
        best_predictions,
        zero_division=0,
        output_dict=True
    )

    # ==========================================
    # 9. SAVE MODEL AND EVALUATION DATA
    # ==========================================

    artifact = {

        "pipeline": best_model,

        "model_name": best_name,

        "features": FEATURES,

        "metrics": best_result,

        # Confusion matrix
        "confusion_matrix": cm.tolist(),

        # Classification report
        "classification_report": report,

        # Actual test labels
        "y_test": y_test.tolist(),

        # Predicted class labels
        "y_pred": best_predictions.tolist(),

        # Predicted failure probabilities
        "y_prob": best_probabilities.tolist(),

        # Number of test records
        "test_rows": int(len(y_test)),

        # Dataset information
        "data_note": (
            "AI4I 2020 is a simulated industrial benchmark dataset, "
            "not actual factory-collected telemetry."
        )

    }

    model_path = (
        MODEL_DIR / "predictive_maintenance_model.joblib"
    )

    joblib.dump(artifact, model_path)

    # ==========================================
    # 10. SAVE MODEL COMPARISON
    # ==========================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False
    )

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    # ==========================================
    # 11. SAVE METRICS JSON
    # ==========================================

    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump({
            "all_models": results,
            "selected_model": best_name
        }, file, indent=4)

    # ==========================================
    # 12. DISPLAY FINAL RESULTS
    # ==========================================

    print("\n========== MODEL COMPARISON ==========\n")

    print(results_df.to_string(index=False))

    print("\n========== BEST MODEL ==========")

    print("Selected model:", best_name)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            best_predictions,
            zero_division=0
        )
    )

    print("\nBest Model Metrics:")
    print(best_result)

    print("\nEvaluation data saved:")
    print("y_test:", len(artifact["y_test"]))
    print("y_pred:", len(artifact["y_pred"]))
    print("y_prob:", len(artifact["y_prob"]))

    print("\nModel saved successfully!")
    print("Location:", model_path)

    print("\nEDA and model training completed!")

    return artifact


# ==========================================
# 13. MAIN
# ==========================================

if __name__ == "__main__":
    train()
