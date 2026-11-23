
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "predictive_maintenance_model.joblib"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model_artifact = load_model()

model = model_artifact["pipeline"]
model_name = model_artifact["model_name"]
metrics = model_artifact["metrics"]

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ==========================================
# SHAP HELPER FUNCTION
# ==========================================

def get_failure_shap_values(shap_output):

    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])

    values = np.asarray(shap_output)

    if values.ndim == 3:
        return values[:, :, 1]

    return values


# ==========================================
# DASHBOARD HEADING
# ==========================================

st.title("⚙️ AI-Powered Predictive Maintenance")

st.write(
    "An intelligent machine failure prediction system "
    "using industrial sensor data and Machine Learning."
)

st.divider()


# ==========================================
# DASHBOARD METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(df):,}")

col2.metric(
    "Machine Failures",
    int(df["Machine failure"].sum())
)

col3.metric(
    "Failure Rate",
    f"{df['Machine failure'].mean() * 100:.2f}%"
)

col4.metric("Selected Model", model_name)

st.divider()


# ==========================================
# SIDEBAR SENSOR INPUTS
# ==========================================

st.sidebar.header("Machine Sensor Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=300.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=314.0,
    value=310.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=3.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=260,
    value=100
)


# ==========================================
# INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])


# ==========================================
# MACHINE FAILURE PREDICTION
# ==========================================

st.header("Machine Failure Prediction")

if st.button("Predict Machine Failure", type="primary"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:

        st.error("⚠️ Potential Machine Failure Detected")

    else:

        st.success("✅ Machine Predicted as Healthy")

    st.metric(
        "Predicted Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "This probability is the model's estimated failure probability, "
        "not a calibrated guarantee of actual failure risk."
    )


    # ======================================
    # SHAP EXPLAINABLE AI
    # ======================================

    st.subheader("Why did the model make this prediction?")

    try:

        with st.spinner("Generating SHAP explanations..."):

            transformed_input = preprocessor.transform(input_data)

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(rf_model)

            local_shap = explainer.shap_values(
                transformed_input
            )

            local_values = get_failure_shap_values(
                local_shap
            )[0]

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Impact": local_values
            })

            explanation_df["Absolute Impact"] = (
                explanation_df["SHAP Impact"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute Impact",
                ascending=True
            )


            # Individual explanation

            st.write("### Individual Prediction Explanation")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(
                explanation_df["Feature"],
                explanation_df["SHAP Impact"]
            )

            ax.axvline(
                x=0,
                linestyle="--",
                color="black"
            )

            ax.set_xlabel("SHAP contribution to model output")

            ax.set_title(
                "Why this machine received its prediction"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            st.write("Feature contribution details:")

            st.dataframe(
                explanation_df.sort_values(
                    "Absolute Impact",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )


            # Global feature importance

            st.write("### Global Feature Importance")

            background = df[
                model_artifact["features"]
            ].sample(
                n=min(100, len(df)),
                random_state=42
            )

            transformed_background = preprocessor.transform(
                background
            )

            global_shap = explainer.shap_values(
                transformed_background
            )

            global_values = get_failure_shap_values(
                global_shap
            )

            importance = np.mean(
                np.abs(global_values),
                axis=0
            )

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean Absolute SHAP": importance
            }).sort_values(
                "Mean Absolute SHAP",
                ascending=True
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(
                importance_df["Feature"],
                importance_df["Mean Absolute SHAP"]
            )

            ax.set_xlabel("Mean Absolute SHAP Value")

            ax.set_title("Global Sensor Feature Importance")

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.dataframe(
                importance_df.sort_values(
                    "Mean Absolute SHAP",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "Positive SHAP contributions increase the model output "
                "for failure class 1. Negative contributions decrease it. "
                "Global importance shows average absolute contributions. "
                "These explain model behavior, not physical causes of failure."
            )

    except Exception as e:

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )


    st.write("### Entered Sensor Readings")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ==========================================
# DATASET ANALYTICS
# ==========================================

st.divider()

st.header("Dataset Analytics")

tab1, tab2, tab3 = st.tabs([
    "Failure Distribution",
    "Machine Types",
    "Sensor Analysis"
])


with tab1:

    failure_counts = (
        df["Machine failure"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(failure_counts)


with tab2:

    type_counts = df["Type"].value_counts()

    st.bar_chart(type_counts)


with tab3:

    selected_sensor = st.selectbox(
        "Choose Sensor",
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    st.subheader(
        f"{selected_sensor} vs Machine Failure"
    )

    sensor_data = df.groupby(
        "Machine failure"
    )[selected_sensor].mean()

    st.bar_chart(sensor_data)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("Machine Learning Model Performance")

st.write("Selected Algorithm:", model_name)

metric_cols = st.columns(5)

metric_names = [
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1"),
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc")
]

for col, (label, key) in zip(metric_cols, metric_names):

    col.metric(
        label,
        f"{metrics[key]:.4f}"
    )


# ==========================================
# ALGORITHM COMPARISON
# ==========================================

if COMPARISON_PATH.exists():

    st.subheader("Algorithm Comparison")

    comparison = pd.read_csv(COMPARISON_PATH)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# ADVANCED MODEL EVALUATION
# ==========================================

st.divider()

st.header("Advanced Model Evaluation")

st.write(
    "Detailed evaluation of the selected Random Forest model "
    "using unseen test data."
)


# Load saved evaluation data

y_test = np.array(model_artifact["y_test"])

y_pred = np.array(model_artifact["y_pred"])

y_prob = np.array(model_artifact["y_prob"])


# Validate evaluation data

if not (
    len(y_test) == len(y_pred) == len(y_prob)
):

    st.error(
        "Evaluation data lengths do not match. "
        "Please retrain the model."
    )

elif len(np.unique(y_test)) < 2:

    st.error(
        "The test dataset must contain both healthy and failure records."
    )

else:

    # ======================================
    # 1. CONFUSION MATRIX
    # ======================================

    st.subheader("1. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Failure"],
        yticklabels=["Healthy", "Failure"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")

    ax.set_ylabel("Actual Label")

    ax.set_title("Random Forest Confusion Matrix")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # Confusion matrix values

    tn, fp, fn, tp = cm.ravel()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True Negatives", tn)

    col2.metric("False Positives", fp)

    col3.metric("False Negatives", fn)

    col4.metric("True Positives", tp)


    # ======================================
    # 2. ROC CURVE
    # ======================================

    st.divider()

    st.subheader("2. ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_auc:.4f})",
        linewidth=2
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.metric(
        "ROC-AUC Score",
        f"{roc_auc:.4f}"
    )


    # ======================================
    # 3. PRECISION-RECALL CURVE
    # ======================================

    st.divider()

    st.subheader("3. Precision-Recall Curve")

    precision_values, recall_values, pr_thresholds = (
        precision_recall_curve(
            y_test,
            y_prob
        )
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        recall_values,
        precision_values,
        label=f"Random Forest (AP = {pr_auc:.4f})",
        linewidth=2
    )

    baseline = np.mean(y_test)

    ax.axhline(
        baseline,
        linestyle="--",
        label=f"Failure Prevalence = {baseline:.4f}"
    )

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")

    ax.set_title("Precision-Recall Curve")

    ax.legend()

    ax.grid(alpha=0.3)

    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.metric(
        "Average Precision Score",
        f"{pr_auc:.4f}"
    )

    st.info(
        "ROC and Precision-Recall curves are calculated using "
        "the saved test labels and predicted probabilities. "
        "The evaluation uses the original held-out test dataset."
    )


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.divider()

st.subheader("About This Project")

st.write("""
This application uses the AI4I 2020 Predictive Maintenance Dataset
from the UCI Machine Learning Repository.

The dataset contains simulated industrial machine sensor records.
It is a benchmark dataset and does not represent actual
factory-collected telemetry.

The system predicts machine failure using six input features
and compares multiple classification algorithms.
""")
