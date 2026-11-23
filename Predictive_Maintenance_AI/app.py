
from pathlib import Path
from datetime import datetime

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.pdf_report import generate_prediction_pdf

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "predictive_maintenance_model.joblib"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL STYLING
# =====================================================

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: white;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #2563EB;
}

.subtitle {
    font-size: 16px;
    color: #64748b;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(37, 99, 235, 0.12),
        rgba(14, 165, 233, 0.06)
    );
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.25);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND DATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model_artifact = load_model()
    df = load_data()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


model = model_artifact["pipeline"]
model_name = model_artifact["model_name"]
metrics = model_artifact["metrics"]

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]


# =====================================================
# 4. DATASET METRICS
# =====================================================

total_records = len(df)

total_failures = int(df["Machine failure"].sum())

healthy_records = total_records - total_failures

failure_rate = df["Machine failure"].mean() * 100


# =====================================================
# 5. SHAP HELPER FUNCTIONS
# =====================================================

def get_failure_shap_values(shap_output):

    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])

    values = np.asarray(shap_output)

    if values.ndim == 3:
        return values[:, :, 1]

    return values


def calculate_individual_shap(input_data):

    transformed_input = preprocessor.transform(input_data)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(rf_model)

    shap_output = explainer.shap_values(
        transformed_input
    )

    shap_values = get_failure_shap_values(
        shap_output
    )[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Impact": shap_values
    })

    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Impact"].abs()
    )

    return explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


@st.cache_data
def calculate_global_importance():

    background = df[
        model_artifact["features"]
    ].sample(
        n=min(100, len(df)),
        random_state=42
    )

    transformed_background = preprocessor.transform(
        background
    )

    explainer = shap.TreeExplainer(rf_model)

    shap_output = explainer.shap_values(
        transformed_background
    )

    global_values = get_failure_shap_values(
        shap_output
    )

    feature_names = preprocessor.get_feature_names_out()

    importance = np.mean(
        np.abs(global_values),
        axis=0
    )

    return pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP": importance
    }).sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )


# =====================================================
# 6. PREDICTION REPORT HELPERS
# =====================================================

def create_prediction_report():

    if not st.session_state.get("prediction_ready", False):
        return None

    input_data = st.session_state["last_input"]

    prediction = st.session_state["last_prediction"]

    probability = st.session_state["last_probability"]

    result = input_data.copy()

    result["Predicted Status"] = (
        "Failure" if prediction == 1 else "Healthy"
    )

    result["Failure Probability (%)"] = round(
        probability * 100, 2
    )

    result["Model"] = model_name

    result["Prediction Date"] = (
        st.session_state["prediction_time"]
    )

    return result


def create_pdf_report():

    if not st.session_state.get("prediction_ready", False):
        return None

    input_data = st.session_state["last_input"]

    explanation_df = st.session_state.get(
        "last_explanation"
    )

    shap_summary = None

    if explanation_df is not None:

        shap_summary = [
            (
                f"{row['Feature']}: "
                f"SHAP impact = {row['SHAP Impact']:.4f}"
            )
            for _, row in explanation_df.head(5).iterrows()
        ]

    pdf_data = generate_prediction_pdf(
        machine_type=input_data["Type"].iloc[0],
        sensor_values=input_data.iloc[0].to_dict(),
        prediction=st.session_state["last_prediction"],
        failure_probability=st.session_state["last_probability"],
        shap_explanation=shap_summary
    )

    return pdf_data.getvalue()


def display_prediction_downloads(key_prefix):

    if not st.session_state.get("prediction_ready", False):
        return

    csv_report = create_prediction_report()

    pdf_report = create_pdf_report()

    if csv_report is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_report.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="machine_prediction_report.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv"
            )

        with col2:

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_report,
                file_name="machine_prediction_report.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_pdf"
            )


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("⚙️ Machine Intelligence")

st.sidebar.caption(
    "AI-Powered Predictive Maintenance"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "⚙️ Machine Prediction",
        "📊 Sensor Analytics",
        "🩺 Machine Health",
        "🎯 Model Performance",
        "🔍 Explainable AI",
        "ℹ️ About Project"
    ],
    key="dashboard_navigation"
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write(f"Algorithm: {model_name}")
st.sidebar.write("Dataset: AI4I 2020")
st.sidebar.write("Explainability: SHAP")

st.sidebar.divider()

st.sidebar.caption(
    "Developed using Python, Scikit-learn and Streamlit"
)


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "🏠 Overview":

    st.markdown(
        '<p class="main-title">⚙️ AI-Powered Predictive Maintenance</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Intelligent industrial equipment monitoring and failure prediction</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Machine Failures", f"{total_failures:,}")
    col3.metric("Failure Rate", f"{failure_rate:.2f}%")
    col4.metric("Selected Model", model_name)

    st.divider()

    st.subheader("Machine Failure Distribution")

    failure_counts = pd.Series({
        "Healthy": healthy_records,
        "Failure": total_failures
    })

    fig = px.bar(
        x=failure_counts.index,
        y=failure_counts.values,
        color=failure_counts.index,
        color_discrete_map={
            "Healthy": "#16A34A",
            "Failure": "#DC2626"
        },
        labels={
            "x": "Machine Status",
            "y": "Number of Records"
        },
        title="Healthy vs Failure Records"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Project Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### ⚙️ Machine Intelligence

        - Machine failure prediction
        - Sensor-based monitoring
        - Failure probability estimation
        - Machine health classification
        """)

    with col2:

        st.markdown("""
        ### 🔍 Explainable AI

        - Individual SHAP explanations
        - Global feature importance
        - Model evaluation
        - Failure detection analytics
        """)

    st.info(
        "Use the sidebar to navigate between project modules."
    )


# =====================================================
# PAGE 2: MACHINE PREDICTION
# =====================================================

elif page == "⚙️ Machine Prediction":

    st.title("⚙️ Machine Failure Prediction")

    st.write(
        "Enter machine sensor readings to predict potential failure."
    )

    st.divider()

    st.subheader("Machine Sensor Inputs")

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=305.0,
            value=300.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=305.0,
            max_value=314.0,
            value=310.0
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=3.0,
            max_value=80.0,
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=260,
            value=100
        )

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    st.divider()

    if st.button(
        "🔍 Predict Machine Failure",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            with st.spinner(
                "Generating prediction and SHAP explanation..."
            ):

                explanation_df = calculate_individual_shap(
                    input_data
                )

            st.session_state["last_input"] = input_data.copy()

            st.session_state["last_prediction"] = int(prediction)

            st.session_state["last_probability"] = float(probability)

            st.session_state["last_explanation"] = explanation_df

            st.session_state["prediction_time"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            st.session_state["prediction_ready"] = True

            st.success("Prediction completed successfully.")

        except Exception as e:

            st.error(f"Prediction error: {e}")

    # Display latest saved prediction

    if st.session_state.get("prediction_ready", False):

        prediction = st.session_state["last_prediction"]

        probability = st.session_state["last_probability"]

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ Potential Machine Failure Detected"
                )

            else:

                st.success(
                    "✅ Machine Predicted as Healthy"
                )

        with col2:

            st.metric(
                "Estimated Failure Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(probability)

        st.caption(
            "This is the model's estimated probability, "
            "not a calibrated guarantee of actual failure risk."
        )

        st.divider()

        st.subheader("Entered Sensor Readings")

        st.dataframe(
            st.session_state["last_input"],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Download Prediction Reports")

        display_prediction_downloads("prediction")

        st.caption(
            f"Prediction generated: {st.session_state['prediction_time']}"
        )


# =====================================================
# PAGE 3: SENSOR ANALYTICS
# =====================================================

elif page == "📊 Sensor Analytics":

    st.title("📊 Interactive Sensor Monitoring")

    st.write(
        "Explore industrial sensor measurements and failure patterns."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        machine_types = st.multiselect(
            "Select Machine Types",
            options=sorted(df["Type"].unique()),
            default=sorted(df["Type"].unique())
        )

    with col2:

        machine_status = st.selectbox(
            "Machine Status",
            ["All", "Healthy", "Failure"]
        )

    filtered_df = df[
        df["Type"].isin(machine_types)
    ].copy()

    if machine_status == "Healthy":

        filtered_df = filtered_df[
            filtered_df["Machine failure"] == 0
        ]

    elif machine_status == "Failure":

        filtered_df = filtered_df[
            filtered_df["Machine failure"] == 1
        ]

    if filtered_df.empty:

        st.warning("No records match the selected filters.")

    else:

        st.metric(
            "Filtered Records",
            f"{len(filtered_df):,}"
        )

        st.divider()

        st.subheader("1. Interactive Sensor Monitoring")

        selected_sensor = st.selectbox(
            "Choose Sensor",
            SENSOR_COLUMNS,
            key="monitor_sensor"
        )

        chart_df = filtered_df[
            ["UDI", "Type", "Machine failure", selected_sensor]
        ].sort_values("UDI").head(1000).copy()

        chart_df["Machine Status"] = (
            chart_df["Machine failure"].map({
                0: "Healthy",
                1: "Failure"
            })
        )

        fig = px.scatter(
            chart_df,
            x="UDI",
            y=selected_sensor,
            color="Machine Status",
            symbol="Type",
            color_discrete_map={
                "Healthy": "#16A34A",
                "Failure": "#DC2626"
            },
            hover_data=["Type"],
            title=f"{selected_sensor} Monitoring"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "UDI represents dataset record order, not a real timestamp."
        )

        st.divider()

        st.subheader("2. Healthy vs Failure Comparison")

        comparison_data = filtered_df.copy()

        comparison_data["Machine Status"] = (
            comparison_data["Machine failure"].map({
                0: "Healthy",
                1: "Failure"
            })
        )

        fig = px.box(
            comparison_data,
            x="Machine Status",
            y=selected_sensor,
            color="Machine Status",
            color_discrete_map={
                "Healthy": "#16A34A",
                "Failure": "#DC2626"
            },
            title=f"{selected_sensor} by Machine Status"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("3. Machine Type Comparison")

        fig = px.violin(
            filtered_df,
            x="Type",
            y=selected_sensor,
            color="Type",
            box=True,
            points=False,
            title=f"{selected_sensor} by Machine Type"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("4. Sensor Distribution")

        fig = px.histogram(
            comparison_data,
            x=selected_sensor,
            color="Machine Status",
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={
                "Healthy": "#16A34A",
                "Failure": "#DC2626"
            },
            title=f"Distribution of {selected_sensor}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("5. Sensor Correlation Analysis")

        correlation = filtered_df[
            SENSOR_COLUMNS
        ].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Sensor Feature Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("6. Sensor Statistics")

        st.dataframe(
            filtered_df[SENSOR_COLUMNS].describe(),
            use_container_width=True
        )

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered Sensor Dataset",
            data=csv_data,
            file_name="filtered_sensor_data.csv",
            mime="text/csv"
        )


# =====================================================
# PAGE 4: MACHINE HEALTH
# =====================================================

elif page == "🩺 Machine Health":

    st.title("🩺 Machine Health Monitoring")

    st.write(
        "Monitor the latest model-generated machine status "
        "and inspect its sensor measurements."
    )

    st.divider()

    if st.session_state.get("prediction_ready", False):

        prediction = st.session_state["last_prediction"]

        probability = st.session_state["last_probability"]

        input_data = st.session_state["last_input"]

        st.subheader("Latest Machine Status")

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction == 0:
                st.success("PREDICTED HEALTHY")
            else:
                st.error("POTENTIAL FAILURE")

        with col2:

            st.metric(
                "Estimated Failure Probability",
                f"{probability * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Machine Type",
                input_data["Type"].iloc[0]
            )

        st.progress(probability)

        st.caption(
            "Status is based on the latest model prediction. "
            "Probability is a model estimate, not a validated "
            "industrial safety or maintenance risk level."
        )

        st.divider()

        st.subheader("Machine Sensor Health Readings")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Air Temperature",
                f"{input_data['Air temperature [K]'].iloc[0]:.2f} K"
            )

            st.metric(
                "Process Temperature",
                f"{input_data['Process temperature [K]'].iloc[0]:.2f} K"
            )

            st.metric(
                "Rotational Speed",
                f"{input_data['Rotational speed [rpm]'].iloc[0]} rpm"
            )

        with col2:

            st.metric(
                "Torque",
                f"{input_data['Torque [Nm]'].iloc[0]:.2f} Nm"
            )

            st.metric(
                "Tool Wear",
                f"{input_data['Tool wear [min]'].iloc[0]} min"
            )

            st.metric(
                "Machine Type",
                input_data["Type"].iloc[0]
            )

        st.divider()

        st.subheader("Sensor Reading Visualization")

        sensor_values = pd.DataFrame({
            "Sensor": [
                "Air Temperature",
                "Process Temperature",
                "Rotational Speed",
                "Torque",
                "Tool Wear"
            ],
            "Reading": [
                input_data["Air temperature [K]"].iloc[0],
                input_data["Process temperature [K]"].iloc[0],
                input_data["Rotational speed [rpm]"].iloc[0],
                input_data["Torque [Nm]"].iloc[0],
                input_data["Tool wear [min]"].iloc[0]
            ]
        })

        fig = px.bar(
            sensor_values,
            x="Sensor",
            y="Reading",
            color="Sensor",
            title="Latest Machine Sensor Readings"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Latest Prediction Information")

        st.write(
            f"Prediction Date: {st.session_state['prediction_time']}"
        )

        st.subheader("Download Machine Health Reports")

        display_prediction_downloads("health")

    else:

        st.info(
            "No machine prediction is available yet."
        )

        st.write(
            "Navigate to Machine Prediction, enter sensor readings "
            "and click Predict Machine Failure."
        )

    st.divider()

    st.subheader("Dataset-Level Health Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", f"{total_records:,}")
    col2.metric("Healthy Records", f"{healthy_records:,}")
    col3.metric("Failure Records", f"{total_failures:,}")

    st.caption(
        "These are aggregate benchmark dataset statistics, "
        "not the live status of connected industrial machines."
    )


# =====================================================
# PAGE 5: MODEL PERFORMANCE
# =====================================================

elif page == "🎯 Model Performance":

    st.title("🎯 Machine Learning Model Performance")

    st.write(
        "Evaluate the selected classification model using held-out test data."
    )

    st.divider()

    st.success(f"Selected Model: {model_name}")

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

    st.divider()

    st.subheader("Algorithm Comparison")

    if COMPARISON_PATH.exists():

        comparison = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Model comparison file not found.")

    st.divider()

    st.header("Advanced Model Evaluation")

    y_test = np.array(model_artifact["y_test"])

    y_pred = np.array(model_artifact["y_pred"])

    y_prob = np.array(model_artifact["y_prob"])

    if not (
        len(y_test) == len(y_pred) == len(y_prob)
    ):

        st.error("Evaluation data lengths do not match.")

    elif len(np.unique(y_test)) < 2:

        st.error("Test data must contain both classes.")

    else:

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

        tn, fp, fn, tp = cm.ravel()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("True Negatives", tn)
        col2.metric("False Positives", fp)
        col3.metric("False Negatives", fn)
        col4.metric("True Positives", tp)

        st.divider()

        st.subheader("2. ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_auc = roc_auc_score(y_test, y_prob)

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
        ax.set_title("ROC Curve")

        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.metric("ROC-AUC Score", f"{roc_auc:.4f}")

        st.divider()

        st.subheader("3. Precision-Recall Curve")

        precision_values, recall_values, _ = (
            precision_recall_curve(y_test, y_prob)
        )

        pr_auc = average_precision_score(y_test, y_prob)

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

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.metric(
            "Average Precision Score",
            f"{pr_auc:.4f}"
        )

        st.info(
            "Evaluation uses saved predictions from the original "
            "held-out test dataset."
        )


# =====================================================
# PAGE 6: EXPLAINABLE AI
# =====================================================

elif page == "🔍 Explainable AI":

    st.title("🔍 Explainable AI - SHAP")

    st.write(
        "Understand how individual sensor measurements "
        "influence machine failure predictions."
    )

    st.divider()

    st.subheader("Individual Prediction Explanation")

    if "last_explanation" in st.session_state:

        explanation_df = st.session_state["last_explanation"]

        plot_df = explanation_df.sort_values(
            "Absolute Impact",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            plot_df["Feature"],
            plot_df["SHAP Impact"]
        )

        ax.axvline(
            0,
            linestyle="--",
            color="black"
        )

        ax.set_xlabel("SHAP Contribution to Model Output")

        ax.set_title(
            "Individual Machine Prediction Explanation"
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Generate a machine prediction first to see its "
            "individual SHAP explanation."
        )

    st.divider()

    st.subheader("Global Feature Importance")

    try:

        with st.spinner(
            "Calculating global SHAP feature importance..."
        ):

            importance_df = calculate_global_importance()

        fig, ax = plt.subplots(figsize=(10, 5))

        plot_df = importance_df.sort_values(
            "Mean Absolute SHAP",
            ascending=True
        )

        ax.barh(
            plot_df["Feature"],
            plot_df["Mean Absolute SHAP"]
        )

        ax.set_xlabel("Mean Absolute SHAP Value")

        ax.set_title("Global Sensor Feature Importance")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "SHAP explains model behavior, not physical causation. "
            "Positive contributions increase the model output for "
            "failure class 1; negative contributions decrease it."
        )

    except Exception as e:

        st.error(
            f"Unable to generate global SHAP explanation: {e}"
        )


# =====================================================
# PAGE 7: ABOUT PROJECT
# =====================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")

    st.write(
        "AI-Powered Predictive Maintenance and Equipment Intelligence System"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write("""
    This application uses machine learning to predict potential
    industrial machine failures from sensor measurements.

    A trained Random Forest classifier estimates failure probability
    based on machine operating conditions.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### Programming and Data Science

        - Python
        - Pandas
        - NumPy
        - Matplotlib
        - Seaborn
        - Plotly
        """)

    with col2:

        st.markdown("""
        ### Machine Learning and Application

        - Scikit-learn
        - Random Forest
        - SHAP
        - Streamlit
        - Joblib
        - ReportLab
        """)

    st.divider()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", f"{total_records:,}")
    col2.metric("Failures", f"{total_failures:,}")
    col3.metric("Failure Rate", f"{failure_rate:.2f}%")

    st.write("""
    Dataset: AI4I 2020 Predictive Maintenance Dataset

    Source: UCI Machine Learning Repository

    The dataset contains simulated industrial benchmark records.
    It does not represent actual factory-collected telemetry.
    """)

    st.divider()

    st.subheader("Project Limitations")

    st.warning("""
    1. Predictions are estimates, not guaranteed failure warnings.

    2. The dataset contains simulated benchmark data.

    3. SHAP explains learned model behavior, not physical causation.

    4. Real industrial deployment requires validation using
       actual machine sensor data and operating conditions.

    5. The application does not connect to live industrial equipment.
    """)

    st.success(
        "AI-Powered Predictive Maintenance System"
    )


# =====================================================
# END OF APPLICATION
# =====================================================
