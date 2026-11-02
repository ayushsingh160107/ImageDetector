"""TruthLens AI - Model Performance & Benchmark Evaluation Page
Renders genuine evaluation metrics computed with scikit-learn from held-out data:
Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
"""

from __future__ import annotations

import json
import os
import plotly.graph_objects as go
import streamlit as st

from components.header import render_header
from ml.evaluate_model import METRICS_OUTPUT_PATH, evaluate_detector

st.set_page_config(
    page_title="TruthLens AI — Model Performance",
    page_icon="📊",
    layout="wide",
)

render_header(
    title="TruthLens Model Performance",
    subtitle="Rigorous Benchmarking & Scikit-Learn Evaluation on Held-Out Test Data",
    badge="Model Performance",
)

st.markdown(
    """
    Genuine forensic evaluation requires calculating metrics on held-out test datasets
    with zero overlap from training distribution.
    """
)

# Button to trigger evaluation if needed
eval_col1, eval_col2 = st.columns([3, 1])
with eval_col2:
    if st.button("🔄 Run Evaluation Benchmark", width="stretch"):
        with st.spinner("Evaluating model on held-out benchmark split with scikit-learn..."):
            evaluate_detector()
            st.success("Benchmark evaluation completed!")

metrics_data = None
if os.path.exists(METRICS_OUTPUT_PATH):
    try:
        with open(METRICS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        st.error(f"Error reading metrics: {e}")

if metrics_data:
    metrics = metrics_data.get("metrics", {})
    cm_data = metrics_data.get("confusion_matrix", {})

    st.markdown("### 🏆 Core Forensic Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
    with m2:
        st.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
    with m3:
        st.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0.0) * 100:.1f}%")
    with m4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0.0) * 100:.1f}%")
    with m5:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.3f}")

    st.markdown("---")

    # Confusion Matrix Visualization with Plotly
    c_col1, c_col2 = st.columns([1, 1], gap="large")

    with c_col1:
        st.markdown("### 🔲 Confusion Matrix")
        matrix = cm_data.get("matrix", [[0, 0], [0, 0]])
        labels = ["Actual Real", "Actual Synthetic"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Real", "Predicted Synthetic"],
                y=labels,
                text=matrix,
                texttemplate="%{text}",
                textfont={"size": 18, "color": "white"},
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#cbd5e1", size=13)),
            yaxis=dict(tickfont=dict(color="#cbd5e1", size=13), autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})

    with c_col2:
        st.markdown("### 📋 Evaluation Dataset & Specifications")
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 16px 20px;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                <p style="margin: 4px 0;"><b>Benchmark Dataset:</b> {metrics_data.get('dataset_name')}</p>
                <p style="margin: 4px 0;"><b>Dataset Source:</b> {metrics_data.get('dataset_source')}</p>
                <p style="margin: 4px 0;"><b>Evaluation Samples:</b> {metrics_data.get('num_test_samples')} balanced held-out images</p>
                <p style="margin: 4px 0;"><b>Model Architecture:</b> {metrics_data.get('architecture')}</p>
                <p style="margin: 4px 0;"><b>Evaluation Timestamp:</b> {metrics_data.get('evaluated_at')[:19].replace('T', ' ')} UTC</p>
                <p style="margin: 4px 0;"><b>True Positives:</b> {cm_data.get('true_positives', 0)} &nbsp;|&nbsp; <b>True Negatives:</b> {cm_data.get('true_negatives', 0)}</p>
                <p style="margin: 4px 0;"><b>False Positives:</b> {cm_data.get('false_positives', 0)} &nbsp;|&nbsp; <b>False Negatives:</b> {cm_data.get('false_negatives', 0)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚠️ Known Limitations & Failure Modes")
    limitations = metrics_data.get("known_limitations", [])
    for lim in limitations:
        st.markdown(f"- 🔸 {lim}")

else:
    st.info("No held-out evaluation file found yet. Click **Run Evaluation Benchmark** above to run scikit-learn metrics on the test split.")
