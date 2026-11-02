"""TruthLens AI - Single Image Forensic Detector Page
Full analysis pipeline: upload validation, deep neural inference, probability scoring,
Grad-CAM visual attribution, and verifiable JSON/PDF report export.
"""

from __future__ import annotations

import streamlit as st

from components.confidence_chart import render_confidence_chart
from components.header import render_header
from components.heatmap_viewer import render_heatmap_viewer
from components.image_uploader import render_image_uploader
from components.prediction_card import render_prediction_card
from services.prediction_service import predict_image
from services.report_service import generate_json_report, generate_pdf_report

st.set_page_config(
    page_title="TruthLens AI — Image Detector",
    page_icon="🔍",
    layout="wide",
)

render_header(
    title="TruthLens AI Detector",
    subtitle="Single Asset Neural Forensics & Explainable Attribution",
    badge="Image Detector",
)

# Sidebar threshold & parameter controls
st.sidebar.markdown("### ⚙️ Forensic Detection Parameters")
threshold_ai = st.sidebar.slider(
    "AI-Generated Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.60,
    step=0.05,
    help="AI probability at or above which image is marked Likely AI-Generated.",
)
threshold_real = st.sidebar.slider(
    "Authentic Real Threshold",
    min_value=0.10,
    max_value=0.50,
    value=0.40,
    step=0.05,
    help="AI probability at or below which image is marked Likely Real (Camera-Captured).",
)
st.sidebar.caption(
    f"Current decision rule: [Real ≤ {threshold_real*100:.0f}%] ↔ [Inconclusive] ↔ [AI ≥ {threshold_ai*100:.0f}%]"
)

st.sidebar.markdown("---")
save_history = st.sidebar.checkbox("Log Record to SQLite History", value=True)
examiner_notes = st.sidebar.text_area("Forensic Examiner Notes", placeholder="Optional notes for report...")

# 1. Image Uploader & Preview
image, metadata = render_image_uploader(
    label="Upload Image or Select a Benchmark Sample",
    key="detector_uploader",
    allow_samples=True,
)

if image is not None and metadata is not None and metadata.is_valid:
    analyze_btn = st.button("🔬 Analyze Image with DeepForensics", type="primary", width="stretch")

    # Check if analysis should be triggered or results exist in session state
    should_run = analyze_btn or (f"pred_{metadata.filename}" in st.session_state)

    if should_run:
        with st.spinner("Executing neural inference and frequency analysis..."):
            pred_result, tensor_input = predict_image(
                image=image,
                filename=metadata.filename,
                file_size_kb=metadata.file_size_kb,
                threshold_ai=threshold_ai,
                threshold_real=threshold_real,
                save_to_history=save_history,
                notes=examiner_notes,
            )
            # Store in session state
            st.session_state[f"pred_{metadata.filename}"] = pred_result
            st.session_state[f"tensor_{metadata.filename}"] = tensor_input

        # Render Executive Verdict
        st.markdown("---")
        render_prediction_card(pred_result)

        # Render Probability Bars
        render_confidence_chart(pred_result)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Visual Attribution (Grad-CAM)
        target_idx = 1 if pred_result.ai_probability >= 0.50 else 0
        expl_result = render_heatmap_viewer(
            image=image,
            input_tensor=tensor_input,
            target_class_idx=target_idx,
            key_prefix="detector",
        )

        st.markdown("---")

        # Report Downloads Section
        st.markdown("### 📥 Export Forensic Reports")
        st.caption("Download cryptographic and verifiable examination dossiers in JSON or PDF format.")

        json_str = generate_json_report(
            prediction=pred_result,
            metadata=metadata,
            region_statistics=expl_result.region_statistics,
            notes=examiner_notes,
        )

        pdf_bytes = generate_pdf_report(
            prediction=pred_result,
            metadata=metadata,
            thumbnail_image=image,
            heatmap_image=expl_result.heatmap_image,
            notes=examiner_notes,
        )

        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.download_button(
                label="📄 Download JSON Forensic Report",
                data=json_str,
                file_name=f"TruthLens_Report_{metadata.filename.split('.')[0]}.json",
                mime="application/json",
                width="stretch",
            )
        with rep_col2:
            st.download_button(
                label="📑 Download PDF Examination Report",
                data=pdf_bytes,
                file_name=f"TruthLens_Report_{metadata.filename.split('.')[0]}.pdf",
                mime="application/pdf",
                width="stretch",
            )
else:
    st.info("👆 Upload an image or click one of the verified sample buttons above to begin analysis.")
