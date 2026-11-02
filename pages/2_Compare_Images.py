"""TruthLens AI - Image Comparison Studio Page
Compares two images side by side with independent inference pipelines,
score divergence analytics, and differential metric reporting.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.header import render_header
from components.image_uploader import render_image_uploader
from services.comparison_service import compare_two_images

st.set_page_config(
    page_title="TruthLens AI — Compare Images",
    page_icon="⚖️",
    layout="wide",
)

render_header(
    title="TruthLens Comparison Studio",
    subtitle="Independent Dual-Asset Forensic Comparison & Divergence Analytics",
    badge="Comparison Studio",
)

st.markdown(
    """
    Compare two images independently through the forensic detection pipeline.
    Evaluate divergence in synthetic probability, sensor noise consistency, and attribution heatmaps.
    """
)

# Columns for Image A and Image B uploaders
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("#### 🅰️ Primary Asset (Image A)")
    img_a, meta_a = render_image_uploader(
        label="Select Image A",
        key="compare_a",
        allow_samples=True,
    )

with col_b:
    st.markdown("#### 🅱️ Secondary Asset (Image B)")
    img_b, meta_b = render_image_uploader(
        label="Select Image B",
        key="compare_b",
        allow_samples=True,
    )

st.markdown("---")

if img_a and meta_a and meta_a.is_valid and img_b and meta_b and meta_b.is_valid:
    if st.button("⚖️ Run Comparative Dual Analysis", type="primary", width="stretch"):
        with st.spinner("Processing both assets through independent inference & explainability pipelines..."):
            comparison = compare_two_images(
                image_a=img_a,
                meta_a=meta_a,
                image_b=img_b,
                meta_b=meta_b,
            )
            st.session_state["active_comparison"] = comparison

    comp = st.session_state.get("active_comparison")
    if comp:
        # Comparison Summary Banner
        div_color = "#ef4444" if comp.divergence_category == "High Divergence" else ("#f59e0b" if comp.divergence_category == "Moderate Difference" else "#10b981")
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                border-left: 6px solid {div_color};
                padding: 18px 22px;
                border-radius: 8px;
                margin-bottom: 20px;
            ">
                <h3 style="margin: 0 0 6px 0; color: #ffffff;">{comp.divergence_category} (Score Δ {comp.ai_prob_diff * 100:.1f}%)</h3>
                <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">{comp.comparison_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparative Metrics Table
        st.markdown("### 📊 Forensic Differential Table")
        df_comp = pd.DataFrame(comp.comparative_table)
        st.dataframe(df_comp, width="stretch", hide_index=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Dual Heatmap Visualizations
        st.markdown("### 🔬 Side-by-Side Attribution Heatmaps")
        st.caption("Inspect activation regions independently computed for Image A and Image B.")

        hcol1, hcol2 = st.columns(2, gap="large")
        with hcol1:
            st.markdown(f"**Image A ({comp.meta_a.filename})** — *{comp.result_a.predicted_label}*")
            st.image(comp.explanation_a.overlay_image, caption=f"AI Probability: {comp.result_a.ai_probability*100:.1f}%", width="stretch")
        with hcol2:
            st.markdown(f"**Image B ({comp.meta_b.filename})** — *{comp.result_b.predicted_label}*")
            st.image(comp.explanation_b.overlay_image, caption=f"AI Probability: {comp.result_b.ai_probability*100:.1f}%", width="stretch")

        st.caption(f"ℹ️ *{comp.disclaimer}*")
else:
    st.info("👈 Please load both Image A and Image B (or choose benchmark samples above) to initiate the comparative studio.")
