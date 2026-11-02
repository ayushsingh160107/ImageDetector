"""TruthLens AI - Heatmap Viewer Component
Renders Grad-CAM and saliency map visual attributions with interactive opacity,
colormap controls, and multi-view comparative tabs.
"""

from __future__ import annotations

from PIL import Image
import streamlit as st
import torch

from services.explanation_service import ExplanationResult, generate_explanation


def render_heatmap_viewer(
    image: Image.Image,
    input_tensor: torch.Tensor,
    target_class_idx: int = 1,
    key_prefix: str = "main",
) -> ExplanationResult:
    """
    Renders visual explainability controls, interactive heatmap viewer, and regional statistics.

    Returns:
        The generated ExplanationResult.
    """
    st.markdown("### 🔬 Visual Attribution & Explainability (Grad-CAM)")
    st.caption(
        "Inspect which spatial regions contributed most heavily to the model's neural activation."
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        alpha = st.slider(
            "Heatmap Blend Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            key=f"{key_prefix}_alpha_slider",
            help="Adjust transparency between original image and forensic activation heatmap.",
        )
    with ctrl_col2:
        colormap = st.selectbox(
            "Colormap",
            options=["JET", "VIRIDIS"],
            index=0,
            key=f"{key_prefix}_colormap_select",
        )
    with ctrl_col3:
        method = st.selectbox(
            "Attribution Method",
            options=["Grad-CAM", "Input Saliency"],
            index=0,
            key=f"{key_prefix}_method_select",
        )

    # Compute explanation
    with st.spinner("Computing visual gradients & activation maps..."):
        expl = generate_explanation(
            image=image,
            input_tensor=input_tensor,
            target_class_idx=target_class_idx,
            alpha=alpha,
            colormap_name=colormap,
            method="gradcam" if method == "Grad-CAM" else "saliency",
        )

    # Multi-tab inspection view
    tab_overlay, tab_side_by_side, tab_heatmap, tab_orig = st.tabs([
        "🎨 Blended Overlay",
        "🔲 Side-by-Side Comparison",
        "🔥 Raw Heatmap",
        "🖼️ Original Image",
    ])

    with tab_overlay:
        st.image(
            expl.overlay_image,
            caption=f"Overlay ({expl.colormap} Colormap, Opacity: {expl.alpha:.2f})",
            width="stretch",
        )

    with tab_side_by_side:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(expl.original_image, caption="Original Input", width="stretch")
        with c2:
            st.image(expl.heatmap_image, caption="Activation Intensity", width="stretch")
        with c3:
            st.image(expl.overlay_image, caption="Blended Attribution", width="stretch")

    with tab_heatmap:
        st.image(expl.heatmap_image, caption="Normalized Activation Map", width="stretch")

    with tab_orig:
        st.image(expl.original_image, caption="Original Input Image", width="stretch")

    # Regional analysis summary card
    stats = expl.region_statistics
    st.markdown(
        f"""
        <div style="
            background: #1e293b;
            border-left: 4px solid #6366f1;
            padding: 12px 16px;
            border-radius: 6px;
            margin-top: 14px;
            font-size: 0.88rem;
        ">
            <b>Attribution Focus:</b> {stats.get('primary_focus', 'N/A')}<br/>
            <b>Peak Activation:</b> {stats.get('peak_intensity', 0):.2f} &nbsp;|&nbsp;
            <b>Mean Activation:</b> {stats.get('mean_intensity', 0):.2f} &nbsp;|&nbsp;
            <b>Central Subject Weight:</b> {stats.get('center_activation', 0):.2f} &nbsp;|&nbsp;
            <b>Periphery Weight:</b> {stats.get('periphery_activation', 0):.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"ℹ️ *{expl.disclaimer}*")

    return expl
