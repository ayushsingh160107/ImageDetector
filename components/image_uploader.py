"""TruthLens AI - Image Uploader Component
Renders upload area, sample image selector, preview thumbnail, and metadata card.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
from PIL import Image
import streamlit as st

from services.image_service import ImageMetadata, validate_and_load_image

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "samples")


def render_image_uploader(
    label: str = "Upload an Image for Forensic Analysis",
    key: str = "main_uploader",
    allow_samples: bool = True,
) -> Tuple[Optional[Image.Image], Optional[ImageMetadata]]:
    """
    Renders file uploader with drag-and-drop, sample selectors, preview, and metadata.

    Returns:
        Tuple of (PIL.Image or None, ImageMetadata or None)
    """
    st.markdown(f"### {label}")
    st.caption("Supported formats: **JPEG, PNG, WebP** (Max file size: 10 MB)")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        key=key,
        label_visibility="collapsed",
    )

    selected_sample_path: Optional[str] = None

    if allow_samples and os.path.exists(SAMPLE_DIR):
        sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if sample_files:
            st.markdown("<div style='margin-top: 10px; margin-bottom: 6px; font-size: 0.88rem; font-weight: 600; color: #94a3b8;'>Or test instantly with verified benchmark samples:</div>", unsafe_allow_html=True)
            cols = st.columns(len(sample_files))
            for i, filename in enumerate(sample_files):
                btn_label = "📷 Camera Photo" if "camera" in filename else "🤖 AI Synthetic"
                sub_label = filename.replace("sample_", "").replace(".jpg", "").replace(".png", "").replace("_", " ").title()
                if cols[i].button(f"{btn_label}\n{sub_label}", key=f"btn_sample_{key}_{i}", width="stretch"):
                    st.session_state[f"active_sample_{key}"] = os.path.join(SAMPLE_DIR, filename)

    active_sample = st.session_state.get(f"active_sample_{key}")

    img: Optional[Image.Image] = None
    meta: Optional[ImageMetadata] = None

    if uploaded_file is not None:
        # Reset sample if user uploaded a file
        if f"active_sample_{key}" in st.session_state:
            del st.session_state[f"active_sample_{key}"]

        img, meta = validate_and_load_image(
            file_source=uploaded_file.getvalue(),
            filename=uploaded_file.name,
        )
    elif active_sample and os.path.exists(active_sample):
        img, meta = validate_and_load_image(
            file_source=active_sample,
            filename=os.path.basename(active_sample),
        )

    if meta is not None:
        if not meta.is_valid:
            st.error(f"⚠️ Validation Error: {meta.validation_error}")
            return None, meta

        # Display preview and metadata card
        preview_col, info_col = st.columns([1, 1], gap="medium")
        with preview_col:
            st.image(img, caption=f"Preview: {meta.filename}", width="stretch")

        with info_col:
            st.markdown(
                f"""
                <div style="
                    background: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 10px;
                    padding: 16px 20px;
                    font-size: 0.9rem;
                ">
                    <h4 style="margin: 0 0 10px 0; color: #f8fafc; font-size: 1.05rem;">Asset Inspection</h4>
                    <p style="margin: 4px 0; color: #cbd5e1;"><b>Filename:</b> <code>{meta.filename}</code></p>
                    <p style="margin: 4px 0; color: #cbd5e1;"><b>Resolution:</b> {meta.dimensions_str} ({meta.megapixels} MP)</p>
                    <p style="margin: 4px 0; color: #cbd5e1;"><b>File Size:</b> {meta.file_size_kb:.1f} KB</p>
                    <p style="margin: 4px 0; color: #cbd5e1;"><b>Color Profile:</b> {meta.format} ({meta.mode})</p>
                    <p style="margin: 4px 0; color: #cbd5e1;"><b>Validation:</b> <span style="color: #4ade80;">✓ Passed Security Checks</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return img, meta
