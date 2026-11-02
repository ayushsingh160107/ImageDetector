"""TruthLens AI - Complete Python-Based AI-Generated Image Detector and Explainer
Streamlit Main Application Entry Point
"""

from __future__ import annotations

import os
import streamlit as st

from components.header import render_header
from ml.model_loader import get_model_metadata

st.set_page_config(
    page_title="TruthLens AI — AI Image Detector & Explainer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Global CSS for rich aesthetics and deep navy/indigo dark mode
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Brand Header
render_header(
    title="TruthLens AI",
    subtitle="Forensic Detection & Attribution for AI-Generated vs. Camera-Captured Imagery",
    badge="v1.2.0 • Python Native",
)

# Hero Section
hero_col1, hero_col2 = st.columns([3, 2], gap="large")

with hero_col1:
    st.markdown(
        """
        ### Welcome to TruthLens AI
        
        As generative vision models (Midjourney, Stable Diffusion, DALL-E, FLUX) produce photorealistic imagery, distinguishing authentic camera captures from synthetic outputs has become an urgent forensic challenge.

        **TruthLens AI** is an open, transparent, Python-native deep learning platform that:
        - **Detects synthetic artifacts** using spatial frequencies and deep neural representations.
        - **Explains its reasoning** with mathematical Grad-CAM saliency heatmaps.
        - **Quantifies confidence & uncertainty** to avoid black-box false positives.
        - **Generates synthetic experiments** via an integrated DCGAN architecture.
        """
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🚀 Launch Image Detector", type="primary", width="stretch"):
            st.switch_page("pages/1_Image_Detector.py")
    with action_col2:
        if st.button("⚖️ Compare Two Images", width="stretch"):
            st.switch_page("pages/2_Compare_Images.py")

with hero_col2:
    meta = get_model_metadata()
    st.markdown(
        f"""
        <div style="
            background: #131b2e;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        ">
            <h4 style="margin: 0 0 12px 0; color: #a5b4fc; font-size: 1.1rem;">
                ⚡ Forensic Engine Status
            </h4>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                <b>Active Model:</b> {meta.get('name')}<br/>
                <b>Architecture:</b> {meta.get('architecture')}<br/>
                <b>Input Resolution:</b> {meta.get('input_resolution')}<br/>
                <b>Target Classes:</b> {', '.join(meta.get('classes', []))}<br/>
                <b>Status:</b> <span style="color: #4ade80;">● Calibrated & Operational</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# 3-Step Forensic Workflow
st.markdown("### 📋 The 3-Step Forensic Workflow")
wf1, wf2, wf3 = st.columns(3, gap="medium")

with wf1:
    st.markdown(
        """
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">1️⃣ Upload & Validate</div>
            <h4 style="margin: 0 0 8px 0; color: #f8fafc;">Cryptographic Validation</h4>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                Images are checked against decompression bombs, mime-type spoofs, and EXIF orientation anomalies.
                Raw images remain local in memory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf2:
    st.markdown(
        """
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">2️⃣ Deep Neural Inference</div>
            <h4 style="margin: 0 0 8px 0; color: #f8fafc;">Artifact Detection</h4>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                Normalized tensors pass through convolutional & spatial attention layers, computing genuine logits,
                calibrated probabilities, and uncertainty bounds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf3:
    st.markdown(
        """
        <div style="background: #1e293b; padding: 20px; border-radius: 10px; height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">3️⃣ Understand & Report</div>
            <h4 style="margin: 0 0 8px 0; color: #f8fafc;">Visual Attribution</h4>
            <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                Grad-CAM computes spatial activation maps to explain evidence in plain language.
                Download verifiable JSON and PDF forensic reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# Feature Explorer Cards
st.markdown("### 🧰 Explore TruthLens Modules")
m1, m2, m3, m4 = st.columns(4, gap="medium")

with m1:
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px;">
            <h4 style="color: #818cf8; margin-top: 0;">🔍 Image Detector</h4>
            <p style="font-size: 0.85rem; color: #9ca3af;">Single image forensic pipeline with instant confidence meters and Grad-CAM overlays.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Detector", key="nav_detector", width="stretch"):
        st.switch_page("pages/1_Image_Detector.py")

with m2:
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px;">
            <h4 style="color: #38bdf8; margin-top: 0;">⚖️ Compare Studio</h4>
            <p style="font-size: 0.85rem; color: #9ca3af;">Upload two images independently, compute differential scores, and view side-by-side heatmaps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Comparison", key="nav_compare", width="stretch"):
        st.switch_page("pages/2_Compare_Images.py")

with m3:
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px;">
            <h4 style="color: #34d399; margin-top: 0;">📊 Performance</h4>
            <p style="font-size: 0.85rem; color: #9ca3af;">Held-out evaluation metrics calculated via scikit-learn: Accuracy, F1, and Confusion Matrix.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Performance", key="nav_perf", width="stretch"):
        st.switch_page("pages/4_Model_Performance.py")

with m4:
    st.markdown(
        """
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 18px;">
            <h4 style="color: #fbbf24; margin-top: 0;">🧬 DCGAN Lab</h4>
            <p style="font-size: 0.85rem; color: #9ca3af;">Synthesize generative images from latent vectors to experiment with synthetic frequency signatures.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open DCGAN Lab", key="nav_gan", width="stretch"):
        st.switch_page("pages/5_GAN_Generator.py")

st.markdown("---")

# Legal and Forensic Disclaimer
st.info(
    "⚖️ **Forensic & Ethical Disclaimer:** TruthLens AI predictions are mathematical probability estimates derived from deep "
    "neural representations. They indicate whether an image exhibits known synthetic generator signatures or camera sensor noise. "
    "Predictions are intended for forensic screening, research, and educational demonstration and do not constitute absolute legal proof of authenticity."
)
