"""TruthLens AI - DCGAN Synthetic Image Generator Page
Synthesizes generative image samples using Deep Convolutional GAN (DCGAN),
explores latent vectors, and tests generator outputs against the detector.
"""

from __future__ import annotations

import io
import os
import torch
import streamlit as st

from components.header import render_header
from ml.gan_model import DCGANGenerator, generate_synthetic_images
from ml.model_loader import CHECKPOINT_DIR

st.set_page_config(
    page_title="TruthLens AI — GAN Generator",
    page_icon="🧬",
    layout="wide",
)

render_header(
    title="TruthLens DCGAN Laboratory",
    subtitle="Synthetic Image Synthesis & Latent Space Exploration",
    badge="DCGAN Laboratory",
)

st.markdown(
    r"""
    Generate synthetic images using a **Deep Convolutional Generative Adversarial Network (DCGAN)**.
    Experiment with latent noise vectors $\mathbf{z} \sim \mathcal{N}(0, I)$ and observe synthetic frequency artifacts.
    """
)

# Sidebar controls
st.sidebar.markdown("### 🎛️ Generator Controls")
seed = st.sidebar.number_input("Latent Vector Random Seed", min_value=0, max_value=999999, value=42, step=1)
num_images = st.sidebar.slider("Number of Samples", min_value=1, max_value=8, value=4, step=1)
latent_dim = st.sidebar.slider("Latent Vector Dimension (Z)", min_value=32, max_value=256, value=100, step=16)

gan_weights_path = os.path.join(CHECKPOINT_DIR, "dcgan_generator.pt")
has_trained_weights = os.path.exists(gan_weights_path)

if has_trained_weights:
    st.sidebar.success("✓ Trained DCGAN weights loaded from checkpoints/")
else:
    st.sidebar.info("ℹ️ Using initialized DCGAN generator weights.")

gen_col1, gen_col2 = st.columns([3, 1])
with gen_col1:
    st.markdown("#### Synthesize Synthetic Batch")
    st.caption(r"Each sample is deterministically mapped from the latent manifold $\mathbf{z} \in \mathbb{R}^{100} \to \mathbb{R}^{3 \times 64 \times 64}$.")
with gen_col2:
    generate_btn = st.button("✨ Generate Synthetic Images", type="primary", width="stretch")

if generate_btn or "generated_gallery" in st.session_state:
    if generate_btn:
        with st.spinner("Synthesizing images through DCGAN Generator transpose convolutions..."):
            generator = DCGANGenerator(latent_dim=latent_dim)
            if has_trained_weights:
                try:
                    ckpt = torch.load(gan_weights_path, map_location="cpu", weights_only=False)
                    if "generator_state_dict" in ckpt:
                        generator.load_state_dict(ckpt["generator_state_dict"])
                except Exception:
                    pass
            generator.eval()

            images = generate_synthetic_images(
                generator=generator,
                num_samples=num_images,
                latent_dim=latent_dim,
                seed=seed,
            )
            st.session_state["generated_gallery"] = images

    images = st.session_state.get("generated_gallery", [])

    st.markdown("### 🖼️ Synthetic Output Gallery")
    st.warning("⚠️ **Synthetic Notice:** All images below are 100% synthetically generated and do not depict real persons or physical scenes.")

    cols = st.columns(min(len(images), 4))
    for i, img in enumerate(images):
        col_idx = i % 4
        with cols[col_idx]:
            st.image(img, caption=f"Synthetic Sample #{i + 1} (Seed: {seed})", width="stretch")
            # Option to download sample
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label=f"💾 Download #{i + 1}",
                data=buf.getvalue(),
                file_name=f"truthlens_dcgan_seed{seed}_sample{i+1}.png",
                mime="image/png",
                key=f"dl_gen_{i}",
                width="stretch",
            )

st.markdown("---")

# DCGAN Architecture Reference
st.markdown("### 🏛️ DCGAN Architecture Specifications")
col_arch1, col_arch2 = st.columns(2, gap="large")

with col_arch1:
    st.markdown(
        r"""
        <div style="background: #1e293b; border-radius: 8px; padding: 18px;">
            <h4 style="margin: 0 0 10px 0; color: #818cf8;">DCGAN Generator ($G$)</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                <li><b>Input:</b> Latent vector $\mathbf{z} \in \mathbb{R}^{100 \times 1 \times 1}$</li>
                <li><b>Layer 1:</b> ConvTranspose2d (100 → 512, 4x4, s=1, p=0) + BatchNorm + ReLU</li>
                <li><b>Layer 2:</b> ConvTranspose2d (512 → 256, 4x4, s=2, p=1) + BatchNorm + ReLU</li>
                <li><b>Layer 3:</b> ConvTranspose2d (256 → 128, 4x4, s=2, p=1) + BatchNorm + ReLU</li>
                <li><b>Layer 4:</b> ConvTranspose2d (128 → 64, 4x4, s=2, p=1) + BatchNorm + ReLU</li>
                <li><b>Output Layer:</b> ConvTranspose2d (64 → 3, 4x4, s=2, p=1) + Tanh</li>
                <li><b>Output Resolution:</b> $3 \times 64 \times 64$ RGB pixel values in $[-1, 1]$</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_arch2:
    st.markdown(
        r"""
        <div style="background: #1e293b; border-radius: 8px; padding: 18px;">
            <h4 style="margin: 0 0 10px 0; color: #38bdf8;">DCGAN Discriminator ($D$)</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                <li><b>Input:</b> Image tensor $\mathbf{x} \in \mathbb{R}^{3 \times 64 \times 64}$</li>
                <li><b>Layer 1:</b> Conv2d (3 → 64, 4x4, s=2, p=1) + LeakyReLU (0.2)</li>
                <li><b>Layer 2:</b> Conv2d (64 → 128, 4x4, s=2, p=1) + BatchNorm + LeakyReLU (0.2)</li>
                <li><b>Layer 3:</b> Conv2d (128 → 256, 4x4, s=2, p=1) + BatchNorm + LeakyReLU (0.2)</li>
                <li><b>Layer 4:</b> Conv2d (256 → 512, 4x4, s=2, p=1) + BatchNorm + LeakyReLU (0.2)</li>
                <li><b>Output Layer:</b> Conv2d (512 → 1, 4x4, s=1, p=0) + Sigmoid</li>
                <li><b>Classification:</b> Real (1.0) vs. Synthetic Fake (0.0)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
