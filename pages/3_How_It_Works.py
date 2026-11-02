"""TruthLens AI - Educational Guide: How It Works Page
Explains AI generation mechanics, Vision Transformers, GANs, frequency analysis,
explainability theory, and common failure modes in accessible, scientific language.
"""

from __future__ import annotations

import streamlit as st
from components.header import render_header

st.set_page_config(
    page_title="TruthLens AI — How It Works",
    page_icon="📖",
    layout="wide",
)

render_header(
    title="TruthLens Knowledge Hub",
    subtitle="Educational Architecture: Generative Vision, Forensics, and Explainable AI",
    badge="Educational Guide",
)

st.markdown(
    """
    Understanding how AI image detectors operate requires examining both how synthetic images are constructed
    and how deep neural networks extract forensic signatures.
    """
)

# Tabs for clear, structured learning
t1, t2, t3, t4, t5 = st.tabs([
    "🤖 Generative AI & GANs",
    "👁️ Vision Transformers & CNNs",
    "🔬 Preprocessing & Inference",
    "🔥 Grad-CAM Explainability",
    "⚠️ Why Detectors Fail",
])

with t1:
    st.markdown("### 1. What is an AI-Generated Image?")
    st.markdown(
        r"""
        AI-generated images are synthetic visual representations synthesized by deep generative models rather than captured by physical camera optical sensors.
        Modern generative models broadly fall into two main families:
        
        - **Diffusion Models (e.g., Stable Diffusion, Midjourney, DALL-E, FLUX):** Start with random Gaussian noise and iteratively denoise the latent representation conditioned on text prompts until a coherent image emerges.
        - **Generative Adversarial Networks (GANs, e.g., StyleGAN, DCGAN):** Consist of two competing neural networks:
          1. **Generator ($G$):** Takes a random latent vector $\mathbf{z}$ and tries to synthesize realistic images.
          2. **Discriminator ($D$):** Evaluates images and predicts whether they are authentic training samples or synthetic fakes.
        """
    )

    st.markdown("#### The GAN Minimax Objective")
    st.latex(r"\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]")
    st.caption("The Generator learns to fool the Discriminator, while the Discriminator learns to detect subtle synthetic flaws.")

with t2:
    st.markdown("### 2. Vision Transformers (ViT) & Convolutional Forensics")
    st.markdown(
        """
        While humans look for semantic clues (e.g., deformed hands, inconsistent shadows, garbled text), deep neural detectors primarily evaluate:
        
        1. **High-Frequency Spectral Artifacts:** Convolutional and deconvolutional upsampling in generators leaves distinctive grid patterns and checkerboard artifacts in the Fourier frequency domain.
        2. **Patch Embeddings in Vision Transformers (ViT):**
           - An input image is divided into a grid of non-overlapping patches (e.g., $16 \times 16$ pixels).
           - Patches are flattened and projected into linear vector embeddings.
           - Multi-head self-attention calculates contextual relationships across distant regions simultaneously.
        """
    )
    st.markdown("#### Self-Attention Formulation")
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    st.caption("Self-attention captures global coherence and subtle structural mismatches across the entire image.")

with t3:
    st.markdown("### 3. Image Preprocessing & Decision Bounds")
    st.markdown(
        r"""
        Before passing into the neural network, every image undergoes rigorous normalization:
        
        1. **Integrity Validation:** Checking file magic bytes, dimension sanity limits (max 4096px), and decompressing without executing code.
        2. **Color Profile Harmonization:** Converting RGBA or Grayscale images to standard 3-channel RGB.
        3. **EXIF Transposition:** Correcting camera orientation tags to prevent rotated feature distortion.
        4. **ImageNet Normalization:** Scaling pixel values from $[0, 255]$ into zero-mean, unit-variance tensors:
           $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
        5. **Softmax & Decision Thresholding:**
           $$\text{Probability}(y = \text{AI}) = \frac{e^{z_{\text{AI}}}}{e^{z_{\text{Real}}} + e^{z_{\text{AI}}}}$$
           TruthLens enforces an **uncertainty margin** ($0.40 < P < 0.60$) where outputs are labeled **Inconclusive** to prevent overconfident erroneous claims.
        """
    )

with t4:
    st.markdown("### 4. Explainability with Grad-CAM")
    st.markdown(
        """
        **Grad-CAM (Gradient-weighted Class Activation Mapping)** reveals what visual evidence the neural network used to make its decision:
        
        1. Forward propagate the image to compute class score $y^c$ (e.g., score for "AI-Generated").
        2. Compute gradients of score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:
        """
    )
    st.latex(r"\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}")
    st.markdown(
        """
        3. Weight each feature map by its gradient importance $\alpha_k^c$ and apply a ReLU function to keep only features that positively contributed to the class:
        """
    )
    st.latex(r"L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)")
    st.markdown(
        """
        4. The resulting 2D activation matrix is normalized to $[0, 1]$, colorized with JET/Viridis, and blended over the original image.
        """
    )

with t5:
    st.markdown("### 5. Why AI Image Detectors Can Fail")
    st.markdown(
        """
        Even state-of-the-art forensic classifiers face technical limitations in the real world:
        
        | Obstacle | Forensic Impact | Real-World Context |
        | :--- | :--- | :--- |
        | **Heavy Compression** | Destroys subtle high-frequency generator artifacts. | Social platforms (WhatsApp, Twitter, Instagram) recompress images to low JPEG qualities. |
        | **Downscaling** | Removes sub-pixel frequency anomalies. | Screenshots or low-resolution thumbnails. |
        | **Hybrid Workflows** | Blurs boundary between real and synthetic. | Real photos retouched with generative infilling, upscalers, or face filters. |
        | **Novel Architectures** | Out-of-distribution generator shift. | Newly released models with different diffusion samplers not present during training. |
        """
    )
    st.warning("Always use AI detection as forensic guidance and screening, never as solitary proof in legal proceedings.")
