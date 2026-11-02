"""TruthLens AI - Explanation Service
High-level service coordinating visual attribution (Grad-CAM and saliency maps),
colorized heatmap rendering, alpha-blended overlays, and forensic region summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from ml.explainability import (
    GradCAM,
    analyze_salient_regions,
    apply_heatmap_overlay,
    compute_saliency_map,
)
from services.prediction_service import get_cached_model


@dataclass
class ExplanationResult:
    """Encapsulates attribution images and explanatory textual commentary."""
    original_image: Image.Image
    heatmap_image: Image.Image
    overlay_image: Image.Image
    heatmap_raw_2d: np.ndarray
    alpha: float
    colormap: str
    target_class_name: str
    region_statistics: Dict[str, Any]
    plain_summary: str
    disclaimer: str


def generate_explanation(
    image: Image.Image,
    input_tensor: torch.Tensor,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.5,
    colormap_name: str = "JET",
    method: str = "gradcam",
    device: str = "cpu",
) -> ExplanationResult:
    """
    Computes visual attribution for an image and returns renderable overlays.

    Args:
        image: Original RGB PIL Image.
        input_tensor: Preprocessed tensor (1, 3, 224, 224).
        target_class_idx: 0 for Real, 1 for AI-Generated, or None for top prediction.
        alpha: Blending weight between original image and heatmap [0.0, 1.0].
        colormap_name: 'JET' or 'VIRIDIS'.
        method: 'gradcam' (default) or 'saliency'.
        device: 'cpu' or 'cuda'.

    Returns:
        ExplanationResult with PIL images, region analysis, and explanations.
    """
    model = get_cached_model(device=device)
    model.eval()

    target_layer = getattr(model, "target_cam_layer", None)
    if method == "gradcam" and target_layer is not None:
        try:
            cam = GradCAM(model=model, target_layer=target_layer)
            heatmap_2d = cam.generate_heatmap(
                input_tensor=input_tensor.clone().to(device),
                target_class_idx=target_class_idx,
            )
            cam.remove_hooks()
        except Exception as e:
            print(f"Notice: GradCAM fallback to saliency due to: {e}")
            heatmap_2d = compute_saliency_map(
                model=model,
                input_tensor=input_tensor.clone().to(device),
                target_class_idx=target_class_idx,
            )
    else:
        heatmap_2d = compute_saliency_map(
            model=model,
            input_tensor=input_tensor.clone().to(device),
            target_class_idx=target_class_idx,
        )

    # Render colorized heatmap and blended overlay
    heatmap_pil, overlay_pil = apply_heatmap_overlay(
        original_image=image,
        heatmap_2d=heatmap_2d,
        alpha=alpha,
        colormap_name=colormap_name,
    )

    region_stats = analyze_salient_regions(heatmap_2d)

    target_name = "AI-Generated (Synthetic)" if target_class_idx == 1 else "Real (Camera-Captured)"
    if target_class_idx is None:
        target_name = "Predicted Dominant Class"

    plain_summary = (
        f"The visual heatmap highlights areas that exerted the strongest mathematical influence "
        f"on the model's prediction for '{target_name}'. "
        f"The primary concentration is localized on: {region_stats.get('primary_focus', 'global features')}."
    )

    disclaimer = (
        "Visual heatmaps reflect intermediate gradient activations and evidence attribution; "
        "they indicate which image features the neural network prioritized, but do NOT constitute "
        "forensic proof of digital alteration or manipulation."
    )

    return ExplanationResult(
        original_image=image,
        heatmap_image=heatmap_pil,
        overlay_image=overlay_pil,
        heatmap_raw_2d=heatmap_2d,
        alpha=alpha,
        colormap=colormap_name,
        target_class_name=target_name,
        region_statistics=region_stats,
        plain_summary=plain_summary,
        disclaimer=disclaimer,
    )
