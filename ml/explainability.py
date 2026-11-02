"""TruthLens AI - Explainability Module (Grad-CAM & Saliency Maps)
Computes visual attributions to explain which spatial regions influence the detector's
prediction (real vs. AI-generated).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for convolutional backbones and feature extraction stages.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.clone().detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].clone().detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Executes forward and backward passes to compute the normalized 2D Grad-CAM heatmap.
        Returns:
            heatmap_2d: 2D float32 numpy array with values in [0.0, 1.0]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not triggered
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Global average pooling of gradients: weights alpha_k
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_activations = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU to keep only positive influence
        cam = F.relu(weighted_activations)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            heatmap_2d = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap_2d = np.zeros_like(cam_np)

        return heatmap_2d.astype(np.float32)


def compute_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Computes Vanilla Input Gradient / Saliency Map:
    dL / d(Input Image), suitable for Vision Transformers and all feedforward vision models.
    """
    model.eval()
    tensor_clone = input_tensor.clone().detach().requires_grad_(True)
    logits = model(tensor_clone)

    if target_class_idx is None:
        target_class_idx = int(torch.argmax(logits, dim=1).item())

    score = logits[0, target_class_idx]
    model.zero_grad()
    score.backward()

    if tensor_clone.grad is not None:
        # Maximum gradient across the 3 color channels
        grad = tensor_clone.grad.abs().squeeze(0).cpu().numpy()  # (3, H, W)
        saliency = np.max(grad, axis=0)  # (H, W)
        
        # Gaussian smoothing / box filter for better visual coherence
        if HAS_OPENCV:
            saliency = cv2.GaussianBlur(saliency, (9, 9), 0)

        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)
        return saliency.astype(np.float32)

    return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)


def apply_heatmap_overlay(
    original_image: Image.Image,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "JET",
) -> Tuple[Image.Image, Image.Image]:
    """
    Resizes 2D heatmap to match original image dimensions and applies color overlay.

    Returns:
        heatmap_image: PIL Image of colorized heatmap alone
        blended_image: PIL Image of original image with heatmap blended on top
    """
    w, h = original_image.size
    orig_np = np.array(original_image.convert("RGB"))

    # Resize 2D heatmap to original image resolution
    if HAS_OPENCV:
        heatmap_resized = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        cv_colormap = cv2.COLORMAP_JET if colormap_name.upper() == "JET" else cv2.COLORMAP_VIRIDIS
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv_colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Alpha blend: (1 - alpha) * original + alpha * heatmap
        alpha = float(np.clip(alpha, 0.0, 1.0))
        blended_np = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_color_rgb, alpha, 0)

        heatmap_image = Image.fromarray(heatmap_color_rgb)
        blended_image = Image.fromarray(blended_np)
    else:
        # Fallback using PIL and basic colormap
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_2d)).resize((w, h), Image.Resampling.BILINEAR)
        # Create a simple red-hot pseudo colormap
        r = heatmap_pil
        g = heatmap_pil.point(lambda p: int(p * 0.4))
        b = heatmap_pil.point(lambda p: 255 - p)
        heatmap_image = Image.merge("RGB", (r, g, b))
        blended_image = Image.blend(original_image.convert("RGB"), heatmap_image, alpha)

    return heatmap_image, blended_image


def analyze_salient_regions(heatmap_2d: np.ndarray) -> Dict[str, Any]:
    """
    Extracts statistical descriptors from the heatmap to summarize which image
    regions (center, borders, fine textures) contributed most heavily to the attribution.
    """
    h, w = heatmap_2d.shape
    center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
    center_x1, center_x2 = int(w * 0.25), int(w * 0.75)

    center_region = heatmap_2d[center_y1:center_y2, center_x1:center_x2]
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[center_y1:center_y2, center_x1:center_x2] = False
    border_region = heatmap_2d[border_mask]

    center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0
    border_mean = float(np.mean(border_region)) if border_region.size > 0 else 0.0

    if center_mean > border_mean * 1.25:
        primary_focus = "Central subject and facial/focal textures"
    elif border_mean > center_mean * 1.25:
        primary_focus = "Background periphery and edge boundary transitions"
    else:
        primary_focus = "Diffuse across both subject details and background artifacts"

    peak_intensity = float(np.max(heatmap_2d))
    mean_intensity = float(np.mean(heatmap_2d))

    return {
        "primary_focus": primary_focus,
        "peak_intensity": round(peak_intensity, 4),
        "mean_intensity": round(mean_intensity, 4),
        "center_activation": round(center_mean, 4),
        "periphery_activation": round(border_mean, 4),
    }
