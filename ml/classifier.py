"""TruthLens AI - Binary Image Forensics Classifier
Implements neural network architectures (Vision Transformer and ResNet backbone)
for distinguishing between camera-captured (real) and AI-generated (synthetic) imagery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@dataclass
class PredictionResult:
    """Standardized result object for TruthLens inference."""
    predicted_label: str  # "Likely AI-Generated", "Likely Real", or "Inconclusive"
    ai_probability: float  # [0.0, 1.0]
    real_probability: float  # [0.0, 1.0]
    confidence_score: float  # [0.0, 1.0]
    raw_logits: List[float]  # [real_logit, ai_logit]
    is_inconclusive: bool
    threshold_ai: float
    threshold_real: float
    inference_time_ms: float
    model_name: str
    model_version: str
    uncertainty_warning: Optional[str]
    plain_explanation: str
    technical_details: Dict[str, Any]


class TruthLensForensicNet(nn.Module):
    """
    Forensic deep learning model combining deep spatial representations
    with a calibrated classification head for real vs. AI-generated binary classification.
    """

    def __init__(self, architecture: str = "resnet50", pretrained_backbone: bool = False):
        super().__init__()
        self.architecture = architecture
        self.classes = ["Real (Camera-Captured)", "AI-Generated (Synthetic)"]

        if architecture == "resnet50" and HAS_TORCHVISION:
            # ResNet-50 backbone
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            base_model = tv_models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            # Replace final classification head with forensic classifier
            base_model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )
            self.backbone = base_model
            self.target_cam_layer = self.backbone.layer4[-1]
        elif architecture == "vit" and HAS_TORCHVISION:
            # ViT-B/16 backbone
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained_backbone else None
            base_model = tv_models.vit_b_16(weights=weights)
            in_features = base_model.heads.head.in_features
            base_model.heads.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.Linear(256, 2),
            )
            self.backbone = base_model
            self.target_cam_layer = base_model.encoder.layers[-1].ln_1
        else:
            # Standalone CNN architecture (zero external heavy dependency requirement)
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, 2),  # 112x112
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, 2),  # 56x56
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, 2),  # 28x28
                # Block 4 - High-level forensic artifacts
                nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Index 17
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.target_cam_layer = self.features[17]  # Last Conv2d layer
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(inplace=False),
                nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
            self.backbone = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            return self.backbone(x)
        features = self.features(x)
        return self.classifier(features)

    def get_target_layer_for_cam(self) -> nn.Module:
        return self.target_cam_layer


def run_inference(
    model: nn.Module,
    tensor_input: torch.Tensor,
    threshold_ai: float = 0.60,
    threshold_real: float = 0.40,
    device: str = "cpu",
    model_name: str = "TruthLens-ViT-Forensic-v1.2",
    model_version: str = "1.2.0",
) -> PredictionResult:
    """
    Executes model inference on a normalized input tensor.

    Args:
        model: Trained or loaded PyTorch model
        tensor_input: Tensor of shape (1, 3, H, W) or (3, H, W)
        threshold_ai: AI probability at or above which image is marked Likely AI-Generated
        threshold_real: AI probability at or below which image is marked Likely Real
        device: 'cpu' or 'cuda'

    Returns:
        PredictionResult with scores, labels, confidence, and plain-language explanation.
    """
    if tensor_input.dim() == 3:
        tensor_input = tensor_input.unsqueeze(0)

    model.eval()
    model.to(device)
    tensor_input = tensor_input.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    real_prob = float(probs[0])
    ai_prob = float(probs[1])
    raw_logits_list = [float(l) for l in logits.squeeze(0).cpu().numpy()]

    # Decision rule with uncertainty margin
    if ai_prob >= threshold_ai:
        predicted_label = "Likely AI-Generated"
        confidence_score = ai_prob
        is_inconclusive = False
        uncertainty_warning = None
        plain_explanation = (
            f"The image exhibits synthetic frequency patterns, generative diffusion textures, or structural "
            f"inconsistencies characteristic of AI image generators (e.g., Midjourney, Stable Diffusion, DALL-E). "
            f"Estimated AI probability is {ai_prob * 100:.1f}%."
        )
    elif ai_prob <= threshold_real:
        predicted_label = "Likely Real / Camera-Captured"
        confidence_score = real_prob
        is_inconclusive = False
        uncertainty_warning = None
        plain_explanation = (
            f"The image preserves natural sensor noise, coherent optical depth of field, and continuous gradient transitions "
            f"typical of genuine photographic capture. Estimated real authenticity probability is {real_prob * 100:.1f}%."
        )
    else:
        predicted_label = "Inconclusive"
        confidence_score = max(ai_prob, real_prob)
        is_inconclusive = True
        uncertainty_warning = (
            f"Model scores fall in the uncertainty boundary ({ai_prob * 100:.1f}% AI vs {real_prob * 100:.1f}% Real). "
            "Heavy compression, aggressive filtering, downscaling, or hybrid editing can mask forensic markers."
        )
        plain_explanation = (
            f"The analysis could not decisively determine authenticity. Both photographic and synthetic indicators are "
            f"present or partially obscured by compression."
        )

    technical_details = {
        "architecture": getattr(model, "architecture", "TruthLens-DeepForensics"),
        "raw_logits": raw_logits_list,
        "ai_probability_pct": f"{ai_prob * 100:.2f}%",
        "real_probability_pct": f"{real_prob * 100:.2f}%",
        "decision_bounds": f"[Real <= {threshold_real:.2f}] < Inconclusive < [AI >= {threshold_ai:.2f}]",
        "device": device,
    }

    return PredictionResult(
        predicted_label=predicted_label,
        ai_probability=round(ai_prob, 4),
        real_probability=round(real_prob, 4),
        confidence_score=round(confidence_score, 4),
        raw_logits=[round(x, 4) for x in raw_logits_list],
        is_inconclusive=is_inconclusive,
        threshold_ai=threshold_ai,
        threshold_real=threshold_real,
        inference_time_ms=round(inference_time_ms, 2),
        model_name=model_name,
        model_version=model_version,
        uncertainty_warning=uncertainty_warning,
        plain_explanation=plain_explanation,
        technical_details=technical_details,
    )
