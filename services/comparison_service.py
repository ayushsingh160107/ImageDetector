"""TruthLens AI - Comparison Service
Performs independent, side-by-side comparative forensic analysis of two images.
Calculates differential metrics, score divergence, and comparative summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from ml.classifier import PredictionResult
from services.explanation_service import ExplanationResult, generate_explanation
from services.image_service import ImageMetadata
from services.prediction_service import predict_image


@dataclass
class ComparisonResult:
    """Comprehensive comparison data between Image A and Image B."""
    result_a: PredictionResult
    result_b: PredictionResult
    explanation_a: ExplanationResult
    explanation_b: ExplanationResult
    meta_a: ImageMetadata
    meta_b: ImageMetadata
    ai_prob_diff: float  # abs(A - B)
    real_prob_diff: float  # abs(A - B)
    confidence_diff: float  # abs(A - B)
    dominant_image: str  # Which image has higher synthetic likelihood
    divergence_category: str  # 'High Divergence', 'Moderate Difference', 'High Similarity'
    comparison_summary: str
    comparative_table: list[Dict[str, Any]]
    disclaimer: str


def compare_two_images(
    image_a: Image.Image,
    meta_a: ImageMetadata,
    image_b: Image.Image,
    meta_b: ImageMetadata,
    threshold_ai: float = 0.60,
    threshold_real: float = 0.40,
    alpha: float = 0.5,
    colormap_name: str = "JET",
) -> ComparisonResult:
    """
    Executes independent inference and attribution on both images and computes comparative metrics.
    """
    # Independent inference pipeline for Image A
    pred_a, tensor_a = predict_image(
        image=image_a,
        filename=meta_a.filename,
        file_size_kb=meta_a.file_size_kb,
        threshold_ai=threshold_ai,
        threshold_real=threshold_real,
        save_to_history=False,
    )

    # Independent inference pipeline for Image B
    pred_b, tensor_b = predict_image(
        image=image_b,
        filename=meta_b.filename,
        file_size_kb=meta_b.file_size_kb,
        threshold_ai=threshold_ai,
        threshold_real=threshold_real,
        save_to_history=False,
    )

    # Independent visual explanation for Image A
    expl_a = generate_explanation(
        image=image_a,
        input_tensor=tensor_a,
        target_class_idx=1 if pred_a.ai_probability >= 0.5 else 0,
        alpha=alpha,
        colormap_name=colormap_name,
    )

    # Independent visual explanation for Image B
    expl_b = generate_explanation(
        image=image_b,
        input_tensor=tensor_b,
        target_class_idx=1 if pred_b.ai_probability >= 0.5 else 0,
        alpha=alpha,
        colormap_name=colormap_name,
    )

    # Compute deltas
    ai_diff = round(abs(pred_a.ai_probability - pred_b.ai_probability), 4)
    real_diff = round(abs(pred_a.real_probability - pred_b.real_probability), 4)
    conf_diff = round(abs(pred_a.confidence_score - pred_b.confidence_score), 4)

    if ai_diff > 0.40:
        divergence_category = "High Divergence"
    elif ai_diff > 0.15:
        divergence_category = "Moderate Difference"
    else:
        divergence_category = "High Forensic Similarity"

    if pred_a.ai_probability > pred_b.ai_probability:
        dominant_image = f"Image A ({meta_a.filename})"
    elif pred_b.ai_probability > pred_a.ai_probability:
        dominant_image = f"Image B ({meta_b.filename})"
    else:
        dominant_image = "Equal Probabilities"

    comparison_summary = (
        f"Comparative forensic analysis demonstrates {divergence_category.lower()} (Δ {ai_diff * 100:.1f}% AI score). "
        f"Image A scored {pred_a.ai_probability * 100:.1f}% AI probability ({pred_a.predicted_label}), while "
        f"Image B scored {pred_b.ai_probability * 100:.1f}% AI probability ({pred_b.predicted_label})."
    )

    comparative_table = [
        {"Metric": "Filename", "Image A": meta_a.filename, "Image B": meta_b.filename, "Differential": "—"},
        {"Metric": "Dimensions", "Image A": meta_a.dimensions_str, "Image B": meta_b.dimensions_str, "Differential": "—"},
        {"Metric": "File Size", "Image A": f"{meta_a.file_size_kb:.1f} KB", "Image B": f"{meta_b.file_size_kb:.1f} KB", "Differential": f"{abs(meta_a.file_size_kb - meta_b.file_size_kb):.1f} KB"},
        {"Metric": "Predicted Class", "Image A": pred_a.predicted_label, "Image B": pred_b.predicted_label, "Differential": "Match" if pred_a.predicted_label == pred_b.predicted_label else "Divergent"},
        {"Metric": "AI-Generated Prob", "Image A": f"{pred_a.ai_probability * 100:.1f}%", "Image B": f"{pred_b.ai_probability * 100:.1f}%", "Differential": f"Δ {ai_diff * 100:.1f}%"},
        {"Metric": "Camera-Captured Prob", "Image A": f"{pred_a.real_probability * 100:.1f}%", "Image B": f"{pred_b.real_probability * 100:.1f}%", "Differential": f"Δ {real_diff * 100:.1f}%"},
        {"Metric": "Confidence Level", "Image A": f"{pred_a.confidence_score * 100:.1f}%", "Image B": f"{pred_b.confidence_score * 100:.1f}%", "Differential": f"Δ {conf_diff * 100:.1f}%"},
        {"Metric": "Inference Latency", "Image A": f"{pred_a.inference_time_ms:.1f} ms", "Image B": f"{pred_b.inference_time_ms:.1f} ms", "Differential": f"{abs(pred_a.inference_time_ms - pred_b.inference_time_ms):.1f} ms"},
    ]

    disclaimer = (
        "Each image underwent independent model evaluation. Similar forensic scores do not imply or prove that "
        "two images share the same generative architecture, prompt, or physical camera sensor."
    )

    return ComparisonResult(
        result_a=pred_a,
        result_b=pred_b,
        explanation_a=expl_a,
        explanation_b=expl_b,
        meta_a=meta_a,
        meta_b=meta_b,
        ai_prob_diff=ai_diff,
        real_prob_diff=real_diff,
        confidence_diff=conf_diff,
        dominant_image=dominant_image,
        divergence_category=divergence_category,
        comparison_summary=comparison_summary,
        comparative_table=comparative_table,
        disclaimer=disclaimer,
    )
