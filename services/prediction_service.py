"""TruthLens AI - Prediction Service
Coordinates model loading, image preprocessing, forensic neural inference,
and database history persistence.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torch

from database.history import save_analysis
from ml.classifier import PredictionResult, run_inference
from ml.model_loader import get_model_metadata, load_model
from ml.preprocessing import get_model_preprocessing_specs, preprocess_image

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def _get_device() -> str:
    """Selects best available compute device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


if HAS_STREAMLIT:
    @st.cache_resource(show_spinner=False)
    def get_cached_model(device: str = "cpu", architecture: str = "custom"):
        """Loads and caches the PyTorch forensic model in Streamlit memory."""
        return load_model(device=device, architecture=architecture)
else:
    _GLOBAL_MODEL = None

    def get_cached_model(device: str = "cpu", architecture: str = "custom"):
        global _GLOBAL_MODEL
        if _GLOBAL_MODEL is None:
            _GLOBAL_MODEL = load_model(device=device, architecture=architecture)
        return _GLOBAL_MODEL


def predict_image(
    image: Image.Image,
    filename: str = "image.png",
    file_size_kb: float = 0.0,
    threshold_ai: float = 0.60,
    threshold_real: float = 0.40,
    device: Optional[str] = None,
    save_to_history: bool = True,
    notes: Optional[str] = None,
) -> Tuple[PredictionResult, torch.Tensor]:
    """
    Executes end-to-end inference for a single input image.

    Args:
        image: RGB PIL image.
        filename: Name of the uploaded file.
        file_size_kb: File size in kilobytes.
        threshold_ai: Decision threshold for AI detection.
        threshold_real: Decision threshold for Real detection.
        device: 'cpu' or 'cuda' (defaults to automatic selection).
        save_to_history: If True, stores record in SQLite database.
        notes: Optional user notes.

    Returns:
        Tuple of (PredictionResult, input_tensor)
    """
    selected_device = device or _get_device()
    model = get_cached_model(device=selected_device)
    metadata = get_model_metadata()

    # Preprocess image into normalized PyTorch tensor (1, 3, 224, 224)
    tensor_input = preprocess_image(image, target_size=(224, 224))

    # Run inference
    result = run_inference(
        model=model,
        tensor_input=tensor_input,
        threshold_ai=threshold_ai,
        threshold_real=threshold_real,
        device=selected_device,
        model_name=metadata.get("name", "TruthLens-DeepForensics"),
        model_version=metadata.get("version", "1.2.0"),
    )

    # Persist to database history if requested
    if save_to_history:
        try:
            dimensions = f"{image.width}x{image.height}"
            save_analysis(
                filename=filename,
                file_size_kb=file_size_kb,
                dimensions=dimensions,
                predicted_label=result.predicted_label,
                ai_probability=result.ai_probability,
                real_probability=result.real_probability,
                confidence_score=result.confidence_score,
                inference_time_ms=result.inference_time_ms,
                model_name=result.model_name,
                status="Success",
                uncertainty_flag=result.is_inconclusive,
                notes=notes,
            )
        except Exception as e:
            # Non-fatal database logging failure
            print(f"Notice: Failed to save to history DB: {e}")

    return result, tensor_input
