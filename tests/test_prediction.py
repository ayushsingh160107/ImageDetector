"""TruthLens AI - Prediction & Model Inference Unit Tests"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from ml.classifier import PredictionResult, TruthLensForensicNet, run_inference
from services.prediction_service import predict_image


def test_model_forward_pass():
    """Verifies that model accepts (1, 3, 224, 224) tensor and returns logits of shape (1, 2)."""
    model = TruthLensForensicNet(architecture="custom")
    dummy_input = torch.randn(1, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (1, 2)


def test_inference_probabilities_sum_to_one():
    """Verifies that run_inference computes probabilities that sum to 1.0 within floating point precision."""
    model = TruthLensForensicNet(architecture="custom")
    dummy_input = torch.randn(1, 3, 224, 224)
    res = run_inference(model, dummy_input)
    
    assert isinstance(res, PredictionResult)
    prob_sum = res.ai_probability + res.real_probability
    assert pytest.approx(prob_sum, 0.01) == 1.0
    assert 0.0 <= res.confidence_score <= 1.0
    assert res.inference_time_ms > 0.0


def test_inconclusive_threshold_trigger():
    """Verifies that scores between threshold_real and threshold_ai yield an Inconclusive verdict."""
    model = TruthLensForensicNet(architecture="custom")
    dummy_input = torch.randn(1, 3, 224, 224)
    # Set wide inconclusive bounds: [0.01, 0.99]
    res = run_inference(model, dummy_input, threshold_ai=0.99, threshold_real=0.01)
    
    assert res.is_inconclusive is True
    assert res.predicted_label == "Inconclusive"
    assert res.uncertainty_warning is not None


def test_end_to_end_predict_service():
    """Verifies end-to-end predict_image service function with PIL input."""
    img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    res, tensor = predict_image(img, filename="service_test.png", save_to_history=False)
    
    assert res.model_name != ""
    assert res.plain_explanation != ""
    assert tensor.shape == (1, 3, 224, 224)
