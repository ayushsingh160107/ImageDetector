"""TruthLens AI - Report Generation Unit Tests"""

from __future__ import annotations

import json
from PIL import Image

from ml.classifier import PredictionResult
from services.image_service import ImageMetadata
from services.report_service import generate_json_report, generate_pdf_report


def test_json_report_format():
    """Verifies that generate_json_report produces valid JSON with required keys."""
    pred = PredictionResult(
        predicted_label="Likely AI-Generated",
        ai_probability=0.88,
        real_probability=0.12,
        confidence_score=0.88,
        raw_logits=[0.1, 2.5],
        is_inconclusive=False,
        threshold_ai=0.60,
        threshold_real=0.40,
        inference_time_ms=18.5,
        model_name="TruthLens-Forensic-v1",
        model_version="1.2.0",
        uncertainty_warning=None,
        plain_explanation="Synthetic diffusion texture detected.",
        technical_details={"device": "cpu"},
    )
    meta = ImageMetadata(
        filename="sample.png",
        format="PNG",
        mode="RGB",
        width=512,
        height=512,
        aspect_ratio=1.0,
        file_size_kb=120.0,
        is_valid=True,
    )

    json_str = generate_json_report(pred, meta)
    data = json.loads(json_str)

    assert "report_id" in data
    assert "image_metadata" in data
    assert "forensic_prediction" in data
    assert data["forensic_prediction"]["predicted_label"] == "Likely AI-Generated"
    assert "forensic_disclaimer" in data


def test_pdf_report_generation():
    """Verifies that generate_pdf_report produces a non-empty binary PDF with %PDF magic header."""
    pred = PredictionResult(
        predicted_label="Likely Real",
        ai_probability=0.15,
        real_probability=0.85,
        confidence_score=0.85,
        raw_logits=[2.1, 0.2],
        is_inconclusive=False,
        threshold_ai=0.60,
        threshold_real=0.40,
        inference_time_ms=14.2,
        model_name="TruthLens-Forensic-v1",
        model_version="1.2.0",
        uncertainty_warning=None,
        plain_explanation="Authentic camera sensor characteristics detected.",
        technical_details={"device": "cpu"},
    )
    meta = ImageMetadata(
        filename="real_sample.jpg",
        format="JPEG",
        mode="RGB",
        width=800,
        height=600,
        aspect_ratio=1.33,
        file_size_kb=240.0,
        is_valid=True,
    )
    thumb = Image.new("RGB", (100, 100), color=(120, 180, 240))

    pdf_bytes = generate_pdf_report(pred, meta, thumbnail_image=thumb)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
