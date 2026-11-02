"""TruthLens AI - Preprocessing & Image Validation Unit Tests"""

from __future__ import annotations

import io
import pytest
import numpy as np
from PIL import Image

from services.image_service import validate_and_load_image
from ml.preprocessing import preprocess_image, get_model_preprocessing_specs


def test_valid_png_loading():
    """Verifies that a valid PNG image loads properly and outputs RGB mode."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    loaded_img, meta = validate_and_load_image(buf.getvalue(), filename="test.png")
    assert loaded_img is not None
    assert meta.is_valid is True
    assert loaded_img.mode == "RGB"
    assert meta.format == "PNG"
    assert meta.width == 100
    assert meta.height == 100


def test_corrupted_image_rejected():
    """Verifies that corrupted random bytes are safely rejected."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    loaded_img, meta = validate_and_load_image(corrupted_bytes, filename="corrupt.jpg")
    assert loaded_img is None
    assert meta.is_valid is False
    assert "Corrupted or invalid" in (meta.validation_error or "")


def test_oversized_file_rejected():
    """Verifies that files exceeding 10MB limit are rejected."""
    large_size = 11 * 1024 * 1024  # 11 MB
    dummy_bytes = b"0" * (1024 * 1024)  # 1 MB chunk for testing with lower threshold
    loaded_img, meta = validate_and_load_image(
        dummy_bytes,
        filename="large.png",
        max_size_bytes=512 * 1024,  # Enforce 512 KB max in test
    )
    assert loaded_img is None
    assert meta.is_valid is False
    assert "exceeds maximum allowed size" in (meta.validation_error or "")


def test_preprocessing_tensor_output():
    """Verifies that preprocess_image produces a (1, 3, 224, 224) float tensor."""
    img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    tensor = preprocess_image(img, target_size=(224, 224))
    
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32 or str(tensor.dtype) == "torch.float32"
    specs = get_model_preprocessing_specs()
    assert specs["input_resolution"] == "224x224x3"
