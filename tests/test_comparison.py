"""TruthLens AI - Image Comparison Unit Tests"""

from __future__ import annotations

import pytest
from PIL import Image

from services.comparison_service import compare_two_images
from services.image_service import ImageMetadata


def test_independent_dual_comparison():
    """Verifies that two images pass through independent inference and compute differential metrics."""
    img_a = Image.new("RGB", (224, 224), color=(20, 40, 60))
    meta_a = ImageMetadata(
        filename="img_a.png",
        format="PNG",
        mode="RGB",
        width=224,
        height=224,
        aspect_ratio=1.0,
        file_size_kb=50.0,
        is_valid=True,
    )

    img_b = Image.new("RGB", (224, 224), color=(200, 220, 240))
    meta_b = ImageMetadata(
        filename="img_b.png",
        format="PNG",
        mode="RGB",
        width=224,
        height=224,
        aspect_ratio=1.0,
        file_size_kb=75.0,
        is_valid=True,
    )

    comp = compare_two_images(img_a, meta_a, img_b, meta_b)

    assert comp.result_a is not None
    assert comp.result_b is not None
    assert comp.ai_prob_diff >= 0.0
    assert comp.real_prob_diff >= 0.0
    assert len(comp.comparative_table) > 0
    assert "disclaimer" in comp.disclaimer.lower() or "independent" in comp.disclaimer.lower()
