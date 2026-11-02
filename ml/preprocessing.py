"""TruthLens AI - Image Preprocessing & Validation Pipeline
Handles robust validation, decompression bomb protection, EXIF orientation correction,
color mode normalization, and tensor preparation for deep learning inference.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

# Prevent decompression bomb DOS attacks while allowing reasonable photos
Image.MAX_IMAGE_PIXELS = 40_000_000  # Up to ~40 Megapixels

ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 4096


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


def validate_image_bytes(
    file_bytes: bytes,
    filename: str = "upload.jpg",
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Dict[str, Any]:
    """
    Validates image file bytes thoroughly:
    1. Size limit check
    2. Header and byte structure verification (Pillow verify)
    3. Dimension checks
    4. Format verification against allowed MIME/formats
    """
    file_size = len(file_bytes)
    if file_size == 0:
        raise ImageValidationError("The uploaded file is empty (0 bytes).")

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed limit of {max_mb:.0f} MB."
        )

    # Validate image integrity by opening with Pillow
    try:
        raw_stream = io.BytesIO(file_bytes)
        img_verify = Image.open(raw_stream)
        img_format = (img_verify.format or "").upper()
        
        # Verify Pillow recognizes it as a valid image
        img_verify.verify()
    except Exception as e:
        raise ImageValidationError(f"Invalid or corrupted image file: {str(e)}")

    if img_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"Unsupported format '{img_format}'. TruthLens supports JPG, JPEG, PNG, and WebP."
        )

    # Re-open after verify() closes the stream
    raw_stream.seek(0)
    try:
        img = Image.open(raw_stream)
        width, height = img.size
    except Exception as e:
        raise ImageValidationError(f"Failed to decode image data: {str(e)}")

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_DIMENSION}px."
        )

    if width < 16 or height < 16:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small for meaningful forensic analysis."
        )

    return {
        "filename": filename,
        "format": img_format,
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "file_size_kb": file_size / 1024.0,
    }


def load_and_preprocess_image(
    file_input: Union[bytes, io.BytesIO, str, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
    """
    Decodes image, fixes orientation, normalizes to RGB,
    creates display copy, and produces a resized numpy array normalized for PyTorch models.

    Returns:
        display_image: PIL.Image in RGB (original aspect ratio preserved for viewing)
        normalized_array: float32 numpy array with shape (3, target_height, target_width)
        metadata: dictionary of image attributes
    """
    if isinstance(file_input, bytes):
        metadata = validate_image_bytes(file_input)
        img = Image.open(io.BytesIO(file_input))
    elif isinstance(file_input, io.BytesIO):
        bytes_data = file_input.getvalue()
        metadata = validate_image_bytes(bytes_data)
        file_input.seek(0)
        img = Image.open(file_input)
    elif isinstance(file_input, str):
        with open(file_input, "rb") as f:
            bytes_data = f.read()
        metadata = validate_image_bytes(bytes_data, filename=file_input)
        img = Image.open(io.BytesIO(bytes_data))
    elif isinstance(file_input, Image.Image):
        img = file_input
        metadata = {
            "filename": getattr(img, "filename", "in_memory.png"),
            "format": img.format or "PNG",
            "width": img.width,
            "height": img.height,
            "file_size_bytes": 0,
            "file_size_kb": 0.0,
        }
    else:
        raise ImageValidationError(f"Unsupported image input type: {type(file_input)}")

    # Correct EXIF orientation (e.g. mobile photos taken vertically)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Ensure RGB color mode
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Create white background for transparent images
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3])
            display_image = bg
        else:
            display_image = img.convert("RGB")
    else:
        display_image = img.copy()

    # Model input preprocessing (Bilinear / Bicubic resize to target_size)
    resized_img = display_image.resize(target_size, Image.Resampling.BILINEAR)
    img_np = np.array(resized_img, dtype=np.float32) / 255.0  # [0, 1]

    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (img_np - mean) / std

    # Transpose to channels-first: (H, W, C) -> (C, H, W)
    tensor_array = np.transpose(normalized, (2, 0, 1))

    return display_image, tensor_array, metadata


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Preprocesses a PIL Image into a normalized PyTorch tensor of shape (1, 3, H, W).
    """
    import torch
    _, tensor_np, _ = load_and_preprocess_image(image, target_size=target_size)
    return torch.from_numpy(tensor_np).unsqueeze(0).float()


def get_model_preprocessing_specs() -> Dict[str, Any]:
    """Returns preprocessing specifications for documentation and UI display."""
    return {
        "input_resolution": "224x224x3",
        "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        "interpolation": "Bilinear",
        "color_space": "RGB",
        "tensor_shape": "(1, 3, 224, 224)",
    }
