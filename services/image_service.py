"""TruthLens AI - Image Service
Handles image loading, format normalization, dimension limits, EXIF orientation correction,
and metadata extraction.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image, ImageOps

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MAX_IMAGE_DIMENSION = 4096


@dataclass
class ImageMetadata:
    """Detailed metadata for an inspected image."""
    filename: str
    format: str
    mode: str
    width: int
    height: int
    aspect_ratio: float
    file_size_kb: float
    is_valid: bool
    validation_error: Optional[str] = None

    @property
    def dimensions_str(self) -> str:
        return f"{self.width} × {self.height} px"

    @property
    def megapixels(self) -> float:
        return round((self.width * self.height) / 1_000_000, 2)


def validate_and_load_image(
    file_source: Union[bytes, io.BytesIO, str],
    filename: str = "uploaded_image.png",
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[Optional[Image.Image], ImageMetadata]:
    """
    Validates file content, enforces size/dimension bounds, and loads PIL Image in RGB format.

    Args:
        file_source: Raw image bytes, BytesIO buffer, or file path.
        filename: Original file name.
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        Tuple of (PIL.Image or None, ImageMetadata)
    """
    raw_bytes: bytes
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None, ImageMetadata(
                filename=filename,
                format="UNKNOWN",
                mode="UNKNOWN",
                width=0,
                height=0,
                aspect_ratio=0.0,
                file_size_kb=0.0,
                is_valid=False,
                validation_error=f"File not found: {file_source}",
            )
        file_size = os.path.getsize(file_source)
        with open(file_source, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_source, io.BytesIO):
        raw_bytes = file_source.getvalue()
        file_size = len(raw_bytes)
    elif isinstance(file_source, bytes):
        raw_bytes = file_source
        file_size = len(raw_bytes)
    else:
        return None, ImageMetadata(
            filename=filename,
            format="UNKNOWN",
            mode="UNKNOWN",
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_size_kb=0.0,
            is_valid=False,
            validation_error="Unsupported file source type.",
        )

    file_size_kb = round(file_size / 1024, 2)

    # 1. Enforce file size limit
    if file_size > max_size_bytes:
        return None, ImageMetadata(
            filename=filename,
            format="UNKNOWN",
            mode="UNKNOWN",
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_size_kb=file_size_kb,
            is_valid=False,
            validation_error=f"File exceeds maximum allowed size ({max_size_bytes / (1024 * 1024):.1f} MB).",
        )

    if file_size == 0:
        return None, ImageMetadata(
            filename=filename,
            format="UNKNOWN",
            mode="UNKNOWN",
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_size_kb=0.0,
            is_valid=False,
            validation_error="Uploaded file is empty (0 bytes).",
        )

    # 2. Open with Pillow and verify integrity
    try:
        buffer = io.BytesIO(raw_bytes)
        img = Image.open(buffer)
        raw_format = (img.format or "PNG").upper()
        
        # Verify image stream
        img.verify()
        
        # Re-open after verify() because verify() clears the file pointer
        buffer.seek(0)
        img = Image.open(buffer)

        # 3. Check format
        if raw_format not in SUPPORTED_FORMATS and raw_format != "JPEG":
            return None, ImageMetadata(
                filename=filename,
                format=raw_format,
                mode=img.mode,
                width=img.width,
                height=img.height,
                aspect_ratio=round(img.width / max(img.height, 1), 2),
                file_size_kb=file_size_kb,
                is_valid=False,
                validation_error=f"Unsupported format '{raw_format}'. Please upload JPG, PNG, or WebP.",
            )

        # 4. Check dimensions
        if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
            return None, ImageMetadata(
                filename=filename,
                format=raw_format,
                mode=img.mode,
                width=img.width,
                height=img.height,
                aspect_ratio=round(img.width / max(img.height, 1), 2),
                file_size_kb=file_size_kb,
                is_valid=False,
                validation_error=f"Image resolution ({img.width}x{img.height}) exceeds max dimension limit of {MAX_IMAGE_DIMENSION}px.",
            )

        # 5. Fix EXIF orientation
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # 6. Convert to RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        metadata = ImageMetadata(
            filename=filename,
            format=raw_format,
            mode="RGB",
            width=img.width,
            height=img.height,
            aspect_ratio=round(img.width / max(img.height, 1), 2),
            file_size_kb=file_size_kb,
            is_valid=True,
            validation_error=None,
        )

        return img, metadata

    except Exception as exc:
        return None, ImageMetadata(
            filename=filename,
            format="CORRUPTED",
            mode="UNKNOWN",
            width=0,
            height=0,
            aspect_ratio=0.0,
            file_size_kb=file_size_kb,
            is_valid=False,
            validation_error=f"Corrupted or invalid image file: {str(exc)}",
        )


def create_thumbnail(img: Image.Image, max_dim: int = 512) -> Image.Image:
    """Returns a copy of the image resized for quick display while preserving aspect ratio."""
    thumb = img.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return thumb
