"""Shared in-memory image normalization for Milestones 23 and 26.

Filesystem trust (path validation, safe-open, TOCTOU) stays in
``vision_input``. Camera capture stays in ``camera_capture``. This module
only accepts already-loaded Pillow images or verified encoded bytes and
emits metadata-free PNG bytes.
"""

from __future__ import annotations

import logging
import warnings
from io import BytesIO
from typing import Literal

from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError, DecompressionBombWarning

from src.config import (
    ALLOWED_VISION_PILLOW_FORMATS,
    MAX_VISION_HEIGHT,
    MAX_VISION_NORMALIZED_BYTES,
    MAX_VISION_SOURCE_BYTES,
    MAX_VISION_SOURCE_PIXELS,
    MAX_VISION_WIDTH,
)

logger = logging.getLogger("ProjectCortana")

_EXTENSION_TO_FORMATS: dict[str, frozenset[str]] = {
    ".png": frozenset({"PNG"}),
    ".jpg": frozenset({"JPEG"}),
    ".jpeg": frozenset({"JPEG"}),
    ".webp": frozenset({"WEBP"}),
}


class VisionNormalizeError(RuntimeError):
    """Raised when in-memory image normalization fails."""

    def __init__(self, error_type: str, user_message: str) -> None:
        super().__init__(user_message)
        self.error_type = error_type
        self.user_message = user_message


def normalized_mode(image: Image.Image) -> Literal["RGB", "RGBA"]:
    """Map arbitrary Pillow modes to RGB or RGBA for provider submission."""
    mode = image.mode
    if mode in {"RGBA", "LA", "PA"} or "A" in mode:
        return "RGBA"
    if mode == "P" and "transparency" in image.info:
        return "RGBA"
    return "RGB"


def encode_metadata_free_png(image: Image.Image) -> tuple[bytes, int, int]:
    """Re-encode one Pillow image as a fresh metadata-free PNG."""
    mode = normalized_mode(image)
    converted = image if image.mode == mode else image.convert(mode)
    fresh = Image.frombytes(mode, converted.size, converted.tobytes())
    buffer = BytesIO()
    fresh.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    width, height = fresh.size
    return png_bytes, width, height


def fit_image_to_bounds(
    image: Image.Image,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
) -> Image.Image:
    """Downscale an image to fit hard dimension/pixel bounds. Never upscales."""
    width, height = image.size
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or isinstance(height, bool)
        or width < 1
        or height < 1
    ):
        raise VisionNormalizeError("corrupt", "Image dimensions are invalid.")

    scale = 1.0
    if width > max_width:
        scale = min(scale, max_width / width)
    if height > max_height:
        scale = min(scale, max_height / height)
    if width * height > max_pixels:
        scale = min(scale, (max_pixels / (width * height)) ** 0.5)
    if scale >= 1.0:
        return image

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    # Final clamp if rounding left a slightly oversized pixel count.
    while new_width * new_height > max_pixels and (new_width > 1 or new_height > 1):
        if new_width >= new_height and new_width > 1:
            new_width -= 1
        elif new_height > 1:
            new_height -= 1
        else:
            break
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def normalize_pillow_image(
    image: Image.Image,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
    max_normalized_bytes: int,
    apply_exif_orientation: bool = False,
    reject_oversized: bool = False,
) -> tuple[bytes, int, int]:
    """Normalize one loaded Pillow image to bounded metadata-free PNG bytes."""
    working = image
    if apply_exif_orientation:
        oriented = ImageOps.exif_transpose(working)
        working = oriented if oriented is not None else working

    width, height = working.size
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or isinstance(height, bool)
        or width < 1
        or height < 1
    ):
        raise VisionNormalizeError("corrupt", "Image dimensions are invalid.")

    if reject_oversized:
        if width > max_width or height > max_height or width * height > max_pixels:
            raise VisionNormalizeError(
                "image_too_large",
                "The image dimensions exceed the permitted visual limits.",
            )
    else:
        working = fit_image_to_bounds(
            working,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
        )

    png_bytes, out_width, out_height = encode_metadata_free_png(working)
    if len(png_bytes) > max_normalized_bytes:
        raise VisionNormalizeError(
            "normalize_too_large",
            "The normalized image exceeds the maximum permitted size.",
        )
    return png_bytes, out_width, out_height


def normalize_encoded_image_bytes(
    raw_bytes: bytes,
    *,
    extension: str,
    max_source_bytes: int = MAX_VISION_SOURCE_BYTES,
    max_width: int = MAX_VISION_WIDTH,
    max_height: int = MAX_VISION_HEIGHT,
    max_pixels: int = MAX_VISION_SOURCE_PIXELS,
    max_normalized_bytes: int = MAX_VISION_NORMALIZED_BYTES,
) -> tuple[str, int, int, bytes]:
    """Validate encoded bytes, orient, strip metadata, and re-encode to PNG.

    Used by Milestone 23 after safe-open has already produced verified bytes.
    Oversized source dimensions are rejected (not downscaled).
    """
    if len(raw_bytes) > max_source_bytes:
        raise VisionNormalizeError(
            "file_too_large",
            "The image file exceeds the maximum permitted size.",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DecompressionBombWarning)
        try:
            with Image.open(BytesIO(raw_bytes)) as header_image:
                detected_format = _validate_header_image(
                    header_image,
                    extension,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                )
                header_image.verify()

            with Image.open(BytesIO(raw_bytes)) as image:
                _validate_header_image(
                    image,
                    extension,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                )
                image.load()
                png_bytes, width, height = normalize_pillow_image(
                    image,
                    max_width=max_width,
                    max_height=max_height,
                    max_pixels=max_pixels,
                    max_normalized_bytes=max_normalized_bytes,
                    apply_exif_orientation=True,
                    reject_oversized=True,
                )
        except VisionNormalizeError:
            raise
        except (DecompressionBombError, DecompressionBombWarning) as error:
            raise VisionNormalizeError(
                "image_too_large",
                "The image dimensions exceed the permitted visual limits.",
            ) from error
        except Exception as error:
            logger.error(
                "Vision image decode/normalize failed error_type=%s",
                type(error).__name__,
            )
            raise VisionNormalizeError(
                "corrupt",
                "The image file could not be decoded safely.",
            ) from error

    return detected_format, width, height, png_bytes


def _validate_header_image(
    image: Image.Image,
    extension: str,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
) -> str:
    """Validate format, animation, and hard source dimension bounds before load."""
    detected = image.format
    if not isinstance(detected, str) or detected not in ALLOWED_VISION_PILLOW_FORMATS:
        raise VisionNormalizeError(
            "unsupported_type",
            "Unsupported image type.",
        )

    allowed_for_extension = _EXTENSION_TO_FORMATS.get(extension)
    if allowed_for_extension is None or detected not in allowed_for_extension:
        raise VisionNormalizeError(
            "format_mismatch",
            "The image file extension does not match its detected format.",
        )

    if bool(getattr(image, "is_animated", False)):
        raise VisionNormalizeError(
            "animated",
            "Animated or multi-frame images are not supported.",
        )
    n_frames = getattr(image, "n_frames", 1)
    if isinstance(n_frames, int) and n_frames > 1:
        raise VisionNormalizeError(
            "animated",
            "Animated or multi-frame images are not supported.",
        )

    width, height = image.size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise VisionNormalizeError("corrupt", "Image dimensions are invalid.")
    if width > max_width or height > max_height:
        raise VisionNormalizeError(
            "image_too_large",
            "The image dimensions exceed the permitted visual limits.",
        )
    if width * height > max_pixels:
        raise VisionNormalizeError(
            "image_too_large",
            "The image dimensions exceed the permitted visual limits.",
        )
    return detected
