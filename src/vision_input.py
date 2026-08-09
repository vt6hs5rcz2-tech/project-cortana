"""Safe static visual input loading and normalization for Milestone 23.

Windows safe-open primitives from ``tool_process_safe_open`` are reused
directly. Non-Windows platforms fail closed in this milestone.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError, DecompressionBombWarning

from src.config import (
    ALLOWED_VISION_EXTENSIONS,
    ALLOWED_VISION_PILLOW_FORMATS,
    MAX_VISION_HEIGHT,
    MAX_VISION_NORMALIZED_BYTES,
    MAX_VISION_SOURCE_BYTES,
    MAX_VISION_SOURCE_PIXELS,
    MAX_VISION_WIDTH,
    VISION_ANALYSIS_ENABLED,
)
from src.path_argument_utils import strip_path_argument_quotes
from src.tool_process_safe_open import (
    FileTooLarge,
    SafeOpenError,
    SafeOpenUnavailableError,
    assert_safe_handle_unchanged_after_read,
    read_bytes_from_safe_handle,
    safe_open_for_read,
)

logger = logging.getLogger("ProjectCortana")

VisionSourceKind = Literal["local_file"]
VisionMimeType = Literal["image/png"]

VISION_DISABLED = "Cortana: Visual analysis is currently disabled."
VISION_PATH_REQUIRED = (
    "Cortana: Please provide an image path. "
    "Usage: /vision-info <path>"
)
VISION_UNSUPPORTED_PLATFORM = (
    "Cortana: Visual file input requires Windows in this milestone."
)
VISION_FILE_ACCESS = "Cortana: The specified image file could not be accessed."
VISION_FILE_NOT_FOUND = "Cortana: The specified image file could not be found."
VISION_FILE_TOO_LARGE = (
    "Cortana: The image file exceeds the maximum size of "
    f"{MAX_VISION_SOURCE_BYTES} bytes."
)
VISION_UNSUPPORTED_TYPE = (
    "Cortana: Unsupported image type. Supported types: "
    + ", ".join(sorted(ALLOWED_VISION_EXTENSIONS))
    + "."
)
VISION_FORMAT_MISMATCH = (
    "Cortana: The image file extension does not match its detected format."
)
VISION_IMAGE_TOO_LARGE = (
    "Cortana: The image dimensions exceed the permitted visual limits."
)
VISION_ANIMATED = (
    "Cortana: Animated or multi-frame images are not supported."
)
VISION_CORRUPT = "Cortana: The image file could not be decoded safely."
VISION_NORMALIZE_TOO_LARGE = (
    "Cortana: The normalized image exceeds the maximum size of "
    f"{MAX_VISION_NORMALIZED_BYTES} bytes."
)

_EXTENSION_TO_FORMATS: dict[str, frozenset[str]] = {
    ".png": frozenset({"PNG"}),
    ".jpg": frozenset({"JPEG"}),
    ".jpeg": frozenset({"JPEG"}),
    ".webp": frozenset({"WEBP"}),
}


class VisionInputError(RuntimeError):
    """Raised when a visual input cannot be loaded safely."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class VisionInputDisabledError(VisionInputError):
    """Raised when visual analysis is disabled."""


class VisionInputValidationError(VisionInputError):
    """Raised when visual input validation fails."""


@dataclass(frozen=True)
class NormalizedVisualInput:
    """Ephemeral normalized image accepted by VisualAnalysisService.

    Contains no filesystem path, basename, EXIF, or provider file ID.
    """

    mime_type: VisionMimeType
    image_bytes: bytes
    width: int
    height: int
    source_kind: VisionSourceKind

    def __post_init__(self) -> None:
        if self.mime_type != "image/png":
            raise VisionInputValidationError(VISION_CORRUPT)
        if self.source_kind != "local_file":
            raise VisionInputValidationError(VISION_CORRUPT)
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise VisionInputValidationError(VISION_CORRUPT)
        if len(self.image_bytes) > MAX_VISION_NORMALIZED_BYTES:
            raise VisionInputValidationError(VISION_NORMALIZE_TOO_LARGE)
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
        ):
            raise VisionInputValidationError(VISION_CORRUPT)


@dataclass(frozen=True)
class LoadedVisualFile:
    """Local wrapper for /vision-info; never passed to the AI path."""

    normalized: NormalizedVisualInput
    display_basename: str
    detected_source_format: str
    source_bytes: int


class VisualInputLoader:
    """Load one authorized local image through Windows safe-open."""

    def load_local_file(self, path_argument: str) -> LoadedVisualFile:
        """Safe-open, validate, normalize, and return one loaded visual file."""
        if not VISION_ANALYSIS_ENABLED:
            raise VisionInputDisabledError(VISION_DISABLED)

        cleaned = strip_path_argument_quotes(path_argument).strip()
        if not cleaned:
            raise VisionInputValidationError(VISION_PATH_REQUIRED)

        display_basename = _display_basename(cleaned)
        extension = Path(cleaned).suffix.casefold()
        if extension not in ALLOWED_VISION_EXTENSIONS:
            raise VisionInputValidationError(VISION_UNSUPPORTED_TYPE)

        raw_bytes = _read_source_bytes(cleaned)
        detected_format, width, height, normalized = _normalize_image_bytes(
            raw_bytes,
            extension=extension,
        )
        logger.info(
            "Vision input loaded format=%s source_bytes=%s "
            "normalized_width=%s normalized_height=%s normalized_bytes=%s",
            detected_format,
            len(raw_bytes),
            width,
            height,
            len(normalized.image_bytes),
        )
        return LoadedVisualFile(
            normalized=normalized,
            display_basename=display_basename,
            detected_source_format=detected_format,
            source_bytes=len(raw_bytes),
        )


def _display_basename(path_argument: str) -> str:
    """Return a safe basename for local display only."""
    cleaned = path_argument.strip().replace("\\", "/")
    if not cleaned:
        return "[image]"
    name = cleaned.rstrip("/").rsplit("/", 1)[-1]
    return name or "[image]"


def _read_source_bytes(path_argument: str) -> bytes:
    """Read bounded source bytes from one Windows safe-open handle."""
    opened = None
    try:
        opened = safe_open_for_read(path_argument, expected_identity=None)
        if opened.identity.size_bytes > MAX_VISION_SOURCE_BYTES:
            raise VisionInputValidationError(VISION_FILE_TOO_LARGE)
        raw_bytes = read_bytes_from_safe_handle(
            opened,
            max_bytes=MAX_VISION_SOURCE_BYTES,
        )
        assert_safe_handle_unchanged_after_read(
            opened,
            total_bytes_read=len(raw_bytes),
            max_bytes=MAX_VISION_SOURCE_BYTES,
        )
        return raw_bytes
    except VisionInputError:
        raise
    except FileTooLarge as error:
        raise VisionInputValidationError(VISION_FILE_TOO_LARGE) from error
    except SafeOpenUnavailableError as error:
        raise VisionInputValidationError(VISION_UNSUPPORTED_PLATFORM) from error
    except SafeOpenError as error:
        message = str(error).casefold()
        if "could not be found" in message or "not found" in message:
            raise VisionInputValidationError(VISION_FILE_NOT_FOUND) from error
        # Secure open failures intentionally avoid path leakage.
        logger.error(
            "Vision safe-open failed error_type=%s",
            type(error).__name__,
        )
        raise VisionInputValidationError(VISION_FILE_ACCESS) from error
    except Exception as error:
        logger.error(
            "Vision file read failed unexpectedly error_type=%s",
            type(error).__name__,
        )
        raise VisionInputValidationError(VISION_FILE_ACCESS) from error
    finally:
        if opened is not None:
            opened.close()


def _normalize_image_bytes(
    raw_bytes: bytes,
    *,
    extension: str,
) -> tuple[str, int, int, NormalizedVisualInput]:
    """Validate, orient, strip metadata, and re-encode to PNG in memory."""
    if len(raw_bytes) > MAX_VISION_SOURCE_BYTES:
        raise VisionInputValidationError(VISION_FILE_TOO_LARGE)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DecompressionBombWarning)
        try:
            with Image.open(BytesIO(raw_bytes)) as header_image:
                detected_format = _validate_header_image(header_image, extension)
                header_image.verify()

            with Image.open(BytesIO(raw_bytes)) as image:
                _validate_header_image(image, extension)
                image.load()
                oriented = ImageOps.exif_transpose(image)
                working = oriented if oriented is not None else image
                normalized_mode = _normalized_mode(working)
                converted = (
                    working
                    if working.mode == normalized_mode
                    else working.convert(normalized_mode)
                )
                fresh = Image.frombytes(
                    normalized_mode,
                    converted.size,
                    converted.tobytes(),
                )
                buffer = BytesIO()
                fresh.save(buffer, format="PNG")
                png_bytes = buffer.getvalue()
                width, height = fresh.size
        except VisionInputError:
            raise
        except (DecompressionBombError, DecompressionBombWarning) as error:
            raise VisionInputValidationError(VISION_IMAGE_TOO_LARGE) from error
        except Exception as error:
            logger.error(
                "Vision image decode/normalize failed error_type=%s",
                type(error).__name__,
            )
            raise VisionInputValidationError(VISION_CORRUPT) from error

    if len(png_bytes) > MAX_VISION_NORMALIZED_BYTES:
        raise VisionInputValidationError(VISION_NORMALIZE_TOO_LARGE)

    normalized = NormalizedVisualInput(
        mime_type="image/png",
        image_bytes=png_bytes,
        width=width,
        height=height,
        source_kind="local_file",
    )
    return detected_format, width, height, normalized


def _validate_header_image(image: Image.Image, extension: str) -> str:
    """Validate format, animation, and hard source dimension bounds before load."""
    detected = image.format
    if not isinstance(detected, str) or detected not in ALLOWED_VISION_PILLOW_FORMATS:
        raise VisionInputValidationError(VISION_UNSUPPORTED_TYPE)

    allowed_for_extension = _EXTENSION_TO_FORMATS.get(extension)
    if allowed_for_extension is None or detected not in allowed_for_extension:
        raise VisionInputValidationError(VISION_FORMAT_MISMATCH)

    if bool(getattr(image, "is_animated", False)):
        raise VisionInputValidationError(VISION_ANIMATED)
    n_frames = getattr(image, "n_frames", 1)
    if isinstance(n_frames, int) and n_frames > 1:
        raise VisionInputValidationError(VISION_ANIMATED)

    width, height = image.size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise VisionInputValidationError(VISION_CORRUPT)
    if width > MAX_VISION_WIDTH or height > MAX_VISION_HEIGHT:
        raise VisionInputValidationError(VISION_IMAGE_TOO_LARGE)
    if width * height > MAX_VISION_SOURCE_PIXELS:
        raise VisionInputValidationError(VISION_IMAGE_TOO_LARGE)
    return detected


def _normalized_mode(image: Image.Image) -> Literal["RGB", "RGBA"]:
    """Map arbitrary Pillow modes to RGB or RGBA for provider submission."""
    mode = image.mode
    if mode in {"RGBA", "LA", "PA"} or "A" in mode:
        return "RGBA"
    if mode == "P" and "transparency" in image.info:
        return "RGBA"
    return "RGB"
