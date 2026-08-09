"""Tests for Milestone 23 visual input loading and normalization."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.config import (
    ALLOWED_VISION_EXTENSIONS,
    MAX_VISION_HEIGHT,
    MAX_VISION_SOURCE_BYTES,
    MAX_VISION_SOURCE_PIXELS,
    MAX_VISION_WIDTH,
    VISION_ANALYSIS_ENABLED,
)
from src.vision_input import (
    VISION_ANIMATED,
    VISION_CORRUPT,
    VISION_DISABLED,
    VISION_FILE_TOO_LARGE,
    VISION_FORMAT_MISMATCH,
    VISION_IMAGE_TOO_LARGE,
    VISION_UNSUPPORTED_TYPE,
    NormalizedVisualInput,
    VisionInputDisabledError,
    VisionInputValidationError,
    VisualInputLoader,
)


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="M23 vision file input requires Windows safe-open",
)


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (32, 24),
    image_format: str = "PNG",
    mode: str = "RGB",
    color: tuple[int, ...] = (255, 0, 0),
    exif: Image.Exif | None = None,
) -> Path:
    image = Image.new(mode, size, color)
    save_kwargs: dict[str, object] = {"format": image_format}
    if exif is not None:
        save_kwargs["exif"] = exif
    if image_format == "JPEG" and mode == "RGBA":
        image = image.convert("RGB")
    image.save(path, **save_kwargs)
    return path


def test_vision_config_defaults() -> None:
    assert VISION_ANALYSIS_ENABLED is True
    assert ALLOWED_VISION_EXTENSIONS == frozenset({".png", ".jpg", ".jpeg", ".webp"})
    assert MAX_VISION_SOURCE_BYTES == 10 * 1024 * 1024
    assert MAX_VISION_WIDTH == 4096
    assert MAX_VISION_HEIGHT == 4096
    assert MAX_VISION_SOURCE_PIXELS == 16_777_216


def test_load_valid_png_jpeg_webp(tmp_path: Path) -> None:
    loader = VisualInputLoader()
    png = _write_image(tmp_path / "a.png", image_format="PNG")
    jpg = _write_image(tmp_path / "b.jpg", image_format="JPEG")
    webp = _write_image(tmp_path / "c.webp", image_format="WEBP")

    for path, expected_format in (
        (png, "PNG"),
        (jpg, "JPEG"),
        (webp, "WEBP"),
    ):
        loaded = loader.load_local_file(str(path.resolve()))
        assert loaded.detected_source_format == expected_format
        assert loaded.display_basename == path.name
        assert loaded.normalized.mime_type == "image/png"
        assert loaded.normalized.source_kind == "local_file"
        assert loaded.normalized.width == 32
        assert loaded.normalized.height == 24
        assert loaded.normalized.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_extension_format_mismatch_rejected(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "spoof.png"
    _write_image(tmp_path / "real.jpg", image_format="JPEG")
    jpeg_bytes = (tmp_path / "real.jpg").read_bytes()
    jpeg_path.write_bytes(jpeg_bytes)

    with pytest.raises(VisionInputValidationError, match="does not match"):
        VisualInputLoader().load_local_file(str(jpeg_path.resolve()))
    assert VISION_FORMAT_MISMATCH


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.gif"
    _write_image(tmp_path / "tmp.png", image_format="PNG")
    path.write_bytes((tmp_path / "tmp.png").read_bytes())
    with pytest.raises(VisionInputValidationError, match="Unsupported image type"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_UNSUPPORTED_TYPE


def test_oversized_dimensions_rejected_before_normalize(tmp_path: Path) -> None:
    path = _write_image(
        tmp_path / "wide.png",
        size=(MAX_VISION_WIDTH + 1, 1),
        image_format="PNG",
    )
    with pytest.raises(VisionInputValidationError, match="dimensions exceed"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_IMAGE_TOO_LARGE


def test_oversized_source_bytes_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_image(tmp_path / "big.png", image_format="PNG")
    monkeypatch.setattr("src.vision_input.MAX_VISION_SOURCE_BYTES", 16)
    with pytest.raises(VisionInputValidationError, match="exceeds the maximum size"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_FILE_TOO_LARGE


def test_corrupt_image_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-an-image")
    with pytest.raises(VisionInputValidationError, match="could not be decoded"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_CORRUPT


def test_animated_webp_rejected(tmp_path: Path) -> None:
    path = tmp_path / "anim.webp"
    frame1 = Image.new("RGB", (16, 16), "red")
    frame2 = Image.new("RGB", (16, 16), "blue")
    frame1.save(
        path,
        format="WEBP",
        save_all=True,
        append_images=[frame2],
        duration=50,
        loop=0,
    )
    with pytest.raises(VisionInputValidationError, match="Animated"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_ANIMATED


def test_exif_orientation_applied_and_metadata_stripped(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (10, 20), "red")
    exif = image.getexif()
    exif[274] = 6  # rotate 90 CW
    image.save(path, format="JPEG", exif=exif)

    loaded = VisualInputLoader().load_local_file(str(path.resolve()))
    # Orientation 6 swaps width/height when baked into pixels.
    assert loaded.normalized.width == 20
    assert loaded.normalized.height == 10

    with Image.open(BytesIO(loaded.normalized.image_bytes)) as normalized:
        assert normalized.format == "PNG"
        assert dict(normalized.getexif()) == {}
        assert "exif" not in normalized.info
        assert "icc_profile" not in normalized.info


def test_normalized_input_rejects_non_png_mime() -> None:
    with pytest.raises(VisionInputValidationError):
        NormalizedVisualInput(
            mime_type="image/jpeg",  # type: ignore[arg-type]
            image_bytes=b"x",
            width=1,
            height=1,
            source_kind="local_file",
        )


def test_loader_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _write_image(tmp_path / "a.png", image_format="PNG")
    monkeypatch.setattr("src.vision_input.VISION_ANALYSIS_ENABLED", False)
    with pytest.raises(VisionInputDisabledError, match="disabled"):
        VisualInputLoader().load_local_file(str(path.resolve()))
    assert VISION_DISABLED


def test_quoted_path_supported(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "my image.png", image_format="PNG")
    quoted = f'"{path.resolve()}"'
    loaded = VisualInputLoader().load_local_file(quoted)
    assert loaded.display_basename == "my image.png"


def test_unc_path_rejected() -> None:
    with pytest.raises(VisionInputValidationError):
        VisualInputLoader().load_local_file(r"\\server\share\image.png")
