"""Local camera frame-quality gate for multimodal insertion."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from src.camera_capture import (
    CameraCaptureSession,
    normalize_camera_rgb_image,
)
from src.vision_normalize import encode_metadata_free_png
from src.visual_frame_quality import assess_visual_frame_quality
from tests.test_camera_capture import FakeCapture, _bgr_solid


def _frame(image: Image.Image, *, sequence: int = 0, when: float = 1.0):
    return normalize_camera_rgb_image(
        image,
        sequence=sequence,
        captured_at_monotonic=when,
    )


def _ordinary_scene() -> Image.Image:
    image = Image.new("RGB", (128, 72), (40, 80, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 15, 70, 55), fill=(20, 20, 20))
    draw.ellipse((80, 20, 115, 50), fill=(200, 40, 40))
    return image


def _privacy_placeholder(*, dx: int = 0, dy: int = 0, grey: int = 140) -> Image.Image:
    image = Image.new("RGB", (320, 180), (grey, grey, grey))
    draw = ImageDraw.Draw(image)
    cx, cy = 160 + dx, 90 + dy
    draw.rectangle((cx - 18, cy - 8, cx + 18, cy + 14), outline=(230, 230, 230), width=2)
    draw.ellipse((cx - 6, cy - 20, cx + 6, cy - 8), outline=(230, 230, 230), width=2)
    return image


def _red_square() -> Image.Image:
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 96, 96), fill=(220, 16, 16))
    return image


def _dark_remote_scene() -> Image.Image:
    image = Image.new("RGB", (160, 90), (210, 200, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, 110, 70), fill=(18, 18, 18))
    draw.ellipse((70, 40, 78, 48), fill=(60, 60, 60))
    return image


def _kitchen_like(*, seed: int) -> Image.Image:
    image = Image.new("RGB", (160, 90), (40 + seed, 70, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 70, 50), fill=(180, 170, 150))
    draw.rectangle((80, 20, 150, 80), fill=(30 + seed, 40, 50))
    draw.ellipse((20, 55, 55, 85), fill=(200, 80, 40))
    for x in range(0, 160, 7):
        draw.point((x, (x + seed) % 90), fill=(x % 200, 90, 40))
    return image


def _bgr_from_image(image: Image.Image) -> object:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _reencoded(image: Image.Image, *, sequence: int) -> object:
    png, _width, _height = encode_metadata_free_png(image)
    with Image.open(BytesIO(png)) as decoded:
        return _frame(decoded.convert("RGB"), sequence=sequence, when=float(sequence))


def test_black_frame_rejected() -> None:
    quality = assess_visual_frame_quality(_frame(Image.new("RGB", (64, 64), "black")))
    assert quality.usable is False
    assert quality.reason == "near_black"


def test_near_black_low_variance_frame_rejected() -> None:
    quality = assess_visual_frame_quality(
        _frame(Image.new("RGB", (64, 64), (8, 8, 8)))
    )
    assert quality.usable is False
    assert quality.reason == "near_black"


def test_ordinary_image_accepted() -> None:
    quality = assess_visual_frame_quality(_frame(_ordinary_scene()))
    assert quality.usable is True
    assert quality.reason == "ok"


def test_simple_low_detail_valid_image_accepted() -> None:
    quality = assess_visual_frame_quality(_frame(Image.new("RGB", (64, 64), "red")))
    assert quality.usable is True


def test_stale_frame_rejected() -> None:
    quality = assess_visual_frame_quality(
        _frame(Image.new("RGB", (64, 64), "red"), when=1.0),
        now=10.0,
        max_age_seconds=3.0,
    )
    assert quality.usable is False
    assert quality.reason == "stale"


def test_fresh_valid_frame_accepted() -> None:
    quality = assess_visual_frame_quality(
        _frame(Image.new("RGB", (64, 64), "red"), when=9.5),
        now=10.0,
        max_age_seconds=3.0,
    )
    assert quality.usable is True


def test_bad_first_frame_later_good_frame_selected() -> None:
    capture = FakeCapture(
        frames=[
            _bgr_solid(64, 64, (0, 0, 0)),
            _bgr_solid(64, 64, (0, 0, 255)),
        ]
    )
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        sleep_fn=lambda _s: None,
        sample_interval_seconds=0.0,
    )
    first = session.open_and_capture_first()
    assert assess_visual_frame_quality(first).usable is False
    assert session.get_usable_fresh_frame(now=first.captured_at_monotonic) is None
    good = session._read_normalize_one(capture)
    session._replace_latest(good)
    selected = session.get_usable_fresh_frame(now=good.captured_at_monotonic)
    assert selected is not None
    assert selected.sequence == good.sequence
    assert assess_visual_frame_quality(selected).usable is True


def test_no_good_frame_returns_none() -> None:
    capture = FakeCapture(frames=[_bgr_solid(64, 64, (0, 0, 0))])
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        sleep_fn=lambda _s: None,
        sample_interval_seconds=0.0,
    )
    first = session.open_and_capture_first()
    assert session.wait_for_usable_fresh_frame(wait_seconds=0.0) is None
    assert assess_visual_frame_quality(first).usable is False


def test_white_washout_rejected() -> None:
    quality = assess_visual_frame_quality(_frame(Image.new("RGB", (64, 64), "white")))
    assert quality.usable is False
    assert quality.reason == "near_white"


def test_grey_placeholder_without_temporal_evidence_is_not_rejected() -> None:
    quality = assess_visual_frame_quality(_frame(_privacy_placeholder()))
    assert quality.usable is True
    assert quality.reason == "ok"


def test_static_privacy_placeholder_rejected() -> None:
    first = _frame(_privacy_placeholder(), sequence=1, when=1.0)
    second = _reencoded(_privacy_placeholder(), sequence=2)
    quality = assess_visual_frame_quality(second, prior_frames=(first,))
    assert quality.usable is False
    assert quality.reason == "privacy_placeholder"


def test_placeholder_survives_slight_reencode_differences() -> None:
    first = _frame(_privacy_placeholder(), sequence=1, when=1.0)
    second = _reencoded(_privacy_placeholder(), sequence=2)
    third = _reencoded(_privacy_placeholder(), sequence=3)
    quality = assess_visual_frame_quality(third, prior_frames=(first, second))
    assert quality.usable is False
    assert quality.reason == "privacy_placeholder"


def test_mildly_changing_privacy_placeholder_sequence_is_rejected() -> None:
    first = _frame(_privacy_placeholder(dx=0, grey=140), sequence=1, when=1.0)
    second = _frame(_privacy_placeholder(dx=2, grey=143), sequence=2, when=1.3)
    third = _frame(_privacy_placeholder(dx=-1, grey=138), sequence=3, when=1.6)
    quality = assess_visual_frame_quality(third, prior_frames=(first, second))
    assert quality.usable is False
    assert quality.reason == "privacy_placeholder"
    assert quality.temporal_mae is not None
    assert quality.temporal_mae > 1.5


def test_red_square_accepted() -> None:
    quality = assess_visual_frame_quality(_frame(_red_square()))
    assert quality.usable is True
    assert quality.reason == "ok"


def test_plain_wall_with_object_accepted() -> None:
    quality = assess_visual_frame_quality(_frame(_ordinary_scene()))
    assert quality.usable is True


def test_dark_remote_scene_accepted() -> None:
    first = _frame(_dark_remote_scene(), sequence=1, when=1.0)
    second = _frame(_dark_remote_scene(), sequence=2, when=1.2)
    quality = assess_visual_frame_quality(second, prior_frames=(first,))
    assert quality.usable is True
    assert quality.reason == "ok"


def test_kitchen_like_frame_accepted() -> None:
    quality = assess_visual_frame_quality(_frame(_kitchen_like(seed=3)))
    assert quality.usable is True


def test_moving_photographic_sequence_accepted() -> None:
    first = _frame(_kitchen_like(seed=1), sequence=1, when=1.0)
    second = _frame(_kitchen_like(seed=40), sequence=2, when=1.4)
    quality = assess_visual_frame_quality(second, prior_frames=(first,))
    assert quality.usable is True
    assert quality.reason == "ok"


def test_placeholder_then_photographic_frame_is_selected() -> None:
    capture = FakeCapture(
        frames=[
            _bgr_from_image(_privacy_placeholder()),
            _bgr_from_image(_privacy_placeholder()),
            _bgr_from_image(_kitchen_like(seed=8)),
        ]
    )
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        sleep_fn=lambda _s: None,
        sample_interval_seconds=0.0,
    )
    first = session.open_and_capture_first()
    assert session.get_usable_fresh_frame(now=first.captured_at_monotonic) is not None
    second = session._read_normalize_one(capture)
    session._replace_latest(second)
    assert session.get_usable_fresh_frame(now=second.captured_at_monotonic) is None
    good = session._read_normalize_one(capture)
    session._replace_latest(good)
    selected = session.get_usable_fresh_frame(now=good.captured_at_monotonic)
    assert selected is not None
    assert selected.sequence == good.sequence
    assert assess_visual_frame_quality(selected).usable is True


def test_placeholder_throughout_wait_returns_none() -> None:
    capture = FakeCapture(
        frames=[
            _bgr_from_image(_privacy_placeholder()),
            _bgr_from_image(_privacy_placeholder()),
            _bgr_from_image(_privacy_placeholder()),
        ]
    )
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        sleep_fn=lambda _s: None,
        sample_interval_seconds=0.0,
    )
    session.open_and_capture_first()
    second = session._read_normalize_one(capture)
    session._replace_latest(second)
    third = session._read_normalize_one(capture)
    session._replace_latest(third)
    assert session.wait_for_usable_fresh_frame(wait_seconds=0.0) is None
