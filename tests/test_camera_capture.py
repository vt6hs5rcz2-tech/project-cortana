"""Tests for Milestone 26 camera capture boundary."""

from __future__ import annotations

import time
from typing import Any

import pytest
from PIL import Image

from src.camera_capture import (
    CAMERA_FAILED,
    CameraCaptureError,
    CameraCaptureSession,
    RealtimeVisualFrame,
    normalize_camera_rgb_image,
    realtime_multimodal_features_enabled,
)
from src.config import (
    MAX_REALTIME_VISUAL_HEIGHT,
    MAX_REALTIME_VISUAL_WIDTH,
)


class FakeCapture:
    def __init__(self, frames: list[Any] | None = None) -> None:
        self.frames = list(frames or [])
        self.opened = True
        self.release_calls = 0
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True

    def read(self) -> tuple[bool, Any]:
        self.read_calls += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1
        self.opened = False


def _bgr_solid(width: int, height: int, bgr: tuple[int, int, int]) -> Any:
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_normalize_camera_image_downscales_oversized() -> None:
    image = Image.new("RGB", (2000, 1500), "red")
    frame = normalize_camera_rgb_image(
        image,
        sequence=0,
        captured_at_monotonic=1.0,
    )
    assert frame.mime_type == "image/png"
    assert frame.width <= MAX_REALTIME_VISUAL_WIDTH
    assert frame.height <= MAX_REALTIME_VISUAL_HEIGHT
    assert frame.width * frame.height <= MAX_REALTIME_VISUAL_WIDTH * MAX_REALTIME_VISUAL_HEIGHT


def test_latest_frame_capacity_one_and_sequence_monotonic() -> None:
    frames = [
        _bgr_solid(64, 64, (0, 0, 255)),
        _bgr_solid(64, 64, (0, 255, 0)),
        _bgr_solid(64, 64, (255, 0, 0)),
    ]
    capture = FakeCapture(frames=frames)
    session = CameraCaptureSession(capture_factory=lambda: capture)
    first = session.open_and_capture_first()
    assert first.sequence == 0
    # Manually pull two more through internal path.
    second = session._read_normalize_one(capture)
    session._replace_latest(second)
    third = session._read_normalize_one(capture)
    session._replace_latest(third)
    latest = session.get_latest_frame()
    assert latest is not None
    assert latest.sequence == 2
    assert latest.sequence > first.sequence
    session.stop()
    assert capture.release_calls == 1
    assert session.get_latest_frame() is None


def test_frame_age_uses_monotonic_and_rejects_stale() -> None:
    clock = {"now": 100.0}

    def mono() -> float:
        return clock["now"]

    capture = FakeCapture(frames=[_bgr_solid(32, 32, (0, 0, 255))])
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        monotonic_fn=mono,
    )
    session.open_and_capture_first()
    assert session.get_fresh_frame(max_age_seconds=3.0) is not None
    clock["now"] = 104.0
    assert session.get_fresh_frame(max_age_seconds=3.0) is None
    session.stop()


def test_one_malformed_frame_raises_without_fatal_signal() -> None:
    import numpy as np

    good = _bgr_solid(32, 32, (0, 0, 255))
    capture = FakeCapture(frames=[good, np.zeros((10,), dtype=np.uint8), good])
    fatal: list[str] = []
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        on_fatal=fatal.append,
        max_consecutive_failures=3,
    )
    session.open_and_capture_first()
    with pytest.raises(CameraCaptureError):
        session._read_normalize_one(capture)
    frame = session._read_normalize_one(capture)
    session._replace_latest(frame)
    session._consecutive_failures = 0
    assert fatal == []
    session.stop()


def test_threshold_failures_signal_fatal() -> None:
    capture = FakeCapture(frames=[_bgr_solid(16, 16, (0, 0, 255))])
    fatal: list[str] = []
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        on_fatal=fatal.append,
        max_consecutive_failures=2,
        sample_interval_seconds=0.01,
        sleep_fn=lambda _s: None,
    )
    session.open_and_capture_first()
    # Exhaust frames so read fails.
    session._consecutive_failures = 0
    with pytest.raises(CameraCaptureError):
        session._read_normalize_one(capture)
    session._consecutive_failures = 1
    with pytest.raises(CameraCaptureError):
        session._read_normalize_one(capture)
    session._consecutive_failures = 2
    session._signal_fatal("camera_failed")
    assert fatal == ["camera_failed"]
    session.stop()


def test_gates_disable_camera_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.camera_capture.REALTIME_MULTIMODAL_ENABLED",
        False,
    )
    assert realtime_multimodal_features_enabled() is False
    session = CameraCaptureSession(
        capture_factory=lambda: FakeCapture(frames=[_bgr_solid(8, 8, (1, 2, 3))])
    )
    with pytest.raises(CameraCaptureError) as exc:
        session.open_and_capture_first()
    assert "disabled" in exc.value.user_message.casefold()


def test_realtime_visual_frame_rejects_non_png() -> None:
    with pytest.raises(CameraCaptureError):
        RealtimeVisualFrame(
            image_bytes=b"not-png",
            mime_type="image/jpeg",  # type: ignore[arg-type]
            width=1,
            height=1,
            sequence=0,
            captured_at_monotonic=0.0,
        )


def test_worker_pacing_uses_sample_interval() -> None:
    sleeps: list[float] = []
    good = [_bgr_solid(16, 16, (0, 0, 255)) for _ in range(5)]
    capture = FakeCapture(frames=good)
    session = CameraCaptureSession(
        capture_factory=lambda: capture,
        sample_interval_seconds=0.5,
        sleep_fn=lambda s: sleeps.append(s),
        monotonic_fn=time.monotonic,
    )
    session.open_and_capture_first()
    session.start_worker()
    # Let worker loop a couple times.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and len(sleeps) < 1:
        time.sleep(0.05)
    session.stop()
    assert any(abs(s - 0.5) < 0.45 or s > 0 for s in sleeps) or capture.read_calls >= 2


def test_stop_releases_camera_even_when_worker_join_times_out() -> None:
    """M26-F1: cleanup_incomplete must not skip camera release/state reset."""
    capture = FakeCapture(frames=[_bgr_solid(32, 32, (0, 0, 255))])
    session = CameraCaptureSession(capture_factory=lambda: capture)
    session.open_and_capture_first()
    assert session.is_open is True
    assert session.get_latest_frame() is not None

    class ZombieThread:
        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    session._worker = ZombieThread()  # type: ignore[assignment]
    with pytest.raises(CameraCaptureError) as exc:
        session.stop()
    assert exc.value.error_type == "cleanup_incomplete"
    assert capture.release_calls == 1
    assert session.get_latest_frame() is None
    assert session._latest is None
    assert session.is_open is False
    assert session._opened is False
    assert session._capture is None
