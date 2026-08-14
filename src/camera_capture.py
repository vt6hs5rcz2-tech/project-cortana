"""Default local camera capture for Milestone 26.

This is the only Project Cortana module permitted to import ``cv2`` / ``numpy``.
Frames leave this module only as metadata-free normalized PNG
``RealtimeVisualFrame`` values. No provider/network/history/authority.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from PIL import Image

from src.config import (
    MAX_REALTIME_VISUAL_CONSECUTIVE_FRAME_FAILURES,
    MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS,
    MAX_REALTIME_VISUAL_HEIGHT,
    MAX_REALTIME_VISUAL_NORMALIZED_BYTES,
    MAX_REALTIME_VISUAL_SOURCE_PIXELS,
    MAX_REALTIME_VISUAL_WIDTH,
    REALTIME_MULTIMODAL_ENABLED,
    REALTIME_VISUAL_FRESH_WAIT_SECONDS,
    REALTIME_VISUAL_SAMPLE_INTERVAL_SECONDS,
    REALTIME_VOICE_ENABLED,
    VISION_ANALYSIS_ENABLED,
    VOICE_INTERACTION_ENABLED,
)
from src.vision_normalize import VisionNormalizeError, normalize_pillow_image

logger = logging.getLogger("ProjectCortana")

CAMERA_UNAVAILABLE = "Cortana: No usable camera is available for multimodal mode."
CAMERA_FAILED = "Cortana: Camera capture failed."
CAMERA_FRAME_INVALID = "Cortana: A camera frame could not be normalized safely."
CAMERA_UNSUPPORTED_PLATFORM = (
    "Cortana: Realtime multimodal camera capture requires Windows in this milestone."
)
MULTIMODAL_DISABLED = (
    "Cortana: Realtime multimodal perception is currently disabled."
)


class CameraCaptureError(RuntimeError):
    """Raised when camera capture cannot continue safely."""

    def __init__(self, user_message: str, *, error_type: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.error_type = error_type


class CameraCaptureDisabledError(CameraCaptureError):
    """Raised when multimodal/vision/voice gates reject camera use."""


@dataclass(frozen=True)
class RealtimeVisualFrame:
    """One provider-safe local camera frame. Contains no device/path metadata."""

    image_bytes: bytes
    mime_type: Literal["image/png"]
    width: int
    height: int
    sequence: int
    captured_at_monotonic: float

    def __post_init__(self) -> None:
        if self.mime_type != "image/png":
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if len(self.image_bytes) > MAX_REALTIME_VISUAL_NORMALIZED_BYTES:
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
            or self.width > MAX_REALTIME_VISUAL_WIDTH
            or self.height > MAX_REALTIME_VISUAL_HEIGHT
            or self.width * self.height > MAX_REALTIME_VISUAL_SOURCE_PIXELS
        ):
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if self.sequence < 0:
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )
        if (
            isinstance(self.captured_at_monotonic, bool)
            or not isinstance(self.captured_at_monotonic, (int, float))
        ):
            raise CameraCaptureError(
                CAMERA_FRAME_INVALID,
                error_type="frame_invalid",
            )


class CaptureDevice(Protocol):
    """Minimal camera device surface used by CameraCaptureSession."""

    def isOpened(self) -> bool:
        """Return True when the device is open."""

    def read(self) -> tuple[bool, Any]:
        """Return ``(ok, bgr_frame)``."""

    def release(self) -> None:
        """Release the device."""

    def set(self, prop_id: int, value: float) -> bool:
        """Request a capture property."""


CaptureFactory = Callable[[], CaptureDevice]
OnCameraFatal = Callable[[str], None]


def realtime_multimodal_features_enabled() -> bool:
    """Return True only when all four multimodal prerequisite gates are on."""
    return bool(
        VOICE_INTERACTION_ENABLED
        and REALTIME_VOICE_ENABLED
        and VISION_ANALYSIS_ENABLED
        and REALTIME_MULTIMODAL_ENABLED
    )


def _default_capture_factory() -> CaptureDevice:
    """Open the OS default camera. Imports cv2 only inside this function path."""
    try:
        import cv2
    except Exception as error:
        logger.error(
            "Camera OpenCV import failed error_type=%s",
            type(error).__name__,
        )
        raise CameraCaptureError(
            CAMERA_UNAVAILABLE,
            error_type="camera_unavailable",
        ) from error

    try:
        capture = cv2.VideoCapture(0)
    except Exception as error:
        logger.error(
            "Camera open raised error_type=%s",
            type(error).__name__,
        )
        raise CameraCaptureError(
            CAMERA_UNAVAILABLE,
            error_type="camera_unavailable",
        ) from error

    if capture is None or not capture.isOpened():
        try:
            if capture is not None:
                capture.release()
        except Exception:
            pass
        raise CameraCaptureError(
            CAMERA_UNAVAILABLE,
            error_type="camera_unavailable",
        )

    # Requested resolution; driver compliance is not trusted.
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(MAX_REALTIME_VISUAL_WIDTH))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(MAX_REALTIME_VISUAL_HEIGHT))
    except Exception:
        pass
    return capture


def _bgr_frame_to_rgb_image(frame: Any) -> Image.Image:
    """Convert one OpenCV BGR ndarray to a Pillow RGB image. Drops the ndarray."""
    import cv2
    import numpy as np

    if frame is None:
        raise CameraCaptureError(CAMERA_FRAME_INVALID, error_type="frame_invalid")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise CameraCaptureError(CAMERA_FRAME_INVALID, error_type="frame_invalid")
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    # Copy into Pillow-owned bytes so the ndarray can be GC'd.
    return Image.fromarray(rgb.copy())


def normalize_camera_rgb_image(
    image: Image.Image,
    *,
    sequence: int,
    captured_at_monotonic: float,
) -> RealtimeVisualFrame:
    """Normalize one RGB Pillow image into a RealtimeVisualFrame."""
    try:
        png_bytes, width, height = normalize_pillow_image(
            image,
            max_width=MAX_REALTIME_VISUAL_WIDTH,
            max_height=MAX_REALTIME_VISUAL_HEIGHT,
            max_pixels=MAX_REALTIME_VISUAL_SOURCE_PIXELS,
            max_normalized_bytes=MAX_REALTIME_VISUAL_NORMALIZED_BYTES,
            apply_exif_orientation=False,
            reject_oversized=False,
        )
    except VisionNormalizeError as error:
        raise CameraCaptureError(
            CAMERA_FRAME_INVALID,
            error_type="frame_invalid",
        ) from error
    return RealtimeVisualFrame(
        image_bytes=png_bytes,
        mime_type="image/png",
        width=width,
        height=height,
        sequence=sequence,
        captured_at_monotonic=float(captured_at_monotonic),
    )


class CameraCaptureSession:
    """Own one default camera, latest-frame buffer, and camera worker."""

    def __init__(
        self,
        *,
        capture_factory: CaptureFactory | None = None,
        on_fatal: OnCameraFatal | None = None,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        sample_interval_seconds: float = REALTIME_VISUAL_SAMPLE_INTERVAL_SECONDS,
        max_consecutive_failures: int = MAX_REALTIME_VISUAL_CONSECUTIVE_FRAME_FAILURES,
    ) -> None:
        self._capture_factory = capture_factory or _default_capture_factory
        self._on_fatal = on_fatal
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._sample_interval_seconds = sample_interval_seconds
        self._max_consecutive_failures = max_consecutive_failures

        self._lock = threading.Lock()
        self._latest: RealtimeVisualFrame | None = None
        self._sequence = 0
        self._consecutive_failures = 0
        self._capture: CaptureDevice | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._opened = False

    @property
    def is_open(self) -> bool:
        """Return True after a successful open_and_capture_first."""
        return self._opened

    def open_and_capture_first(self) -> RealtimeVisualFrame:
        """Open the default camera and capture one valid normalized frame."""
        if not realtime_multimodal_features_enabled():
            raise CameraCaptureDisabledError(
                MULTIMODAL_DISABLED,
                error_type="multimodal_disabled",
            )
        if sys.platform != "win32":
            raise CameraCaptureError(
                CAMERA_UNSUPPORTED_PLATFORM,
                error_type="unsupported_platform",
            )
        if self._capture is not None:
            raise CameraCaptureError(CAMERA_FAILED, error_type="camera_failed")

        capture = self._capture_factory()
        self._capture = capture
        try:
            frame = self._read_normalize_one(capture)
        except Exception:
            self._release_capture()
            raise
        self._replace_latest(frame)
        self._opened = True
        logger.info(
            "Camera opened normalized_width=%s normalized_height=%s",
            frame.width,
            frame.height,
        )
        return frame

    def start_worker(self) -> None:
        """Start the one owned camera worker (2 FPS latest-frame refresh)."""
        if not self._opened or self._capture is None:
            raise CameraCaptureError(CAMERA_FAILED, error_type="camera_failed")
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="cortana-multimodal-camera",
            daemon=False,
        )
        self._worker.start()

    def get_latest_frame(self) -> RealtimeVisualFrame | None:
        """Return the current latest frame without age filtering."""
        with self._lock:
            return self._latest

    def get_fresh_frame(
        self,
        *,
        max_age_seconds: float = MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS,
        now: float | None = None,
    ) -> RealtimeVisualFrame | None:
        """Return the latest frame only when younger than ``max_age_seconds``."""
        frame = self.get_latest_frame()
        if frame is None:
            return None
        current = self._monotonic() if now is None else now
        age = current - frame.captured_at_monotonic
        if age < 0 or age > max_age_seconds:
            return None
        return frame

    def wait_for_fresh_frame(
        self,
        *,
        wait_seconds: float = REALTIME_VISUAL_FRESH_WAIT_SECONDS,
        max_age_seconds: float = MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS,
    ) -> RealtimeVisualFrame | None:
        """Wait briefly for a fresh frame; never wait indefinitely."""
        deadline = self._monotonic() + max(0.0, wait_seconds)
        while True:
            frame = self.get_fresh_frame(max_age_seconds=max_age_seconds)
            if frame is not None:
                return frame
            if self._monotonic() >= deadline or self._stop.is_set():
                return self.get_fresh_frame(max_age_seconds=max_age_seconds)
            self._sleep(0.05)

    def stop(self) -> None:
        """Stop the worker, clear the latest frame, and release the camera.

        Release/state cleanup always runs even when the worker fails to join.
        A join timeout still surfaces as ``cleanup_incomplete`` after cleanup.
        """
        self._stop.set()
        worker = self._worker
        self._worker = None
        join_incomplete = False
        if worker is not None:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.error("Camera worker join timed out")
                join_incomplete = True

        release_ok = self._release_capture()
        with self._lock:
            self._latest = None
        self._sequence = 0
        self._consecutive_failures = 0
        self._opened = False
        logger.info("Camera stopped")

        if join_incomplete:
            raise CameraCaptureError(
                CAMERA_FAILED,
                error_type="cleanup_incomplete",
            )
        if not release_ok:
            raise CameraCaptureError(
                CAMERA_FAILED,
                error_type="camera_failed",
            )

    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                started = self._monotonic()
                capture = self._capture
                if capture is None:
                    self._signal_fatal("camera_failed")
                    return
                try:
                    frame = self._read_normalize_one(capture)
                    self._replace_latest(frame)
                    self._consecutive_failures = 0
                except CameraCaptureError:
                    self._consecutive_failures += 1
                    logger.error(
                        "Camera frame failed consecutive=%s",
                        self._consecutive_failures,
                    )
                    if (
                        self._consecutive_failures
                        >= self._max_consecutive_failures
                    ):
                        self._signal_fatal("camera_failed")
                        return
                except Exception as error:
                    self._consecutive_failures += 1
                    logger.error(
                        "Camera worker unexpected error_type=%s consecutive=%s",
                        type(error).__name__,
                        self._consecutive_failures,
                    )
                    if (
                        self._consecutive_failures
                        >= self._max_consecutive_failures
                    ):
                        self._signal_fatal("camera_failed")
                        return
                elapsed = self._monotonic() - started
                remaining = self._sample_interval_seconds - elapsed
                if remaining > 0 and not self._stop.wait(remaining):
                    continue
        finally:
            # Capture release is owned by stop()/cleanup; worker must not race it.
            pass

    def _read_normalize_one(self, capture: CaptureDevice) -> RealtimeVisualFrame:
        try:
            ok, frame = capture.read()
        except Exception as error:
            raise CameraCaptureError(
                CAMERA_FAILED,
                error_type="camera_failed",
            ) from error
        if not ok:
            raise CameraCaptureError(CAMERA_FAILED, error_type="camera_failed")
        try:
            image = _bgr_frame_to_rgb_image(frame)
        finally:
            # Drop OpenCV frame reference as soon as conversion finishes.
            del frame
        captured_at = self._monotonic()
        sequence = self._sequence
        self._sequence += 1
        return normalize_camera_rgb_image(
            image,
            sequence=sequence,
            captured_at_monotonic=captured_at,
        )

    def _replace_latest(self, frame: RealtimeVisualFrame) -> None:
        with self._lock:
            self._latest = frame

    def _release_capture(self) -> bool:
        """Release the capture device. Return False when release fails."""
        capture = self._capture
        self._capture = None
        if capture is None:
            return True
        try:
            capture.release()
            return True
        except Exception as error:
            logger.error(
                "Camera release failed error_type=%s",
                type(error).__name__,
            )
            return False

    def _signal_fatal(self, error_type: str) -> None:
        self._stop.set()
        if self._on_fatal is not None:
            self._on_fatal(error_type)
