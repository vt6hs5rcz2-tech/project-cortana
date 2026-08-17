"""Cheap local quality checks for realtime camera frames.

Rejects washout/warmup/blank frames and obvious camera-off/privacy
placeholder graphics before visual insertion. Does not perform object
recognition or any provider/network work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from src.camera_capture import RealtimeVisualFrame
from src.config import MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS

# Near-black / near-white washout. A mid-tone solid color (simple valid
# scene) can have low variance and must still be accepted.
_NEAR_BLACK_MEAN = 16.0
_NEAR_WHITE_MEAN = 240.0
_MIN_WASH_DYNAMIC_RANGE = 24
_STAT_THUMB_SIZE = (80, 80)
_TILE = 8
_PLACEHOLDER_UNIFORM_MIN = 0.82
_PLACEHOLDER_CHROMA_MAX = 8.0
_PLACEHOLDER_UNIQUE_COLORS_MAX = 40
_PLACEHOLDER_CENTER_RATIO_MIN = 1.6
# Spatial signature is the discriminator. Live photographic frames on this
# host sit far outside those bounds (unique colors hundreds, uniform ~0.12).
# Temporal MAE is only a bounded corroboration so icon animation / exposure
# flicker cannot keep an obvious placeholder usable.
_PLACEHOLDER_TEMPORAL_MAE_MAX = 12.0
_PLACEHOLDER_MAJORITY_MIN = 2
_PLACEHOLDER_MAJORITY_WINDOW = 3


@dataclass(frozen=True)
class VisualFrameQuality:
    """Deterministic local quality result for one camera frame."""

    usable: bool
    reason: str
    mean_luminance: float
    luminance_variance: float
    dynamic_range: int
    chroma_std: float = 0.0
    unique_colors: int = 0
    uniform_tile_fraction: float = 0.0
    center_energy_ratio: float = 0.0
    temporal_mae: float | None = None
    prior_frame_count: int = 0


@dataclass(frozen=True)
class _SpatialSignals:
    mean: float
    variance: float
    dynamic_range: int
    chroma_std: float
    unique_colors: int
    uniform_tile_fraction: float
    center_energy_ratio: float
    gray: tuple[int, ...]


def assess_visual_frame_quality(
    frame: RealtimeVisualFrame,
    *,
    now: float | None = None,
    max_age_seconds: float = MAX_REALTIME_VISUAL_FRAME_AGE_SECONDS,
    prior_frames: Sequence[RealtimeVisualFrame] | None = None,
) -> VisualFrameQuality:
    """Return whether ``frame`` is usable visual context for one turn."""
    if frame.width < 1 or frame.height < 1:
        return VisualFrameQuality(False, "malformed_dimensions", 0.0, 0.0, 0)
    if now is not None:
        age = now - frame.captured_at_monotonic
        if age < 0 or age > max_age_seconds:
            return VisualFrameQuality(False, "stale", 0.0, 0.0, 0)
    try:
        spatial = _spatial_signals(frame.image_bytes)
    except Exception:
        return VisualFrameQuality(False, "decode_failed", 0.0, 0.0, 0)
    prior_count = 0 if prior_frames is None else len(prior_frames)
    temporal_mae = _temporal_mae_against_latest(spatial, prior_frames)
    if (
        spatial.mean < _NEAR_BLACK_MEAN
        and spatial.dynamic_range < _MIN_WASH_DYNAMIC_RANGE
    ):
        return _quality(
            False,
            "near_black",
            spatial,
            temporal_mae=temporal_mae,
            prior_frame_count=prior_count,
        )
    if (
        spatial.mean > _NEAR_WHITE_MEAN
        and spatial.dynamic_range < _MIN_WASH_DYNAMIC_RANGE
    ):
        return _quality(
            False,
            "near_white",
            spatial,
            temporal_mae=temporal_mae,
            prior_frame_count=prior_count,
        )
    if _is_privacy_placeholder(spatial, prior_frames):
        return _quality(
            False,
            "privacy_placeholder",
            spatial,
            temporal_mae=temporal_mae,
            prior_frame_count=prior_count,
        )
    return _quality(
        True,
        "ok",
        spatial,
        temporal_mae=temporal_mae,
        prior_frame_count=prior_count,
    )


def _quality(
    usable: bool,
    reason: str,
    spatial: _SpatialSignals,
    *,
    temporal_mae: float | None,
    prior_frame_count: int,
) -> VisualFrameQuality:
    return VisualFrameQuality(
        usable,
        reason,
        spatial.mean,
        spatial.variance,
        spatial.dynamic_range,
        chroma_std=spatial.chroma_std,
        unique_colors=spatial.unique_colors,
        uniform_tile_fraction=spatial.uniform_tile_fraction,
        center_energy_ratio=spatial.center_energy_ratio,
        temporal_mae=temporal_mae,
        prior_frame_count=prior_frame_count,
    )


def _temporal_mae_against_latest(
    current: _SpatialSignals,
    prior_frames: Sequence[RealtimeVisualFrame] | None,
) -> float | None:
    if prior_frames is None or not prior_frames:
        return None
    try:
        prior = _spatial_signals(prior_frames[-1].image_bytes)
    except Exception:
        return None
    return _gray_mae(current.gray, prior.gray)


def _is_privacy_placeholder(
    current: _SpatialSignals,
    prior_frames: Sequence[RealtimeVisualFrame] | None,
) -> bool:
    """Require a spatial placeholder signature plus recent corroboration.

    Temporal MAE is a bounded check, not a single hard 1.5-grey requirement.
    Current plus at least one recent prior must look placeholder-like.
    """
    if prior_frames is None or not prior_frames:
        return False
    if not _spatial_placeholder(current):
        return False
    spatial_count = 1
    latest_mae: float | None = None
    window = prior_frames[-_PLACEHOLDER_MAJORITY_WINDOW:]
    for prior_frame in reversed(window):
        try:
            prior = _spatial_signals(prior_frame.image_bytes)
        except Exception:
            continue
        mae = _gray_mae(current.gray, prior.gray)
        if latest_mae is None:
            latest_mae = mae
        if _spatial_placeholder(prior):
            spatial_count += 1
    if spatial_count < _PLACEHOLDER_MAJORITY_MIN or latest_mae is None:
        return False
    return latest_mae <= _PLACEHOLDER_TEMPORAL_MAE_MAX


def _spatial_placeholder(spatial: _SpatialSignals) -> bool:
    return (
        spatial.uniform_tile_fraction >= _PLACEHOLDER_UNIFORM_MIN
        and spatial.chroma_std <= _PLACEHOLDER_CHROMA_MAX
        and spatial.unique_colors <= _PLACEHOLDER_UNIQUE_COLORS_MAX
        and spatial.center_energy_ratio >= _PLACEHOLDER_CENTER_RATIO_MIN
    )


def _spatial_signals(image_bytes: bytes) -> _SpatialSignals:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        thumb = rgb.copy()
        thumb.thumbnail(_STAT_THUMB_SIZE)
    pixels = list(thumb.getdata())
    if not pixels:
        return _SpatialSignals(0.0, 0.0, 0, 0.0, 0, 0.0, 0.0, ())
    gray_list = [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in pixels]
    mean = sum(gray_list) / len(gray_list)
    variance = sum((value - mean) ** 2 for value in gray_list) / len(gray_list)
    dynamic_range = max(gray_list) - min(gray_list)
    rg = [float(r - g) for r, g, b in pixels]
    bg = [float(b - g) for r, g, b in pixels]
    chroma = rg + bg
    chroma_mean = sum(chroma) / len(chroma)
    chroma_std = math.sqrt(
        sum((value - chroma_mean) ** 2 for value in chroma) / len(chroma)
    )
    unique_colors = len({(r // 8, g // 8, b // 8) for r, g, b in pixels})
    return _SpatialSignals(
        mean=mean,
        variance=variance,
        dynamic_range=dynamic_range,
        chroma_std=chroma_std,
        unique_colors=unique_colors,
        uniform_tile_fraction=_uniform_tile_fraction(gray_list, thumb.size),
        center_energy_ratio=_center_energy_ratio(gray_list, thumb.size),
        gray=tuple(gray_list),
    )


def _uniform_tile_fraction(gray: list[int], size: tuple[int, int]) -> float:
    width, height = size
    if width < _TILE or height < _TILE:
        return 0.0
    cols = width // _TILE
    rows = height // _TILE
    if cols < 1 or rows < 1:
        return 0.0
    uniform = 0
    total = 0
    for row in range(rows):
        for col in range(cols):
            cells: list[int] = []
            for y in range(_TILE):
                start = (row * _TILE + y) * width + col * _TILE
                cells.extend(gray[start : start + _TILE])
            if not cells:
                continue
            mean = sum(cells) / len(cells)
            variance = sum((value - mean) ** 2 for value in cells) / len(cells)
            total += 1
            if variance <= 18.0:
                uniform += 1
    return uniform / total if total else 0.0


def _center_energy_ratio(gray: list[int], size: tuple[int, int]) -> float:
    width, height = size
    if not gray:
        return 0.0
    global_var = _variance(gray)
    if global_var <= 1e-6:
        return 0.0
    x0, x1 = width // 3, (2 * width) // 3
    y0, y1 = height // 3, (2 * height) // 3
    center: list[int] = []
    for y in range(y0, y1):
        start = y * width + x0
        center.extend(gray[start : start + (x1 - x0)])
    if not center:
        return 0.0
    return _variance(center) / global_var


def _variance(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _gray_mae(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        return 999.0
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)
