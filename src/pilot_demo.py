"""Guarded preparation of an isolated sanitized pilot/demo data profile.

This is not a slash command. It never operates on the default real user
profile. Destructive reset requires an explicit custom CORTANA_DATA_DIR and
``--confirm-demo-reset``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    CORTANA_OWNED_STORE_DIRNAMES,
    CORTANA_OWNED_STORE_FILENAMES,
    PROJECT_ROOT,
    cortana_owned_store_paths,
    get_builtin_app_data_dir,
    get_configured_data_dir_override,
    is_custom_data_profile,
)

REFUSE_MISSING_OVERRIDE = "CORTANA_DATA_DIR is required for demo reset."
REFUSE_DEFAULT_APP_DATA = (
    "Demo reset refuses the default application data directory."
)
REFUSE_REPOSITORY_ROOT = "Demo reset refuses the repository root."
REFUSE_FILESYSTEM_ROOT = "Demo reset refuses the filesystem root."
REFUSE_HOME_ROOT = "Demo reset refuses the user home directory."
REFUSE_CONFIRMATION = (
    "Demo reset requires --confirm-demo-reset. No files were changed."
)


@dataclass(frozen=True)
class DemoResetResult:
    """Bounded result of a demo-reset plan or reset."""

    allowed: bool
    performed: bool
    reason: str
    store_names: tuple[str, ...]
    removed_names: tuple[str, ...] = ()


def list_demo_reset_store_names() -> tuple[str, ...]:
    """Return the known Cortana-owned store names a demo reset may clear."""
    return CORTANA_OWNED_STORE_FILENAMES + CORTANA_OWNED_STORE_DIRNAMES


def assess_demo_reset_target() -> DemoResetResult:
    """Return whether the current CORTANA_DATA_DIR may be reset."""
    names = list_demo_reset_store_names()
    if not is_custom_data_profile():
        return DemoResetResult(
            allowed=False,
            performed=False,
            reason=REFUSE_MISSING_OVERRIDE,
            store_names=names,
        )
    override = get_configured_data_dir_override()
    if override is None:
        return DemoResetResult(
            allowed=False,
            performed=False,
            reason=REFUSE_MISSING_OVERRIDE,
            store_names=names,
        )
    refusal = _forbidden_target_reason(override)
    if refusal is not None:
        return DemoResetResult(
            allowed=False,
            performed=False,
            reason=refusal,
            store_names=names,
        )
    return DemoResetResult(
        allowed=True,
        performed=False,
        reason="Demo reset is allowed for the custom data profile.",
        store_names=names,
    )


def prepare_pilot_demo(*, confirm: bool) -> DemoResetResult:
    """Plan or perform a known-store reset inside a custom data directory."""
    assessment = assess_demo_reset_target()
    if not assessment.allowed:
        return assessment
    if not confirm:
        return DemoResetResult(
            allowed=True,
            performed=False,
            reason=REFUSE_CONFIRMATION,
            store_names=assessment.store_names,
        )
    override = get_configured_data_dir_override()
    if override is None:
        return DemoResetResult(
            allowed=False,
            performed=False,
            reason=REFUSE_MISSING_OVERRIDE,
            store_names=assessment.store_names,
        )
    removed = _remove_known_stores(override)
    return DemoResetResult(
        allowed=True,
        performed=True,
        reason="Cleared known Cortana-owned demo stores.",
        store_names=assessment.store_names,
        removed_names=removed,
    )


def format_demo_reset_report(result: DemoResetResult) -> str:
    """Return path-free demo-reset output for the operator."""
    lines = [
        "Data profile: custom" if is_custom_data_profile() else "Data profile: default",
        f"Allowed: {'yes' if result.allowed else 'no'}",
        f"Performed: {'yes' if result.performed else 'no'}",
        result.reason,
        "Stores:",
    ]
    lines.extend(f"  {name}" for name in result.store_names)
    if result.removed_names:
        lines.append("Removed:")
        lines.extend(f"  {name}" for name in result.removed_names)
    return "\n".join(lines)


def _forbidden_target_reason(path: Path) -> str | None:
    resolved = path.resolve()
    if _is_filesystem_root(resolved):
        return REFUSE_FILESYSTEM_ROOT
    if resolved == Path.home().resolve():
        return REFUSE_HOME_ROOT
    if resolved == PROJECT_ROOT.resolve():
        return REFUSE_REPOSITORY_ROOT
    if resolved == get_builtin_app_data_dir().resolve():
        return REFUSE_DEFAULT_APP_DATA
    return None


def _is_filesystem_root(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == resolved.parent


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _remove_known_stores(root: Path) -> tuple[str, ...]:
    removed: list[str] = []
    for path in cortana_owned_store_paths(root):
        if not path.exists():
            continue
        if not _is_within(path, root):
            continue
        if path.is_symlink():
            path.unlink()
            removed.append(path.name)
            continue
        if path.is_file():
            path.unlink()
            removed.append(path.name)
            continue
        if path.is_dir() and path.name in CORTANA_OWNED_STORE_DIRNAMES:
            shutil.rmtree(path)
            removed.append(path.name)
    return tuple(removed)
