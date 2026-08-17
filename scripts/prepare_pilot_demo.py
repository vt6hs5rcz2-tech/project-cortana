"""Prepare a clean sanitized pilot/demo profile.

This is not a slash command and must not appear in /help.

Requires an explicit custom CORTANA_DATA_DIR. Never operates on the default
real user profile. Without --confirm-demo-reset this prints the intended
stores and refuses destructive action.

Reset behavior:
- Clears known Cortana-owned store files and directories inside the custom
  data directory only: memories, documents, incidents, tool control,
  workflow runs, reminders, calendar local state, study state, evidence
  copies, and tool-process scratch.
- Does not disconnect Google accounts or delete OS keyring credentials.
- Does not delete source documents outside the demo data directory.
- Does not recursively delete unrelated files in the custom directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pilot_demo import format_demo_reset_report, prepare_pilot_demo


def main(argv: list[str] | None = None) -> int:
    """Run the guarded demo-reset CLI."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Prepare a sanitized Cortana demo profile."
    )
    parser.add_argument(
        "--confirm-demo-reset",
        action="store_true",
        help="Perform the known-store reset. Without this flag, describe only.",
    )
    args = parser.parse_args(argv)
    result = prepare_pilot_demo(confirm=args.confirm_demo_reset)
    print(format_demo_reset_report(result))
    if result.performed:
        return 0
    return 0 if result.allowed and not args.confirm_demo_reset else 1


if __name__ == "__main__":
    raise SystemExit(main())
