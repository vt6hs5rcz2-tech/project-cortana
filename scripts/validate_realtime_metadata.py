"""Manual M30 live Realtime metadata-echo release gate.

Do not run from pytest or application startup. This script never prints
OPENAI_API_KEY, never prints the full environment, and does not write
Cortana user stores.

PASS requires exact echo of cortana_user_item_id and cortana_generation on
response.created. FAIL leaves M30 release readiness BLOCKED. There is no
FIFO fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime_metadata_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
