# M30 release gate

M30 cannot close until every item below is satisfied. The live Realtime metadata check is a manual gate. It is not a startup network probe and must not be weakened to FIFO correlation.

## Required before M30 can close

### A. Automated

- Full pytest is green
- `python -m mypy --strict src` is green
- `git diff --check` is clean

### B. Live M25

- `python scripts/validate_realtime_metadata.py` reports PASS
- Client-supplied `cortana_user_item_id` and `cortana_generation` are echoed exactly on `response.created`
- FAIL (missing metadata, malformed metadata, different item id, different generation, or a connection/API error) keeps M30 release readiness BLOCKED

### C. Voice

- `docs/PILOT_VOICE_SMOKE_TEST.md` is PASS

### D. Multimodal

- `docs/PILOT_MULTIMODAL_SMOKE_TEST.md` is PASS if multimodal is included in the demo
- or multimodal is explicitly OMITTED from the demo

### E. Pilot profile

- Isolated custom `CORTANA_DATA_DIR` is verified
- Diagnostics reports `Data profile: custom` without printing the full path

### F. Demo

- Sanitized or simulated data only
- `scripts/prepare_pilot_demo.py` used only against the custom profile

### G. Defects

- No unresolved BLOCKER or HIGH pilot-visible defects discovered during manual smoke

## Not part of startup readiness

The live metadata gate is manual. Normal `/status` and startup readiness must not contact the Realtime API. `/diagnostics` reports the recorded release-gate result (`Realtime metadata gate: PASS`) and does not perform a live measurement at startup.
