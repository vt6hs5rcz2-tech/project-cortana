# Test matrix

In-repo provenance for the Pre-M30 validation tiers. Counts below were
recorded locally on this working tree after Batch 6 (workflow idempotency and
test-quality cleanup). They are **not** frozen historic numbers: adding
permanent tests changes them.

Do not treat this file as a runtime artifact. Nothing in `src/` reads it.

## What each suite contains

| Suite | What it is | Exact command |
| --- | --- | --- |
| Normal | Everyday tests. Excludes all `tests/test_pre_m30_*_adversarial.py` files and all `tests/test_pre_m30_deep_*.py` discovery files. | `python -m pytest --ignore-glob="tests/test_pre_m30_*_adversarial.py" --ignore-glob="tests/test_pre_m30_deep_*.py"` |
| Hardening | Permanent M25–M29 conversational/realtime hardening contracts in one file. | `python -m pytest tests/test_pre_m30_hardening_adversarial.py` |
| Bug Hunt #2 | Nine domain adversarial files. Does **not** include hardening. | See command below. |
| Bug Hunt #3 | First-run, end-to-end, restart/recovery, user-experience, and long-session adversarial files. | See command below. |
| Deep Audit #4 | Eight discovery files under `tests/test_pre_m30_deep_*.py`. | `python -m pytest tests/test_pre_m30_deep_*.py` |
| Targeted workflow/tool | Workflow executor/commands/repository/idempotency plus tool model, command, security, and kill-switch tests. | See command below. |
| Full tree | Entire `tests/` tree, including hardening, Bug Hunts, and Deep Audit. | `python -m pytest` |
| Types | Strict mypy on production sources only. | `python -m mypy --strict src` |

### Bug Hunt #2 command

```text
python -m pytest tests/test_pre_m30_foundation_adversarial.py tests/test_pre_m30_memory_adversarial.py tests/test_pre_m30_documents_adversarial.py tests/test_pre_m30_calendar_reminders_adversarial.py tests/test_pre_m30_study_adversarial.py tests/test_pre_m30_tools_workflows_adversarial.py tests/test_pre_m30_voice_vision_adversarial.py tests/test_pre_m30_cross_system_adversarial.py tests/test_pre_m30_startup_persistence_adversarial.py
```

### Bug Hunt #3 command

```text
python -m pytest tests/test_pre_m30_first_run_adversarial.py tests/test_pre_m30_end_to_end_adversarial.py tests/test_pre_m30_restart_recovery_adversarial.py tests/test_pre_m30_user_experience_adversarial.py tests/test_pre_m30_long_session_adversarial.py
```

### Targeted workflow/tool command

```text
python -m pytest tests/test_workflow_idempotency.py tests/test_workflow_executor.py tests/test_workflow_commands.py tests/test_workflow_json_repository.py tests/test_workflow_repository.py tests/test_workflow_models.py tests/test_workflow_registry.py tests/test_workflow_audit.py tests/test_workflow_incident_linkage.py tests/test_workflow_security_invariants.py tests/test_tool_capability_kill_switches.py tests/test_tool_commands.py tests/test_tool_models.py tests/test_tool_security_invariants.py
```

## Locally reproducible counts (Batch 6 working tree)

These were produced by the commands above on the same uncommitted tree as
Batches 1–6. They are locally reproducible. They are not CI artifacts and
they are not an external review package.

| Suite | Result | Reproducible here? |
| --- | --- | --- |
| Normal | 1228 passed, 6 skipped | Yes |
| Hardening | 79 passed | Yes |
| Bug Hunt #2 | 90 passed | Yes |
| Bug Hunt #3 | 68 passed | Yes |
| Targeted workflow/tool | 134 passed | Yes |
| `python -m mypy --strict src` | Success: no issues found in 116 source files | Yes |
| Deep Audit #4 | 95 passed | Yes |
| Full tree | 1560 passed, 6 skipped | Yes |

## Not reproduced / not claimed

- An outside review package previously mentioned a numeric matrix that was
  not shipped in this repository. This file replaces that missing provenance.
- The unverifiable claim “18 targeted tests × 20 iterations” is **not**
  reproduced here. No historical command or test list for that claim exists
  in this repo.
- A **new** locally captured timing bundle was run during the final
  re-freeze validation: 15 named M25/M26 tests × 20 iterations, 15 passed
  each time, 0 failures. Tests:
  `test_barge_in_aborts_and_rejects_stale_audio`,
  `test_hard_session_timeout_lifecycle`,
  `test_realtime_session_happy_path_and_cleanup`,
  `test_m25_barge_in_correction_is_interruption_context_only`,
  `test_stale_a_before_b_is_cancelled_and_b_remains_eligible`,
  `test_stale_a_after_b_is_rejected_once_b_has_output`,
  `test_repeated_barge_ins_do_not_permanently_silence`,
  `test_created_after_cleanup_is_tombstoned`,
  `test_m26_missing_transcript_falls_back_without_fabricating_plan`,
  `test_m26_interrupted_turn_before_timeout_does_not_create`,
  `test_rapid_double_turn_response_linkage_fifo`,
  `test_startup_order_camera_before_mic_and_banner`,
  `test_visual_ack_timeout_does_not_create_response`,
  `test_cleanup_releases_visual_memory_and_ack_state`,
  `test_hundred_non_ack_turns_are_hard_capped`.
- Stale packaged counts such as 1123 or 1360 are not current and are not
  repeated in `README.md`.

## Notes

- Historic counts change when permanent tests are added. Batch 6 added
  workflow idempotency tests and renamed existing test-quality cases; the
  normal suite rose from the pre-Batch-6 1212 passed / 6 skipped figure
  because those new tests are part of the everyday tree.
- Discovery files under `tests/test_pre_m30_deep_*.py` are allowed to change
  their assertions when a stronger accurate contract replaces a stale
  overstated name. That is a test-quality fix, not a product feature.
