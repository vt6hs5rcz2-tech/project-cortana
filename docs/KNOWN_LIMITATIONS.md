# Current accepted limitations

Companion to the “Current accepted limitations” section in `README.md`.
These are accepted product limits, not unfinished Batch 1–6 defects.

- **M25 `response.created` correlation:** `/voice-realtime` binds
  `response.created` through trusted client-generated metadata
  (`cortana_user_item_id`, `cortana_generation`). Missing, malformed, or
  unknown metadata fails closed: the response is tombstoned and is never
  FIFO-bound. Metadata echo is supported by the installed SDK/schema; live
  production API echo is not yet measured. If the provider omits metadata,
  Cortana rejects that response rather than guessing.
- **M26 response correlation:** `/multimodal-realtime` still uses
  client-created responses without metadata correlation and retains its own
  visual-ack/FIFO architecture. It did not gain the M25 explicit-correlation
  model.
- **M26 visual-ack correlation:** late or ambiguous visual acks are
  discarded. They are never written onto a stale turn and never bound to a
  newer turn.
- **Orphan visual-ack debt:** `_orphan_visual_ack_debt` is an intentionally
  uncapped skip counter. Capping it would risk binding a late ack to the
  wrong later turn.
- **Workflow side-effect retry (after Batch 6):** a persisted claim means
  the mutating step was already attempted. Reconstruction/retry does not
  re-execute that operation and does not assert that the prior side effect
  succeeded. A new execution requires `--new-operation` or a new
  `operation_id`. Oldest claims are dropped after
  `MAX_WORKFLOW_COMPLETED_EFFECT_KEYS` (200).
- **Provider-side delete:** Realtime `conversation.item.delete` and
  calendar/keyring revocation remain best-effort.
- **Extreme speech:** very long or adversarial spoken input is bounded and
  fail-open to the existing response path; it is not a complete
  abuse-resistant speech stack.
- **Duplicate memories/reminders:** explicit user commands may create
  another memory or reminder with the same text. Duplicate IDs in a memory
  file fail closed on load. There is no silent merge of same-text records.

DST nonexistent and ambiguous local wall times are rejected. That former
limitation is closed and is not listed above.
