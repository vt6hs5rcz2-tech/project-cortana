# Pilot checklist

Human-readable M30 pilot checklist. Use a dedicated custom `CORTANA_DATA_DIR` so the real user profile is not touched.

Do not demonstrate a capability that failed its smoke test. See `docs/PILOT_VOICE_SMOKE_TEST.md`, `docs/PILOT_MULTIMODAL_SMOKE_TEST.md`, and `docs/DEMO_PLAN.md`.

## STARTUP

- [ ] Clean pilot profile: `CORTANA_DATA_DIR` points at an isolated directory
- [ ] Readiness is `READY` or `READY_WITH_OPTIONAL_FEATURES_UNAVAILABLE`, not blocked
- [ ] `/help` shows the short core command list
- [ ] `/about` shows the pilot product identity
- [ ] `/status` shows compact status
- [ ] `/diagnostics` shows Data profile `custom` and does not print the full data path

## TEXT

- [ ] Ordinary chat receives a reply
- [ ] A correction is understood in the same session
- [ ] `/remember` saves a memory and `/memories` lists it
- [ ] `/clear` resets conversation state without deleting saved memories

## KNOWLEDGE

- [ ] A sanitized local document can be added
- [ ] A grounded answer cites the added document
- [ ] A study session can start and ask a question
- [ ] Study end reports honestly and does not invent leftover state

## ASSISTANT

- [ ] A reminder can be created and listed
- [ ] Calendar is used only if already configured and validated

## VOICE

- [ ] `/voice-turn` completes one spoken turn
- [ ] `/voice-realtime` completes a spoken turn
- [ ] Barge-in works
- [ ] Fallback to `/voice-turn` works

## MULTIMODAL

- [ ] Run only if `docs/PILOT_MULTIMODAL_SMOKE_TEST.md` passed
- [ ] Otherwise mark multimodal as omitted from the pilot/demo

## SECURITY

- [ ] A denied tool is refused
- [ ] An approval-required tool asks before execution
- [ ] A workflow dry-run does not claim a live change

## RESTART

- [ ] Memories and reminders persist after restart
- [ ] Conversation state is reset after restart
- [ ] Study-state reporting is honest after restart

## SUPPORT

- [ ] `/diagnostics` snapshot is suitable for support (no secrets, no user text, no full data path)
- [ ] Version/build identity is `Cortana 1.0.0-pilot`
