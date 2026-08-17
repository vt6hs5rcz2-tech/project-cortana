# Demo plan

Use sanitized or simulated data only. Do not use real user memories, real documents, real reminders, or live personal calendar contents.

Do **not** demonstrate a capability that failed its smoke test.

Prepare the profile with `scripts/prepare_pilot_demo.py` only when `CORTANA_DATA_DIR` points at a dedicated custom directory. The utility never resets the default real user profile.

## Demo reset behavior

The preparation utility clears only known Cortana-owned stores inside the custom data directory:

- `memories.json`
- `documents.json`
- `incidents.json`
- `tool_control.json`
- `workflow_runs.json`
- `reminders.json`
- `calendar_control.json`
- `study_state.json`
- `evidence/`
- `tool_process_scratch/`

It does not disconnect Google accounts, delete OS keyring credentials, delete source documents outside the demo directory, or recursively delete unrelated files.

## Recommended flow

1. Startup and version identity
2. `/help`
3. Natural chat plus one correction
4. Memory save and recall
5. Sanitized document grounding
6. One study question
7. One reminder
8. Calendar only if already validated
9. Realtime voice
10. Barge-in
11. Multimodal only if smoke-tested immediately beforehand
12. Safe tool denial or workflow dry-run
13. `/clear`
14. Restart
15. Demonstrate expected persistence (memories/reminders) and conversation reset
16. Compact `/status` close

## Rules

- If voice smoke failed, skip steps 9–11 and do not claim realtime voice.
- If multimodal smoke failed or was not just run, omit step 11.
- If calendar was not validated, omit step 8.
- Keep all demo content fictional and non-sensitive.

## Private pilot launch

Use a dedicated custom data directory. Do not use the default user profile.

```text
$env:CORTANA_DATA_DIR = "$env:LOCALAPPDATA\CortanaPilotProfile"
python scripts/prepare_pilot_demo.py
python scripts/prepare_pilot_demo.py --confirm-demo-reset
python main.py
```

Confirm `/diagnostics` reports `Data profile: custom` and does not print the full path.

## Private demo sequence

1. Startup/readiness — `Cortana 1.0.0-pilot`. Expect Ready, or Ready with optional features unavailable when calendar is not connected. Voice and multimodal should still show available.
2. `/about`
3. `/diagnostics`
4. Natural conversation plus one correction. Prefer a bounded first prompt such as “In one sentence, what can you help with in this session?” Avoid an open “what can you do?” opener.
5. Memory save + recall
6. Document knowledge (`/ask-docs` on the sanitized tea guide)
7. Study Partner — one question
8. Reminder
9. `/voice-turn` — a short, easy phrase
10. `/voice-realtime`
11. `/multimodal-realtime` object recognition (remote, thermometer, or cup)
12. Multimodal non-visual relevance (a question that does not depend on the camera)
13. Stop / barge-in
14. Clean `/exit`

Do not demo calendar, tools, or other optional capability unless it was validated immediately beforehand.

## Multimodal conditions

- Confirm the camera shows a usable photographic frame before starting.
- ThinkShutter on this host is unreliable as an OpenCV software gate. Do not use the physical shutter as the demo of camera-unavailable fallback.
- Use one simple object against an uncluttered scene.

## Known non-blocking voice observations

- Occasional slight barge-in delay
- Recoverable PortAudio restart warning
- Possible speaker-to-mic false Heard turn
- Occasional STT miss

These are pilot observations, not hidden defects. Do not choose an unusually difficult STT phrase.

## Fallbacks

Do not retry indefinitely.

- Voice failure → return to text mode
- Multimodal camera issue → `/voice-realtime` or text
- STT miss → repeat the question once, naturally
- PortAudio recovered warning → continue if audible output is healthy
- Barge-in feels late → one retry, then continue
- Calendar/tools not validated → omit, do not improvise
