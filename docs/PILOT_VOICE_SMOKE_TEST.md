# Pilot voice smoke test

Manual M30 release-gate checklist for `/voice-turn` and `/voice-realtime`.

Record observations only. Do **not** retain audio recordings, transcripts beyond this checklist, or raw user speech.

Do not demonstrate voice in a demo if this smoke test fails.

## Preconditions

- Windows host with a usable microphone and speakers
- `OPENAI_API_KEY` configured
- Isolated custom `CORTANA_DATA_DIR` recommended
- Startup readiness is not blocked

## Steps

1. `/voice-turn`
   - Speak one short prompt and wait for the spoken reply.
   - Notes:

2. `/voice-realtime`
   - Start a realtime session.
   - Notes:

3. One normal realtime turn
   - Speak one complete utterance and hear the reply.
   - Notes:

4. Two sequential turns
   - After the first reply finishes, speak a second related prompt.
   - Notes:

5. Barge-in mid-response
   - Start speaking while Cortana is still talking.
   - Notes:

6. Rapid correction
   - Interrupt and immediately correct the previous request.
   - Notes:

7. Five-turn conversation
   - Complete five spoken turns in one realtime session.
   - Notes:

8. Interrupt while audio is playing
   - Cut off playback and confirm the session stays usable.
   - Notes:

9. Ctrl+C clean exit
   - Stop the process with Ctrl+C and confirm it exits without a hang.
   - Notes:

10. Restart `/voice-realtime`
    - Start the application again and open a new realtime session.
    - Notes:

11. Fallback to `/voice-turn`
    - Leave realtime and complete one `/voice-turn`.
    - Notes:

## Results

| Check | PASS | FAIL | Notes |
| --- | --- | --- | --- |
| Heard response |  |  |  |
| Correct response |  |  |  |
| No wrong-turn audio |  |  |  |
| Barge-in works |  |  |  |
| No freeze |  |  |  |
| Clean exit |  |  |  |
| Reconnect works |  |  |  |

Overall voice smoke: PASS / FAIL

Observer:

Date:
