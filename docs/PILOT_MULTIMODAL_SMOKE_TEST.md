# Pilot multimodal smoke test

Manual-only M30 checklist for `/multimodal-realtime`.

Do **not** redesign Milestone 26. M26 retains its current separate FIFO/visual-ack correlation architecture. This checklist only records whether the existing path is usable for a demo.

Record observations only. Do **not** retain camera frames, audio, or transcripts beyond this checklist.

Do not include multimodal in a demo unless this smoke test was run immediately beforehand and passed.

## Preconditions

- Windows host with a usable camera, microphone, and speakers
- Voice smoke test already understood
- Isolated custom `CORTANA_DATA_DIR` recommended
- Startup readiness is not blocked

## Steps

1. Camera present check
   - Confirm the OS sees a camera before starting Cortana.
   - Notes:

2. `/multimodal-realtime` start
   - Start the multimodal session.
   - Notes:

3. Frame capture
   - Confirm a bounded live frame is captured for a turn.
   - Notes:

4. Visual question
   - Ask a question that depends on what the camera can see.
   - Notes:

5. Spoken response
   - Confirm a spoken reply is heard.
   - Notes:

6. Interruption
   - Barge in while a multimodal reply is playing.
   - Notes:

7. Visual ack timeout handling
   - Observe the existing visual-ack wait/timeout behavior if a provider ack is late or missing.
   - Notes:

8. Clean stop
   - Stop the multimodal session without a hang.
   - Notes:

9. Restart
   - Start `/multimodal-realtime` again in a new session.
   - Notes:

10. Camera unavailable fallback to `/voice-realtime`
    - With the camera unavailable, confirm fallback to `/voice-realtime` rather than a hang or crash.
    - Notes:

## Results

| Check | PASS | FAIL | OMIT | Notes |
| --- | --- | --- | --- | --- |
| Camera present |  |  |  |  |
| Session starts |  |  |  |  |
| Frame captured |  |  |  |  |
| Visual question answered |  |  |  |  |
| Spoken response heard |  |  |  |  |
| Interruption works |  |  |  |  |
| Visual ack timeout handled |  |  |  |  |
| Clean stop |  |  |  |  |
| Restart works |  |  |  |  |
| Camera-unavailable fallback |  |  |  |  |

Overall multimodal smoke: PASS / FAIL / OMIT FROM DEMO

Observer:

Date:
