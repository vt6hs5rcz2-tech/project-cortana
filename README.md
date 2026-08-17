# project-cortana

AI-powered authorized cybersecurity and defensive operations platform.

## Quickstart

Prerequisites:

- Python 3.13
- An OpenAI API key
- Optional: microphone, camera, and Google Calendar OAuth client file

Install and run:

```text
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then:

```text
python main.py
```

First commands:

- `/help` — core commands
- `/status` — compact session status
- `/about` — what Cortana is

Basic troubleshooting:

- Missing API key: add `OPENAI_API_KEY` to `.env`
- Invalid API key: the key was rejected; check the value in `.env`
- Voice unavailable: Windows plus a usable microphone are required
- Camera unavailable: Windows plus a usable camera are required for multimodal
- Camera image unusable: check the physical shutter / ThinkShutter, Lenovo camera privacy, the Windows Camera app, and Settings > Privacy & security > Camera > desktop app access
- Calendar unavailable: optional Google packages and an OAuth client file are required

Do not put secrets in chat, issues, or command output.

Optional isolated pilot/demo profile: set `CORTANA_DATA_DIR` to a dedicated empty directory. See `docs/PILOT_CHECKLIST.md`, `docs/PILOT_VOICE_SMOKE_TEST.md`, `docs/PILOT_MULTIMODAL_SMOKE_TEST.md`, and `docs/DEMO_PLAN.md`.

## Overview

Project Cortana is an early software milestone focused on:

- Centralized assistant identity
- Local slash-command handling
- Temporary in-session conversation history
- Explicit, user-controlled persistent memory
- Explicit, session-only active memory context for AI requests
- Local Knowledge Vault for explicit document ingestion and inspection
- Deterministic local lexical document retrieval
- Explicit source-grounded AI questions, document summaries, and two-document comparison over retrieved document passages
- Session-scoped Study Partner for grounded explanations, practice questions, graded answers, and progress over authorized documents
- Static visual understanding for authorized local images (describe, ask, and local validation/info)
- Explicit push-to-talk natural voice conversation for one spoken turn at a time
- Explicit realtime spoken conversation with barge-in/interruption
- Explicit realtime multimodal perception (voice + bounded live camera snapshots)
- Bounded conversational intelligence for continuity, follow-ups, corrections, response depth, and consistent style (no privileged authority)
- Realtime conversational planning that guides `/voice-realtime` and `/multimodal-realtime` from the same bounded session state without taking over response ownership
- Natural speech delivery and conversational timing so spoken replies sound less like text being read aloud
- Local human-controlled security event, incident, indicator, evidence, note, and timeline foundation
- Local human-supervised defensive tool framework with scope controls, dry-run planning, and approval
- Optional process-isolated execution for a tiny allowlisted defensive tool subset
- Optional Windows Job Object resource governance for process-isolated tools
- Optional process-isolated file tools (`file-sha256`, `compare-sha256`, `text-search`) using the Windows safe-open foundation
- Trusted defensive playbook orchestration over allowlisted Milestone 9 tools
- Durable workflow-run and workflow-audit history with optional authorized incident linkage
- Optional controlled security analyst assistance over sanitized single-incident packets
- Coordinated evidence↔incident linking, incident-scoped evidence text-search request creation, and AI-off bounded incident context display
- Deterministic ordinary-language routing for a small high-confidence set of existing capabilities (slash commands remain privileged)
- Local timezone-aware reminder scheduling foundation separate from persistent memory
- Google Calendar integration with OS credential storage and explicit prepare→confirm writes

## Conversation history, persistent memory, active context, and documents

| Kind | Lifetime | How it changes | Sent to AI? |
| --- | --- | --- | --- |
| Temporary conversation history | Current session only | Built from unmatched chat turns; cleared with `/clear`; slash commands and orchestrator-handled routes are excluded | Yes, as prior turns |
| Session conversational state (M27) | Current session only | Updated by the conversational-intelligence layer for continuity; cleared with `/clear`; bounded; never permanent memory | Yes, as separated developer metadata only |
| Speech delivery state (M29) | Current session only | Tracks recent spoken openings, acknowledgments, chunk fingerprints, and interrupted-response fingerprints for pacing/repetition control; cleared with `/clear`; never stores audio | No; used only to shape a spoken delivery copy or advisory pacing instructions |
| Explicit persistent memory | Survives restarts | Saved only through local `/remember` or explicit NL `remember …` routing | No, unless activated |
| Active memory context | Current session only | Selected with `/recall`; cleared with `/clear`, `/release`, `/release-all`, or restart. Persistent memories stay stored. | Yes, only while active |
| Knowledge Vault documents | Survives restarts | Ingested only through local `/add-document` | No by default |
| Retrieved document passages | Current request/session only | Selected by `/search-docs` (local), explicit NL document-search routing (lexical only), or grounded AI commands (`/ask-docs`, `/doc-summary`, `/docs-compare`) | Only selected chunks through grounded document AI commands |
| Security incidents and evidence | Survives restarts | Created only through explicit local Milestone 8 commands | Never in this milestone |
| Defensive tool scopes, requests, approvals, results, and audits | Survives restarts | Created only through explicit local Milestone 9 commands | Never in this milestone |
| Workflow/playbook run state and workflow audits | Survives restarts when persistence is enabled; otherwise current process only | Created only through explicit local Milestone 10/11 commands | Never in this milestone |
| Incident AI analysis preparations and results | Current process only | Created only through explicit local Milestone 12 analysis commands | Only through `/incident-analysis-run`, as a sanitized allowlisted packet |
| Reminders | Survives restarts | Created only through explicit local Milestone 19 reminder commands | Never in this milestone |
| Calendar account metadata, proposals, and calendar audits | Survives restarts | Connected/changed only through explicit local Milestone 20 calendar commands | Never in this milestone |
| Study Partner sessions, questions, attempts, and chunk stats | Survives restarts | Created only through explicit local Milestone 22 `/study-*` commands | Only selected study chunks / short-answer grading inputs through study AI commands |
| Visual analysis images | Ephemeral for one command only | Loaded only through explicit local Milestone 23 `/vision-*` commands | Only a normalized in-memory PNG plus the explicit visual task through `/vision-describe` or `/vision-ask` |
| Voice microphone/TTS audio | Ephemeral for one `/voice-turn` only | Captured/synthesized only through explicit local Milestone 24 `/voice-turn` | Bounded mic WAV for transcription; ordinary chat text for the reply; reply text for TTS. Raw audio is never stored |
| Realtime voice audio | Ephemeral for one `/voice-realtime` session only | Streamed only while an explicit Milestone 25 realtime session is active | Bounded PCM frames to the OpenAI Realtime API; finalized transcripts only may enter local conversation history. Raw audio is never stored |
| Realtime multimodal camera frames | Ephemeral for one `/multimodal-realtime` session only | Captured only while an explicit Milestone 26 multimodal session is active; latest-frame buffer capacity 1; no local archive | At most one current normalized PNG may be sent to the OpenAI Realtime API per user turn. Finalized transcripts only may enter local conversation history. Cortana deletes superseded visual items from the active Realtime conversation context but does not claim provider-side permanent deletion |

Persistent memories, Knowledge Vault documents, incident records, evidence copies, and Study Partner state are stored locally by Project Cortana in user-local application data locations. They are not placed in Git-tracked source directories. Visual analysis does not persist image bytes. Voice capture, TTS audio, and multimodal camera frames are ephemeral and are not archived.

Saved memories remain inactive by default. Nothing from persistent memory is sent to the AI model unless the user explicitly activates it with `/recall` for the current session.

Documents remain inactive by default. No document text is sent to the AI unless the user explicitly invokes `/ask-docs`, `/doc-summary`, or `/docs-compare`. Ordinary conversation never reads the Knowledge Vault.

Security incident records are never injected into ordinary chat, active-memory requests, or `/ask-docs`. When Milestone 12 analysis is explicitly enabled and the user runs `/incident-analysis-run`, only a sanitized allowlisted single-incident packet is sent. Evidence bytes, chain-of-custody records, and raw structured tool data remain structurally excluded from that packet.

Defensive tool definitions, scope notes, request justifications, parameters, execution results, and audit details are never sent to the AI in this milestone. The AI cannot select or execute tools.

Active memory selections:

- Exist only in memory for the current session
- Reset when the application closes
- Are not written to disk
- Are injected as structured, untrusted user-provided reference data
- Use session-specific reference boundaries that are not persisted
- Cannot replace Cortana’s system identity instructions
- Remain structurally separate from retrieved document context

## Knowledge Vault

The Knowledge Vault stores metadata and extracted text from documents that the user explicitly ingests.

Supported file types in this milestone:

- `.txt`
- `.md`
- `.pdf`

Behavior and limits:

- Storage is local and outside the source repository.
- Original binary document files are not copied into the vault.
- Extracted text and document metadata are stored locally as JSON.
- Documents are not sent to the AI model unless selected chunks are explicitly requested through `/ask-docs`, `/doc-summary`, or `/docs-compare`.
- OCR is not implemented.
- Embeddings, vector databases, cloud file search, semantic search, and autonomous retrieval are not implemented.
- If the vault file is missing, Cortana starts with an empty document list.
- If the vault file is empty, malformed, or structurally invalid, Cortana returns a clear local error, leaves the file unchanged for inspection, and does not attempt automatic repair in this milestone.
- Atomic writes protect against partial or corrupted JSON during a single save.
- Atomic writes do not coordinate multiple processes. If more than one Cortana instance uses the same document-vault file, updates may overwrite each other (last-writer-wins lost updates). One application instance should access a vault file at a time. Cross-process locking is not implemented yet.

Centralized limits:

- Maximum source file size: 10 MB
- Maximum extracted text length: 500,000 characters per document
- Maximum stored documents: 100

## Local document retrieval

Project Cortana uses deterministic local lexical retrieval over documents already stored in the Knowledge Vault.

Behavior:

- Tokenizes and normalizes query terms locally
- Ranks document chunks with transparent lexical scoring (term frequency, multi-term preference, optional phrase bonus)
- Uses deterministic tie-breaking
- Does not claim semantic similarity
- Does not use embeddings
- Does not make network or AI calls during `/search-docs`
- Returns no results when there is no meaningful match

Retrieved-context limits:

- Target chunk size: 1,200 characters
- Chunk overlap: 150 characters
- Maximum retrieved chunks per grounded request: 8
- Maximum retrieved-context characters per grounded request: 12,000
- Maximum grounded answer characters: 4,000
- Maximum summary map stages: 8
- Maximum summary output characters: 4,000
- Maximum compare documents: 2
- Maximum compare chunks per document: 4
- Maximum compare context characters: 12,000

Lexical retrieval matches words and phrases present in stored text. Semantic retrieval would attempt meaning-based similarity; it is disabled in this milestone.

Feature flags:

- `LOCAL_DOCUMENT_RETRIEVAL_ENABLED` gates `/search-docs`, M18 lexical document-search routing, and grounded operations that depend on local retrieval
- `DOCUMENT_CONTEXT_INJECTION_ENABLED` gates `/ask-docs`, `/doc-summary`, `/docs-compare`, and Study Partner AI operations
- `SEMANTIC_RETRIEVAL_ENABLED` remains false and unimplemented
- `STUDY_PARTNER_ENABLED` gates all `/study-*` commands
- `VISION_ANALYSIS_ENABLED` gates all `/vision-*` commands
- `VOICE_INTERACTION_ENABLED` gates `/voice-turn`, `/voice-status`, and parent-gates realtime voice/multimodal
- `REALTIME_VOICE_ENABLED` gates `/voice-realtime` (requires the parent voice gate too)
- `REALTIME_MULTIMODAL_ENABLED` gates `/multimodal-realtime` (requires voice, realtime, and vision gates too)

## Source-grounded answers

Milestone 21 routes grounded document AI through `DocumentKnowledgeService` and a dedicated AI path that excludes ordinary conversation history and active memory.

Commands that send selected document text to the AI:

- `/ask-docs <question>` — vault-wide lexical retrieval, then grounded Q&A
- `/doc-summary <document-id>` — sequential chunk map/reduce summary of one authorized document
- `/docs-compare <doc-id> | <doc-id> | <question>` — scoped retrieval inside exactly two documents

Grounded model output must be one JSON object:

```json
{"answer":"...","support":"supported|partial|unsupported","citations":["[DOC-1:C1]"]}
```

`support="supported"` means the model declared the answer source-supported and every citation label it used was present in the supplied source packet after local structural validation. It does **not** mean Cortana independently proved that every sentence is logically entailed by the cited source text. M21 does not implement an NLI/entailment engine.

## Study Partner / Tutor

Milestone 22 adds a session-scoped Study Partner on top of Milestone 21 grounded document intelligence.

M21 answers “what do these documents say?”
M22 helps the user learn and practice what authorized study documents say.

Commands:

- `/study-start <doc-id>[,<doc-id>...]` — start one active session over 1–5 explicit Knowledge Vault documents
- `/study-status` — show active session details and pending prompt preview (never the answer key)
- `/study-explain <topic>` — scoped grounded explanation via `DocumentKnowledgeService.ask_within_documents`
- `/study-question <mcq|short|-> | <topic-or->` — generate one grounded practice question
- `/study-answer <answer>` — grade the pending question
- `/study-progress` — honest counts/accuracy/weak source refs
- `/study-end` — complete the active session (allowed even with a pending question)

Study behavior:

- Exactly one active study session globally
- Study state persists in local `study_state.json` (sessions, questions, attempts, chunk stats)
- Answer keys persist for restart safety but are never shown by `/study-question`, `/study-status`, `/status`, logs, or ordinary history before grading
- MCQ grading is deterministic (`A`/`a`/`A)`/`A.` only); no AI call
- Short-answer grading is AI-assisted evaluation over rehydrated source evidence; it is not objective proof
- Valid study citations do **not** prove semantic entailment of questions or answer keys
- Adaptation prefers weak primary source chunks, then unattempted chunks in a deterministic ring, then least-recently-correct chunks
- Only each question’s `primary_source_ref` updates chunk stats
- Study operations do not write conversation history, active memory, persistent memory, reminders, or calendar events
- Guidance phrases `help me study` and `quiz me` point to slash commands only; they never execute study actions

## Static visual understanding

Milestone 23 adds safe static visual understanding for one authorized local image at a time.

Supported source types:

- `.png`
- `.jpg` / `.jpeg`
- `.webp`

Behavior and limits:

- Windows local drive-letter paths only, using the existing hardened safe-open / identity-verification boundary
- Extension allowlist plus Pillow-detected format agreement; mismatched containers are rejected
- Hard source dimension/pixel bounds are enforced before full decode
- Animated / multi-frame images are rejected
- Images are oriented with EXIF Orientation locally, then re-encoded to a metadata-free PNG in memory
- Raw and normalized image bytes are ephemeral; no vision repository, Document Vault image ingestion, or Study Partner image integration
- `/vision-describe` and `/vision-ask` use a dedicated multimodal AI path that excludes ordinary conversation history and active memory
- `/vision-info` runs the same validation/normalization pipeline with zero AI calls
- Visible text, URLs, and QR codes remain untrusted data and never acquire operational authority
- OCR engines, face recognition, and biometrics are not implemented
- Live camera perception is a separate explicit Milestone 26 mode (`/multimodal-realtime`), not part of `/vision-*`

Visual model output must be one JSON object:

```json
{"answer":"...","visibility":"observed|mixed|undetermined","warning":null}
```

`visibility` is the model's self-reported characterization of its visual basis after local structural validation. It does **not** mean Cortana independently verified the claim against image pixels. No numeric confidence score is used.

Centralized limits:

- Maximum source file size: 10 MB
- Maximum width/height: 4,096
- Maximum source pixels: 16,777,216
- Maximum normalized PNG size: 5 MB
- Maximum question characters: 2,000
- Maximum output characters: 4,000

Feature flag:

- `VISION_ANALYSIS_ENABLED` gates `/vision-describe`, `/vision-ask`, and `/vision-info`

## Natural voice conversation

Milestone 24 adds explicit push-to-talk natural voice conversation for one spoken utterance at a time.

Commands:

- `/voice-turn` — record one bounded utterance, transcribe it, generate an ordinary Cortana chat reply, synthesize speech, and play it synchronously
- `/voice-status` — show safe voice configuration without opening the microphone

Behavior and limits:

- Microphone capture requires an explicit `/voice-turn`; startup, construction, import, and `/voice-status` never open the mic
- Recording stops when Enter is pressed or when the 30-second maximum is reached
- Enter-to-stop uses non-blocking Windows console polling on the capture thread (no background `input()` waiter)
- While "Listening..." is shown, pressing Enter ends recording; avoid typing the next command until recording has stopped
- Anything typed while "Listening..." is shown is discarded and is not queued for the next command
- Canonical audio is in-memory PCM WAV, 16-bit mono 16 kHz; no temp files and no audio archive
- Speech-to-text uses OpenAI transcriptions (`CORTANA_TRANSCRIPTION_MODEL`, default `gpt-4o-mini-transcribe`) with provider language auto-detection
- Text-to-speech uses OpenAI speech (`CORTANA_TTS_MODEL` / `CORTANA_TTS_VOICE`, defaults `gpt-4o-mini-tts` / `coral`) and returns WAV for Windows `winsound` playback
- Milestone 29 synthesizes a speech-only delivery copy of the canonical reply: markdown/lists/URLs are normalized and chunked locally, then spoken in order. History still stores the original assistant text
- The text response is authoritative: TTS or playback failure still keeps the visible/history-committed conversational turn
- Voice is a transport into ordinary conversation (`generate_response` + history + active memory)
- Spoken transcripts intentionally have less operational authority than typed input in M24: they do not execute slash commands and do not enter Milestone 18 natural-language routing
- M24 is sequential push-to-talk. No wake word and no always-on listening. For interruptible realtime conversation, use Milestone 25 `/voice-realtime`.

Centralized limits:

- Maximum utterance duration: 30 seconds
- Sample rate: 16,000 Hz mono 16-bit
- Maximum PCM bytes: 960,000 (`16000 * 1 * 2 * 30`)
- Maximum WAV bytes: 960,044 (PCM + stable 44-byte stdlib WAV header)
- Minimum utterance: 250 ms
- Maximum transcript characters: 4,000
- Maximum TTS characters: 4,096 (no silent truncation)

Feature flag:

- `VOICE_INTERACTION_ENABLED` gates `/voice-turn` and `/voice-status`

## Realtime voice conversation

Milestone 25 adds an explicitly activated realtime spoken conversation session with genuine barge-in.

Commands:

- `/voice-realtime` — start one bounded realtime session; ordinary CLI input is paused until Ctrl+C or session end
- `/voice-status` — also reports realtime gate/model/voice configuration (never opens the microphone)
- `/voice-turn` remains the simple push-to-talk fallback

Behavior and limits:

- Realtime never starts automatically; `/voice-realtime` is required
- Microphone opens only after a successful Realtime API connect and session configuration
- Audio is raw PCM 16-bit mono 24 kHz in 20 ms frames (960 bytes); no WAV wrapping and no disk archive
- Uses OpenAI synchronous `client.realtime.connect(..., max_retries=0)` — no automatic reconnect
- Server VAD with `create_response=false` and `interrupt_response=true`; no local VAD; no per-utterance Enter-to-stop. Cortana issues exactly one client `response.create` per valid committed user turn, with trusted local correlation metadata
- Assistant audio plays through `sounddevice.RawOutputStream`; barge-in uses `abort()` for immediate local silence
- Input transcription (`gpt-4o-mini-transcribe` by default) is guidance of input audio content rather than precisely what the realtime model heard
- Local `ConversationHistory` remains canonical; only finalized transcripts are committed. Interrupted assistant text is not committed
- Milestone 28 plans each finalized transcript locally and refreshes next-turn session instructions after a completed pair. M25 does not wait for that transcript before `response.create`; M28 still cannot inject pre-response guidance into the current M25 utterance
- Milestone 29 adds advisory spoken-delivery guidance (pacing, acknowledgments, interruption recovery) to those same next-turn instructions. Provider realtime audio remains provider-owned; Cortana does not re-chunk or re-synthesize M25 audio locally
- Spoken realtime content remains conversational only: no slash-command, M18, calendar/reminder, tool/workflow, or memory-write authority
- Realtime function/tool calling is disabled (`tools=[]`, `tool_choice=none`)
- Hard session cap: 20 minutes
- On failure/unavailable: report clearly; do not silently fall back to `/voice-turn`

Feature flags:

- `VOICE_INTERACTION_ENABLED` parent gate
- `REALTIME_VOICE_ENABLED` realtime subset gate (both must be true)

Settings:

- `CORTANA_REALTIME_MODEL` (default `gpt-realtime-mini`)
- `CORTANA_REALTIME_VOICE` (default `coral`)

## Realtime multimodal perception

Milestone 26 adds an explicitly activated realtime multimodal session that combines Milestone 25 voice with bounded live camera snapshots.

Commands:

- `/multimodal-realtime` — start one bounded voice+camera session; ordinary CLI input is paused until Ctrl+C or session end
- `/voice-status` — also reports multimodal gate/camera sampling configuration (never opens the camera or microphone)
- `/voice-realtime` remains the voice-only realtime mode
- `/vision-*` remains static local-file vision

Behavior and limits:

- Multimodal never starts automatically; `/multimodal-realtime` is required
- Camera opens only after Realtime connect + session configuration, and only after one valid normalized frame is captured
- Microphone opens only after that first valid camera frame
- Local camera capture is approximately 2 FPS into a latest-frame buffer of capacity 1
- `CameraCaptureSession.stop()` clears the latest frame and resets session-local sequence and consecutive-failure counters so a later start behaves like a fresh capture session
- No local frame archive, video buffer, or disk-backed camera images
- Provider disclosure is at most one current normalized PNG (`detail=low`) per user turn; no provider upload during silence
- Visual frames are bound at `speech_stopped` / `input_audio_buffer.committed` using provider item IDs
- Session uses server VAD with `create_response=false` and `interrupt_response=true`, then bare `response.create()` after optional visual item insertion and the matching visual-item ack (or the visual-ack timeout). When the finalized transcript arrives in time, Milestone 28 injects pre-response planning first, and Milestone 29 may include spoken-delivery guidance in that same `session.update`. If the transcript does not arrive in time, the same `response.create()` still runs once without transcript-derived M28/M29 guidance.
- In-session visual turns are hard-capped (`_MAX_VISUAL_TURNS = 16`). Unacknowledged visual items expire after `REALTIME_MULTIMODAL_VISUAL_ACK_WAIT_SECONDS` (default 8s). Local frame bytes are released on expiry, eviction, stale, or delete. Late or ambiguous visual acks are discarded and are never bound to a newer turn. Dropping an old visual turn is preferred over retaining unbounded frames or guessing correlation.
- Visual-ack timeout does not create another assistant response and does not rewrite `ConversationState` from stale visual context.
- Superseded/completed/cancelled/expired visual conversation items are removed via `conversation.item.delete` from the active Realtime conversation context when a remote id is known. Provider-side deletion is best-effort only.
- Provider-side retention is governed by provider/service policy; Cortana does not claim provider-side permanent deletion
- Barge-in remains immediate for local playback; vision work never delays playback abort
- Local `ConversationHistory` remains text-only finalized transcripts
- Voice+vision remains conversational only: no slash-command, M18, calendar/reminder, tool/workflow, memory-write, face recognition, lip-reading, or surveillance authority
- Visible text/QR/URLs in camera frames are untrusted visual content only
- Hard session cap: 20 minutes (same as realtime voice)
- Camera startup failure fails the multimodal session before microphone open and suggests `/voice-realtime`

Centralized limits:

- Max visual resolution: 1280×720
- Max in-session visual turns: 16 (hard cap; oldest expired if all are still in flight)
- Visual-ack wait: 8 seconds (clamped 0.25–30)
- Max frame age: 3 seconds (monotonic)
- Fresh-frame wait at turn binding: 0.75 seconds
- Consecutive frame-failure threshold: 3
- Image detail: `low`

Feature flags (all required):

- `VOICE_INTERACTION_ENABLED`
- `REALTIME_VOICE_ENABLED`
- `VISION_ANALYSIS_ENABLED`
- `REALTIME_MULTIMODAL_ENABLED`

Dependency:

- `opencv-python-headless` (with NumPy) is confined to `src/camera_capture.py`

## Conversational intelligence

Milestone 27 adds a bounded conversational-intelligence layer so Cortana behaves more like one continuous conversational partner rather than isolated replies. It shares one small, session-scoped `ConversationState` (topic, active goal, unresolved question, recent referents, latest correction, waiting-for-user, offered options, recent acknowledgment/restatement tracking, optional M26 visual-context reference id) across every interaction mode, and it never authorizes privileged actions.

What it improves:

- Conversational continuity via small session-scoped state
- Follow-up resolution for short replies such as yes/no, “the second one”, “that one”, “tomorrow”, “continue”, when recent state provides enough evidence
- Conversational repair/correction handling (“I meant Tuesday”, “the other one”, “go back”, “that’s not what I asked”, “forget that”)
- Deterministic response-depth selection: `brief`, `normal`, or `detailed`
- Repetition control for unnecessary acknowledgments/restatements, without suppressing required safety/policy notices
- Lightweight acknowledgment policy that prefers starting the answer directly
- Higher-level turn-taking interpretation (continuation, correction, interruption, complete request, incomplete thought, topic change)
- Centralized conversational style/personality policy shared by text and voice paths
- Multimodal conversational references (“what is that?”) resolved only against an authorized current visual-context reference

Behavior differs by mode, and this difference is intentional, not accidental:

| Mode | Follow-up/correction/depth/repetition/topic-change interpretation | When it runs | State shared |
| --- | --- | --- | --- |
| Text chat | Full, per-turn | Before each reply is generated, so guidance can shape that reply | Read and written every turn |
| `/voice-turn` (M24) | Full, per-turn | Before each reply is generated, so guidance can shape that reply | Read and written every turn |
| `/voice-realtime` (M25) | Bounded local planning on each **finalized** user transcript; next-turn session instruction refresh after a completed pair | True per-utterance pre-response injection is still unavailable: M25 sends `response.create` immediately after a trusted committed item id, without waiting for the transcript. M28 still plans the finalized transcript and refreshes the next turn. | Observed on finalized transcripts; next-turn context refreshed after completed pairs |
| `/multimodal-realtime` (M26) | Full bounded planning after the finalized user transcript and **before** the existing manual `response.create()`, with a bounded fallback if that transcript does not arrive in time | After visual-frame bind at `speech_stopped`/`committed`; plans when the transcript is final, or falls back to `response.create` without transcript-derived M28 guidance when the wait expires | Planned and injected immediately before the single existing `response.create()` when the transcript arrives in time |

For `/voice-realtime`, Milestone 28 plans each finalized user transcript locally so `ConversationState` stays current (follow-ups, corrections, depth, topic/goal) even while the current response is already in flight. Compact next-turn guidance is then applied through the existing `session.update` instruction field after a completed pair. M25 response ownership is a separate transport concern: one client `response.create` per committed user turn, correlated by trusted metadata. M28 does **not** wait for the transcript, change VAD, or change barge-in/`abort()` ownership.

For `/multimodal-realtime`, Milestone 28 computes the same bounded plan after the finalized user transcript and injects compact advisory guidance through `session.update` immediately before the existing manual `response.create()` — still exactly once per turn, still after optional visual-item insertion, still with `create_response=false`. If the transcript does not arrive within the bounded wait, that same `response.create` still runs once without fabricating transcript-derived guidance.

No mode ever generates or triggers a second/duplicate assistant response as a result of this layer.

Security boundaries:

- Conversational inference clarifies meaning and observes finalized turns only; it does not authorize tools, workflows, calendar/reminder writes, memory writes, incident operations, document writes, or confirmation bypass
- Conversational “forget that” discards local interpretation only and never deletes persistent memory
- Injected conversational metadata is developer/internal context, structurally separated from the user-authored message; it may contain short excerpts of the user's own prior words for continuity, but never carries elevated authority and cannot authorize privileged actions regardless of its content
- Visual references remain non-authoritative and do not create competing responses or privileged actions
- If resolution is uncertain, original user text is preserved; if the intelligence layer fails, ordinary conversation continues

Limits:

- Session-scoped and bounded; not permanent memory
- No raw camera/audio storage
- Does not rebuild M25 VAD/barge-in transport, response ownership, or M26's `create_response=False` behavior
- `/voice-realtime` still cannot inject per-utterance guidance into the in-flight auto-response; see Realtime conversational planning
- No wake word, speaker ID, emotion detection, voice cloning, browsing, Bluetooth, translation, lip-reading, autonomous actions, or secretary workflows

## Realtime conversational planning

Milestone 28 closes the Milestone 27 gap between text/`/voice-turn` (which already interpret a user turn before generating a reply) and the realtime sessions (which previously only observed a turn after it completed).

What it does:

- Builds a compact immutable `RealtimeConversationPlan` from existing Milestone 27 `ConversationIntelligence` and `ConversationState` — no second heuristic engine
- Runs only on **finalized** user transcripts; partial/delta fragments never update planning or state
- Resolves follow-ups such as yes/no, “the first/second one”, “the other one”, “continue”, “go on”, “Tuesday”, “what you said before”, and “tell me more about that” when bounded state has enough evidence
- Treats corrections such as “I meant Tuesday”, “no, the other one”, “that’s not what I asked”, “go back”, “forget that”, and “change that” as conversational repair only
- Selects adaptive response depth (`brief` / `normal` / `detailed`) as guidance, without crude hard token limits
- Marks clarification needed when a referent is unresolved instead of guessing
- Reduces repetitive acknowledgments/restatements using recent-assistant state, without suppressing safety notices, authorization prompts, confirmations, uncertainty, or errors
- Reuses the Milestone 27 centralized style policy; planning may add concise/direct/explanation/clarification hints only
- Preserves cross-modal continuity on the same session `ConversationState` (text → `/voice-turn` → `/voice-realtime` → `/multimodal-realtime` → text)

What text and `/voice-turn` already do:

- Full per-turn interpretation **before** the ordinary chat/TTS reply is generated
- Developer-context injection of conversational metadata on that same response path

What `/voice-realtime` can safely do:

- Local planning as soon as a finalized transcript is available
- Next-turn `session.update` instruction refresh after a completed user/assistant pair
- Interruption context: a barge-in utterance such as “No, I meant Tuesday” is planned as interruption + correction against current state
- It cannot inject per-utterance guidance into the **current** auto-response without racing the provider or changing response ownership

What `/multimodal-realtime` can safely do:

- Bind the current camera frame at `speech_stopped` / `committed` as before
- Wait for the finalized user transcript, with a bounded deadline
- If the transcript arrives in time: compute the bounded plan (including authorized current visual referent when live), inject compact internal guidance via `session.update`, then call the same existing `response.create()` exactly once
- If the transcript does not arrive before the deadline: fall back to the same `response.create()` once **without** fabricating transcript text or transcript-derived M28 guidance for that turn
- Visual context still cannot independently trigger a response or authorize actions

M25 true pre-response guidance:

- **Still not available.** M25 now owns `response.create` at commit time so responses can carry trusted correlation metadata, but it does not wait for the finalized transcript. M28 therefore still cannot inject current-utterance guidance into the in-flight M25 response.

Latency:

- Planning is deterministic, in-process, and uses only bounded session state
- No extra model/network/database/tool calls
- Failures degrade to the existing response path; they do not crash the session, create a second response, or grant authority
- `/multimodal-realtime` waits for a finalized transcript so the current turn can receive the same M28 planning used by text and `/voice-turn`
- That wait is bounded by `REALTIME_MULTIMODAL_TRANSCRIPT_WAIT_SECONDS` (default 2.5s, clamped to 0.25–8.0s). This is a conservative allowance for provider transcription latency, not a measured end-to-end number
- If the transcript does not arrive within that deadline, Cortana falls back to response generation without transcript-derived M28 guidance for that turn. This prevents indefinite silence
- Timeout fallback uses the same existing `response.create` path. It does not create a duplicate response
- A transcript that arrives after fallback cannot trigger a second `response.create`, cannot attach a stale plan to the already-started response, and is not interpreted into `ConversationState` if that would overwrite a newer turn

Single-response ownership:

- `/voice-realtime` uses `create_response=false` and issues exactly one client `response.create` per valid committed user turn, with trusted local correlation metadata
- `/multimodal-realtime` keeps `create_response=false` and its existing unlabeled manual `response.create` as the sole owner. M26 did not gain M25 metadata correlation and still uses its own visual-ack/FIFO architecture
- Planning **guides** those paths; it does not own them

Authority boundaries:

- `RealtimeConversationPlan.authorizes_privileged_action` is structurally false
- Planning never executes or authorizes tools, workflows, calendar/reminder writes, memory writes, incident/document operations, confirmations, payments, or other side effects
- “Forget that” remains conversational only
- User-derived text inside planning metadata is advisory session context, not a trusted instruction

Known limitations:

- `/voice-realtime` cannot shape the in-flight auto-response for the utterance that was just transcribed
- `/multimodal-realtime` waits for the provider transcript before `response.create` when the transcript arrives in time, which adds transcript latency in exchange for true pre-response guidance
- If a multimodal transcript does not arrive within the bounded deadline, that turn still gets exactly one `response.create` without transcript-derived M28 guidance. The late transcript is kept for history pairing when it eventually arrives, but it is not used to plan or rewrite that already-started response, and it is not applied to `ConversationState` when a newer turn is already active
- No wake word, speaker ID, emotion detection, translation, lip-reading, browsing, Bluetooth, or autonomous workflows

## Natural Speech Delivery & Conversational Timing

Milestone 29 improves how Cortana **delivers** spoken responses so they sound less like text being read aloud and more like a natural conversational assistant. It is not a new reasoning engine and it does not take over assistant response ownership.

Centralized speech-delivery policy:

- `SpeechDeliveryPlan` is a small, frozen, typed plan: delivery mode, response depth, chunk size, pause hints, acknowledgment hint, list delivery mode, interruption recovery, concise spoken delivery, avoid-phrases, required-notice preservation, interruption/timing flags
- `authorizes_privileged_action` is structurally false
- Deterministic, bounded, no I/O, no network, no second model call, no tool calls
- Shared by `/voice-turn`, `/voice-realtime`, and `/multimodal-realtime`

Speech-friendly normalization:

- Operates on a **delivery copy only**; canonical assistant text and conversation history are unchanged
- Handles markdown headings, bullets, numbered lists, emphasis markers, code fences, extra blank lines, repeated punctuation, long parentheticals, raw URLs, and dense symbols
- Converts written lists into spoken sequences such as “First, … Second, … Third, …”
- Preserves numbers, dates, short IDs, required safety/authorization language, and uncertainty
- Long URLs, hashes, and large code blocks are not read character-by-character by default; speech may say they are better viewed on screen while the canonical text remains on screen/history
- Explicit exact/verbatim speech requests override that simplification. Phrases such as “spell it”, “read it aloud”, “dictate the command”, “word for word”, “character by character”, or “give me the exact URL/hash” keep hashes, URLs, commands, and code in the spoken delivery copy. Exactness then wins over speech elegance. A bare word such as “read” does not force exact-content mode

Chunking:

- Prefers sentence, then clause, then list-item boundaries, with a safe hard wrap
- Avoids splitting numbers, URLs, abbreviations, dates/times, and token sequences where possible
- Local and deterministic; no extra model call
- First speech-safe chunk is prepared immediately so playback can start without waiting on later formatting

Response-depth delivery:

- **Brief**: fewer, larger chunks; direct opening; minimal acknowledgment; no recap. Safety/factual content is not truncated
- **Normal**: ordinary spoken pacing
- **Detailed**: more deliberate chunking and structured delivery, without reading a written report verbatim

Acknowledgments:

- Reuses Milestone 27 acknowledgment/repetition state
- Does not automatically say “Got it”, “Okay”, “Sure”, or “Absolutely” before every spoken reply
- A brief acknowledgment is allowed after a user correction such as “No, Tuesday.”
- Required safety/authorization confirmations are not suppressed

Interruption recovery:

- Milestone 25 still owns barge-in and abort
- After an interruption, delivery guidance says not to restart the aborted answer or repeat already-heard material
- Uses Milestone 27/M28 correction state so the next reply continues from the corrected point
- Tracks only short fingerprints of interrupted spoken text — never raw audio

Repetition reduction:

- Avoids repeating recent spoken openings, acknowledgments, summaries, and restatements of the user’s request
- Does not suppress safety warnings, uncertainty, required confirmations, errors, or important constraints
- “Repeat that” explicitly allows repetition

`/voice-turn` behavior:

- Canonical chat/history text is generated first and remains authoritative
- A speech delivery copy is normalized and chunked locally
- Chunks are synthesized and played in order through the existing TTS path
- If the turn is already cancelled before playback begins, Cortana synthesizes and plays zero chunks, clears pending delivery state, and returns cleanly. Canonical history is unchanged
- Remaining chunks stop when the turn is cancelled after the first chunk has started
- Temporary audio is in-memory only; no committed audio artifacts

`/voice-realtime` behavior:

- `create_response=false`, server VAD, and `interrupt_response=true` remain the M25 session settings
- Cortana calls `response.create` once per committed user turn for transport correlation only. It does not create a competing local TTS/audio stream
- Provider-generated audio remains provider-owned and is not re-chunked or re-synthesized locally
- Spoken-delivery style is included in session instructions, and per-turn delivery guidance is attached to the existing Milestone 28 next-turn `session.update` after a completed pair
- Ending a realtime session clears session-specific delivery state, including interrupted-response fingerprints and pending local chunks. Conversation-wide repetition/opening fingerprints remain until `/clear`. A later realtime session on the same conversation object is not filtered by a stale interruption from a previous session
- True per-utterance local audio control is **not architecturally available** while the provider owns streaming audio

`/multimodal-realtime` behavior:

- `create_response=false`, `interrupt_response=true`, visual lifecycle, and the single `response.create` owner are unchanged
- Delivery guidance may be included with the existing Milestone 28 advisory `session.update` **before** that single `response.create`
- The 2.5s transcript timeout/fallback is unchanged; fallback does not wait for another transcript event and does not create a second response
- Session-end cleanup uses the same speech-delivery session boundary as `/voice-realtime`: interrupted-response fingerprints do not survive into a later session

Perceived latency:

- Normalization and chunking are local CPU-only work
- The first speech-safe chunk is prepared first
- No extra network, model, or tool calls
- `/voice-turn` still synthesizes chunks sequentially over the existing TTS path. Multi-chunk replies can increase total spoken latency because later chunks wait for earlier synthesis and playback. This is a known limitation, not a parallel TTS architecture
- Correctness is not sacrificed to shave milliseconds
- No blocking sleeps are added to “sound human”

Single response ownership:

- Milestone 29 guides delivery; it does not create a second assistant response path
- Fail-open uses the original assistant text for existing speech output; it does not skip safety or authorization checks

Authority boundaries:

- Speech-delivery instructions are style/delivery only
- They never execute or authorize tools, workflows, calendar/reminder writes, memory writes, incident/document operations, confirmations, or other side effects
- User-derived content inside delivery guidance has no elevated authority

Exact-content limitations:

- Dates, numbers, short IDs, versions, IPs, times, flags, paths, and similar technical tokens are preserved in the spoken copy
- When the user asks to hear exact/verbatim/spelled/dictated/read-aloud content, hashes, URLs, commands, and code are spoken as written instead of being replaced with screen-view language
- Without that explicit intent, longer code/URLs remain visually available rather than poorly verbalized

Known provider limitations:

- OpenAI Realtime audio for `/voice-realtime` and `/multimodal-realtime` is streamed by the provider. Cortana cannot safely re-chunk that audio locally without creating a competing stream
- Pause/pacing on those paths is advisory session guidance, not a local TTS clock
- `/voice-turn` can apply local chunk boundaries because TTS playback is locally owned; chunk gaps provide short natural pauses without fake punctuation or blocking sleeps

Citation labels use a compact deterministic format such as `[DOC-1:C1]`. Each label maps through a session source manifest to:

- document ID
- filename
- chunk index
- character range

Citation-label validation checks that labels in the AI response match the exact labels supplied with the request. Fabricated labels are marked and support is downgraded. It does **not** prove full factual entailment of every natural-language claim against the source text.

Source manifests:

- Exist only in memory for the current session
- Are shown with `/sources`
- Are cleared by `/clear`, `/remove-all-documents confirm`, and application restart
- Are not written to disk
- Do not expose local file paths or vault paths

Grounded document AI requests never include ordinary conversation history or active memory. Document Q&A/summary/compare results are not written into ordinary `ConversationHistory`.

## Security event, incident, and evidence foundation

Milestone 8 adds a defensive, documentation-focused local foundation for cybersecurity casework. Records are created or modified only through explicit user commands. This is an audit-support foundation, not a substitute for certified forensic procedure, and it does not claim legal forensic admissibility.

Supported record types and relationships:

- Security events can optionally link to one incident
- Incidents can link many events, indicators, evidence records, and analyst notes
- Indicators store normalized and original values without reputation lookups
- Evidence metadata is stored in the incident repository JSON
- Incident, event, indicator, evidence, note, and custody collections are fail-closed at documented count caps. Over-cap or malformed loads leave the file unchanged.
- Evidence byte copies are stored separately under a user-local evidence directory
- Chain-of-custody entries are append-only and tied to evidence records
- Timelines are derived locally from events, notes, and custody entries and are never persisted as a separate source of truth

Severity values:

- `informational`
- `low`
- `medium`
- `high`
- `critical`

Event statuses:

- `new`
- `investigating`
- `contained`
- `resolved`
- `false-positive`

Incident statuses:

- `open`
- `triage`
- `investigating`
- `contained`
- `monitoring`
- `resolved`
- `closed`

Indicator types:

- `ipv4`
- `ipv6`
- `domain`
- `url`
- `email`
- `sha256`
- `sha1`
- `md5`
- `filename`
- `process`
- `registry-key`
- `generic`

Evidence storage statuses:

- `metadata-only` — metadata recorded without claiming a preserved local copy
- `copied` — a local evidence byte copy exists and was hash-verified at registration

Behavior and limits:

- All Milestone 8 commands are local and never call the AI service.
- Ordinary chat, active memory, and `/ask-docs` never receive incident packets. Milestone 12 analysis is a separate, default-disabled explicit command path.
- No automatic collection, scanning, remediation, containment, malware execution, packet capture, vulnerability scanning, penetration-testing actions, threat-intelligence network lookups, cloud SIEM integrations, or telemetry.
- Evidence files may be dangerous. Cortana stores opaque bytes and never executes, opens for parsing, imports, or inspects evidence contents beyond hashing and byte-copying.
- SHA-256 is calculated locally with streaming reads.
- If evidence metadata persistence fails after the byte copy, Cortana best-effort deletes the copied `.bin` and reports failure. The original source file is left untouched. Rollback unlink failure does not hide the original error.
- `/evidence-verify` recalculates the stored copy hash, compares it to the recorded digest, appends a custody verification entry, and never repairs or deletes evidence automatically.
- Chain-of-custody entries are append-only through normal application commands.
- Repository and evidence storage live in user-local application data, outside the Git repository.
- Atomic JSON writes prevent partial repository files during a single save.
- Atomic writes do not coordinate concurrent Cortana processes. Last-writer-wins lost updates remain possible. Use one application instance per incident repository. Cross-process locking is not implemented yet.
- `/clear` clears conversation history, conversational state, recalled active memories, and the grounded source manifest. It does not delete persistent memories, incidents, or evidence.
- `/remember` does not create incidents or evidence.
- `/add-document` / Knowledge Vault ingestion does not register evidence.
- `/ask-docs` does not search incident records.
- Recording an indicator does not mean the indicator is malicious.

## Defensive tool framework

Milestone 9 adds a secure, local framework for registering, planning, approving, and auditing defensive cybersecurity tools. The framework remains defensive, allowlisted, human-supervised, and non-destructive.

Built-in tools:

| Tool ID | Purpose |
| --- | --- |
| `system-summary` | Safe local platform/capability summary |
| `file-sha256` | Streaming SHA-256 of an explicitly supplied regular file |
| `text-search` | Literal string search in an explicitly supplied text file |
| `compare-sha256` | Compare a file digest to an expected SHA-256 |
| `incident-summary` | Deterministic local incident count summary |
| `simulated-log-check` | Simulation-only mock log check |

Risk levels:

- `informational`
- `low`
- `moderate`
- `high` (simulation-only in this milestone)
- `prohibited` (cannot be enabled or executed)

Workflow:

1. Create an authorized scope with `/scope-new`
2. Create a request with `/tool-request`
3. Generate a dry-run plan with `/tool-dry-run`
4. Approve or reject when required with `/tool-approve` or `/tool-reject`
5. Execute with `/tool-run`
6. Inspect results and audit entries with `/tool-result` and `/tool-audit`

Behavior and limits:

- Tools execute only when registered, allowlisted, scope-validated, and authorized for their risk level.
- Every tool declaration has an explicit `capability_class`. Classification is metadata on the definition, not inferred from the tool ID.
- Capability classes: `internal-readonly` (built-in bounded tools), `arbitrary-shell`, `external-process`, `autonomous-remediation`, and reserved `ai-context-injection`.
- Kill-switches are enforced at `tool_policy.assert_executable` before any implementation runs. Direct executor invocation and workflow steps use the same gate. Registering a `ToolDefinition` does not bypass it.
- `ARBITRARY_SHELL_EXECUTION_ENABLED` (default False) is **ENFORCED**. Tools classified `arbitrary-shell` cannot execute while it is False. There is no unrestricted PowerShell, cmd, Bash, Python, or subprocess command interface.
- `EXTERNAL_TOOL_EXECUTION_ENABLED` (default False) is **ENFORCED**. Tools classified `external-process` cannot execute while it is False. Enabled `future-external` definitions remain structurally rejected.
- `AUTONOMOUS_REMEDIATION_ENABLED` (default False) is **ENFORCED**. Tools classified `autonomous-remediation` cannot execute while it is False, even with a stored approval. This flag does **not** disable user-explicit, approved bounded read-only tools (`internal-readonly`). Enabling it still requires explicit approval and dry-run; conversational "yes" and tool results cannot approve execution.
- `TOOL_AI_CONTEXT_INJECTION_ENABLED` (default False) is **RESERVED / not implemented**. No production path injects tool results into AI context. `/status` reports it as reserved. This is distinct from `WORKFLOW_AI_CONTEXT_INJECTION_ENABLED`, which only gates Milestone 12 packet selection.
- `PROCESS_CHILD_STARTUP_TIMEOUT_SECONDS` is **RESERVED / unused**. There is no child startup handshake. Process-isolated execution waits with the tool's `timeout_seconds` on `communicate()`. `/status` reports it as reserved.
- The AI does not select tools and does not execute tools.
- All file-based tools are read-only. They reject symlinks/reparse points and never delete, modify, or execute target files.
- Built-in tools remain bounded and read-only. With process isolation disabled (default), execution uses in-process worker threads: timeouts stop the caller from waiting, but workers are not forcibly terminated, and late completion is never published.
- Milestone 13 optionally enables process-isolated execution for a tiny allowlisted subset (`system-summary`, `simulated-log-check`) via `subprocess.Popen` and schema-validated JSON IPC. Milestones 15–16 extend isolation to reviewed file tools under dual gates. See the process-isolation sections below.
- Evidence and incident systems remain separate unless a request explicitly links an existing incident ID as metadata.
- Tool-control persistence uses atomic UTF-8 JSON outside the Git repository.
- Scopes, requests, approvals, and results fail closed at documented count caps. Tool audit entries trim oldest-first on write, matching other bounded audit logs.
- Atomic writes do not coordinate concurrent Cortana processes. Use one application instance per tool-control repository. Cross-process locking is not implemented yet.
- Audit records support accountability and do not claim legal forensic certification.
- Prohibited capabilities remain out of scope: penetration testing, exploit execution, credential dumping, persistence, evasion, malware execution, destructive remediation, firewall changes, account disabling, process termination, file deletion, registry modification, remote access, real network scanning, internet threat-intelligence lookups, and cloud SIEM integrations.

### Process-isolated tool execution (Milestone 13)

Milestone 13 adds an optional process-isolated execution path for a tiny trusted subset of existing defensive tools. `DefensiveToolExecutor` remains the sole public execution boundary. Workflows and Milestone 12 AI analysis do not gain a new execution path.

Flags (both default disabled):

- `PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED`
- `PROCESS_ISOLATED_TOOL_TERMINATION_ENABLED` (cannot independently enable process execution)

When execution is disabled, behavior matches Milestones 9–12. Eligible tools fall back to the in-process route. Required tools fail closed if isolation is unavailable.

Initial process-isolation eligible tools:

- `system-summary`
- `simulated-log-check`

In Milestone 13, all file-touching tools remained `process_isolation=prohibited`. Milestones 15–16 later make `file-sha256`, `compare-sha256`, and `text-search` eligible under dual feature gates. Repository-backed tools (`incident-summary`) and all other tools remain prohibited.

Architecture notes:

- Parent and child communicate with bounded, exact-key, schema-validated JSON only.
- The child entry point is `python -m src.tool_process_runner`.
- Authoritative results use a dedicated parent-created result file channel. stdout and stderr are diagnostic-only and are never parsed as results.
- Every `Popen` call passes an explicit environment allowlist (no full parent env inheritance; no API keys or repository paths).
- On timeout, the first state observed by the parent wins. Late results after termination are never accepted.
- Cancellation is narrow: pre-launch cancelled request status, plus KeyboardInterrupt termination of an active child. No mid-flight cancellation service or worker pool.
- Antivirus/EDR may delay or block child Python processes. Startup timing is calibrated for Windows process creation. A child could become orphaned if the parent process itself crashes. Process isolation improves terminability; it is not a sandbox against malicious trusted tool code.

### Process resource governance and safe file-opening (Milestone 14)

Milestone 14 adds optional Windows Job Object governance on top of Milestone 13 process isolation, plus a Windows-native safe file-opening foundation that is **not** wired into tool eligibility.

Flags (both default disabled):

- `PROCESS_RESOURCE_LIMITS_ENABLED` — Job Object governance for process-isolated tools only
- `PROCESS_FILE_TOOL_ISOLATION_ENABLED` — dual-gate with process isolation for reviewed file integrity tools (Milestone 15); alone it does not make tools eligible

When resource limits are disabled, Milestone 13 isolation behavior is unchanged. When enabled on Windows with `pywin32`, each isolated execution:

1. launches the child with `subprocess.Popen`
2. creates/configures one Job Object and assigns the child
3. verifies limits with `QueryInformationJobObject`
4. only then sends the JSON request through `communicate()`

Job Object settings:

- `ActiveProcessLimit = 1`
- `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
- Job memory limit: `MAX_PROCESS_ISOLATED_JOB_MEMORY_BYTES` = 256 MiB (chosen to cover Python 3.13 startup, `system-summary`, `simulated-log-check`, and typical Windows/AV overhead while remaining a meaningful containment bound)
- one Job Object per child execution; no process pool; no reusable Job Object
- CPU-rate and handle-count limits are deferred

Termination:

- Prefer `TerminateJobObject` for entire-tree termination when a Job Object is active
- Fall back to direct child kill only if Job Object setup/assignment failed or the Job Object is unavailable
- Windows termination is hard; graceful shutdown is not claimed
- Parent crash closes the Job Object; with kill-on-job-close, members are terminated where the OS guarantee applies

Outcomes:

- `resource_limit_exceeded` is a distinct parent-decided outcome and is not classified as `failed`, `timed_out_terminated`, or `cancelled`
- The child cannot claim `resource_limit_exceeded`

Conditional dependency:

- `pywin32` is required only on Windows (`requirements.txt` platform marker)
- It is imported only when resource limits or safe-open Windows APIs are actually used
- Non-Windows, or Windows without `pywin32`, fails safely when limits are enabled; there is no silent unlimited fallback

Safe file-opening foundation (`src/tool_process_safe_open.py`):

- Uses Windows reparse-aware open (`CreateFile` with `FILE_FLAG_OPEN_REPARSE_POINT`)
- Rejects device paths, UNC, reserved device names (every path segment, including trailing spaces/dots and extensions), ADS, symlinks/junctions/reparse points, and non-regular files
- Captures and verifies Windows file identity (volume serial + file indexes), not path strings alone
- When an authorized root is supplied, containment is checked before open on the caller path and again after open using a path independently derived from the handle via `GetFinalPathNameByHandle`
- Plain `open()` / `os.open()` are not the secure boundary
- Milestones 15–16 wire this foundation into process-isolated `file-sha256`, `compare-sha256`, and `text-search`

### Controlled process-isolated file integrity tools (Milestone 15)

Milestone 15 allows a tiny reviewed subset of existing read-only file-integrity tools to use the process-isolated path with the Milestone 14 safe-open foundation.

Dual feature gates (both default disabled):

- `PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED`
- `PROCESS_FILE_TOOL_ISOLATION_ENABLED`

Routing:

- process isolation off → existing in-process file-tool behavior
- process isolation on, file-tool isolation off → existing in-process behavior for eligible file tools
- both on → parent captures Windows file identity, child opens with `safe_open_for_read`, streams SHA-256 through the verified handle
- file safe-open unavailable while file-tool isolation is enabled → fail closed in the parent before the child request is sent
- `PROCESS_RESOURCE_LIMITS_ENABLED` remains an independent optional Job Object protection

Eligible tools (Milestone 15–16):

- `file-sha256` → `process_isolation=eligible`
- `compare-sha256` → `process_isolation=eligible` (still one authorized file path + one expected SHA-256 digest; not a two-file tool)
- `text-search` → `process_isolation=eligible` (Milestone 16; third and final currently registered file tool eligible for process isolation)

All other file-touching tools remain `process_isolation=prohibited`.

Security properties:

- Parent validates schema/scope, then captures immutable file authorization (canonical path, authorized root, volume serial, file indexes, size, baseline last-write time)
- Child receives exact-key nested `file_authorization` only; never scope/repository/approval/AI objects
- Child hashes via streaming `hash_sha256_from_safe_handle` (`ReadFile` + `hashlib.update`); full-file buffering is not used for hashing
- Child text-search uses incremental UTF-8 decoding (`errors="replace"`), bounded pending-line assembly, and literal case-sensitive line search; full-file buffering is not used
- After EOF, the same handle is re-queried; size/identity/last-write changes fail closed as `failed` / `FileChangedDuringRead`
- Identity mismatch fails closed as `failed` / `IdentityMismatch`
- Oversized files fail closed as `failed` / `FileTooLarge`
- Hash results expose basename-only `filename_only`; text-search keeps the existing `filename` basename field
- Canonical paths, roots, identity fields, and search queries never appear in audits, workflow records, or incident notes
- Match previews may appear only in the intended tool result `matches` list, not in audit logs
- In-process open remains in `tool_safe_files.py`; isolated open remains in `tool_process_safe_open.py`

### Personal assistant scheduling core (Milestone 19)

Milestone 19 adds a local persistent reminder/scheduling foundation for one-time and bounded recurring obligations. Reminders are not memories: they are time-bound scheduled records stored separately from `MemoryStore` and never injected into ordinary AI conversation context.

Behavior:

- Persist reminders in user-local `reminders.json` with atomic writes and sticky fail-closed corruption handling
- Require explicit local wall time plus an IANA timezone on create (no guessed timezone; abbreviations such as EST/PST are rejected)
- Nonexistent spring-forward and ambiguous fall-back local times are rejected; reminder and calendar share the same wall-time conversion
- Persist absolute instants as UTC ISO-8601 with trailing `Z`; store the intended IANA timezone separately
- Persist only three statuses: `scheduled`, `completed`, `cancelled`
- Derive overdue as `status == scheduled` and `due_at <= now` (no persisted `due` status; no write-on-read reconciliation)
- Support recurrence: `none`, `daily[:interval]`, `weekly[:interval][:weekdays]`, `monthly[:interval]`
- Anchor recurring series to `recurrence_anchor_at` and preserve local wall-clock time across DST
- Snooze moves only the current occurrence; reschedule resets the series anchor for recurring reminders
- Completing a recurring reminder jumps to the first occurrence strictly after now
- Lifecycle mutations and their audit entries are persisted together in one atomic replace
- Explicit slash control plane only; no background delivery, OS notifications, calendar sync, or free-form temporal parsing

Commands:

```text
/reminder-add <title> | <YYYY-MM-DD HH:MM> | <IANA-timezone> | <recurrence> | <message>
/reminders
/reminder-show <reminder-id>
/reminder-complete <reminder-id>
/reminder-cancel <reminder-id>
/reminder-snooze <reminder-id> | <YYYY-MM-DD HH:MM>
/reminder-reschedule <reminder-id> | <YYYY-MM-DD HH:MM> | <IANA-timezone-or->
```

M18 guidance only (exact phrases; no operational execution): `set a reminder`, `list reminders`.

Not in this milestone: push/email/SMS/voice delivery, appointments/bookings/payments, yearly/COUNT/UNTIL/RRULE engines, background schedulers, AI scheduling, reminder↔calendar linkage, or reminder↔memory linkage.

### Google Calendar integration (Milestone 20)

Milestone 20 adds Cortana's first real external calendar integration: one authorized Google Calendar account, secure OAuth credential storage, event/free-busy reads, conflict checking, and explicit prepare→confirm writes. Google remains the source of truth for event data. Reminders stay a separate local domain.

Behavior:

- Desktop OAuth loopback via `InstalledAppFlow.run_local_server()` using `CORTANA_GOOGLE_OAUTH_CLIENT_FILE`
- Exact scopes: `calendar.calendarlist.readonly`, `calendar.events`, `calendar.freebusy`
- Persist only the Google refresh token in the OS credential store (`keyring` / Windows Credential Manager); never in `calendar_control.json`, audit, logs, AI history, or tool child process env
- Google Calendar and keyring are optional. If those packages are missing, Cortana core still starts and calendar commands/status report unavailable. Provider-side deletion/revocation remains best-effort.
- Local `calendar_control.json` stores one account metadata object, proposals, and bounded audits only (no event catalog, no tokens)
- Default calendar starts as Google's stable `primary` alias; change with `/calendar-use`
- Timed local wall times are interpreted in the selected calendar's IANA timezone (never OS timezone guessing). Nonexistent and ambiguous DST local times are rejected, matching reminders.
- Reads: list calendars, list/show events, free/busy
- Writes: timed non-recurring create/reschedule/cancel only after `/calendar-confirm`
- Create proposals generate a Google-compliant `client_event_id` once at prepare time and reuse it for recovery
- Confirm rechecks free/busy and etag/stale state; conflicts and stale remote changes fail closed and require re-prepare
- Ambiguous provider outcomes become `unknown_outcome` (not blind retry as failed)
- Recurring and all-day events are readable; writes to recurring/all-day events are rejected
- No attendees, booking, payments, background sync, multi-provider support, or AI calendar execution
- Calendar slash output and M18 guidance stay outside ordinary AI conversation history

Commands:

```text
/calendar-connect
/calendar-disconnect
/calendars
/calendar-use <calendar-id>
/calendar-events [<calendar-id>]
/calendar-event <calendar-id-or-> | <event-id>
/calendar-freebusy <calendar-id-or-> | <YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>
/calendar-create <calendar-id-or-> | <title> | <YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>
/calendar-reschedule <calendar-id-or-> | <event-id> | <YYYY-MM-DD HH:MM> | <YYYY-MM-DD HH:MM>
/calendar-cancel <calendar-id-or-> | <event-id>
/calendar-confirm <proposal-id>
```

M18 guidance only (exact phrases; no operational execution): `show my calendar`, `schedule a meeting`.

Study Partner guidance only (exact phrases; no operational execution): `help me study`, `quiz me`.

Network trust: Cortana delegates Google endpoint selection and TLS validation to the official Google SDKs. Calendar network code is not registered as a defensive tool and does not accept arbitrary user URLs.

### Unified assistant orchestration (Milestone 18)

Milestone 18 adds a small deterministic natural-language orchestration layer for high-confidence ordinary-language requests. It is integration only: no autonomous agent, no AI intent classifier, and no replacement for the slash-command control plane.

Behavior:

- Slash input never enters the orchestrator; existing slash dispatch is unchanged
- Only non-slash input is offered to `UnifiedAssistantOrchestrator`
- Matching uses whole-message anchored parsers (`re.fullmatch` / exact allowlists)
- Ambiguous phrases fall through to ordinary AI conversation unchanged
- Orchestrator-handled requests and results are excluded from ordinary AI conversation history

High-confidence operational routes:

- Memory write: `remember <text>`, `remember: <text>`, `remember - <text>` (reuses `MemoryStore`; does not activate memory)
- Memory read: exact phrases `list my memories`, `list memories`, `show my memories`
- Document search: anchored `find document(s) about …` / `search document(s) for …` / `search my documents for …` — **lexical search only** via `LexicalDocumentRetriever` (not `/ask-docs`, no AI client)
- Incident read: exact `show incident <UUID>` (read-only `IncidentRepository`; incident contents stay out of conversation history)

Guidance-only routes (static text; no executors, no AI, no request construction):

- Evidence search → points to controlled `/evidence-search`
- Tools → points to controlled `/tool-*`
- Workflows → points to controlled `/playbook-*`
- Analyst assistance → points to controlled `/incident-analysis-*` and existing feature gates
- Reminders → points to controlled `/reminder-add` / `/reminders` (exact phrases only)
- Calendar → points to controlled `/calendar-events` or `/calendar-create` + `/calendar-confirm` (exact phrases only)

Not routed by natural language in this milestone: evidence-search request construction, ask-docs, study explain/question/answer execution, ingestion/deletion, tool/workflow/AI analysis execution, note saving, reminder creation/mutation, calendar reads/writes, shell/network/remediation, and other consequential flows remain explicit controlled commands.

### Evidence text search and bounded incident context (Milestone 17)

Milestone 17 adds a thin operator bridge over existing incident, evidence, scope, tool, and M12 packet machinery. It does not introduce an Investigation entity or a second search engine.

Capabilities:

- `/incident-link-evidence <incident-id> <evidence-id>` — bidirectional, idempotent membership update on `SecurityIncident.evidence_ids` and `EvidenceRecord.related_incident_ids`
- `/evidence-search <incident-id> | <evidence-id> | <scope-id> | <query> [| <max-matches>]` — creates an ordinary `text-search` `ToolExecutionRequest` with `incident_id` set; does **not** execute the search
- `/incident-context <incident-id> | <scope-id> | <event-ids> | <indicator-ids> | <note-ids> | <workflow-run-ids>` — displays the existing M12 allowlisted packet for operator review (**AI-off**; use `-` for empty ID lists)

Execution rules for evidence search:

- Scope is supplied per operation
- Scope must authorize both the incident (`assert_incident_authorized`) and the resolved evidence-store path (`assert_path_authorized`)
- Evidence must already be linked to the incident on both sides
- Operator continues with existing `/tool-dry-run`, `/tool-approve`, and `/tool-run`
- Process isolation / Job Objects remain governed by existing M13–16 flags when enabled
- Resolved evidence-store paths are not printed by the new commands

AI boundary:

- Bounded context display reuses `build_incident_analysis_packet` and never calls AI
- `AI_INCIDENT_ANALYSIS_ENABLED` and `/incident-analysis-*` remain the only AI analysis route

Demo-ready later (sanitized data only): evidence linking, controlled evidence-search request creation, AI-off incident context display.

### Controlled process-isolated text search (Milestone 16)

Milestone 16 extends the dual-gated file-isolation path to the existing read-only `text-search` tool without redesigning its public interface.

Preserved behavior:

- one authorized file path, one literal query, optional `max_matches`
- case-sensitive, non-regex, line-oriented search
- UTF-8 decoding with `errors="replace"`; binary files are not specially rejected
- bounded by `MAX_TOOL_FILE_BYTES`, `MAX_TOOL_TEXT_SEARCH_MATCHES`, and `MAX_TOOL_TEXT_SEARCH_PREVIEW_CHARS`
- existing result shape (`filename`, `match_count`, `truncated`, `matches[{line_number, preview}]`)

Isolated-path specifics:

- requires both `PROCESS_ISOLATED_TOOL_EXECUTION_ENABLED` and `PROCESS_FILE_TOOL_ISOLATION_ENABLED`
- reuses unchanged `FileAuthorization` and `safe_open_for_read`
- streams via incremental decoder and bounded pending-line memory (`MAX_TOOL_TEXT_SEARCH_PENDING_LINE_CHARS`)
- queries and matched snippets are sensitive; they must not enter audit entries
- hitting `max_matches` remains success with `truncated=True`, not an error
- in-process fallback remains when either isolation flag is off

Documented limitations:

- File tools remain read-only
- Only `file-sha256`, `compare-sha256`, and `text-search` are process-isolation eligible
- `compare-sha256` remains one file plus an expected digest
- `text-search` remains literal and case-sensitive; invalid UTF-8 becomes replacement characters
- Pending-line memory is bounded; pathologically long lines discard overflow with bounded overlap
- Full paths and search queries are internal/sensitive authorization or parameter data
- Hard links cannot be distinguished from another name for the same underlying file identity
- Sparse files report logical size
- Concurrent modification is detected using post-read size and last-write-time re-checks; this cannot guarantee protection against every filesystem or endpoint-security behavior
- Process isolation and Job Objects improve containment but are not a complete sandbox
- Non-Windows environments use existing in-process behavior when file-tool isolation is off; enabling file-tool isolation off Windows fails safely

Limitations (Job Objects):

- Job Objects improve containment but are not a complete sandbox
- The 256 MiB memory limit is not complete memory safety and does not prevent all resource exhaustion
- EDR/antivirus may interfere with child processes or Job Object behavior

Example (safe placeholders):

```text
/scope-new Lab review | file-sha256,text-search | C:\Cases\example-root | Hash review for case notes
/tool-request file-sha256 | <scope-id> | {"path":"C:\\Cases\\example-root\\sample.txt"} | Verify sample integrity
/tool-dry-run <request-id>
/tool-approve <request-id> | Approved after dry-run review
/tool-run <request-id>
```

Multi-field Milestone 8 commands use the delimiter ` | ` so paths and free text can contain spaces. Example:

```text
/evidence-register C:\Cases\packet capture.pcap | Edge capture | Captured by analyst
/event-new high | Phishing report | User clicked a suspicious link
```

## Defensive workflow orchestration

Milestone 10 adds a bounded, deterministic workflow layer that coordinates multiple existing approved defensive tools through trusted, predefined playbooks.

Milestone 11 adds durable workflow-run and workflow-audit history, plus optional linkage of an authorized workflow run to an existing incident. This is record durability and traceability only: workflows remain non-resumable, non-autonomous, and do not create evidence or custody records.

### Milestone 11 release note (default behavior change)

**Upgrading from Milestone 10:** Milestone 11 enables durable workflow-run persistence and incident linkage by default (`WORKFLOW_RUN_PERSISTENCE_ENABLED = True`, `WORKFLOW_INCIDENT_LINKAGE_ENABLED = True`).

- Workflow runs and workflow audit entries now persist across application restarts in a user-local JSON repository.
- Non-terminal runs found after restart are marked `abandoned` and are never resumed, retried, or continued.
- Incident linkage is enabled by default; a completed linked run may append one bounded summary note to an existing authorized incident.
- Either capability can be disabled independently through its config flag. When both are disabled, Milestone 10 in-memory behavior remains unchanged.
- Persisted workflow data uses a reduced safe projection and excludes raw parameters, `structured_data`, approvals, secrets, and file contents.

Built-in playbooks:

| Playbook | Steps |
| --- | --- |
| `platform-baseline` | `system-summary` → `simulated-log-check` (`auth-noise`) |
| `mock-log-triage` | `simulated-log-check` across `auth-noise`, `malware-keyword`, and `empty` fixtures |

Behavior and limits:

- Playbooks are defined only in trusted Python source. They are not loaded from JSON, YAML, user files, AI output, or external sources.
- Every step references an existing registered Milestone 9 `tool_id` with static predeclared parameters.
- `DefensiveToolExecutor` remains the sole tool execution boundary.
- Dry-run is the default. `/playbook-run <name> | <scope-id>` calls `plan_dry_run` for each reached step and never calls `execute`.
- Explicit execution uses `/playbook-run <name> --execute | <scope-id>` and still requires scope, policy, and step-specific fingerprint-bound approvals where Milestone 9 requires them.
- Built-in playbooks are read-only and remain replay-safe: a later `/playbook-run ... --execute` starts from step 0 without extra confirmation.
- Side-effecting steps (any tool whose `capability_class` is not `internal-readonly`) persist a bounded operation claim before live execution. Repeating the same playbook/step/parameters without a new operation instance is denied. A genuinely new execution uses `/playbook-run <name> --execute --new-operation | <scope-id>`.
- Optional incident linkage uses `/playbook-run <name> | <scope-id> | <incident-id>` or `/playbook-run <name> --execute | <scope-id> | <incident-id>`.
- Incident linkage reuses `assert_incident_authorized` and requires the incident to exist before step one. On successful completion, exactly one bounded `summary` incident note is appended.
- Execution is strictly sequential and stop-on-failure. There is no parallel execution, silent retry, nested playbook, dynamic output piping, background worker, or arbitrary scripting interface.
- When `WORKFLOW_RUN_PERSISTENCE_ENABLED` is true, workflow runs and workflow audit entries are stored in a dedicated user-local JSON file using atomic writes. Persisted records use a strict safe-field allowlist and never store raw tool structured data, parameters, paths, or approval fingerprints.
- Persisted non-terminal runs from a previous process lifetime are deterministically marked `abandoned` on load. Workflows are never resumed, retried, or continued after restart.
- One application instance should access the workflow repository file at a time. Atomic writes protect against partial-write corruption; they do not provide full concurrent multi-process coordination. Last-writer-wins is not a supported operating mode.
- `WORKFLOW_RUN_PERSISTENCE_ENABLED` and `WORKFLOW_INCIDENT_LINKAGE_ENABLED` are independent. When both are disabled, Milestone 10 in-memory behavior remains unchanged.
- Workflow commands never call the AI service.

Example:

```text
/scope-new Baseline lab | system-summary,simulated-log-check | none | Local baseline review
/playbook-run platform-baseline | <scope-id>
/playbook-run platform-baseline | <scope-id> | <incident-id>
/playbook-status <run-id>
```

## Controlled security analyst assistance

Milestone 12 adds optional, human-controlled AI assistance over one sanitized incident packet at a time. Analysis and note saving are disabled by default and never run automatically.

Commands (explicit only; pipe-delimited):

1. `/incident-analysis-prepare <kind> | <incident-id> | <scope-id> | <event-ids> | <indicator-ids> | <note-ids> | <workflow-run-ids>` — authorize, build packet, preview counts/warning (no AI call)
2. `/incident-analysis-run <analysis-id>` — revalidate and call the AI for one prepared request
3. `/incident-analysis-show <analysis-id>` — inspect one in-memory prepared or completed analysis
4. `/incident-analysis-save-note <analysis-id>` — save the exact stored advisory output as one incident note

Use `-` for an empty ID list, or comma-separated UUIDs for multiple IDs. Supported kinds: `summary`, `gaps`, `investigation_questions`, `report_draft`, `remediation_checklist`, `outcome_explanation`.

Behavior and limits:

- `AI_INCIDENT_ANALYSIS_ENABLED` and `AI_INCIDENT_NOTE_SAVE_ENABLED` default to disabled. Note saving also requires the analysis flag.
- `WORKFLOW_AI_CONTEXT_INJECTION_ENABLED` (default false) independently gates whether Milestone 11 safe workflow/tool summaries may be selected into a packet. It does not enable analysis by itself.
- Ordinary conversation never receives incident packets (`INCIDENT_AI_CONTEXT_INJECTION_ENABLED` remains false).
- Prepare requires an authorized scope and reuses `assert_incident_authorized`. Selected records must belong to the same incident.
- Prepare shows transmission warning and category/packet counts; it does not print the full packet or call the AI.
- Each preparation uses a random untrusted-data boundary token for packet markers.
- Packets are allowlisted single-incident projections. Evidence, custody, and raw structured tool data are structurally excluded.
- Authorization is revalidated before the AI call and again before note saving.
- Analysis results are retained in process memory only. The durable artifact is only an explicitly saved note.
- Saved notes use fixed author `ai-analyst-assistance`, tag `ai-assisted`, type `hypothesis`, and a provenance banner; the body is verbatim analysis text.
- Named limits cover selected events/indicators/notes/workflow/tool summaries, packet size, output size, and retained analyses.
- No automatic persistence, tool/workflow execution, background work, or autonomous response.

Centralized defaults:

- Maximum selected events: 10; indicators: 20; notes: 10; workflow summaries: 5; tool summaries: 10
- Maximum packet characters: 32,000
- Maximum analysis output characters: 4,000
- Maximum retained in-memory analyses: 50

## Local commands

| Command | Description |
| --- | --- |
| `/help` | List core commands |
| `/help more` | Show the full command catalog |
| `/status` | Show compact session status |
| `/status verbose` | Show detailed operator status |
| `/diagnostics` | Show safe support diagnostics |
| `/clear` | Clear session conversation context, including recalled active memories; persistent memories stay stored |
| `/remember <text>` | Save one explicit persistent memory |
| `/memories` | List saved persistent memories |
| `/forget <memory-id>` | Delete one saved memory by ID |
| `/forget-all` | Begin deletion of all saved memories |
| `/forget-all confirm` | Confirm deletion of all saved memories |
| `/recall <memory-id>` | Activate one saved memory for temporary AI context |
| `/active-memories` | List memories currently active for AI context |
| `/release <memory-id>` | Remove one memory from active AI context |
| `/release-all` | Clear all active AI memory context for this session |
| `/add-document <path>` | Ingest one local document into the Knowledge Vault |
| `/documents` | List stored Knowledge Vault documents |
| `/document <document-id>` | Inspect one stored document by ID |
| `/remove-document <document-id>` | Delete one stored document by ID |
| `/remove-all-documents` | Begin deletion of all stored documents |
| `/remove-all-documents confirm` | Confirm deletion of all stored documents |
| `/search-docs <query>` | Search Knowledge Vault documents locally without calling the AI |
| `/ask-docs <question>` | Ask a source-grounded question using retrieved document passages |
| `/doc-summary <document-id>` | Summarize one authorized Knowledge Vault document |
| `/docs-compare <doc-id> \| <doc-id> \| <question>` | Compare two authorized documents for one question |
| `/sources` | Show sources from the latest successful grounded document request |
| `/study-start <doc-id>[,<doc-id>...]` | Start one Study Partner session over authorized documents |
| `/study-status` | Show the active study session status (no answer key) |
| `/study-explain <topic>` | Explain a topic from the active study documents |
| `/study-question <mcq\|short\|-> \| <topic-or->` | Generate one grounded practice question |
| `/study-answer <answer>` | Submit an answer to the pending study question |
| `/study-progress` | Show honest study progress metrics |
| `/study-end` | Complete the active study session |
| `/vision-describe <path>` | Describe one authorized local image |
| `/vision-ask <path> \| <question>` | Ask a question about one authorized local image |
| `/vision-info <path>` | Validate/normalize one authorized local image without calling the AI |
| `/voice-turn` | Capture one spoken utterance and hear Cortana's ordinary conversational reply |
| `/voice-realtime` | Start an explicit realtime spoken conversation with barge-in; Ctrl+C returns to text mode |
| `/multimodal-realtime` | Start realtime voice with bounded live camera context; Ctrl+C returns to text mode |
| `/voice-status` | Show safe voice/realtime/multimodal configuration without opening the microphone or camera |
| `/event-new <severity> \| <title> \| <description>` | Record one local security event |
| `/events` | List saved security events |
| `/event <event-id>` | Show one security event |
| `/event-status <event-id> <status>` | Update one security event status |
| `/incident-new <severity> \| <title> \| <summary>` | Open one local security incident |
| `/incidents` | List saved security incidents |
| `/incident <incident-id>` | Show one security incident |
| `/incident-status <incident-id> <status>` | Update one security incident status |
| `/incident-link-event <incident-id> <event-id>` | Link an event to an incident |
| `/incident-unlink-event <incident-id> <event-id>` | Unlink an event from an incident |
| `/indicator-add <type> \| <value> \| <confidence>` | Record one local indicator |
| `/indicators` | List saved indicators |
| `/indicator <indicator-id>` | Show one indicator |
| `/evidence-register <path> \| <title> \| <description>` | Register and copy local evidence bytes |
| `/evidence` | List saved evidence metadata |
| `/evidence-show <evidence-id>` | Show one evidence record |
| `/evidence-verify <evidence-id>` | Verify a stored evidence copy by SHA-256 |
| `/incident-add-note <incident-id> \| <note-type> \| <text>` | Add one analyst note to an incident |
| `/incident-notes <incident-id>` | List analyst notes for an incident |
| `/incident-timeline <incident-id>` | Show a derived incident timeline |
| `/tools` | List enabled defensive tools |
| `/tool <tool-id>` | Show one defensive tool |
| `/scope-new <name> \| <tool-id-list> \| <allowed-root-or-none> \| <justification>` | Create one authorized tool scope |
| `/scopes` | List authorized scopes |
| `/scope <scope-id>` | Show one authorized scope |
| `/scope-disable <scope-id>` | Disable one authorized scope |
| `/tool-request <tool-id> \| <scope-id> \| <parameter-json> \| <justification>` | Create one tool execution request |
| `/tool-requests` | List tool execution requests |
| `/tool-request-show <request-id>` | Show one tool execution request |
| `/tool-dry-run <request-id>` | Generate a dry-run plan for a request |
| `/tool-approve <request-id> \| <reason>` | Approve one tool execution request |
| `/tool-reject <request-id> \| <reason>` | Reject one tool execution request |
| `/tool-cancel <request-id>` | Cancel one tool execution request |
| `/tool-run <request-id>` | Execute one authorized tool request |
| `/tool-result <result-id>` | Show one tool execution result |
| `/tool-audit` | List tool-control audit entries |
| `/playbooks` | List enabled defensive playbooks |
| `/playbook-show <name>` | Show one defensive playbook |
| `/playbook-run <name> \| <scope-id>` | Dry-run one trusted playbook |
| `/playbook-run <name> --execute \| <scope-id>` | Execute one trusted playbook after validations |
| `/playbook-run <name> --execute --new-operation \| <scope-id>` | Execute a new side-effecting operation instance of the same playbook |
| `/playbook-run <name> \| <scope-id> \| <incident-id>` | Dry-run one trusted playbook linked to an existing incident |
| `/playbook-run <name> --execute \| <scope-id> \| <incident-id>` | Execute one trusted playbook linked to an existing incident |
| `/playbook-status <run-id>` | Show one workflow run |
| `/incident-analysis-prepare <kind> \| <incident-id> \| <scope-id> \| <event-ids> \| <indicator-ids> \| <note-ids> \| <workflow-run-ids>` | Prepare and preview one sanitized incident analysis packet |
| `/incident-analysis-run <analysis-id>` | Confirm and run AI analysis for one prepared request |
| `/incident-analysis-show <analysis-id>` | Show one in-memory incident analysis result |
| `/incident-analysis-save-note <analysis-id>` | Save one analysis verbatim as an incident note |
| `/about` | Describe Cortana |
| `/exit` | End the session cleanly |

Notes:

- `/clear` affects temporary conversation history and the latest `/sources` manifest. It does not delete documents, persistent memories, incidents, or evidence, and does not remove active memories.
- `/remember` saves a persistent memory but does not activate it, ingest documents, or create incidents/evidence.
- `/release` and `/release-all` clear temporary active context only; they do not delete persistent memories.
- `/forget` deletes a persistent memory and also removes it from active context when selected.
- `/forget-all confirm` deletes all persistent memories and clears active memory selections.
- `/remove-document` removes one vault document and invalidates matching source-manifest entries.
- `/remove-all-documents confirm` deletes all vault documents and clears the source manifest. It does not affect persistent memories, active-memory context, incidents, or evidence.
- `/search-docs` never calls the AI service.
- `/ask-docs`, `/doc-summary`, `/docs-compare`, and Study Partner AI commands send only selected retrieved/authorized chunks, never the entire vault and never incident records.
- Study Partner never persists raw user answers, generated feedback, or source text in `study_state.json`.
- Grounded document commands do not write into ordinary conversation history and do not inject active memory.
- Vision commands do not write into ordinary conversation history, do not inject active memory, and do not persist image bytes.
- `/vision-info` never calls the AI service.
- `/voice-turn` is sequential push-to-talk: capture → STT → ordinary chat → TTS → synchronous playback. It does not open the microphone at startup and does not listen in the background.
- `/voice-realtime` opens one bounded Realtime API session with server VAD and barge-in. Ordinary CLI input is paused until Ctrl+C or session end. It does not auto-start, auto-reconnect, or grant spoken operational authority.
- `/multimodal-realtime` opens one bounded Realtime API session with microphone + default camera. Local capture is ~2 FPS into a capacity-1 latest-frame buffer; at most one current image may be sent per user turn. It does not auto-start, archive frames, perform face recognition/lip-reading, or grant voice+vision operational authority.
- While "Listening..." is shown, pressing Enter ends recording; avoid typing the next command until recording has stopped. Anything typed while "Listening..." is shown is discarded and is not queued for the next command.
- Spoken transcripts enter ordinary conversation history after a successful chat response, but they do not execute slash commands or Milestone 18 natural-language actions in this milestone.
- `/voice-status` never opens the microphone and never calls the AI service.
- Milestone 8 security commands never call the AI service.
- Milestone 9 defensive tool commands never call the AI service.
- Milestone 10/11 workflow/playbook commands never call the AI service.
- Milestone 12 prepare/show/save-note commands are local. Only `/incident-analysis-run` calls the AI, and only when analysis is enabled.
- Absolute path-like input such as `/etc/passwd` is treated as conversation content for the AI, not as a local command, unless it is explicitly provided as an `/add-document` or `/evidence-register` argument.

## Active memory limits

Active memory context is limited by centralized defaults:

- Maximum of 10 active memories
- Maximum of 8,000 combined characters of active memory text

Activation is rejected with a clear local message when either limit would be exceeded. Existing active selections are not silently removed, and saved memory text is not silently truncated.

## Memory storage and corruption handling

- Memories are saved only through explicit local commands.
- Storage uses a local JSON file under the user-local application data directory.
- If the memory file is missing, Cortana starts with an empty memory list.
- If the memory file is empty, malformed, structurally invalid, over the count cap, contains overlong text, or contains duplicate IDs, Cortana returns a clear local error, leaves the file unchanged for inspection, and does not attempt automatic repair in this milestone.
- Atomic writes protect against partial or corrupt files during a single save.
- This milestone assumes a single Cortana instance per memory file. Running multiple instances against the same file may cause last-writer-wins lost updates; cross-process locking is not implemented yet.

## Current accepted limitations

These are accepted product limits, not unfinished Batch 1–6 defects:

- **M25 `response.created` correlation:** `/voice-realtime` now binds `response.created` through trusted client-generated metadata (`cortana_user_item_id`, `cortana_generation`). Missing, malformed, or unknown metadata fails closed: the response is tombstoned and is never FIFO-bound. Metadata echo is supported by the installed SDK/schema. The live M30 metadata gate measured exact production echo (`Realtime metadata gate: PASS`). If the provider omits metadata, Cortana rejects that response rather than guessing.
- **M26 response correlation:** `/multimodal-realtime` still uses client-created responses without metadata correlation and retains its own visual-ack/FIFO architecture. It did not gain the M25 explicit-correlation model.
- **M26 visual-ack correlation:** late or ambiguous visual acks are discarded. They are never written onto a stale turn and never bound to a newer turn.
- **Orphan visual-ack debt:** `_orphan_visual_ack_debt` is an intentionally uncapped skip counter. Capping it would risk binding a late ack to the wrong later turn.
- **Workflow side-effect retry:** a persisted claim means the mutating step was already attempted. Reconstruction/retry does not re-execute that operation and does not assert that the prior side effect succeeded. A new execution requires `--new-operation` or a new `operation_id`. Oldest claims are dropped after `MAX_WORKFLOW_COMPLETED_EFFECT_KEYS` (200).
- **Provider-side delete:** Realtime `conversation.item.delete` and calendar/keyring revocation remain best-effort.
- **Extreme speech:** very long or adversarial spoken input is bounded and fail-open to the existing response path; it is not a complete abuse-resistant speech stack.
- **Duplicate memories/reminders:** explicit user commands may create another memory or reminder with the same text. Duplicate IDs in a memory file fail closed on load. There is no silent merge of same-text records.
- **DST:** nonexistent and ambiguous local wall times are rejected for reminders and calendar events. Recurring expansions skip DST-invalid local days. This is current behavior, not a limitation to “silently fold.”

## Running tests

Everyday development uses the normal suite, which excludes the Pre-M30
hardening files:

```bash
python -m pytest --ignore-glob="tests/test_pre_m30_*_adversarial.py"
```

`python -m pytest` still runs the full tree, including hardening and extended
stress tests.

Intended tiers:

- **Normal:** existing everyday tests, excluding `tests/test_pre_m30_*_adversarial.py`.
- **Hardening:** `tests/test_pre_m30_hardening_adversarial.py`, the Bug Hunt #2
  domain suites, and Bug Hunt #3 first-run, end-to-end, restart, and
  user-experience suites.
- **Extended stress:** `tests/test_pre_m30_long_session_adversarial.py`.

Windows `%TEMP%\cortana-*` directories from older development are a historical
environment artifact. They are not in git. Current leak tests are green; those
directories can be cleaned on the developer machine separately.
