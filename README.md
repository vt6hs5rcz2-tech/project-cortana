# project-cortana

AI-powered authorized cybersecurity and defensive operations platform.

## Overview

Project Cortana is an early software milestone focused on:

- Centralized assistant identity
- Local slash-command handling
- Temporary in-session conversation history
- Explicit, user-controlled persistent memory
- Explicit, session-only active memory context for AI requests
- Local Knowledge Vault for explicit document ingestion and inspection
- Deterministic local lexical document retrieval
- Explicit source-grounded AI questions over retrieved document passages

## Conversation history, persistent memory, active context, and documents

| Kind | Lifetime | How it changes | Sent to AI? |
| --- | --- | --- | --- |
| Temporary conversation history | Current session only | Built from chat turns; cleared with `/clear` | Yes, as prior turns |
| Explicit persistent memory | Survives restarts | Saved only through local `/remember` commands | No, unless activated |
| Active memory context | Current session only | Selected with `/recall`; cleared with `/release`, `/release-all`, or restart | Yes, only while active |
| Knowledge Vault documents | Survives restarts | Ingested only through local `/add-document` | No by default |
| Retrieved document passages | Current request/session only | Selected only by `/search-docs` (local) or `/ask-docs` (AI) | Only selected chunks through `/ask-docs` |

Persistent memories and Knowledge Vault documents are stored locally by Project Cortana in user-local application data files. They are not placed in Git-tracked source directories.

Saved memories remain inactive by default. Nothing from persistent memory is sent to the AI model unless the user explicitly activates it with `/recall` for the current session.

Documents remain inactive by default. No document text is sent to the AI unless the user explicitly invokes `/ask-docs`. Ordinary conversation never reads the Knowledge Vault.

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
- Documents are not sent to the AI model unless selected chunks are explicitly requested through `/ask-docs`.
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

Lexical retrieval matches words and phrases present in stored text. Semantic retrieval would attempt meaning-based similarity; it is disabled in this milestone.

## Source-grounded answers

`/ask-docs <question>` is the only command that retrieves document passages and sends selected chunks to the AI.

Citation labels use a compact deterministic format such as `[DOC-1:C1]`. Each label maps through a session source manifest to:

- document ID
- filename
- chunk index
- character range

Citation-label validation checks that labels in the AI response match the exact labels supplied with the request. It does **not** prove full factual entailment of every natural-language claim against the source text.

Source manifests:

- Exist only in memory for the current session
- Are shown with `/sources`
- Are cleared by `/clear`, `/remove-all-documents confirm`, and application restart
- Are not written to disk
- Do not expose local file paths or vault paths

Active-memory context and retrieved-document context can appear in the same AI request, but remain separate structured developer messages.

## Local commands

| Command | Description |
| --- | --- |
| `/help` | List available commands |
| `/status` | Show safe local session information |
| `/clear` | Clear temporary conversation history and the latest grounded source manifest |
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
| `/sources` | Show sources from the latest successful `/ask-docs` request |
| `/about` | Describe Project Cortana and this milestone |
| `/exit` | End the session cleanly |

Notes:

- `/clear` affects temporary conversation history and the latest `/sources` manifest. It does not delete documents or persistent memories, and does not remove active memories.
- `/remember` saves a persistent memory but does not activate it or ingest documents.
- `/release` and `/release-all` clear temporary active context only; they do not delete persistent memories.
- `/forget` deletes a persistent memory and also removes it from active context when selected.
- `/forget-all confirm` deletes all persistent memories and clears active memory selections.
- `/remove-document` removes one vault document and invalidates matching source-manifest entries.
- `/remove-all-documents confirm` deletes all vault documents and clears the source manifest. It does not affect persistent memories or active-memory context.
- `/search-docs` never calls the AI service.
- `/ask-docs` sends only selected retrieved chunks, never the entire vault.
- Absolute path-like input such as `/etc/passwd` is treated as conversation content for the AI, not as a local command, unless it is explicitly provided as an `/add-document` argument.

## Active memory limits

Active memory context is limited by centralized defaults:

- Maximum of 10 active memories
- Maximum of 8,000 combined characters of active memory text

Activation is rejected with a clear local message when either limit would be exceeded. Existing active selections are not silently removed, and saved memory text is not silently truncated.

## Memory storage and corruption handling

- Memories are saved only through explicit local commands.
- Storage uses a local JSON file under the user-local application data directory.
- If the memory file is missing, Cortana starts with an empty memory list.
- If the memory file is empty, malformed, or structurally invalid, Cortana returns a clear local error, leaves the file unchanged for inspection, and does not attempt automatic repair in this milestone.
- Atomic writes protect against partial or corrupt files during a single save.
- This milestone assumes a single Cortana instance per memory file. Running multiple instances against the same file may cause last-writer-wins lost updates; cross-process locking is not implemented yet.

## Running tests

```bash
python -m pytest
```
