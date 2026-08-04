# project-cortana

AI-powered authorized cybersecurity and defensive operations platform.

## Overview

Project Cortana is an early software milestone focused on:

- Centralized assistant identity
- Local slash-command handling
- Temporary in-session conversation history
- Explicit, user-controlled persistent memory
- Explicit, session-only active memory context for AI requests

## Conversation history, persistent memory, and active context

| Kind | Lifetime | How it changes | Sent to AI? |
| --- | --- | --- | --- |
| Temporary conversation history | Current session only | Built from chat turns; cleared with `/clear` | Yes, as prior turns |
| Explicit persistent memory | Survives restarts | Saved only through local `/remember` commands | No, unless activated |
| Active memory context | Current session only | Selected with `/recall`; cleared with `/release`, `/release-all`, or restart | Yes, only while active |

Persistent memories are stored locally by Project Cortana in a user-local application data file. They are not placed in Git-tracked source directories.

Saved memories remain inactive by default. Nothing from persistent memory is sent to the AI model unless the user explicitly activates it with `/recall` for the current session.

Active memory selections:

- Exist only in memory for the current session
- Reset when the application closes
- Are not written to disk
- Are injected as structured, untrusted user-provided reference data
- Use session-specific reference boundaries that are not persisted
- Cannot replace Cortana’s system identity instructions

## Local commands

| Command | Description |
| --- | --- |
| `/help` | List available commands |
| `/status` | Show safe local session information |
| `/clear` | Clear temporary conversation history for this session |
| `/remember <text>` | Save one explicit persistent memory |
| `/memories` | List saved persistent memories |
| `/forget <memory-id>` | Delete one saved memory by ID |
| `/forget-all` | Begin deletion of all saved memories |
| `/forget-all confirm` | Confirm deletion of all saved memories |
| `/recall <memory-id>` | Activate one saved memory for temporary AI context |
| `/active-memories` | List memories currently active for AI context |
| `/release <memory-id>` | Remove one memory from active AI context |
| `/release-all` | Clear all active AI memory context for this session |
| `/about` | Describe Project Cortana and this milestone |
| `/exit` | End the session cleanly |

Notes:

- `/clear` affects only temporary conversation history and does not remove active memories.
- `/remember` saves a persistent memory but does not activate it.
- `/release` and `/release-all` clear temporary active context only; they do not delete persistent memories.
- `/forget` deletes a persistent memory and also removes it from active context when selected.
- `/forget-all confirm` deletes all persistent memories and clears active memory selections.
- Absolute path-like input such as `/etc/passwd` is treated as conversation content for the AI, not as a local command.

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
