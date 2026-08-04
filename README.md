# project-cortana

AI-powered authorized cybersecurity and defensive operations platform.

## Overview

Project Cortana is an early software milestone focused on:

- Centralized assistant identity
- Local slash-command handling
- Temporary in-session conversation history
- Explicit, user-controlled persistent memory

## Conversation history vs persistent memory

| Kind | Lifetime | How it changes |
| --- | --- | --- |
| Temporary conversation history | Current session only | Built from chat turns; cleared with `/clear` |
| Explicit persistent memory | Survives restarts | Saved only through local `/remember` commands |

Persistent memories are stored locally by Project Cortana in a user-local application data file. They are not placed in Git-tracked source directories.

Saved memories are **not** sent to the AI model in this milestone. They are not added to system instructions or conversation history yet.

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
| `/about` | Describe Project Cortana and this milestone |
| `/exit` | End the session cleanly |

Notes:

- `/clear` affects only temporary conversation history.
- `/forget-all` affects only persistent memories and requires the explicit `confirm` follow-up.
- Absolute path-like input such as `/etc/passwd` is treated as conversation content for the AI, not as a local command.

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
