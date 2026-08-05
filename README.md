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
- Local human-controlled security event, incident, indicator, evidence, note, and timeline foundation
- Local human-supervised defensive tool framework with scope controls, dry-run planning, and approval
- Optional process-isolated execution for a tiny allowlisted defensive tool subset
- Trusted defensive playbook orchestration over allowlisted Milestone 9 tools
- Durable workflow-run and workflow-audit history with optional authorized incident linkage
- Optional controlled security analyst assistance over sanitized single-incident packets

## Conversation history, persistent memory, active context, and documents

| Kind | Lifetime | How it changes | Sent to AI? |
| --- | --- | --- | --- |
| Temporary conversation history | Current session only | Built from chat turns; cleared with `/clear` | Yes, as prior turns |
| Explicit persistent memory | Survives restarts | Saved only through local `/remember` commands | No, unless activated |
| Active memory context | Current session only | Selected with `/recall`; cleared with `/release`, `/release-all`, or restart | Yes, only while active |
| Knowledge Vault documents | Survives restarts | Ingested only through local `/add-document` | No by default |
| Retrieved document passages | Current request/session only | Selected only by `/search-docs` (local) or `/ask-docs` (AI) | Only selected chunks through `/ask-docs` |
| Security incidents and evidence | Survives restarts | Created only through explicit local Milestone 8 commands | Never in this milestone |
| Defensive tool scopes, requests, approvals, results, and audits | Survives restarts | Created only through explicit local Milestone 9 commands | Never in this milestone |
| Workflow/playbook run state and workflow audits | Survives restarts when persistence is enabled; otherwise current process only | Created only through explicit local Milestone 10/11 commands | Never in this milestone |
| Incident AI analysis preparations and results | Current process only | Created only through explicit local Milestone 12 analysis commands | Only through `/incident-analysis-run`, as a sanitized allowlisted packet |

Persistent memories, Knowledge Vault documents, incident records, and evidence copies are stored locally by Project Cortana in user-local application data locations. They are not placed in Git-tracked source directories.

Saved memories remain inactive by default. Nothing from persistent memory is sent to the AI model unless the user explicitly activates it with `/recall` for the current session.

Documents remain inactive by default. No document text is sent to the AI unless the user explicitly invokes `/ask-docs`. Ordinary conversation never reads the Knowledge Vault.

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

## Security event, incident, and evidence foundation

Milestone 8 adds a defensive, documentation-focused local foundation for cybersecurity casework. Records are created or modified only through explicit user commands. This is an audit-support foundation, not a substitute for certified forensic procedure, and it does not claim legal forensic admissibility.

Supported record types and relationships:

- Security events can optionally link to one incident
- Incidents can link many events, indicators, evidence records, and analyst notes
- Indicators store normalized and original values without reputation lookups
- Evidence metadata is stored in the incident repository JSON
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
- `/evidence-verify` recalculates the stored copy hash, compares it to the recorded digest, appends a custody verification entry, and never repairs or deletes evidence automatically.
- Chain-of-custody entries are append-only through normal application commands.
- Repository and evidence storage live in user-local application data, outside the Git repository.
- Atomic JSON writes prevent partial repository files during a single save.
- Atomic writes do not coordinate concurrent Cortana processes. Last-writer-wins lost updates remain possible. Use one application instance per incident repository. Cross-process locking is not implemented yet.
- `/clear` clears conversation history and the grounded source manifest only. It does not delete incidents or evidence.
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
- Arbitrary shell execution is disabled. There is no unrestricted PowerShell, cmd, Bash, Python, or subprocess command interface.
- External tool execution and autonomous remediation are disabled.
- The AI does not select tools and does not execute tools.
- All file-based tools are read-only. They reject symlinks/reparse points and never delete, modify, or execute target files.
- Built-in tools remain bounded and read-only. With process isolation disabled (default), execution uses in-process worker threads: timeouts stop the caller from waiting, but workers are not forcibly terminated, and late completion is never published.
- Milestone 13 optionally enables process-isolated execution for a tiny allowlisted subset (`system-summary`, `simulated-log-check`) via `subprocess.Popen` and schema-validated JSON IPC. File-touching tools remain process-isolation prohibited. See the process-isolation section below.
- Evidence and incident systems remain separate unless a request explicitly links an existing incident ID as metadata.
- Tool-control persistence uses atomic UTF-8 JSON outside the Git repository.
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

File-touching tools (`file-sha256`, `compare-sha256`, `text-search`), repository-backed tools (`incident-summary`), and all other tools remain `process_isolation=prohibited`.

Architecture notes:

- Parent and child communicate with bounded, exact-key, schema-validated JSON only.
- The child entry point is `python -m src.tool_process_runner`.
- Authoritative results use a dedicated parent-created result file channel. stdout and stderr are diagnostic-only and are never parsed as results.
- Every `Popen` call passes an explicit environment allowlist (no full parent env inheritance; no API keys or repository paths).
- On timeout, the first state observed by the parent wins. Late results after termination are never accepted.
- Cancellation is narrow: pre-launch cancelled request status, plus KeyboardInterrupt termination of an active child. No mid-flight cancellation service or worker pool.
- Memory limiting is not implemented. Windows job-object support is deferred. Process termination limits runtime but does not guarantee a hard memory ceiling.
- Antivirus/EDR may delay or block child Python processes. Startup timing is calibrated for Windows process creation. A child could become orphaned if the parent process itself crashes. Process isolation improves terminability; it is not a sandbox against malicious trusted tool code.

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
| `/playbook-run <name> \| <scope-id> \| <incident-id>` | Dry-run one trusted playbook linked to an existing incident |
| `/playbook-run <name> --execute \| <scope-id> \| <incident-id>` | Execute one trusted playbook linked to an existing incident |
| `/playbook-status <run-id>` | Show one workflow run |
| `/incident-analysis-prepare <kind> \| <incident-id> \| <scope-id> \| <event-ids> \| <indicator-ids> \| <note-ids> \| <workflow-run-ids>` | Prepare and preview one sanitized incident analysis packet |
| `/incident-analysis-run <analysis-id>` | Confirm and run AI analysis for one prepared request |
| `/incident-analysis-show <analysis-id>` | Show one in-memory incident analysis result |
| `/incident-analysis-save-note <analysis-id>` | Save one analysis verbatim as an incident note |
| `/about` | Describe Project Cortana and this milestone |
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
- `/ask-docs` sends only selected retrieved chunks, never the entire vault and never incident records.
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
- If the memory file is empty, malformed, or structurally invalid, Cortana returns a clear local error, leaves the file unchanged for inspection, and does not attempt automatic repair in this milestone.
- Atomic writes protect against partial or corrupt files during a single save.
- This milestone assumes a single Cortana instance per memory file. Running multiple instances against the same file may cause last-writer-wins lost updates; cross-process locking is not implemented yet.

## Running tests

```bash
python -m pytest
```
