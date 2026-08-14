# Pre-M30 hardening history

Documentation-only history of the Pre-M30 hardening and deep-fix work.
This is not a runtime module and is not imported by `src/`.

Authoritative hardened baseline commit:

`fb4a6188cab86796157ea4fdab9147a1e17ee1a7`

Batches 1–6 in this working tree are uncommitted relative to that baseline.
This file records what was validated locally. It does not invent external
or CI evidence.

## How to read the numbers

- **Locally reproducible:** a command in `docs/TEST_MATRIX.md` was run on
  this tree and the count below is that output.
- **Manual / external:** not claimed. No packaged outside-review run is
  checked in, and no “18 targeted tests × 20 iterations” harness exists.
- Counts change when permanent tests are added. Prefer the latest local
  command over any older number in chat notes.

## Baseline (commit `fb4a618`)

The frozen baseline is the commit above. Batches 1–6 are later uncommitted
fixes on that tree. This file does not restate unverifiable pre-baseline
package counts.

## Batch 1–5 (same uncommitted tree)

Those batches remain in the working tree. They are not restated as a
separate git history because they have not been committed. Their product
changes are already present when Batch 6 is validated.

Pre-Batch-6 local counts on that same tree were:

| Suite | Result |
| --- | --- |
| Normal | 1212 passed, 6 skipped |
| Hardening | 79 passed |
| Bug Hunt #2 | 90 passed |
| Bug Hunt #3 | 68 passed |
| Deep Audit #4 | 91 passed, 4 failed |
| Targeted subsystem (Batch 5 persistence/calendar set) | 263 passed, 2 skipped |
| Full tree | 1540 passed, 4 failed, 6 skipped |
| `python -m mypy --strict src` | Success, 116 source files |

Those four Deep Audit failures were the Batch 6 scope:

1. workflow rerun / side-effecting step idempotency
2. misleading M25 cleanup test name
3. misleading authority / persistent-memory test names
4. missing `TEST_MATRIX.md` / `PRE_M30_HARDENING_HISTORY.md`

## Batch 6 (this file’s current tree)

Product change: side-effecting workflow steps persist a bounded operation
claim before live execution. Read-only built-in playbooks are unchanged.
Test-quality renames and this documentation provenance were added.

Locally reproducible after Batch 6, including after these documentation
files were added so Deep Audit #4 could see them:

| Suite | Result | Kind |
| --- | --- | --- |
| Normal | 1228 passed, 6 skipped | Local |
| Hardening | 79 passed | Local |
| Bug Hunt #2 | 90 passed | Local |
| Bug Hunt #3 | 68 passed | Local |
| Deep Audit #4 | 95 passed | Local |
| Targeted workflow/tool | 134 passed | Local |
| Full tree | 1560 passed, 6 skipped | Local |
| `python -m mypy --strict src` | Success, 116 source files | Local |

## What Batch 6 did not do

- Did not begin Milestone 30.
- Did not freeze, commit, or push.
- Did not add a new mutating production playbook. Built-in tools remain
  `internal-readonly`.
- Did not claim an unverifiable soak (`18 × 20` or similar) unless a
  command and test list were captured. They were not.
