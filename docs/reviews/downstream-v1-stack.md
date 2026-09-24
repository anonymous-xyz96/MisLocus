# Downstream v1: integration review and merge guide

## Scope and fixed points

Review baseline: producer PR #19, `fa11c25b5d21281b90ba0478ebe6a6f266f81402`.
Scientific reference: publication commit `469f563bca33f3be7eee03385152d3f023011269`.
Donor: the preserved `fix/downstream-repro-v1` working tree at base
`35416e2f71b4ecc025a2f78fb8c41b1a6557fc27`, not a wholesale branch merge.
The integrated code endpoint is `29ef77d775f241b99c83c6f05172835e43eb5c49`.

Review `git diff fa11c25...29ef77d`, plus this documentation layer. Requirements
are the approved downstream handoff/scientific decisions, consolidated in
[the current contract](../downstream-controls-first.md), not a new method choice.
No local coding-standards or issue-tracker guide was present. For optional
issue integration, run `/setup-matt-pocock-skills` separately.

## Dependency-ordered review units

Each PR targets the preceding branch, not main directly. The first depends on
producer PR #19. Merge in order; do not merge a child into its unmerged parent
or squash away a parent without rebasing/retargeting and rechecking child diffs.
Every comparison must remain **strictly below 800 additions + deletions**,
including tests, docs and formatting. No binaries or lockfile changes are hidden
from that count.

| Step | Branch suffix (`review/downstream-v1-`) | Changed lines | Cumulative downstream tests |
|---|---|---:|---:|
| 01 | `01-identities` | 303 | 2 |
| 02 | `02-stages` | 453 | 5 |
| 03 | `03-normalization` | 463 | 10 |
| 04 | `04-preprocessing` | 413 | 12 |
| 05 | `05-gpu-admission` | 379 | 14 |
| 06 | `06-calibration` | 519 | 20 |
| 07 | `07-classifier-lineage` | 512 | 21 |
| 08 | `08-copairs` | 743 | 26 |
| 09 | `09-consumers` | 430 | 29 |
| 10 | `10-acceptance` | 669 | 32 |
| 11 | `11-evidence` | below 800; verify PR diff | unchanged |

Larger units keep coupled boundaries together: plotting callers change with
calibration signatures, executor/writer/CLI change together, copairs traces and
query eligibility share one execution path, and acceptance lands with its full
end-to-end test. Step 05 also updates the old executor to avoid its removed GPU
discovery helper; that temporary legacy execution path disappears in step 07.
Every intermediate layer passed its available tests and classification imports;
steps 05–10 also passed classifier CLI parsing.

## Standards — single-agent review

Matt Pocock standards and spec passes, plus Ponytail complexity review, were
performed separately by one agent. Delegation tools were unavailable. These
are **not independent approvals**; external review is still required.

No unresolved manual standards finding was identified. Reused producer hashing,
source-capture and invocation helpers instead of replacing them with the older
donor versions. Used native cgroups, stdlib executors/locks and installed
libraries; no new scheduler, dependencies or generalized pipeline framework.
Retained necessary validation rather than trimming safeguards to meet line caps.

Configured Ruff **0.16.3** lint and formatting pass for all 29 changed/new Python
files; whitespace checks pass. Formatting of touched legacy files contributes
to the stated totals. Twenty-seven files, including all scientific evaluator
code and the original four test files, are byte-identical to the qualified donor.
Only shared `config.py`/`provenance.py` were reconciled. The newer producer
`read_json_with_hash`, `capture_source`, `invocation(snapshot=None)` and
`verify_source` bodies are AST-identical to the producer base.

## Spec — single-agent review

No unresolved port discrepancy was identified against the approved scope.
The original pre-pilot review's parent-binding, omitted-fit, invalid-cosine,
partial-control, T4-routing, missing-model and numeric-null fixes remain covered.
The later approved missing-reference query exclusion preserves the comparison
pool and refuses missing positives or missing calibration. Producer p95 and BH
fields retain their original meanings; manuscript conjunction is a separate task.

Current evidence: **32 downstream + 4 shared source/crop + 11 producer
provenance/campaign tests passed**, CPU-only in bounded native services. The
32-test endpoint took 37.760 s; service peak was 347 MiB. No downstream tests
skipped; publication checkout parity was available. One compatibility-test
invocation initially omitted the documented vendor PYTHONPATH; the corrected
invocation passed, and the failed harness log was preserved.

The existing six-batch v2 acceptance and 48 receipt pins were rehashed, along
with 60 small score files. This is not a new full payload audit or GPU/production
qualification of the merged source. Old source identities remain authoritative
for old results. No published CI success or independent review is claimed.

## Preservation and exclusions

Original working trees were not reset or cleaned. All 50 donor modified/untracked
files were archived and rehashed unchanged under the local
`analyses/downstream-pr-preparation-v1` evidence root. Its snapshot hash is in
[the evidence JSON](../evidence/downstream-v1-port.json).

Kept local rather than committing stale/duplicate material: the initial training
plan in main, old downstream handoff/pilot drafts and `_orig` backup, one-off
`check_export_admission.py`/`check_xgboost_gpu.py` harnesses, qualification binaries,
logs and older generated reports. Current code, reusable MAD diagnostics, tests
and an explicitly sourced evidence summary are in the stack. Historical files
remain recoverable; no pending code change was silently discarded.

The Overleaf checkout's **836 tracked files are unchanged**. Publication planning
and the eleven-asset reporting refresh remain separate; no manuscript import,
rendering, baseline rerun, new HPA computation or reporting-policy implementation
was authorized by this integration task.
