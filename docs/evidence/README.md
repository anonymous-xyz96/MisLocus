# SubCell v2 completion evidence

This package is a **portable inventory and bounded audit summary**, not the model
weights, crop release or embedding payloads themselves. The new outputs remain in
external storage; they have not been uploaded to the frozen HF dataset by this PR.
No raw data, private host paths or internal planning history is committed here.

- [`subcell-v2-completion.json`](subcell-v2-completion.json): six native selected
  checkpoints, both complete seed42 all-split exports, source identities, counts,
  metrics, runtime, resource events and the remaining interpretation limits.
- [`subcell-v2-receipts.sha256`](subcell-v2-receipts.sha256): 115 original receipts,
  source snapshots, operational scripts, configuration and verification records.
- [`subcell-v2-payloads.sha256`](subcell-v2-payloads.sha256): 25 large artifacts:
  12 best/last checkpoint paths, the canonical manifest and12 embedding Parquets.
  At pass100 best and last have the same hashes in these runs; extraction was still
  explicitly bound to the native **selected** checkpoint, not an automatic last-file choice.

All inventory paths are relative to an `ARTIFACT_ROOT` containing the sibling
`subcell-training/` and `mislocus-derived/` trees. Verify retained copies without
modifying them, using standard SHA256 tools:

```bash
CODE=/absolute/path/to/this/code-checkout
ARTIFACT_ROOT=/absolute/path/to/artifact-store
cd "$ARTIFACT_ROOT"
sha256sum --check "$CODE/docs/evidence/subcell-v2-receipts.sha256"
# Optional larger read: includes weights and all12 embedding payloads.
sha256sum --check "$CODE/docs/evidence/subcell-v2-payloads.sha256"
```

Missing files fail these checks; a hash is not a download URL. Raw receipts retain
their original operational paths. This inventory relocates **lookups**, not historical
configuration, preflight path bindings or the source identity of a produced artifact.
Do not rewrite original receipts to make them resemble the current PR head.

## What was established

- Production training: commit `03b1961`, code fingerprint
  `1d7ca8218d49cce12fa1ee609777c841d35e103e696e64a7e8a193d480c4f175`.
  Six fresh HPA starts, 100 passes /9,700 updates each, selected pass100, no terminal
  resume allowed. The campaign audit was rerun read-only for this PR: all600 pass
  receipts, all60 validation receipts and every logged learning rate matched the
  approved schedule. Its new report does not overwrite the earlier audit.
- Extraction: commit `e306037`, using the approved seed42 selection receipts, not
  historical gene-trained models. Two exports ×3,332,309 cells ×1,536 finite FP32
  features, all six batches, exact ordered canonical coverage and split labels.
  Full readback and source/input/output verification completed at22:20 UTC on
  2026-09-23. The checksum package rechecked all listed checkpoint/export hashes.
- A fresh crop audit verified17,060 files /436,847,345,580 bytes against the unchanged
  original extraction receipt. The checksum inventories do **not** rehash those
  407GiB of crop payloads; rerun01 `verify` for another point-in-time content audit.
- Clean extraction source:26 tests passed. Bounded GPU pilots covered288 cells/model;
  repeated pilots had maximum absolute embedding difference0. Earlier CPU native
  Lightning trajectory probes had maximum resumed parameter difference0 for both
  tiny model families. These different scopes are not interchangeable.
- Native checkpoint inspection now covers all six final runs: NumPy/CPU-Torch/CUDA
  RNG states agree across the two ranks within each run and differ across seeds
  42/43/44 within each family. The summary preserves the state digests. No GPU
  augmentation trajectory or performance-penalty ablation was rerun for this audit.
- Memory-high reclaim events occurred during extraction; no hard memory-limit or
  OOM events were recorded. Do not interpret the successful run as an absence of
  memory pressure or proof of optimal throughput.

The external producer control directory retains the exact controller/readback scripts,
script archive, fixed run spec, shared locked ledger, pilot/production statuses and
verification receipts. Their hashes are included here; the operational scripts are
not misrepresented as part of the training commit or as a general-purpose framework.
Original source archives and the approved contract remain authoritative for old runs.

## What this does not establish

- No independent parallel reviewer was available; this is direct source/artifact
  review, not independent certification. The [two-axis report](../reviews/subcell-v2-merge-review.md)
  records its scope and merge risks.
- Same-run rank RNG correlation has **not** been removed; its scientific effect is
  unmeasured. Historical missing determinism flags have not been reconstructed.
- Seed42-only exported/downstream results do not demonstrate multi-seed robustness.
  T3 allele AP is neither mislocalization performance nor unseen-allele generalization.
- Matched frozen four-channel bulk exports, downstream policy/implementation review,
  manuscript synchronization and public artifact distribution remain separate work.
  Nothing here certifies XGBoost/copairs results or licenses T4-driven tuning.

Consumers must check `extraction.json` plus the corresponding full verification
receipt and file hashes, then write only into a separate analysis root. A pilot,
partial directory or lone Parquet file is not a complete representation export.
