# Downstream controls-first execution

## Status and scope

The MAE-s42/ViT-s42 six-batch campaign **completed and passed acceptance on
2026-09-24 at 15:15:38 UTC**, with 48/48 stages. Do not restart it.
The immutable campaign is `downstream-s42-all-v2`; its acceptance SHA256 is
`ce9aa1be8da53b10b76d7c19ce756b9978282548d37dc48675a4c2511fcdfe51`.
See [port evidence](evidence/downstream-v1-port.json) and the
[review/stack guide](reviews/downstream-v1-stack.md).

Per model: 3,330,277 retained cells, 354 control fits, 1,824 experimental fits,
1,617 XGBoost summary rows, 1,913 PA results and 275 LOO control results.
These are computational results, not manuscript/biological approval.

This port selectively integrates the downstream implementation onto producer
stack tip `fa11c25b5d21281b90ba0478ebe6a6f266f81402`. It preserves newer
producer provenance helpers and the existing lockfile. Historical acceptance
belongs to its archived source, **not** to this newer integration commit.
Publication methods reference: `469f563bca33f3be7eee03385152d3f023011269` in
`broadinstitute/2026_03_03_prot_loc_rep_benchmark`.

No producer retraining, extraction, new scientific campaign, frozen-model
benchmark, extra encoder seeds or manuscript changes are part of this port.

## Input and preprocessing contract

- Set an absolute, fresh `MISLOCUS_DATA_ROOT`, outside producer exports and the
  frozen release. Annotations remain repository-relative. Do not link whole
  producer directories into writable analysis roots; file-only raw-input links
  are supported.
- New `subcell_allele_rybg_v2_*` inputs require `--extraction-spec`. Admission
  binds producer completion, independent verification, specification, source,
  checkpoint/pass, selection, preflight/crop evidence and raw artifact hashes.
- Require 1,536 ordered finite Float32 `SubCell_*` coordinates, original local
  and batch-qualified IDs, unique coordinates, consistent well metadata,
  exact ordered identities/split counts and all T1–T4 plates.
- Preserve the reference embedding path: cell/well QC → physical-plate
  RobustMAD → batch std dead-feature filter → ±100 guard → single-pass
  decorrelation → control annotation. CP retains its separate feature selection,
  blocklists and GFP-aware two-pass decorrelation.
- Require complete, unique, finite plate statistics with nonnegative MAD.
  Preserve original row order/metadata; reject nonfinite outputs. The reference
  zero-MAD pass-through is unchanged. Never renormalize accepted features.
- Save cell mappings, feature order, cohort hashes and explicit QC exclusions.

The six-batch raw audit covered 3,332,309 cells/model and 56 plates. Neither new
model had invalid or zero MADs; maximum absolute normalized bounds were
24.516739 (MAE) and 28.308138 (ViT), so ±100 clipping changes no values here.
This does not validate every historical representation's normalization policy.
`docs/checks/audit_plate_mad.py` and `audit_export_mad.py` remain read-only
statistics diagnostics, not alternative preprocessing pipelines.

## Authoritative completion and lineage

`started.json` exclusively claims a stage; old/partial artifacts cannot be
reused implicitly. Caught failures preserve `failed.json`; abrupt termination
may leave only the start marker. **Only `stage.json`, published last, is stage
completion.** Merely finding a feature or score table is insufficient.

Receipts bind input/parent/output hashes, model/batch, UUID stage identity,
actual archived source, runtime/build identity, resource settings and initial/
final cgroup state. Startup and completion recheck dependencies. Consumers
reject incomplete, changed or mismatched parents. Independent provenance
ledger entries use unique stage IDs.

Prediction keys are `(AnalysisStageID, Representation, Classifier_ID,
Metadata_BatchQualifiedCellID)`, not plate/well/object alone. Preserve image,
site, plate, well, object and original extraction split. `fold_membership.parquet`
records actual train/test roles. `classifier_info.csv` and native `models/*.ubj`
bind ordered cell hashes and feature order; `allele_inventory.parquet` accounts
for retained alleles even when reference support prevents pair construction.
Declared eligible fits cannot silently disappear.

`Prediction = P(Label=1)` means reference allele/well, **not pathogenicity**.
Encoder T1/T2 training and T3 validation differ from the evaluator's T1–T3
training/T4 test roles. This is technical-replicate transfer, not unseen-allele
generalization.

## Scientific evaluation rules

### XGBoost

- Default direct T1–T3 → T4. Multiple platemaps may yield multiple fits;
  aggregate with `min_classifiers=1`; undefined single-fit SD stays numeric null.
- Run `--scope control` before experimental `--scope all` (Exp+cPC). NC+PC
  same-allele/different-well comparisons supply calibration; cPC is not null.
  Four supported wells yield six pairs per allele, separately for ALK and
  ALK_Arg1275Gln. Unsupported pairs remain explicit exclusions.
- Finite eligible control AUROCs; empirical p95 with **nearest** interpolation;
  strict `AUROC > threshold`. No AUROC=0.5 calibration fallback.
- Bind exact features/path/order, source, parameters, protocol, backend/build,
  allocation, workers and threads. Never swap CPU and GPU calibrations.
- XGBoost seed is **0**, independent of encoder/copairs seed 42.
  Legacy `--test-split none` requires its own calibration; it is not T4 acceptance.

### Copairs / phenotypic activity

- Median FOV profiles; full comparison pool; T4 queries. Same-allele/different-
  plate positives; same-plate/same-gene reference negatives; no reference cap.
- NC+PC leave-one-well-out control null, not new random two-well sampling.
  Seed 42; 10,000 variant and 1,000 control null draws; empirical p95 uses
  **linear** interpolation.
- Producer `is_hit` remains strict **p95-only**. Permutation/BH fields remain
  separate. The selected manuscript reporting rule is **p95 AND BH**, applied
  within batch before replicate collapse using a separate reporting flag.
  That reporting adapter is not implemented by this stack; do not overwrite
  producer fields or describe joint filtering as calibrated biological 5% FDR.
- Skip only queries lacking the same-plate reference in the actual pool,
  preserving all comparison profiles and other supported plates. Record
  `excluded_queries.parquet` and `missing_same_plate_reference`; all-excluded
  calls are `not_estimable`, never fabricated AP=1. Missing positives, invalid
  cosine norms, nonfinite eligible AP and absent calibration still fail.
- Every call records profile/member identities, full pool, AP, queries,
  parameters, outcome and isolated null cache. Unexpected control errors abort.

The approved missing-reference rule fixed the B7 MCEE stop after existing
minimum-20-cell QC removed P3 references. Failed v1 and accepted pilot artifacts
remain intact. V2 explicitly reused nine unchanged older preprocessing/XGBoost
receipts and reran all twelve copairs stages. `--reuse-manifest` permits only
canonical absolute receipt paths with exact hashes; no validation is bypassed.

## Execution and validation

Use native bounded systemd units, capped nested pools and serial campaign jobs,
not a pressure-based scheduler or automatic partial retry. New SubCell stages
require both finite job limits and a shared `mislocus-downstream-*.slice`.
The guard ceiling (64 CPUs, 384/512 GiB high/hard RAM, 2,048 tasks) is not a
recommended allocation. The accepted campaign used **16 CPUs, 96/128 GiB,
512 tasks**, with serial jobs and two XGBoost workers/one thread on one GPU.

GPU execution requires the locked GPU environment, exactly one explicit
`CUDA_VISIBLE_DEVICES` allocation, CUDA-enabled XGBoost and a private
`XDG_RUNTIME_DIR`. Advisory leases reject existing clients but cannot reserve
against external schedulers. Actual booster device is checked; no CPU fallback.
B15 measured ~27.8×/28.4× throughput versus its serial one-thread CPU baseline;
independently trained CPU/GPU predictions differ. This is historical pilot
qualification, not optimized-CPU equivalence or a new GPU test of this port.

Example **CPU test** setup, not production launch authorization:

```bash
pixi install --locked
SLICE=mislocus-downstream-checks-v1.slice
systemctl --user set-property --runtime "$SLICE" \
  CPUQuota=400% MemoryHigh=24G MemoryMax=32G TasksMax=256
systemd-run --user --wait --pipe --collect --slice="$SLICE" \
  --working-directory="$PWD" \
  -p CPUQuota=400% -p MemoryHigh=24G -p MemoryMax=32G -p TasksMax=256 \
  env PATH="$PWD/.pixi/envs/default/bin:$PATH" PYTHONPATH="$PWD/src" \
  PYTHONNOUSERSITE=1 MISLOCUS_CAMPAIGN_SLICE="$SLICE" \
  MISLOCUS_CLASSIFIER_BACKEND=cpu CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  POLARS_MAX_THREADS=2 NUMEXPR_NUM_THREADS=1 \
  "$PWD/.pixi/envs/default/bin/python" -m unittest discover \
  -s tests -p 'test_downstream_*.py'
```

The reference-parity test requires the pinned read-only sibling checkout
`../prot-loc-publish-readiness-reference`; otherwise it explicitly skips.
The recorded port validation had that checkout and no skipped downstream tests.

`09f_verify_downstream.py` validates all four stages/model/batch, sources,
artifacts, inputs, parents, scopes, protocol and model/membership bindings before
writing fresh `acceptance.json`. Synthetic tests replay saved predictions;
the verifier itself does not recompute every production prediction. Do not run
it over an already accepted root or relabel accepted artifacts with new source.

`10_benchmark_clinvar.py --fold-mode t4-only` requires complete requested
biological pairs; `--pa` loads verified T4-query PA. Missing requested models
fail, and undefined SD/AUPRC stay numeric null. `09e_summarize_t4.py` is a guarded
legacy post-hoc utility, not the new campaign path. Historical baselines require
explicit historical admission, never invented stage receipts.

Paper-refresh planning, HPA for the two new models and item-by-item manuscript
approval remain separate. Preserve main-any/supplement-all organization; do not
silently decide complete-pair eligibility or multiple-testing families here.
