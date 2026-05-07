#!/usr/bin/env python3
"""Preprocess profiles (CellProfiler or deep-embedding) for variant classification.

**CellProfiler pipeline** (``--representation cellprofiler``, default):
 1. Filter to manifest-approved cells (cell QC gate)
 2. Drop low-cell-count wells
 3. Remove NaN features/rows
 4. Compute per-plate statistics
 5. Select variant features (MAD≠0, abs_coef_var > threshold)
 6. RobustMAD normalization (per-plate)
 7. Clip outliers (±threshold)
 8a. Apply blocklists (Neighbors + pycytominer blocklist drops)
 8b. Save normalized.parquet (post-clipping, blocklist-filtered)
 8c. Variance threshold (pycytominer heuristic cleanup)
 9. Decorrelate features (two-pass GFP-aware correlation threshold)
 10. Annotate controls (TC, NC, PC, cPC)

**Deep-embedding pipeline** (``--representation {cytoself, subcell_portable_*,
vit}``): core preprocessing only. Learned embeddings have
different statistical properties than CellProfiler features (zero-centered,
continuous floats, no stain-channel semantics), so several CellProfiler steps
either break or no-op on them. A preamble aliases Metadata_well_position →
Metadata_Well if the raw extractor output uses only the former; features.parquet
is written at the end. The nine numbered pipeline steps:
 1. Drop low-cell-count wells (<20)
 2. Drop NaN features/rows (usually a no-op)
 3. Compute per-plate statistics
 4. RobustMAD normalization (per-plate)
 5. Drop zero-std features (principled substitute for select_variant_features;
    absolute std threshold — see EMBEDDING_STD_EPSILON for why std, not MAD)
 6. Clip outliers (±threshold)
 7. Save normalized.parquet
 8. Single-pass decorrelation across all features (channel_aware=False)
 9. Annotate controls

Skipped for embeddings, with their actual reasons:

- ``filter_to_manifest`` — embeddings are already manifest-filtered during
  extraction (07/08 scripts); rerunning would be an expensive no-op.

- ``select_variant_features`` — two separate failure modes on embeddings.
  First, its ``abs_coef_var = |MAD/median|`` has a singularity at ``median = 0``
  (the ``.replace(inf, 0)`` guard rewrites those to 0 and drops them). Second
  and more importantly, its ``mad != 0`` gate misfires on sparse count features:
  Cytoself_spectrum_* are per-cell VQ1 codebook-index histograms where each cell
  activates only ~16 of 512 bins, so median=0 and MAD=0 by construction even
  when std > 0 and the bin carries real signal. Running it unmodified would
  drop ~500 legitimate spectrum features. Replaced by Step 5 (absolute std,
  which correctly measures spread for both continuous and zero-inflated data).

- ``apply_blocklists`` — no-op on embeddings. Its 54 pycytominer-hardcoded
  literal column names (``Nuclei_Correlation_Manders_*``, ``Nuclei_Granularity_14-16_*``,
  etc.) and the custom ``_Neighbors_`` pattern match zero embedding columns.
  Skipping saves a pandas pass but would be harmless to run.

- ``apply_variance_threshold`` — actively harmful post-clip. Its ``freq_cut=0.05``
  rule compares ``count(2nd_most_common) / count(most_common)``. On continuous
  embedding floats each cell's value is unique (both counts ≈ 1) pre-clip, so
  it finds nothing. Post-clip, ~10K cells per affected feature get pinned to
  exactly ±100.0, producing ``count(100.0) ≈ 10000, count(other) = 1``, ratio
  ``≈ 1e-4 < 0.05`` → flagged. On Cytoself B13 this would drop ~1080/1536
  features via false positive. Same CellProfiler GFP-intensity bug at scale.

Usage:
    # CellProfiler (default)
    pixi run python scripts/06_preprocess_profiles.py --batch 2025_01_27_Batch_13
    pixi run python scripts/06_preprocess_profiles.py --batch 2025_01_27_Batch_13 --normalized-only

    # Deep embeddings
    pixi run python scripts/06_preprocess_profiles.py --batch 2025_01_27_Batch_13 --representation cytoself
    pixi run python scripts/06_preprocess_profiles.py --batch 2025_01_27_Batch_13 --representation subcell_portable_bg_vit
"""

import argparse
import logging
import sys
import time

import polars as pl

from prot_loc_benchmark.config import (
    BATCH_CONTROLS,
    CELLPROFILER_DIR,
    CPC_GENE_ALLELES,
    CROP_MANIFEST_DIR,
    INTERIM_DIR,
    PREPROCESS_CC_THRESHOLD,
    PREPROCESS_NAN_THRESHOLD,
    PREPROCESS_OUTLIER_THRESHOLD,
    PREPROCESS_VARIANT_ACV_THRESHOLD,
    REP_RAW_FILES,
)
from prot_loc_benchmark.preprocessing import (
    annotate_controls,
    apply_blocklists,
    apply_variance_threshold,
    clip_outliers,
    compute_plate_stats,
    decorrelate_features,
    drop_dead_features,
    drop_low_cell_count_wells,
    drop_nan_features,
    filter_to_manifest,
    robustmad,
    select_variant_features,
)
from prot_loc_benchmark.preprocessing.normalize import EMBEDDING_STD_EPSILON


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def preprocess_batch(batch_id: str, normalized_only: bool = False) -> None:
    """Run the full preprocessing pipeline for one batch."""
    t0 = time.time()

    profiles_path = CELLPROFILER_DIR / batch_id / "profiles.parquet"
    manifest_path = CROP_MANIFEST_DIR / batch_id / "manifest.parquet"
    features_path = CELLPROFILER_DIR / batch_id / "features.parquet"
    plate_stats_path = CELLPROFILER_DIR / batch_id / "plate_stats.parquet"
    normalized_path = CELLPROFILER_DIR / batch_id / "normalized.parquet"

    if not profiles_path.exists():
        logger.error("Profiles not found: %s", profiles_path)
        sys.exit(1)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    controls = BATCH_CONTROLS.get(batch_id)
    if controls is None:
        logger.error("No control config for batch %s in BATCH_CONTROLS", batch_id)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Preprocessing batch: %s", batch_id)
    logger.info("=" * 70)

    # ── Step 1: Filter to manifest (cell QC gate) ──────────────────────
    logger.info("Step 1/10: Filter to manifest...")
    lf = filter_to_manifest(str(profiles_path), str(manifest_path))

    # ── Step 2: Drop low-cell-count wells ──────────────────────────────
    logger.info("Step 2/10: Drop low-cell-count wells (<%d)...", PREPROCESS_CC_THRESHOLD)
    lf = drop_low_cell_count_wells(lf, cc_threshold=PREPROCESS_CC_THRESHOLD)

    # ── Step 3: Remove NaN features/rows ───────────────────────────────
    logger.info("Step 3/10: Remove NaN features/rows...")
    lf = drop_nan_features(lf, cell_threshold=PREPROCESS_NAN_THRESHOLD)

    # ── Step 4: Compute plate statistics ───────────────────────────────
    logger.info("Step 4/10: Computing plate statistics...")
    plate_stats = compute_plate_stats(lf)
    plate_stats.write_parquet(str(plate_stats_path))
    logger.info("Plate stats saved: %s", plate_stats_path)

    # ── Step 5: Select variant features ────────────────────────────────
    logger.info("Step 5/10: Selecting variant features...")
    lf = select_variant_features(
        lf, plate_stats, acv_threshold=PREPROCESS_VARIANT_ACV_THRESHOLD
    )

    # ── Step 6: RobustMAD normalization ────────────────────────────────
    logger.info("Step 6/10: RobustMAD normalization...")
    lf = robustmad(lf, plate_stats)

    # ── Step 7: Clip outliers ──────────────────────────────────────────
    logger.info("Step 7/10: Clipping outliers (±%.0f)...", PREPROCESS_OUTLIER_THRESHOLD)
    lf = clip_outliers(lf, threshold=PREPROCESS_OUTLIER_THRESHOLD)

    # ── Step 8a: Apply blocklists (unconditional drops) ─────────────────
    logger.info("Step 8a/10: Apply blocklists (Neighbors + pycytominer)...")
    # pycytominer requires pandas
    df = lf.collect().to_pandas()
    df = apply_blocklists(df)

    # ── Step 8b: Save normalized.parquet ──────────────────────────────
    # Snapshot after plate normalization + outlier clipping + blocklists,
    # but BEFORE pycytominer's variance_threshold (which has a known
    # false-positive interaction with clip_outliers — it drops features
    # like Cells_Intensity_MeanIntensity_GFP whose clipped-tail mode
    # trips pycytominer's freq_cut heuristic, despite healthy variance).
    #
    logger.info("Step 8b/10: Saving normalized.parquet...")
    pl.from_pandas(df).write_parquet(str(normalized_path), compression="zstd")
    n_norm_feats = len([c for c in df.columns if not c.startswith("Metadata_")])
    logger.info("Normalized saved: %s (%d features)", normalized_path, n_norm_feats)

    if normalized_only:
        elapsed = time.time() - t0
        logger.info("=" * 70)
        logger.info("Normalized-only mode: stopping after blocklists.")
        logger.info("  Output: %s", normalized_path)
        logger.info("  Features: %d", n_norm_feats)
        logger.info("  Time: %.1f min", elapsed / 60)
        logger.info("=" * 70)
        return

    # ── Step 8c: Variance threshold (heuristic cleanup) ───────────────
    logger.info("Step 8c/10: Variance threshold...")
    df = apply_variance_threshold(df)

    # ── Step 9: Decorrelate features (expensive two-pass correlation) ──
    logger.info("Step 9/10: Decorrelate features (two-pass correlation)...")
    df = decorrelate_features(df)

    # ── Step 10: Annotate controls ─────────────────────────────────────
    logger.info("Step 10/10: Annotating controls...")
    lf = pl.from_pandas(df).lazy()
    lf = annotate_controls(
        lf,
        tc=controls["TC"],
        nc=controls["NC"],
        pc=controls["PC"],
        cpc_gene_alleles=CPC_GENE_ALLELES,
    )

    # ── Save output ────────────────────────────────────────────────────
    result = lf.collect()
    result.write_parquet(str(features_path), compression="zstd")

    feat_cols = [c for c in result.columns if not c.startswith("Metadata_")]
    elapsed = time.time() - t0

    logger.info("=" * 70)
    logger.info("Preprocessing complete: %s", batch_id)
    logger.info("  Output: %s", features_path)
    logger.info("  Cells: %d", result.height)
    logger.info("  Features: %d", len(feat_cols))
    logger.info("  Time: %.1f min", elapsed / 60)
    logger.info("=" * 70)


def preprocess_embedding_batch(
    batch_id: str,
    representation: str,
    normalized_only: bool = False,
) -> None:
    """Run the deep-embedding preprocessing pipeline for one batch + rep.

    Core preprocessing (plate-normalization + single-pass decorrelation) applied
    to learned embeddings. See the module docstring for the full step list.
    Writes three parquets to ``data/interim/{rep}/{batch}/``:
    ``plate_stats.parquet``, ``normalized.parquet``, ``features.parquet``.
    """
    t0 = time.time()

    raw_file = REP_RAW_FILES.get(representation)
    if raw_file is None:
        logger.error(
            "No raw-embedding file mapping for representation %r. "
            "Add an entry to REP_RAW_FILES in src/prot_loc_benchmark/config.py.",
            representation,
        )
        sys.exit(1)

    rep_dir = INTERIM_DIR / representation / batch_id
    input_path = rep_dir / raw_file
    plate_stats_path = rep_dir / "plate_stats.parquet"
    normalized_path = rep_dir / "normalized.parquet"
    features_path = rep_dir / "features.parquet"

    if not input_path.exists():
        logger.error("Raw embedding not found: %s", input_path)
        sys.exit(1)

    controls = BATCH_CONTROLS.get(batch_id)
    if controls is None:
        logger.error("No control config for batch %s in BATCH_CONTROLS", batch_id)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Preprocessing embedding batch: %s (rep=%s)", batch_id, representation)
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", rep_dir)
    logger.info("=" * 70)

    # ── Load + alias Metadata_well_position → Metadata_Well if needed ────
    # TODO: de-dupe with scripts/09c_classify_PA.py:_resolve_well_col by moving
    # an alias helper into prot_loc_benchmark.config or a shared util.
    lf = pl.scan_parquet(str(input_path))
    cols = lf.collect_schema().names()
    if "Metadata_Well" not in cols:
        if "Metadata_well_position" not in cols:
            logger.error(
                "Input has neither Metadata_Well nor Metadata_well_position; "
                "cannot apply well-level filters."
            )
            sys.exit(1)
        logger.info("Aliasing Metadata_well_position → Metadata_Well")
        lf = lf.with_columns(pl.col("Metadata_well_position").alias("Metadata_Well"))

    # ── Step 1: Drop low-cell-count wells ───────────────────────────────
    logger.info("Step 1/9: Drop low-cell-count wells (<%d)...", PREPROCESS_CC_THRESHOLD)
    lf = drop_low_cell_count_wells(lf, cc_threshold=PREPROCESS_CC_THRESHOLD)

    # ── Step 2: Remove NaN features/rows (usually a no-op) ───────────────
    logger.info("Step 2/9: Remove NaN features/rows...")
    lf = drop_nan_features(lf, cell_threshold=PREPROCESS_NAN_THRESHOLD)

    # ── Step 3: Compute plate statistics ────────────────────────────────
    logger.info("Step 3/9: Computing plate statistics...")
    plate_stats = compute_plate_stats(lf)
    plate_stats.write_parquet(str(plate_stats_path))
    logger.info("Plate stats saved: %s", plate_stats_path)

    # ── Step 4: RobustMAD normalization ─────────────────────────────────
    logger.info("Step 4/9: RobustMAD normalization...")
    lf = robustmad(lf, plate_stats)

    # ── Step 5: Drop zero-std features (embedding-only substitute for
    # select_variant_features; std handles sparse Cytoself_spectrum_* dims
    # where median=0 makes MAD=0 even with real signal).
    logger.info("Step 5/9: Drop zero-std features (std < %.0e)...", EMBEDDING_STD_EPSILON)
    lf = drop_dead_features(lf, epsilon=EMBEDDING_STD_EPSILON)

    # ── Step 6: Clip outliers ───────────────────────────────────────────
    logger.info("Step 6/9: Clipping outliers (±%.0f)...", PREPROCESS_OUTLIER_THRESHOLD)
    lf = clip_outliers(lf, threshold=PREPROCESS_OUTLIER_THRESHOLD)

    # ── Step 7: Save normalized.parquet ─────────────────────────────────
    logger.info("Step 7/9: Saving normalized.parquet...")
    normalized_df = lf.collect()
    normalized_df.write_parquet(str(normalized_path), compression="zstd")
    n_norm_feats = len([c for c in normalized_df.columns if not c.startswith("Metadata_")])
    logger.info("Normalized saved: %s (%d features)", normalized_path, n_norm_feats)

    if normalized_only:
        elapsed = time.time() - t0
        logger.info("=" * 70)
        logger.info("Normalized-only mode: stopping before decorrelation.")
        logger.info("  Time: %.1f min", elapsed / 60)
        logger.info("=" * 70)
        return

    # ── Step 8: Single-pass decorrelation (no channel split) ────────────
    logger.info("Step 8/9: Decorrelate features (single-pass, all features)...")
    df = decorrelate_features(normalized_df.to_pandas(), channel_aware=False)

    # ── Step 9: Annotate controls ───────────────────────────────────────
    logger.info("Step 9/9: Annotating controls...")
    result = annotate_controls(
        pl.from_pandas(df).lazy(),
        tc=controls["TC"],
        nc=controls["NC"],
        pc=controls["PC"],
        cpc_gene_alleles=CPC_GENE_ALLELES,
    ).collect()

    # ── Save features.parquet ───────────────────────────────────────────
    result.write_parquet(str(features_path), compression="zstd")

    feat_cols = [c for c in result.columns if not c.startswith("Metadata_")]
    elapsed = time.time() - t0

    logger.info("=" * 70)
    logger.info("Embedding preprocessing complete: %s (%s)", batch_id, representation)
    logger.info("  Output:   %s", features_path)
    logger.info("  Cells:    %d", result.height)
    logger.info("  Features: %d", len(feat_cols))
    logger.info("  Time:     %.1f min", elapsed / 60)
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess profiles (CellProfiler or deep-embedding) for variant classification."
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch ID (e.g., 2025_01_27_Batch_13)",
    )
    parser.add_argument(
        "--representation",
        default="cellprofiler",
        choices=["cellprofiler"] + sorted(REP_RAW_FILES),
        help=(
            "Which representation to preprocess. 'cellprofiler' (default) runs "
            "the full 10-step CellProfiler pipeline on profiles.parquet. Other "
            "values (e.g. 'cytoself', 'subcell_portable_bg_vit') run the "
            "deep-embedding pipeline on the raw extracted embeddings."
        ),
    )
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help=(
            "Stop after saving normalized.parquet (plate-normalized, clipped). "
            "Skips decorrelation and control annotation. For CellProfiler, also "
            "skips variance threshold."
        ),
    )
    args = parser.parse_args()

    if args.representation == "cellprofiler":
        preprocess_batch(args.batch, normalized_only=args.normalized_only)
        output_dir = CELLPROFILER_DIR / args.batch
    else:
        preprocess_embedding_batch(
            args.batch,
            representation=args.representation,
            normalized_only=args.normalized_only,
        )
        output_dir = INTERIM_DIR / args.representation / args.batch

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[output_dir])


if __name__ == "__main__":
    main()
