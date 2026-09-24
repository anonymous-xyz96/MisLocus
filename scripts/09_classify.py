#!/usr/bin/env python3
"""Run XGBoost classification for one batch and representation.

Trains binary classifiers to distinguish variant allele cells from
reference allele cells: train T1/T2/T3, test T4 (default).
Run --scope control first; subsequent experimental runs load its calibration.
Legacy LOPO is available explicitly via --test-split none.

Usage:
    # Full run (all alleles + controls)
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler

    # Quick test on a single allele
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler \\
        --scope CCM2_Ile432Thr

    # Controls only (null distribution)
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler \\
        --scope control

    # With GPU acceleration
    pixi run -e gpu python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler --gpu

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from prot_loc_benchmark.stages import bound_environment

bound_environment()
# ruff: noqa: E402 -- native thread bounds must precede numerical-library imports.

import polars as pl

from prot_loc_benchmark.classification import (
    aggregate_allele_metrics,
    build_control_pairs,
    build_cpc_pairs,
    build_experimental_pairs,
    compute_null_threshold,
    filter_pairs_by_scope,
    generate_folds,
    get_feature_channels,
    get_pair_data,
    plot_auroc_distributions,
    run_classifier_tasks,
    select_device,
    split_fold,
    write_wide_summary,
)
from prot_loc_benchmark.classification.calibration import calibration_context, load_calibration, save_calibration
from prot_loc_benchmark.classification.metrics import validate_thresholds
from prot_loc_benchmark.classification.train import allocated_gpu
from prot_loc_benchmark.config import (
    BATCH_LAYOUT,
    CLASSIFICATION_OUTPUT_DIR,
    INTERIM_DIR,
    MIN_CELL_COUNT,
    REP_FEATURE_FILES,
    XGBOOST_PARAMS,
)
from prot_loc_benchmark.identity import CELL_ID, identify_cells
from prot_loc_benchmark.provenance import save_json, sha256
from prot_loc_benchmark.stages import require_bounded_execution, require_stage, stage

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_template_number(plate: str) -> int | None:
    """Extract template number (1-4) from plate barcode."""
    import re

    # Use [0-9] instead of \d — conda-forge Python 3.12 has a broken \d
    m = re.search(r"T([0-9]+)$", plate)
    return int(m.group(1)) if m else None


def classify_batch(
    batch_id: str,
    representation: str,
    scope: str = "all",
    use_gpu: bool = False,
    unseen_only: bool = False,
    channels: list[str] | None = None,
    test_split: str | None = "t4",
    workers: int = 1,
    threads: int = 1,
) -> None:
    """Run controls OR experiments; a completed matched calibration is required."""
    t0 = time.time()
    require_bounded_execution(representation)
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    if test_split not in (None, "t4") or (unseen_only and test_split is not None):
        raise ValueError("T4 holdout cannot be combined with --unseen-only")
    if "control" in scope.split(",") and scope != "control":
        raise ValueError("Run --scope control separately before experimental scopes")
    control_only = scope == "control"
    xgb_params = {**XGBOOST_PARAMS, "n_jobs": threads, "random_state": 0}

    # ── Resolve paths and config ─────────────────────────────────────
    layout = BATCH_LAYOUT.get(batch_id)
    if layout is None:
        logger.error("Unknown batch %s (not in BATCH_LAYOUT)", batch_id)
        sys.exit(1)

    feature_file = REP_FEATURE_FILES.get(representation)
    if feature_file is None:
        logger.error("Unknown representation: %s", representation)
        sys.exit(1)

    features_path = INTERIM_DIR / representation / batch_id / feature_file
    if not features_path.exists():
        logger.error("Features not found: %s", features_path)
        sys.exit(1)

    suffix = "_t4" if test_split == "t4" else ("_unseen" if unseen_only else "")
    base_dir = CLASSIFICATION_OUTPUT_DIR / (representation + suffix) / batch_id
    control_dir = base_dir / "controls"
    output_dir = control_dir if control_only else base_dir

    logger.info("=" * 70)
    logger.info(
        "Classification: batch=%s rep=%s layout=%s scope=%s unseen_only=%s",
        batch_id,
        representation,
        layout,
        scope,
        unseen_only,
    )
    logger.info("=" * 70)

    # ── Device selection ─────────────────────────────────────────────
    if use_gpu:
        os.environ["MISLOCUS_CLASSIFIER_BACKEND"] = "gpu"
    os.environ.setdefault("MISLOCUS_CLASSIFIER_BACKEND", "cpu")
    device = select_device()
    logger.info("Device: %s", device)

    # ── Load features ────────────────────────────────────────────────
    logger.info("Loading features from %s", features_path)
    input_sha256 = sha256(features_path)
    lf = pl.scan_parquet(str(features_path))
    schema = lf.collect_schema()
    all_cols = schema.names()
    meta_cols = [c for c in all_cols if c.startswith("Metadata_")]
    feat_cols = [c for c in all_cols if not c.startswith("Metadata_")]
    logger.info("Schema: %d metadata, %d feature columns", len(meta_cols), len(feat_cols))

    # Eagerly collect the full dataframe once to avoid per-pair parquet re-scans.
    # ~727K rows x 1060 cols ≈ 3 GB in memory — acceptable for this pipeline.
    logger.info("Collecting full dataframe into memory...")
    t_load = time.time()
    parent_stage = require_stage(
        features_path.parent, features_path.name, representation=representation, batch=batch_id
    )
    df_full = identify_cells(
        lf.collect(), batch_id, canonical=representation.startswith("subcell_allele_rybg_v2_")
    ).sort(CELL_ID)
    if "Metadata_well_position" not in df_full.columns and "Metadata_Well" in df_full.columns:
        df_full = df_full.with_columns(pl.col("Metadata_Well").alias("Metadata_well_position"))
    if not feat_cols or df_full.is_empty():
        raise ValueError("No cells/features available for classification")
    if not df_full.select(pl.all_horizontal(pl.col(feat_cols).is_finite().fill_null(False)).all()).item():
        raise ValueError("Classification requires finite, non-null features")
    if test_split == "t4" and set(
        df_full["Metadata_Plate"].map_elements(_parse_template_number, return_dtype=pl.Int64)
    ) != {1, 2, 3, 4}:
        raise ValueError("T4 testing requires T1/T2/T3 training inputs as well as T4")
    logger.info("Loaded %d rows in %.1fs", df_full.height, time.time() - t_load)

    # Filter to unseen plates (T3+T4) for strict DL evaluation
    if unseen_only:
        df_full = df_full.with_columns(
            pl.col("Metadata_Plate").map_elements(_parse_template_number, return_dtype=pl.Int64).alias("_template")
        )
        n_before = df_full.height
        df_full = df_full.filter(pl.col("_template").is_in([3, 4])).drop("_template")
        logger.info(
            "Unseen-only filter: %d → %d cells (T3+T4 plates only)",
            n_before,
            df_full.height,
        )

    # Wrap as lazy for pair building (which only needs group_by counts)
    lf = df_full.lazy()

    # ── Build pairs ──────────────────────────────────────────────────
    logger.info("Building classification pairs...")
    experimental_pairs = build_experimental_pairs(lf, min_cells=MIN_CELL_COUNT)
    cpc_pairs = build_cpc_pairs(lf, min_cells=MIN_CELL_COUNT)
    control_pairs = build_control_pairs(lf, control_types=["NC", "PC"], min_cells=MIN_CELL_COUNT)
    all_pairs = control_pairs if control_only else experimental_pairs + cpc_pairs

    # 'all' means all experimental + cPC pairs, using previously run controls.
    all_pairs = sorted(filter_pairs_by_scope(all_pairs, scope), key=lambda pair: pair.pair_id)
    if len({pair.pair_id for pair in all_pairs}) != len(all_pairs):
        raise ValueError("Duplicate classifier pair identity")
    if not all_pairs:
        logger.error("No pairs to classify after scope filter")
        sys.exit(1)

    # ── Feature channels ─────────────────────────────────────────────
    channel_features = get_feature_channels(feat_cols, representation)
    if channels:
        missing = [c for c in channels if c not in channel_features]
        if missing:
            logger.error(
                "Requested --channels %s not available for representation %r (available: %s)",
                missing,
                representation,
                sorted(channel_features),
            )
            sys.exit(1)
        channel_features = {c: channel_features[c] for c in channels}
        logger.info("Channel filter: restricting to %s", list(channel_features))

    protocol = "t1-t3_train_t4_test" if test_split == "t4" else ("unseen_lopo" if unseen_only else "lopo")
    context = calibration_context(
        features_path, representation, batch_id, channel_features, protocol, xgb_params, device, input_sha256
    )
    context["workers"] = workers
    if not control_only:
        null_thresholds, control_metrics = load_calibration(control_dir, context)
    inputs = [features_path] + ([] if control_only else [control_dir / "calibration.json"])
    parents = [parent_stage] + ([] if control_only else [require_stage(control_dir)])
    parameters = {"context": context, "scope": scope, "workers": workers, "threads": threads}
    parameters["label_definition"] = {
        "1": "reference allele/control reference well",
        "0": "variant allele/control comparison well",
    }
    with (
        allocated_gpu(device) as gpu,
        stage(
            output_dir,
            inputs,
            {**parameters, "gpu_allocation": gpu},
            parents=parents,
            allowed=() if control_only else ("controls",),
        ) as run,
    ):
        skipped = []
        # Pair builders intentionally omit unsupported alleles. Account for those
        # before fold-level exclusions so absence from scores is never invisible.
        references = (
            df_full.filter(pl.col("Metadata_node_type") == "disease_wt")
            .group_by("Metadata_symbol", "Metadata_Control")
            .len(name="reference_cells")
        )
        inventory = (
            df_full.group_by("Metadata_gene_allele", "Metadata_symbol", "Metadata_node_type", "Metadata_Control")
            .len(name="cells")
            .join(references, on=["Metadata_symbol", "Metadata_Control"], how="left")
            .with_columns(pl.col("reference_cells").fill_null(0))
        )
        selected_controls = [p.gene for p in all_pairs if p.is_control]
        selected_variants = [p.allele_var for p in all_pairs if not p.is_control]
        inventory.with_columns(
            pl.when(pl.col("Metadata_gene_allele").is_in(selected_controls))
            .then(pl.lit("control_pairs_generated"))
            .when(pl.col("Metadata_node_type") != "allele")
            .then(pl.lit("reference_or_control"))
            .when(pl.col("cells") < MIN_CELL_COUNT)
            .then(pl.lit("insufficient_variant_cells"))
            .when(pl.col("reference_cells") < MIN_CELL_COUNT)
            .then(pl.lit("insufficient_matching_reference_cells"))
            .when(pl.col("Metadata_gene_allele").is_in(selected_variants))
            .then(pl.lit("pair_generated"))
            .otherwise(pl.lit("outside_requested_scope"))
            .alias("pair_status")
        ).sort("Metadata_gene_allele").write_parquet(output_dir / "allele_inventory.parquet")

        def tasks():
            for pair in all_pairs:
                pair_df = get_pair_data(df_full, pair)
                folds = generate_folds(pair_df, layout, test_split=test_split) if not pair_df.is_empty() else []
                if not folds:
                    skipped.append({"pair_id": pair.pair_id, "reason": "no_supported_fold"})
                for channel, ch_features in channel_features.items():
                    for fold in folds:
                        train_df, test_df = split_fold(pair_df, fold, layout)
                        n_pos = int((train_df["Label"] == 1).sum())
                        n_neg = train_df.height - n_pos
                        reason = None
                        if train_df.height < MIN_CELL_COUNT or test_df.height < 10 or test_df["Label"].n_unique() < 2:
                            reason = "insufficient_train_or_test_support"
                        elif min(n_pos, n_neg) == 0 or max(n_pos, n_neg) / min(n_pos, n_neg) > 100:
                            reason = "single_class_or_extreme_training_imbalance"
                        if reason:
                            skipped.append(
                                {"pair_id": pair.pair_id, "channel": channel, "fold_id": fold.fold_id, "reason": reason}
                            )
                            continue
                        yield {
                            "pair": pair,
                            "channel": channel,
                            "ch_features": ch_features,
                            "fold": fold,
                            "train_df": train_df,
                            "test_df": test_df,
                            "xgb_params": xgb_params,
                        }

        metrics_rows, importance_rows, info_rows, n_classifiers = run_classifier_tasks(
            tasks(),
            device=device,
            output_dir=output_dir,
            max_workers=workers,
            stage_id=run["stage_id"],
            representation=representation,
        )
        save_json(output_dir / "excluded_classifiers.json", skipped)
        if not metrics_rows:
            raise ValueError("No classifiers produced results; stage is incomplete")
        metrics_df = pl.DataFrame(metrics_rows)
        metrics_df.write_csv(output_dir / "metrics.csv")
        pl.DataFrame(importance_rows).write_csv(output_dir / "feat_importance.csv")
        pl.DataFrame(info_rows).write_csv(output_dir / "classifier_info.csv")
        if control_only:
            control_metrics = metrics_df
            null_thresholds = compute_null_threshold(control_metrics)
            validate_thresholds(null_thresholds, list(channel_features))
        exp_metrics = metrics_df.filter(pl.col("category").is_in(["Exp", "cPC"]))
        if not exp_metrics.is_empty():
            kwargs = {"min_classifiers": 1} if test_split == "t4" else {}
            summary = aggregate_allele_metrics(exp_metrics, null_thresholds, **kwargs)
            summary.write_csv(output_dir / "metrics_summary.csv")
            if not summary.is_empty():
                write_wide_summary(summary, output_dir, batch_id)
        plot_auroc_distributions(control_metrics, exp_metrics, output_dir, batch_id, null_thresholds)
        if control_only:
            save_calibration(output_dir, context)
        else:
            save_json(
                output_dir / "completion.json",
                {
                    "status": "complete",
                    "context": context,
                    "calibration_sha256": sha256(control_dir / "calibration.json"),
                },
            )
        logger.info("Completed %d classifiers in %.1f min", n_classifiers, (time.time() - t0) / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost classification for variant mislocalization prediction.")
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch ID (e.g., 2025_01_27_Batch_13)",
    )
    parser.add_argument(
        "--representation",
        required=True,
        choices=sorted(REP_FEATURE_FILES),
        help="Feature representation to classify",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "Run 'control' first. Then 'all' (default: Exp+cPC), 'exp', 'cpc', "
            "or comma-separated allele names (e.g., 'CCM2_Ile432Thr,KRAS_Gly12Val')"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU-accelerated XGBoost",
    )
    parser.add_argument(
        "--unseen-only",
        action="store_true",
        help=(
            "Legacy T3+T4 LOPO (requires --test-split none). "
            "Not the final held-out protocol: T3 was used for checkpoint selection. "
            "Results saved to {rep}_unseen/."
        ),
    )
    parser.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated subset of feature channels to classify. If omitted, "
            "all channels discovered by get_feature_channels() are used. Example: "
            "--channels combined (for Cytoself, skips global/spectrum and saves "
            "~2/3 of runtime)."
        ),
    )
    parser.add_argument(
        "--test-split",
        choices=["t4", "none"],
        default="t4",
        help="Default: train T1/T2/T3, test T4. 'none' restores legacy LOPO.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent classifiers (default: 1)")
    parser.add_argument("--threads", type=int, default=1, help="Threads per XGBoost fit (default: 1)")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()] if args.channels else None

    classify_batch(
        batch_id=args.batch,
        representation=args.representation,
        scope=args.scope,
        use_gpu=args.gpu,
        unseen_only=args.unseen_only,
        channels=channels,
        test_split=None if args.test_split == "none" else args.test_split,
        workers=args.workers,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
