#!/usr/bin/env python3
"""Post-hoc T4-only summary of XGBoost classification metrics.

Re-aggregates the per-fold metrics produced by ``09_classify.py`` to the
single fold whose held-out test plate is a T4 plate. Mirrors the
canonical DL training protocol (T1+T2 train, T3 val, T4 test) by
reporting only the T4-test fold.

No retraining. Reads existing ``metrics.csv`` + ``classifier_info.csv``
from ``data/processed/classification/{rep}/{batch}/`` and writes filtered
copies plus a re-aggregated ``metrics_summary.csv`` to
``data/processed/classification/{rep}_t4/{batch}/``.

The aggregator runs with ``min_classifiers=1`` because each
(pair, channel) typically has exactly one T4 fold per platemap; the
default ``MIN_CLASSIFIERS=2`` would drop nearly every single-platemap
pair. Pairs that span two platemaps still get 2 T4 folds and average
across them naturally.

B11/B12 (multi_rep layout) have no T4 plates and are skipped with a
warning.

Usage:
    pixi run python scripts/09e_summarize_t4.py \\
        --representation cellprofiler \\
        --batches 2025_01_27_Batch_13 2025_01_28_Batch_14
"""

from __future__ import annotations

import argparse
import logging
import sys

import polars as pl

from prot_loc_benchmark.classification.metrics import (
    aggregate_allele_metrics,
    compute_null_threshold,
)
from prot_loc_benchmark.config import BATCH_LAYOUT, CLASSIFICATION_OUTPUT_DIR
from prot_loc_benchmark.provenance import record

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


T4_SUFFIX = "T4"


def summarize_one(rep: str, batch: str) -> bool:
    """Return True if a T4 summary was produced, False if skipped."""
    layout = BATCH_LAYOUT.get(batch)
    if layout != "single_rep":
        logger.warning(
            "Skipping %s/%s: layout=%s (T4 evaluation requires single_rep)",
            rep,
            batch,
            layout,
        )
        return False

    src_dir = CLASSIFICATION_OUTPUT_DIR / rep / batch
    metrics_path = src_dir / "metrics.csv"
    info_path = src_dir / "classifier_info.csv"
    if not metrics_path.exists() or not info_path.exists():
        logger.warning("Skipping %s/%s: missing %s or %s", rep, batch, metrics_path.name, info_path.name)
        return False

    metrics = pl.read_csv(metrics_path)
    info = pl.read_csv(info_path).select("classifier_id", "test_plates")

    joined = metrics.join(info, on="classifier_id", how="left", validate="1:1")
    n_missing = joined.filter(pl.col("test_plates").is_null()).height
    if n_missing:
        raise ValueError(f"{rep}/{batch}: {n_missing} metrics rows have no classifier_info")

    t4 = joined.filter(pl.col("test_plates").str.ends_with(T4_SUFFIX))
    if t4.is_empty():
        logger.warning("Skipping %s/%s: no T4 test plates found in classifier_info", rep, batch)
        return False

    logger.info(
        "%s/%s: %d → %d classifier rows after T4 filter (%d unique T4 plates)",
        rep,
        batch,
        joined.height,
        t4.height,
        t4["test_plates"].n_unique(),
    )

    out_dir = CLASSIFICATION_OUTPUT_DIR / f"{rep}_t4" / batch
    metrics_cols = [c for c in t4.columns if c != "test_plates"]
    control_metrics = t4.filter(pl.col("category").is_in(["NC", "PC"])).select(metrics_cols)
    exp_metrics = t4.filter(pl.col("category").is_in(["Exp", "cPC"])).select(metrics_cols)

    null_thresholds = compute_null_threshold(control_metrics)
    from prot_loc_benchmark.classification.metrics import validate_thresholds

    validate_thresholds(null_thresholds, exp_metrics["channel"].unique().to_list())
    out_dir.mkdir(parents=True, exist_ok=False)
    t4.select(metrics_cols).write_csv(out_dir / "metrics.csv")
    info.join(t4.select("classifier_id").unique(), on="classifier_id").write_csv(out_dir / "classifier_info.csv")

    if exp_metrics.is_empty():
        logger.warning("%s/%s: no Exp/cPC rows after T4 filter; metrics_summary.csv not written", rep, batch)
    else:
        # min_classifiers=1: each (pair, channel) typically has 1 T4 fold per
        # platemap. Default MIN_CLASSIFIERS=2 would drop every single-platemap
        # pair, defeating the purpose of T4-only reporting.
        summary = aggregate_allele_metrics(exp_metrics, null_thresholds, min_classifiers=1)
        if summary.is_empty():
            logger.warning("%s/%s: aggregate_allele_metrics returned empty", rep, batch)
        else:
            summary.write_csv(out_dir / "metrics_summary.csv")
            n_hits = int(summary["is_hit"].sum()) if "is_hit" in summary.columns else 0
            logger.info(
                "%s/%s: wrote metrics_summary.csv (%d alleles, %d hits)",
                rep,
                batch,
                summary.height,
                n_hits,
            )

    record(output_dirs=[out_dir], input_paths=[metrics_path, info_path])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T4-only re-aggregation of XGBoost classification metrics (post-hoc).",
    )
    parser.add_argument(
        "--representation",
        required=True,
        help="Representation name (e.g. cellprofiler, cytoself, vit, subcell_portable_bg_vit)",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        required=True,
        help="One or more batch IDs (e.g. 2025_01_27_Batch_13 2025_01_28_Batch_14)",
    )
    args = parser.parse_args()

    n_ok = 0
    for batch in args.batches:
        if summarize_one(args.representation, batch):
            n_ok += 1
    logger.info("Done. %d/%d batches summarized for T4.", n_ok, len(args.batches))


if __name__ == "__main__":
    main()
