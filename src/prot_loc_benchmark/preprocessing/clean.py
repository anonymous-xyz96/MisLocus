"""NaN handling and outlier removal."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _find_feat_cols(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if not c.startswith("Metadata_")]


def _find_meta_cols(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if c.startswith("Metadata_")]


def drop_nan_features(
    lf: pl.LazyFrame,
    cell_threshold: int = 100,
) -> pl.LazyFrame:
    """Remove NaN/Inf features and rows.

    Strategy (ported from preprocess/annotate.py:drop_nan_features):
    1. Replace inf/-inf with NaN
    2. If a feature has >cell_threshold NaN rows → drop the feature
    3. Drop remaining rows that have any NaN in feature columns

    Returns a LazyFrame with no NaN/Inf in feature columns.
    """
    cols = lf.collect_schema().names()
    feat_cols = _find_feat_cols(cols)
    meta_cols = _find_meta_cols(cols)

    # Replace inf/-inf with null (NaN)
    lf = lf.with_columns(
        [
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in feat_cols
        ]
    )

    # Collect NaN counts per feature to decide which to drop
    nan_counts = (
        lf.select([pl.col(c).is_null().sum().alias(c) for c in feat_cols])
        .collect()
        .row(0, named=True)
    )

    feats_to_drop = [f for f, count in nan_counts.items() if count > cell_threshold]
    feats_to_keep = [f for f in feat_cols if f not in feats_to_drop]

    logger.info(
        "NaN feature removal: dropping %d features (>%d NaN rows), keeping %d",
        len(feats_to_drop),
        cell_threshold,
        len(feats_to_keep),
    )

    # Select only kept columns
    lf = lf.select(pl.col(meta_cols + feats_to_keep))

    # Drop rows with any remaining NaN in feature columns
    n_before = lf.select(pl.len()).collect().item()
    lf = lf.drop_nulls(subset=feats_to_keep)
    n_after = lf.select(pl.len()).collect().item()

    logger.info(
        "NaN row removal: %d → %d cells (%d dropped)",
        n_before,
        n_after,
        n_before - n_after,
    )
    return lf


def clip_outliers(
    lf: pl.LazyFrame,
    threshold: float = 100.0,
) -> pl.LazyFrame:
    """Clip feature values to [-threshold, threshold].

    The reference MisLocus pipeline (``preprocess/clean.py:outlier_removal_polars``)
    did TWO steps: first ``drop_outlier_feats_polar`` dropped any feature whose
    99th percentile of absolute values exceeded the threshold, then
    ``clip_features_polar`` clipped the remaining values. Our previous port
    mirrored that drop-then-clip behavior.

    We no longer drop features here, for three reasons:

    1. **Clip-then-drop is self-contradictory.** If outlier values will be
       capped at ±threshold anyway, there is no rationale for throwing away
       a feature whose 1% tail happens to exceed the cap. The capped feature
       retains 99%+ of its signal.
    2. **It silently discarded biologically meaningful signal.** For example,
       ``Cells_Intensity_MeanIntensity_GFP`` has a naturally fat tail after
       RobustMAD normalization because protein expression in overexpression
       experiments spans orders of magnitude — that is real biology, not a
       measurement artifact. The drop behavior was killing the column entirely.
    3. **Genuinely degenerate features are handled elsewhere.** Later steps
       (variance threshold, pycytominer blocklist, correlation threshold) will
       remove features that are constant, known-bad, or redundant. Dropping on
       tail-percentile in addition is redundant and aggressive.

    This function is now a pure value clipper: every feature column is retained,
    and each value is capped to [-threshold, threshold].
    """
    cols = lf.collect_schema().names()
    feat_cols = _find_feat_cols(cols)
    meta_cols = _find_meta_cols(cols)

    # Count how many values will actually be clipped, for logging
    n_clipped = (
        lf.select(
            [
                (pl.col(c).abs() > threshold).sum().alias(c)
                for c in feat_cols
            ]
        )
        .collect()
        .row(0, named=True)
    )
    total_clipped = sum(v for v in n_clipped.values() if v is not None)
    n_affected_feats = sum(1 for v in n_clipped.values() if v is not None and v > 0)

    logger.info(
        "Outlier clipping: capped %d values across %d features at ±%.0f "
        "(all %d features retained)",
        total_clipped,
        n_affected_feats,
        threshold,
        len(feat_cols),
    )

    lf = lf.select(
        [pl.col(c) for c in meta_cols]
        + [pl.col(c).clip(-threshold, threshold) for c in feat_cols]
    )
    return lf
