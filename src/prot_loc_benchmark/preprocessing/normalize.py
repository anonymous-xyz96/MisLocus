"""Plate-level statistics, variant feature selection, and RobustMAD normalization."""

import logging
from functools import partial

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)

# Std threshold for the embedding pipeline's dead-dim filter. std — not MAD —
# because sparse count features (e.g. Cytoself_spectrum_*) have median=0 hence
# MAD=0 even when they carry real signal.
EMBEDDING_STD_EPSILON = 1e-6


def _find_feat_cols(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if not c.startswith("Metadata_")]


def _find_meta_cols(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if c.startswith("Metadata_")]


def compute_plate_stats(lf: pl.LazyFrame) -> pl.DataFrame:
    """Compute per-plate statistics for each feature.

    Ported from preprocess/normalize.py:get_plate_stats (pandas version)
    using scipy.stats.median_abs_deviation for reliable MAD computation.

    Returns polars DataFrame with columns:
        Metadata_Plate, feature, median, mad, min, max, count, abs_coef_var
    """
    cols = lf.collect_schema().names()
    feat_cols = _find_feat_cols(cols)

    # Collect to pandas for reliable grouped MAD via scipy
    select_cols = feat_cols + ["Metadata_Plate"]
    pdf = lf.select(pl.col(select_cols)).collect().to_pandas()

    mad_fn = partial(median_abs_deviation, nan_policy="omit", axis=0)
    grouped = pdf.groupby("Metadata_Plate", observed=True)

    median_df = grouped[feat_cols].median()
    max_df = grouped[feat_cols].max()
    min_df = grouped[feat_cols].min()
    count_df = grouped[feat_cols].count()
    mad_df = grouped[feat_cols].apply(mad_fn)
    mad_df = pd.DataFrame(
        index=mad_df.index,
        data=np.stack(mad_df.values),
        columns=feat_cols,
    )

    median_df["stat"] = "median"
    mad_df["stat"] = "mad"
    max_df["stat"] = "max"
    min_df["stat"] = "min"
    count_df["stat"] = "count"

    stats = pd.concat([median_df, mad_df, min_df, max_df, count_df])
    stats.reset_index(inplace=True)
    stats = stats.melt(
        id_vars=["Metadata_Plate", "stat"],
        var_name="feature",
    )
    stats = stats.pivot(
        index=["Metadata_Plate", "feature"],
        columns="stat",
        values="value",
    )
    stats.reset_index(inplace=True)
    stats["abs_coef_var"] = (
        (stats["mad"] / stats["median"]).fillna(0).abs().replace(np.inf, 0)
    )
    stats = stats.astype(
        {
            "min": np.float32,
            "max": np.float32,
            "count": np.float32,
            "median": np.float32,
            "mad": np.float32,
            "abs_coef_var": np.float32,
        }
    )

    # Convert to polars
    result = pl.from_pandas(stats).cast({"feature": pl.Categorical})

    n_plates = result["Metadata_Plate"].n_unique()
    n_features = result["feature"].n_unique()
    logger.info("Plate stats: %d plates × %d features", n_plates, n_features)
    return result


def select_variant_features(
    lf: pl.LazyFrame,
    plate_stats: pl.DataFrame,
    acv_threshold: float = 1e-3,
) -> pl.LazyFrame:
    """Keep features with MAD≠0 AND abs_coef_var > threshold in ALL plates.

    Ported from preprocess/normalize.py:select_variant_features_polars.

    Takes the intersection of passing features across all plates,
    ensuring every retained feature has meaningful variation everywhere.
    """
    cols = lf.collect_schema().names()
    meta_cols = _find_meta_cols(cols)

    # Filter stats to features with MAD≠0 and abs_coef_var > threshold
    passing = plate_stats.filter(
        (pl.col("mad") != 0) & (pl.col("abs_coef_var") > acv_threshold)
    )

    # Take intersection across all plates
    per_plate = passing.group_by("Metadata_Plate").agg(pl.col("feature"))
    feature_sets = [set(row) for row in per_plate["feature"].to_list()]
    if not feature_sets:
        logger.warning("No variant features found — returning empty frame")
        return lf.select(pl.col(meta_cols))

    variant_features = sorted(set.intersection(*feature_sets))

    n_total = len(_find_feat_cols(cols))
    logger.info(
        "Variant feature selection: %d → %d features (MAD≠0, abs_coef_var>%.0e in all %d plates)",
        n_total,
        len(variant_features),
        acv_threshold,
        len(feature_sets),
    )

    return lf.select(pl.col(meta_cols + variant_features))


def drop_dead_features(
    lf: pl.LazyFrame,
    epsilon: float = EMBEDDING_STD_EPSILON,
) -> pl.LazyFrame:
    """Drop feature columns whose global std is below ``epsilon``.

    Embedding-pipeline substitute for ``select_variant_features``. The MAD-based
    variant filter misfires on sparse count features like ``Cytoself_spectrum_*``
    whose median=0 drives MAD=0 even when they carry real signal; std is the
    safer zero-variance probe for those distributions.
    """
    feat_cols = _find_feat_cols(lf.collect_schema().names())
    stds = lf.select([pl.col(c).std() for c in feat_cols]).collect().row(0, named=True)
    dead = [f for f, s in stds.items() if s is None or s < epsilon]
    if dead:
        logger.info("Dropped %d zero-std features out of %d", len(dead), len(feat_cols))
        return lf.drop(dead)
    logger.info("No zero-std features to drop (all %d passed)", len(feat_cols))
    return lf


def robustmad(
    lf: pl.LazyFrame,
    plate_stats: pl.DataFrame,
) -> pl.LazyFrame:
    """Apply per-plate RobustMAD normalization: (value - median) / MAD.

    Ported from preprocess/normalize.py:robustmad (rewritten in polars).

    For each plate, subtracts the plate median and divides by the plate MAD
    for each feature. Epsilon=0 (no regularization, matching the existing pipeline).
    """
    cols = lf.collect_schema().names()
    feat_cols = _find_feat_cols(cols)
    meta_cols = _find_meta_cols(cols)

    # Pivot stats to get per-plate median and MAD as dicts
    medians_df = plate_stats.filter(pl.col("feature").is_in(feat_cols)).pivot(
        index="Metadata_Plate",
        on="feature",
        values="median",
    )
    mads_df = plate_stats.filter(pl.col("feature").is_in(feat_cols)).pivot(
        index="Metadata_Plate",
        on="feature",
        values="mad",
    )

    # Build lookup: plate → {feature: median}, plate → {feature: mad}
    plates = medians_df["Metadata_Plate"].to_list()
    median_lookup = {}
    mad_lookup = {}
    for plate in plates:
        row_med = medians_df.filter(pl.col("Metadata_Plate") == plate).drop("Metadata_Plate")
        row_mad = mads_df.filter(pl.col("Metadata_Plate") == plate).drop("Metadata_Plate")
        median_lookup[plate] = row_med.row(0, named=True)
        mad_lookup[plate] = row_mad.row(0, named=True)

    # Collect, normalize per plate, recombine
    df = lf.collect()
    normalized_parts = []

    for plate in plates:
        plate_df = df.filter(pl.col("Metadata_Plate") == plate)
        if plate_df.height == 0:
            continue

        med_vals = median_lookup[plate]
        mad_vals = mad_lookup[plate]

        # Build expressions: (col - median) / mad for each feature
        norm_exprs = []
        for f in feat_cols:
            med = med_vals.get(f)
            mad = mad_vals.get(f)
            if med is not None and mad is not None and mad != 0:
                norm_exprs.append(
                    ((pl.col(f) - med) / mad).cast(pl.Float32).alias(f)
                )
            else:
                # MAD=0 → keep original (shouldn't happen after variant selection)
                norm_exprs.append(pl.col(f).cast(pl.Float32))

        plate_norm = plate_df.select([pl.col(c) for c in meta_cols] + norm_exprs)
        normalized_parts.append(plate_norm)

    result = pl.concat(normalized_parts)
    logger.info(
        "RobustMAD normalization: %d cells across %d plates",
        result.height,
        len(plates),
    )
    return result.lazy()
