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

    if not feat_cols:
        raise ValueError("No features remain for plate statistics")
    # Collect to pandas for reliable grouped MAD via scipy
    select_cols = feat_cols + ["Metadata_Plate"]
    pdf = lf.select(pl.col(select_cols)).collect().to_pandas()
    if pdf.empty:
        raise ValueError("No cells remain for plate statistics")

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
    stats["abs_coef_var"] = (stats["mad"] / stats["median"]).fillna(0).abs().replace(np.inf, 0)
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

    features = _find_feat_cols(cols)
    plates = lf.select("Metadata_Plate").unique().collect()
    expected = plates.join(pl.DataFrame({"feature": features}, schema={"feature": pl.String}), how="cross")
    keys = plate_stats.select("Metadata_Plate", pl.col("feature").cast(pl.String))
    if (
        keys.unique().height != keys.height
        or not expected.join(keys, on=["Metadata_Plate", "feature"], how="anti").is_empty()
    ):
        raise ValueError("Missing or duplicate plate-feature statistics")
    stats = plate_stats.filter(pl.col("Metadata_Plate").is_in(plates["Metadata_Plate"].to_list()))
    # Filter stats to features with MAD≠0 and abs_coef_var > threshold
    passing = stats.filter((pl.col("mad") != 0) & (pl.col("abs_coef_var") > acv_threshold))

    # Count against ALL input plates, including plates where nothing passed.
    variant_features = sorted(
        passing.group_by("feature")
        .agg(pl.col("Metadata_Plate").n_unique().alias("n_plates"))
        .filter(pl.col("n_plates") == plates.height)["feature"]
        .to_list()
    )
    variant_features = [f for f in variant_features if f in features]

    n_total = len(_find_feat_cols(cols))
    logger.info(
        "Variant feature selection: %d → %d features (MAD≠0, abs_coef_var>%.0e in all %d plates)",
        n_total,
        len(variant_features),
        acv_threshold,
        plates.height,
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

    if not feat_cols:
        raise ValueError("No features remain for normalization")
    df = lf.collect()
    if df.is_empty() or df["Metadata_Plate"].null_count():
        raise ValueError("Normalization requires cells with non-null plate identities")
    plates = df["Metadata_Plate"].unique(maintain_order=True).to_list()
    required = {"Metadata_Plate", "feature", "median", "mad"}
    if required - set(plate_stats.columns):
        raise ValueError("Normalization statistics require plate, feature, median and mad")
    stats = plate_stats.filter(pl.col("Metadata_Plate").is_in(plates) & pl.col("feature").is_in(feat_cols))
    # Extra features are expected after CP feature selection, but every requested
    # plate-feature combination must have exactly one valid fitted statistic.
    if (
        stats.height != len(plates) * len(feat_cols)
        or stats.select("Metadata_Plate", "feature").unique().height != stats.height
    ):
        raise ValueError("Missing or duplicate normalization statistics for input plates/features")
    if not stats.select(
        (pl.col("median").is_finite() & pl.col("mad").is_finite() & (pl.col("mad") >= 0)).fill_null(False).all()
    ).item():
        raise ValueError("Normalization statistics must have finite medians and nonnegative finite MADs")

    # Pivot stats to get per-plate median and MAD as dicts
    medians_df = stats.pivot(
        index="Metadata_Plate",
        on="feature",
        values="median",
    )
    mads_df = stats.pivot(
        index="Metadata_Plate",
        on="feature",
        values="mad",
    )

    # Build lookup: plate → {feature: median}, plate → {feature: mad}
    median_lookup = {}
    mad_lookup = {}
    for plate in plates:
        row_med = medians_df.filter(pl.col("Metadata_Plate") == plate).drop("Metadata_Plate")
        row_mad = mads_df.filter(pl.col("Metadata_Plate") == plate).drop("Metadata_Plate")
        median_lookup[plate] = row_med.row(0, named=True)
        mad_lookup[plate] = row_mad.row(0, named=True)

    # Normalize each input plate independently, then restore the original row order.
    normalized_parts = []
    row_indices = []

    for plate in plates:
        mask = df["Metadata_Plate"] == plate
        plate_df = df.filter(mask)
        row_indices.append(mask.arg_true())

        med_vals = median_lookup[plate]
        mad_vals = mad_lookup[plate]

        # Build expressions: (col - median) / mad for each feature
        norm_exprs = []
        for f in feat_cols:
            med = med_vals[f]
            mad = mad_vals[f]
            if mad != 0:
                norm_exprs.append(((pl.col(f) - med) / mad).cast(pl.Float32).alias(f))
            else:
                # Preserve the reference zero-MAD policy, including sparse embeddings.
                norm_exprs.append(pl.col(f).cast(pl.Float32))

        plate_norm = plate_df.select([pl.col(c) for c in meta_cols] + norm_exprs)
        normalized_parts.append(plate_norm)

    result = pl.concat(normalized_parts)[pl.concat(row_indices).arg_sort()]
    if not result.select(pl.all_horizontal(pl.col(feat_cols).is_finite().fill_null(False)).all()).item():
        raise ValueError("Normalization produced nonfinite features; inspect input values and statistics")
    logger.info(
        "RobustMAD normalization: %d cells across %d plates",
        result.height,
        len(plates),
    )
    return result.lazy()
