"""Feature selection using pycytominer.

Split into three phases to give preprocessing a clean save point for
``normalized.parquet`` that preserves the per-cell intensity columns
downstream analyses need:

1. ``apply_blocklists`` — unconditional drops (Neighbors features,
   pycytominer blocklist of known-bad features). Safe to run early.
2. ``apply_variance_threshold`` — pycytominer heuristic cleanup that drops
   near-constant or low-diversity features. Has a subtle false-positive
   mode: when ``clip_outliers`` pins tail values to the exact threshold
   (e.g., 30 cells all clipped to ``+100``), the artificial mode trips
   pycytominer's ``freq_cut`` check and the entire feature is dropped —
   even when it has plenty of legitimate variance in its bulk values.
   ``Cells_Intensity_MeanIntensity_GFP`` hits this case. We therefore
   save ``normalized.parquet`` *before* this step so the intensity
   columns survive for downstream matching.
3. ``decorrelate_features`` — expensive channel-aware two-pass correlation
   threshold + drop_na_columns. Runs only when building the final
   ``features.parquet``.

``prefilter_features`` is a convenience wrapper composing (1) and (2) for
callers that want the same feature set as before the split.
"""

import logging

import pandas as pd
from pycytominer.feature_select import feature_select
from pycytominer.operations import correlation_threshold, variance_threshold

logger = logging.getLogger(__name__)


def _find_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("Metadata_")]


# Features that measure imaging context rather than cell biology.
# Neighbors features capture plating density and spatial arrangement of cells
# in the imaging field — these are confounders for localization phenotyping.
CUSTOM_BLOCKLIST_PATTERNS = [
    "_Neighbors_",
]


def apply_blocklists(df: pd.DataFrame) -> pd.DataFrame:
    """Apply unconditional feature blocklists (safe, no false positives).

    Drops two categories of known-bad features:
      1. Custom blocklist — Neighbors features (plating density / spatial
         arrangement — confounders for localization phenotyping)
      2. Pycytominer blocklist — Nuclei_Correlation_Manders,
         Nuclei_Correlation_RWC, Nuclei_Granularity_14–16

    This is the save point for ``normalized.parquet``: all remaining features
    are legitimate, plate-normalized, outlier-clipped measurements including
    the per-cell intensity columns needed for downstream matching.

    Args:
        df: pandas DataFrame with Metadata_* and feature columns.

    Returns:
        DataFrame with blocklisted features removed.
    """
    features = _find_feat_cols(df)
    n_start = len(features)

    # ── 1. Custom blocklist (Neighbors features) ────────────────────────
    custom_blocked = [
        f for f in features
        if any(pat in f for pat in CUSTOM_BLOCKLIST_PATTERNS)
    ]
    if custom_blocked:
        df = df.drop(columns=custom_blocked)
        features = [f for f in features if f not in custom_blocked]
        logger.info("custom blocklist: removed %d features (%s)",
                    len(custom_blocked), ", ".join(CUSTOM_BLOCKLIST_PATTERNS))

    # ── 2. Pycytominer blocklist ────────────────────────────────────────
    cols_before = set(df.columns)
    df = feature_select(df, operation="blocklist", image_features=False)
    blocked = cols_before - set(df.columns)
    logger.info("pycytominer blocklist: removed %d features", len(blocked))

    n_end = len(_find_feat_cols(df))
    logger.info("apply_blocklists total: %d → %d features", n_start, n_end)
    return df.reset_index(drop=True)


def apply_variance_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pycytominer variance_threshold to drop near-constant features.

    NOTE: this step has a known interaction with ``clip_outliers``. When
    ``clip_outliers`` pins tail values to the exact clip threshold (e.g.,
    +100.0), the artificial mode trips pycytominer's ``freq_cut`` check
    (2nd_most_common / most_common < 0.05) and the feature is dropped even
    when it has healthy variance in its bulk values. This is a false positive
    that affects e.g. ``Cells_Intensity_MeanIntensity_GFP`` (204 cells
    clipped to exactly +100 out of 723K, despite std=6.28).

    ``normalized.parquet`` is saved BEFORE this step so the intensity
    columns survive for downstream matching. ``features.parquet`` is
    produced AFTER this step, so the false-positive drops still occur
    there — but those columns are not needed as classifier features.

    Args:
        df: pandas DataFrame (output of ``apply_blocklists``).

    Returns:
        DataFrame with low-variance features removed.
    """
    features = _find_feat_cols(df)
    n_start = len(features)

    low_variance = variance_threshold(df, features)
    features = [f for f in features if f not in low_variance]
    df = df.drop(columns=low_variance)
    logger.info("variance_threshold: removed %d features", len(low_variance))

    n_end = len(_find_feat_cols(df))
    logger.info("apply_variance_threshold total: %d → %d features", n_start, n_end)
    return df.reset_index(drop=True)


def prefilter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cheap drops before de-correlation (convenience wrapper).

    Composes ``apply_blocklists`` + ``apply_variance_threshold``.
    Use the individual functions when you need to save an intermediate
    between blocklists and variance threshold (e.g., ``normalized.parquet``).
    """
    df = apply_blocklists(df)
    df = apply_variance_threshold(df)
    return df


def decorrelate_features(df: pd.DataFrame, *, channel_aware: bool = True) -> pd.DataFrame:
    """Apply correlation threshold + drop NA columns.

    Two modes:

    - ``channel_aware=True`` (CellProfiler default): two-pass decorrelation that
      preserves GFP feature diversity. GFP is the protein localization channel —
      if we de-correlate all features together, GFP features that happen to
      correlate with morphology features in other channels (DNA, AGP, Mito) get
      dropped, losing information critical for mislocalization analysis.
      Pass 1 decorrelates within GFP; pass 2 within non-GFP. Cross-channel
      correlations are intentionally preserved.

    - ``channel_aware=False`` (deep-embedding default): single-pass correlation
      threshold across all feature columns together. Use this for learned
      embeddings (Cytoself, SubCell, ViT) where the feature names carry no
      stain-channel semantics.

    Args:
        df: pandas DataFrame that has already passed prefilter_features.
        channel_aware: split GFP vs non-GFP for two-pass decorrelation (True)
            or run a single pass across all features (False).

    Returns:
        DataFrame with correlation-reduced feature set.
    """
    features = _find_feat_cols(df)
    n_start = len(features)

    if channel_aware:
        gfp_feats = [f for f in features if "GFP" in f]
        non_gfp_feats = [f for f in features if "GFP" not in f]
        logger.info("correlation split: %d GFP, %d non-GFP features",
                    len(gfp_feats), len(non_gfp_feats))

        gfp_drop = correlation_threshold(df, gfp_feats) if len(gfp_feats) > 1 else []
        gfp_kept = [f for f in gfp_feats if f not in gfp_drop]
        logger.info("correlation_threshold (GFP): %d → %d features (removed %d)",
                    len(gfp_feats), len(gfp_kept), len(gfp_drop))

        non_gfp_drop = correlation_threshold(df, non_gfp_feats) if len(non_gfp_feats) > 1 else []
        non_gfp_kept = [f for f in non_gfp_feats if f not in non_gfp_drop]
        logger.info("correlation_threshold (non-GFP): %d → %d features (removed %d)",
                    len(non_gfp_feats), len(non_gfp_kept), len(non_gfp_drop))

        all_corr_drop = gfp_drop + non_gfp_drop
    else:
        all_corr_drop = correlation_threshold(df, features) if len(features) > 1 else []
        logger.info(
            "correlation_threshold (single-pass): %d → %d features (removed %d)",
            len(features), len(features) - len(all_corr_drop), len(all_corr_drop),
        )

    df = df.drop(columns=all_corr_drop)
    logger.info("correlation_threshold total: removed %d features", len(all_corr_drop))

    remaining_feats = _find_feat_cols(df)
    cols_before = set(df.columns)
    # Pass ``features`` explicitly so pycytominer doesn't try to infer
    # CellProfiler-style column names (fails for Cytoself/SubCell embeddings).
    df = feature_select(
        df,
        features=remaining_feats,
        operation="drop_na_columns",
        image_features=False,
    )
    na_dropped = cols_before - set(df.columns)
    logger.info("drop_na_columns: removed %d features", len(na_dropped))

    n_end = len(_find_feat_cols(df))
    if channel_aware:
        gfp_final = len([f for f in _find_feat_cols(df) if "GFP" in f])
        logger.info(
            "decorrelate_features total: %d → %d features (%d GFP, %d non-GFP)",
            n_start, n_end, gfp_final, n_end - gfp_final,
        )
    else:
        logger.info("decorrelate_features total: %d → %d features", n_start, n_end)
    return df.reset_index(drop=True)
