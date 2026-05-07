"""Cell-level and well-level quality control."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _find_feat_cols(schema_names: list[str]) -> list[str]:
    """Return feature column names (not starting with Metadata_)."""
    return [c for c in schema_names if not c.startswith("Metadata_")]


def _find_meta_cols(schema_names: list[str]) -> list[str]:
    """Return metadata column names (starting with Metadata_)."""
    return [c for c in schema_names if c.startswith("Metadata_")]


def filter_to_manifest(
    profiles_path: str,
    manifest_path: str,
) -> pl.LazyFrame:
    """Filter profiles to only cells present in the crop manifest.

    The manifest contains cells that passed QC (area ratio, boundary,
    GFP MAD, min cells/allele). This ensures the same cell set is used
    across all representations for fair comparison.

    JOIN keys: (Metadata_Plate, Metadata_Well, Metadata_Site, Metadata_ObjectNumber)
    """
    profiles = pl.scan_parquet(profiles_path)
    manifest = pl.scan_parquet(manifest_path)

    join_keys = [
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_ObjectNumber",
    ]

    # Semi-join: keep only profile rows that exist in manifest
    filtered = profiles.join(
        manifest.select(join_keys).unique(),
        on=join_keys,
        how="semi",
    )

    n_before = profiles.select(pl.len()).collect().item()
    n_after = filtered.select(pl.len()).collect().item()
    logger.info(
        "Manifest filter: %d → %d cells (%.1f%% retained)",
        n_before,
        n_after,
        100 * n_after / n_before if n_before > 0 else 0,
    )
    return filtered


def drop_low_cell_count_wells(
    lf: pl.LazyFrame,
    cc_threshold: int = 20,
) -> pl.LazyFrame:
    """Drop wells with fewer than cc_threshold cells.

    Groups by (Metadata_Plate, Metadata_Well) and removes all cells
    from wells below the threshold. TC wells are NOT dropped — they
    stay for potential use as a null distribution reference.
    """
    well_counts = (
        lf.group_by(["Metadata_Plate", "Metadata_Well"])
        .agg(pl.len().alias("_well_count"))
    )

    filtered = (
        lf.join(well_counts, on=["Metadata_Plate", "Metadata_Well"], how="left")
        .filter(pl.col("_well_count") >= cc_threshold)
        .drop("_well_count")
    )

    n_before = lf.select(pl.len()).collect().item()
    n_after = filtered.select(pl.len()).collect().item()
    wells_dropped = (
        well_counts.filter(pl.col("_well_count") < cc_threshold)
        .select(pl.len())
        .collect()
        .item()
    )
    logger.info(
        "Well count filter (<%d): dropped %d wells, %d → %d cells",
        cc_threshold,
        wells_dropped,
        n_before,
        n_after,
    )
    return filtered
