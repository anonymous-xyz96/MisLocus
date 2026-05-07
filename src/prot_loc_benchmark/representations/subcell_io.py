"""Shared I/O utilities for SubCell embedding export.

Used by both frozen extraction (08a) and fine-tuned extraction (08d) to ensure
consistent output format for the classification pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prot_loc_benchmark.config import CELLPROFILER_DIR

# Metadata columns retained in embedding parquets (must match classification pipeline)
SUBCELL_REQUIRED_META_COLS = [
    "Metadata_Plate",
    "Metadata_well_position",
    "Metadata_Site",
    "Metadata_ImageNumber",
    "Metadata_ObjectNumber",
    "Metadata_gene_allele",
    "Metadata_symbol",
    "Metadata_node_type",
    "Metadata_Control",
    "Metadata_plate_map_name",
]


def export_subcell_batch(
    batch_id: str,
    all_embeddings: np.ndarray,
    all_metadata: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Join embeddings with features.parquet metadata, write embeddings.parquet.

    Parameters
    ----------
    batch_id : str
        Batch identifier (e.g. "2025_01_27_Batch_13").
    all_embeddings : np.ndarray
        Embedding array of shape (n_cells, embed_dim), float32.
    all_metadata : pl.DataFrame
        Crop metadata with Metadata_Plate, Metadata_Well, etc.
    output_dir : Path
        Directory to write embeddings.parquet.
    """
    features_path = CELLPROFILER_DIR / batch_id / "features.parquet"
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found — cannot get metadata columns")
        return

    # Build embedding DataFrame with CellID
    feat_cols = [f"SubCell_{i}" for i in range(all_embeddings.shape[1])]
    emb_df = pl.DataFrame(
        {col: all_embeddings[:, i].astype(np.float32) for i, col in enumerate(feat_cols)}
    )

    # Construct CellID from crop metadata
    cell_ids = all_metadata.select(
        (
            pl.col("Metadata_Plate")
            + "_"
            + pl.col("Metadata_Well")
            + "_"
            + pl.col("Metadata_ImageNumber").cast(pl.Utf8)
            + "_"
            + pl.col("Metadata_ObjectNumber").cast(pl.Utf8)
        ).alias("metadata_cell_id")
    )
    emb_df = emb_df.with_columns(cell_ids.to_series())

    # Load metadata from features.parquet
    fp_lf = pl.scan_parquet(str(features_path))
    meta_cols = [c for c in fp_lf.collect_schema().names() if c.startswith("Metadata_")]
    fp_meta = fp_lf.select(meta_cols).collect()

    # Construct matching CellID in features.parquet
    fp_meta = fp_meta.with_columns(
        (
            pl.col("Metadata_Plate")
            + "_"
            + pl.col("Metadata_well_position")
            + "_"
            + pl.col("Metadata_ImageNumber").cast(pl.Utf8)
            + "_"
            + pl.col("Metadata_ObjectNumber").cast(pl.Utf8)
        ).alias("metadata_cell_id")
    )

    # Inner join
    joined = emb_df.join(fp_meta, on="metadata_cell_id", how="inner")

    # Select required metadata + embeddings
    available_meta = [c for c in SUBCELL_REQUIRED_META_COLS if c in joined.columns]
    result = joined.select(available_meta + feat_cols)

    # Drop nulls in embedding columns
    n_before = result.height
    result = result.drop_nulls(subset=feat_cols)
    n_after = result.height
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with null embeddings")

    n_emb = emb_df.height
    n_fp = fp_meta.height
    print(f"  Join: {n_emb:,} subcell cells ∩ {n_fp:,} features cells = {result.height:,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "embeddings.parquet"
    result.write_parquet(str(out_path))
    print(f"  Wrote {out_path}: {result.height:,} cells × {len(feat_cols)} features")
