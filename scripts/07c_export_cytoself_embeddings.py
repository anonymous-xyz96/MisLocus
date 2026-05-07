#!/usr/bin/env python3
"""Export Cytoself embeddings as per-batch latent_codes.parquet for classification.

Converts numpy embeddings from the training pipeline into the parquet format
expected by scripts/09_classify.py. Joins with CellProfiler features.parquet
to get the exact metadata columns the classification pipeline needs.

Produces two embedding groups per cell:
  - Cytoself_global_0..1023: flattened vqvec2 (64×4×4) — coarse localization
  - Cytoself_spectrum_0..511: vqindhist1 (512-dim) — feature spectrum

Usage:
    pixi run python scripts/07c_export_cytoself_embeddings.py \\
        --model-dir data/interim/cytoself/models/gfp_nucdist_128_s43

See docs/cytoself_training_design.md for embedding details.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = REPO_ROOT / "data" / "interim"

# Metadata columns required by the classification pipeline (pairs.py, cv.py)
REQUIRED_META_COLS = [
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


def load_embeddings(emb_dir: Path, splits: list[str]) -> pd.DataFrame:
    """Load and concatenate embeddings + metadata across splits.

    Returns a DataFrame with metadata_cell_id + all embedding columns.
    """
    dfs = []
    for split in splits:
        meta_path = emb_dir / f"{split}_metadata.csv"
        vqvec2_path = emb_dir / f"{split}_vqvec2.npy"
        vqindhist1_path = emb_dir / f"{split}_vqindhist1.npy"

        if not meta_path.exists():
            print(f"  SKIP {split}: {meta_path} not found")
            continue

        meta = pd.read_csv(meta_path, low_memory=False)
        n = len(meta)
        print(f"  {split}: {n:,} cells")

        # vqvec2: (n, 64, 4, 4) → flatten → (n, 1024) → Cytoself_global_*
        vqvec2 = np.load(vqvec2_path)
        vqvec2_flat = vqvec2.reshape(n, -1)
        global_cols = [f"Cytoself_global_{i}" for i in range(vqvec2_flat.shape[1])]

        # vqindhist1: (n, 512) → Cytoself_spectrum_*
        vqindhist1 = np.load(vqindhist1_path)
        spectrum_cols = [f"Cytoself_spectrum_{i}" for i in range(vqindhist1.shape[1])]

        # Build embedding DataFrame
        emb_df = pd.DataFrame(vqvec2_flat, columns=global_cols)
        for i, col in enumerate(spectrum_cols):
            emb_df[col] = vqindhist1[:, i]

        # Attach cell identity
        emb_df["metadata_cell_id"] = meta["metadata_cell_id"].values
        emb_df["batch_id"] = meta["batch_id"].values

        dfs.append(emb_df)

    return pd.concat(dfs, ignore_index=True)


def build_cellid(plate: str, well: str, img_num: int, obj_num: int) -> str:
    """Construct CellID from features.parquet columns."""
    return f"{plate}_{well}_{img_num}_{obj_num}"


def export_batch(
    batch_id: str,
    emb_batch: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Join embeddings with features.parquet metadata and write latent_codes.parquet."""
    features_path = INTERIM_DIR / "cellprofiler" / batch_id / "features.parquet"
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found — cannot get metadata columns")
        return

    # Load only metadata columns from features.parquet
    fp_lf = pl.scan_parquet(str(features_path))
    all_cols = fp_lf.collect_schema().names()
    meta_cols = [c for c in all_cols if c.startswith("Metadata_")]
    fp_meta = fp_lf.select(meta_cols).collect()

    # Construct CellID in features.parquet to match cytoself's metadata_cell_id
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

    # Convert embeddings to polars for join
    emb_pl = pl.from_pandas(emb_batch.drop(columns=["batch_id"]))
    feat_cols = [c for c in emb_pl.columns if c.startswith("Cytoself_")]

    # Join: keep only cells present in BOTH cytoself embeddings AND features.parquet
    joined = emb_pl.join(fp_meta, on="metadata_cell_id", how="inner")

    # Select required metadata + all embedding columns
    available_meta = [c for c in REQUIRED_META_COLS if c in joined.columns]
    output_cols = available_meta + feat_cols
    result = joined.select(output_cols)

    # Drop any rows with NaN in features
    n_before = result.height
    result = result.drop_nulls(subset=feat_cols)
    n_after = result.height
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with null embeddings")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "latent_codes.parquet"
    result.write_parquet(str(out_path))
    print(f"  Wrote {out_path}: {result.height:,} cells × {len(feat_cols)} features")

    # Report join stats
    n_emb = emb_batch.shape[0]
    n_fp = fp_meta.height
    print(f"  Join: {n_emb:,} cytoself cells ∩ {n_fp:,} features cells = {result.height:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Cytoself embeddings as per-batch latent_codes.parquet."
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained model directory (contains embeddings_export/)",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=INTERIM_DIR / "cytoself",
        help="Base output dir (writes {output-base}/{batch}/latent_codes.parquet)",
    )
    p.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Direct path to embeddings dir (overrides model-dir/embeddings_export/)",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to export (default: train,val,test)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = args.embeddings_dir if args.embeddings_dir else args.model_dir / "embeddings_export"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not emb_dir.exists():
        print(f"ERROR: {emb_dir} does not exist")
        return

    print(f"Loading embeddings from {emb_dir}")
    emb_df = load_embeddings(emb_dir, splits)
    print(f"Total: {len(emb_df):,} cells")

    feat_cols = [c for c in emb_df.columns if c.startswith("Cytoself_")]
    global_cols = [c for c in feat_cols if c.startswith("Cytoself_global_")]
    spectrum_cols = [c for c in feat_cols if c.startswith("Cytoself_spectrum_")]
    print(f"Features: {len(global_cols)} global + {len(spectrum_cols)} spectrum = {len(feat_cols)} total")

    # Export per batch
    batches = sorted(emb_df["batch_id"].unique())
    print(f"\nExporting {len(batches)} batches...")
    output_dirs = []
    for batch_id in batches:
        print(f"\n{batch_id}:")
        batch_mask = emb_df["batch_id"] == batch_id
        emb_batch = emb_df[batch_mask]
        output_dir = args.output_base / batch_id
        export_batch(batch_id, emb_batch, output_dir)
        output_dirs.append(output_dir)

    if output_dirs:
        from prot_loc_benchmark.provenance import record
        record(output_dirs=output_dirs)

    print("\nDone.")


if __name__ == "__main__":
    main()
