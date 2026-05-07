#!/usr/bin/env python3
"""Extract ViT embeddings from single-cell crops using Bag-of-Channels (BoC).

Each imaging channel is processed independently through a pre-trained ViT
(default: MorphEm), producing a CLS token embedding per channel. Per-channel
embeddings are concatenated to form the final feature vector.

Uses a DataLoader with prefetching workers to overlap CPU I/O + preprocessing
with GPU inference.

Usage:
    # All 4 channels, 8 workers
    pixi run -e vit python scripts/08b_extract_vit_embeddings.py \\
        --batch 2025_01_27_Batch_13 --channels gfp,dna,agp,mito --num-workers 8

    # Single channel, larger batch
    pixi run -e vit python scripts/08b_extract_vit_embeddings.py \\
        --batch 2025_01_27_Batch_13 --channels gfp --batch-size 1024
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prot_loc_benchmark.config import (
    FOCUS_BATCHES,
    INTERIM_DIR,
    SINGLE_CELL_CROPS_DIR,
)
from prot_loc_benchmark.representations.vit import ViTExtractor, preprocess_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

VALID_CHANNELS = ["gfp", "dna", "agp", "mito"]

# Metadata columns required by the classification pipeline
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


# ── Dataset for DataLoader ──────────────────────────────────────────────────


class CropChannelDataset(Dataset):
    """Flat dataset over all cells across all alleles for one channel.

    Each item returns a preprocessed (1, 224, 224) tensor + cell index.
    Preprocessing (scale, saturation noise, instance norm, resize) is done
    in worker processes, overlapping with GPU inference.
    """

    def __init__(self, allele_cells: list[tuple[np.ndarray, int, int]]):
        """
        Parameters
        ----------
        allele_cells : list of (mmap_array, cell_idx, global_idx)
            mmap_array is the full (N, H, W) mmap, cell_idx indexes into it,
            global_idx is the position in the output embedding array.
        """
        self.cells = allele_cells

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        arr, cell_idx, global_idx = self.cells[idx]
        # Read single cell from mmap → (H, W) uint16
        crop = np.array(arr[cell_idx])  # materialize from mmap
        # Preprocess: (H, W) → (1, 224, 224) float32
        tensor = preprocess_batch(crop[np.newaxis])[0]  # (1, 224, 224)
        return tensor, global_idx


def build_cell_index(
    allele_dirs: list[Path],
    channels: list[str],
) -> tuple[dict[str, list[tuple[np.ndarray, int, int]]], list[str], int]:
    """Build a flat cell index across all alleles for each channel.

    Returns
    -------
    channel_cells : dict[str, list[tuple[mmap, cell_idx, global_idx]]]
    cell_ids : list[str] in global_idx order
    total_cells : int
    """
    channel_cells: dict[str, list[tuple[np.ndarray, int, int]]] = {ch: [] for ch in channels}
    cell_ids: list[str] = []
    global_idx = 0

    for allele_dir in allele_dirs:
        meta_path = allele_dir / "metadata.parquet"
        if not meta_path.exists():
            continue

        meta = pl.read_parquet(str(meta_path))
        n_cells = meta.height
        if n_cells == 0:
            continue

        # Check all channel .npy files exist and have correct length
        mmaps: dict[str, np.ndarray] = {}
        skip = False
        for ch in channels:
            npy_path = allele_dir / f"{ch}.npy"
            if not npy_path.exists():
                skip = True
                break
            arr = np.load(str(npy_path), mmap_mode="r")
            if len(arr) != n_cells:
                skip = True
                break
            mmaps[ch] = arr
        if skip:
            continue

        # Build cell IDs
        ids = (
            meta.select(
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
            .to_series()
            .to_list()
        )

        for i in range(n_cells):
            for ch in channels:
                channel_cells[ch].append((mmaps[ch], i, global_idx))
            cell_ids.append(ids[i])
            global_idx += 1

    return channel_cells, cell_ids, global_idx


def extract_channel_with_loader(
    extractor: ViTExtractor,
    cells: list[tuple[np.ndarray, int, int]],
    total_cells: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Extract embeddings for one channel using a DataLoader for prefetching."""
    dataset = CropChannelDataset(cells)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Pre-allocate output array (filled as batches complete)
    hidden_dim = None
    embeddings = None

    with torch.no_grad():
        for batch_tensors, batch_indices in loader:
            batch_tensors = batch_tensors.to(extractor.device, non_blocking=True)
            output = extractor.model.forward_features(batch_tensors)
            emb = output["x_norm_clstoken"].cpu().numpy()

            if embeddings is None:
                hidden_dim = emb.shape[1]
                embeddings = np.zeros((total_cells, hidden_dim), dtype=np.float32)

            indices = batch_indices.numpy()
            embeddings[indices] = emb

    return embeddings


def join_with_features(
    emb_df: pl.DataFrame,
    batch_id: str,
    feat_col_prefix: str = "ViT_",
) -> pl.DataFrame | None:
    """Join embeddings with features.parquet metadata."""
    features_path = INTERIM_DIR / "cellprofiler" / batch_id / "features.parquet"
    if not features_path.exists():
        logger.error("Features not found: %s — run 05_preprocess_profiles.py first", features_path)
        return None

    fp_lf = pl.scan_parquet(str(features_path))
    all_cols = fp_lf.collect_schema().names()
    meta_cols = [c for c in all_cols if c.startswith("Metadata_")]
    fp_meta = fp_lf.select(meta_cols).collect()

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

    feat_cols = [c for c in emb_df.columns if c.startswith(feat_col_prefix)]

    joined = emb_df.join(fp_meta, on="metadata_cell_id", how="inner")

    available_meta = [c for c in REQUIRED_META_COLS if c in joined.columns]
    result = joined.select(available_meta + feat_cols)

    n_before = result.height
    result = result.drop_nulls(subset=feat_cols)
    n_after = result.height
    if n_before != n_after:
        logger.info("Dropped %d rows with null embeddings", n_before - n_after)

    n_emb = emb_df.height
    n_fp = fp_meta.height
    logger.info("Join: %d ViT cells ∩ %d features cells = %d", n_emb, n_fp, result.height)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract ViT embeddings from single-cell crops (Bag of Channels)."
    )
    p.add_argument(
        "--batch",
        required=True,
        help="Batch ID (e.g., 2025_01_27_Batch_13)",
    )
    p.add_argument(
        "--model-name",
        default="CaicedoLab/MorphEm",
        help="HuggingFace model ID (default: CaicedoLab/MorphEm)",
    )
    p.add_argument(
        "--channels",
        default="gfp",
        help="Comma-separated channels to extract (default: gfp). Options: gfp,dna,agp,mito",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (default: 512)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for I/O + preprocessing (default: 8)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda)",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=INTERIM_DIR / "vit",
        help="Base output dir (writes {output-base}/{batch}/embeddings.parquet)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batch_id = args.batch
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    for ch in channels:
        if ch not in VALID_CHANNELS:
            logger.error("Invalid channel '%s'. Must be one of: %s", ch, VALID_CHANNELS)
            sys.exit(1)

    crops_dir = SINGLE_CELL_CROPS_DIR / batch_id
    if not crops_dir.exists():
        logger.error("Crops directory not found: %s", crops_dir)
        sys.exit(1)

    allele_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
    logger.info(
        "Batch %s: %d alleles, channels=%s, model=%s, batch_size=%d, workers=%d",
        batch_id, len(allele_dirs), channels, args.model_name,
        args.batch_size, args.num_workers,
    )

    # Build flat cell index across all alleles (mmap handles, no data loaded yet)
    logger.info("Building cell index...")
    channel_cells, cell_ids, total_cells = build_cell_index(allele_dirs, channels)
    logger.info("Total: %d cells across %d alleles", total_cells, len(allele_dirs))

    if total_cells == 0:
        logger.error("No cells found for batch %s", batch_id)
        sys.exit(1)

    # Load model
    extractor = ViTExtractor(model_name=args.model_name, device=args.device)

    # Extract per-channel embeddings using DataLoader
    all_embeddings = []
    all_columns = []
    for ch in channels:
        logger.info("Extracting channel: %s (%d cells)", ch, len(channel_cells[ch]))
        emb = extract_channel_with_loader(
            extractor, channel_cells[ch], total_cells,
            args.batch_size, args.num_workers,
        )
        hidden_dim = emb.shape[1]
        col_names = [f"ViT_{ch}_{i}" for i in range(hidden_dim)]
        all_embeddings.append(emb)
        all_columns.extend(col_names)
        logger.info("  %s done: %d cells × %d features", ch, emb.shape[0], hidden_dim)

    # Concatenate channels → (total_cells, hidden_dim * n_channels)
    concatenated = np.concatenate(all_embeddings, axis=1)

    # Build embedding DataFrame with cell IDs
    emb_df = pl.DataFrame(
        {col: concatenated[:, i] for i, col in enumerate(all_columns)}
    ).with_columns(pl.Series("metadata_cell_id", cell_ids))

    logger.info("Embeddings: %d cells × %d features", emb_df.height, len(all_columns))

    # Join with features.parquet for metadata
    result = join_with_features(emb_df, batch_id)
    if result is None or result.height == 0:
        logger.error("Join produced no results")
        sys.exit(1)

    # Write output
    output_dir = args.output_base / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "embeddings.parquet"
    result.write_parquet(str(out_path))

    feat_cols = [c for c in result.columns if c.startswith("ViT_")]
    logger.info("Wrote %s: %d cells × %d features", out_path, result.height, len(feat_cols))

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[output_dir])


if __name__ == "__main__":
    main()
