#!/usr/bin/env python3
"""Run Cytoself VQ-VAE inference on arbitrary batches using a trained model.

Loads a trained Cytoself model and extracts VQ-VAE embeddings for cells in a
manifest CSV. This is the inference-only counterpart to 07b_train_cytoself.py —
it reuses the same ManifestNpyDataset and model architecture but skips training.

The forward pass exits before the FC classification head when extracting VQ
embeddings, so alleles not in the original label book are assigned a dummy
label (0) without affecting embedding quality.

Output format matches the training export (07b) so that 07c can convert to
latent_codes.parquet without modification.

Usage:
    CUDA_VISIBLE_DEVICES=0 pixi run -e cytoself python scripts/07d_infer_cytoself_embeddings.py \\
        --model-dir data/interim/cytoself/models/gfp_nucdist_128_b7b8b13b16_siteqc_s43 \\
        --manifest-csv data/interim/cytoself/manifests/manifest_b11b12.csv \\
        --output-dir data/interim/cytoself/models/gfp_nucdist_128_b7b8b13b16_siteqc_s43/embeddings_export_b11b12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset

# Allow running from repo root without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_CYTOSELF = str(REPO_ROOT / "vendor" / "cytoself")
if VENDOR_CYTOSELF not in sys.path:
    sys.path.insert(0, VENDOR_CYTOSELF)

from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer


# ============================================================================
# DATASET (copied from 07b_train_cytoself.py — must stay in sync)
# ============================================================================


class ManifestNpyDataset(Dataset):
    """Dataset matching upstream Cytoself's 3-channel input.

    Produces 3 channels per cell, matching the upstream DataManagerOpenCell:
      ch0: pro     = target protein fluorescence (GFP)
      ch1: nuc     = nucleus fluorescence (DNA), intensity_adjustment=1.0
      ch2: nucdist = Euclidean distance transform of nuclear mask, scaled by 0.01

    The nucdist channel is computed on-the-fly from the DNA channel via
    Otsu threshold → binary mask → scipy.ndimage.distance_transform_edt.
    (The upstream pre-computes and stores as .npy; functionally identical.)

    Reference: Kobayashi et al., Nature Methods 2022
    Upstream code: github.com/royerlab/cytoself DataManagerOpenCell
    """

    # Intensity adjustments from upstream DataManagerOpenCell.__init__()
    INTENSITY_ADJ = {"pro": 1.0, "nuc": 1.0, "nucdist": 0.01}

    def __init__(
        self,
        df: pd.DataFrame,
        base_path_col: str = "base_path",
        cell_idx_col: str = "cell_idx",
        label_col: str = "gene_label_idx",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.base_path_col = base_path_col
        self.cell_idx_col = cell_idx_col
        self.label_col = label_col
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_array(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        base_path = str(row[self.base_path_col])
        cell_idx = int(row[self.cell_idx_col])

        # ch0: pro (GFP — target protein fluorescence)
        gfp_arr = self._load_array(f"{base_path}/gfp.npy")
        pro = gfp_arr[cell_idx].astype(np.float32, copy=True) * self.INTENSITY_ADJ["pro"]

        # ch1: nuc (DNA — raw nucleus fluorescence)
        dna_arr = self._load_array(f"{base_path}/dna.npy")
        nuc = dna_arr[cell_idx].astype(np.float32, copy=True) * self.INTENSITY_ADJ["nuc"]

        # ch2: nucdist (Euclidean distance transform of nuclear mask)
        # Otsu threshold on nuc → binary mask → EDT → scale by 0.01
        try:
            thresh = threshold_otsu(nuc)
        except ValueError:
            thresh = nuc.mean()
        mask = nuc > thresh
        nucdist = distance_transform_edt(mask).astype(np.float32) * self.INTENSITY_ADJ["nucdist"]

        # Stack 3 channels: (3, H, W) matching upstream [pro, nuc, nucdist]
        image = np.stack([pro, nuc, nucdist], axis=0)
        image_t = torch.from_numpy(image)

        label_t = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return {"image": image_t, "label": label_t}


# ============================================================================
# INFERENCE
# ============================================================================


def export_embeddings(
    trainer: CytoselfFullTrainer,
    manifest_df: pd.DataFrame,
    out_dir: Path,
    layers: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> dict[str, int]:
    """Run inference and export embeddings per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pin_memory = torch.cuda.is_available()
    counts = {}

    for split_name in ["train", "val", "test"]:
        split_df = manifest_df[manifest_df["split"] == split_name].copy()
        if len(split_df) == 0:
            print(f"  {split_name}: 0 cells — skipping")
            continue

        split_df = split_df.sort_values("row_id")
        print(f"  {split_name}: {len(split_df):,} cells")

        # Save metadata
        split_df.to_csv(out_dir / f"{split_name}_metadata.csv", index=False)

        # Build DataLoader (no augmentation for inference)
        ds = ManifestNpyDataset(split_df)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Infer each layer
        for layer in layers:
            with torch.no_grad():
                emb, _ = trainer.infer_embeddings(loader, output_layer=layer)
            np.save(out_dir / f"{split_name}_{layer}.npy", emb)
            print(f"    {layer}: shape={emb.shape}")

        counts[split_name] = len(split_df)

    return counts


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Cytoself inference on arbitrary batches using a trained model."
    )
    p.add_argument("--model-dir", type=Path, required=True,
                    help="Path to trained model dir (contains model_*.pt and data_splits/)")
    p.add_argument("--manifest-csv", type=Path, required=True,
                    help="Manifest CSV (same format as 07a output)")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Where to save {split}_metadata.csv and {split}_{layer}.npy")
    p.add_argument("--model-filename", type=str, default="model_8.pt",
                    help="Model weights file within --model-dir")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--export-layers", type=str, default="vqvec2,vqindhist1")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()
    export_layers = [x.strip() for x in args.export_layers.split(",") if x.strip()]

    # ── Load run summary for model architecture ──────────────────────────
    summary_path = args.model_dir / "run_summary.json"
    with open(summary_path) as f:
        run_summary = json.load(f)

    # ── Read label book for n_classes ────────────────────────────────────
    label_book_path = args.model_dir / "data_splits" / "label_book.csv"
    label_book = pd.read_csv(label_book_path)
    n_classes = len(label_book)
    label_to_idx = dict(zip(label_book["label"], label_book["label_idx"]))
    print(f"Model: {n_classes} classes, input_shape={run_summary['input_shape']}")

    # ── Load manifest and assign labels ──────────────────────────────────
    print(f"Loading manifest from {args.manifest_csv}")
    df = pd.read_csv(args.manifest_csv, low_memory=False)
    print(f"  Total cells: {len(df):,}")

    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["gene_label_idx"] = df["variant"].map(label_to_idx).fillna(0).astype(int)
    n_unknown = df["variant"].map(label_to_idx).isna().sum()
    if n_unknown > 0:
        print(f"  {n_unknown:,} cells have alleles not in label_book (assigned dummy label 0)")

    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        print(f"  {s}: {n:,}")

    # ── Construct model ──────────────────────────────────────────────────
    model_args = {
        "input_shape": tuple(run_summary["input_shape"]),
        "emb_shapes": ((32, 32), (4, 4)),
        "output_shape": tuple(run_summary["input_shape"]),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": n_classes,
        "fc_input_type": "vqvec",
    }
    train_args = {"lr": 1e-3, "max_epoch": 1}  # not used, required by constructor

    print("Loading model...")
    trainer = CytoselfFullTrainer(
        train_args=train_args,
        homepath=str(args.output_dir),
        model_args=model_args,
        device=args.device,
    )
    model_path = args.model_dir / args.model_filename
    trainer.load_model(str(model_path), by_weights=True)
    trainer.model.eval()
    print(f"  Loaded {model_path} on {trainer.device}")

    # ── Run inference ────────────────────────────────────────────────────
    print(f"\nExporting layers: {export_layers}")
    counts = export_embeddings(
        trainer=trainer,
        manifest_df=df,
        out_dir=args.output_dir,
        layers=export_layers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Save inference summary ───────────────────────────────────────────
    inf_summary = {
        "model_dir": str(args.model_dir),
        "model_filename": args.model_filename,
        "manifest_csv": str(args.manifest_csv),
        "n_total": int(len(df)),
        "n_per_split": {k: int(v) for k, v in counts.items()},
        "n_unknown_alleles": int(n_unknown),
        "export_layers": export_layers,
        "batches": sorted(df["batch_id"].unique().tolist()),
        "device": str(trainer.device),
    }
    with open(args.output_dir / "inference_summary.json", "w") as f:
        json.dump(inf_summary, f, indent=2)

    print(f"\nDone. Output: {args.output_dir}")

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[args.output_dir])


if __name__ == "__main__":
    main()
