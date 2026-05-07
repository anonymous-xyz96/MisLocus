#!/usr/bin/env python3
"""Train Cytoself VQ-VAE on MisLocus single-cell crops.

Adapted from an internal reference implementation derived from the
upstream Cytoself baseline example script.

Key differences from reference:
  - All alleles included (no --only-wt filter)
  - Nuclear distance transform computed on-the-fly from DNA channel
  - Technical replicate split (T1+T2=train, T3=val, T4=test)
  - Allele-level labels for pretext classification

Usage:
    CUDA_VISIBLE_DEVICES=0 pixi run -e cytoself train-cytoself \\
        --manifest-csv data/interim/cytoself/manifests/manifest_b13_b16.csv \\
        --output-dir data/interim/cytoself/models/gfp_nucdist_s43 \\
        --max-epoch 20 --seed 43

See docs/cytoself_training_design.md for design rationale.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow running from repo root without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_CYTOSELF = str(REPO_ROOT / "vendor" / "cytoself")
if VENDOR_CYTOSELF not in sys.path:
    sys.path.insert(0, VENDOR_CYTOSELF)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer


# ============================================================================
# DATASET
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
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.base_path_col = base_path_col
        self.cell_idx_col = cell_idx_col
        self.label_col = label_col
        self._cache: dict[str, np.ndarray] = {}

        self.transform = None
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomApply([
                    lambda x: transforms.functional.rotate(x, 0),
                    lambda x: transforms.functional.rotate(x, 90),
                    lambda x: transforms.functional.rotate(x, 180),
                    lambda x: transforms.functional.rotate(x, 270),
                ]),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
            ])

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

        if self.transform is not None:
            image_t = self.transform(image_t)

        label_t = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return {"image": image_t, "label": label_t}


# ============================================================================
# DATA MANAGEMENT
# ============================================================================


@dataclass
class SimpleDataManager:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    unique_labels: np.ndarray
    train_variance: float
    val_variance: float
    test_variance: float


def compute_variance(df: pd.DataFrame, n_sample: int = 512) -> float:
    """Compute pixel variance from a sample of cells."""
    if len(df) == 0:
        return 1.0
    sample_df = df.iloc[:min(len(df), n_sample)].copy()
    ds = ManifestNpyDataset(sample_df, augment=False)
    vals = []
    for i in range(len(ds)):
        x = ds[i]["image"].numpy()
        vals.append(x.astype(np.float64).ravel())
    if not vals:
        return 1.0
    v = np.var(np.concatenate(vals))
    return float(max(v, 1e-8))


def add_label_index(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "variant",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Map allele names to integer indices for classification."""
    unique_labels = np.array(sorted(train_df[label_col].astype(str).unique()))
    label_to_idx = {g: i for i, g in enumerate(unique_labels)}

    def _map(df0: pd.DataFrame) -> pd.DataFrame:
        out = df0.copy()
        out["gene_label_idx"] = out[label_col].astype(str).map(label_to_idx)
        out = out[out["gene_label_idx"].notna()].copy()
        out["gene_label_idx"] = out["gene_label_idx"].astype(int)
        return out

    return _map(train_df), _map(val_df), _map(test_df), unique_labels


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 8,
    disable_augment: bool = False,
) -> tuple[SimpleDataManager, dict[str, DataLoader]]:
    """Build train/val/test DataLoaders and deterministic export loaders."""
    pin_memory = torch.cuda.is_available()

    train_ds = ManifestNpyDataset(train_df, augment=not disable_augment)
    val_ds = ManifestNpyDataset(val_df, augment=False)
    test_ds = ManifestNpyDataset(test_df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    dm = SimpleDataManager(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        unique_labels=np.array(sorted(train_df["variant"].astype(str).unique())),
        train_variance=compute_variance(train_df),
        val_variance=compute_variance(val_df),
        test_variance=compute_variance(test_df),
    )

    # Deterministic export loaders (no augmentation, sorted by row_id)
    export_loaders = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sorted_df = split_df.sort_values("row_id")
        export_loaders[split_name] = DataLoader(
            ManifestNpyDataset(sorted_df, augment=False),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return dm, export_loaders


# ============================================================================
# EMBEDDING EXPORT
# ============================================================================


def export_embeddings(
    trainer: CytoselfFullTrainer,
    export_loaders: dict[str, DataLoader],
    split_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
    layers: Sequence[str],
) -> None:
    """Export embeddings for each split and layer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, loader in export_loaders.items():
        df0 = split_dfs[split].sort_values("row_id").copy()
        df0.to_csv(out_dir / f"{split}_metadata.csv", index=False)
        for layer in layers:
            emb, _ = trainer.infer_embeddings(loader, output_layer=layer)
            np.save(out_dir / f"{split}_{layer}.npy", emb)
            print(f"  Exported {split}/{layer}: shape={emb.shape}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Cytoself VQ-VAE on MisLocus crops.")
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/interim/cytoself/models/gfp_nucdist_s43"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--disable-augment", action="store_true")
    p.add_argument("--max-cells-total", type=int, default=0, help="Subsample for smoke tests")
    p.add_argument("--export-layers", type=str, default="vqvec2,vqindhist1,vqind1")
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()
    export_layers = [x.strip() for x in args.export_layers.split(",") if x.strip()]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "data_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter manifest ──────────────────────────────────────────
    print(f"Loading manifest from {args.manifest_csv}")
    df = pd.read_csv(args.manifest_csv, low_memory=False)
    print(f"  Total cells: {len(df):,}")

    # Subsample for smoke tests
    if args.max_cells_total > 0 and len(df) > args.max_cells_total:
        df = df.sample(n=args.max_cells_total, random_state=args.seed).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} cells")

    df["row_id"] = np.arange(len(df), dtype=np.int64)

    # ── Split by technical replicate ──────────────────────────────────────
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    print(f"  Split: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more splits are empty. Check manifest 'split' column.")

    # ── Allele-level label indices ────────────────────────────────────────
    train_df, val_df, test_df, unique_labels = add_label_index(
        train_df, val_df, test_df, label_col="variant"
    )
    n_classes = len(unique_labels)
    print(f"  Unique alleles (classes): {n_classes}")

    # Save split info
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    pd.DataFrame({"label": unique_labels, "label_idx": np.arange(n_classes)}).to_csv(
        split_dir / "label_book.csv", index=False
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────
    print("Building DataLoaders...")
    datamanager, export_loaders = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disable_augment=args.disable_augment,
    )

    # ── Configure model ───────────────────────────────────────────────────
    # 3-channel model matching upstream Cytoself (DataManagerOpenCell):
    # [pro, nuc, nucdist] at 128x128 (native crop size, no downscaling)
    model_args = {
        "input_shape": (3, 128, 128),
        "emb_shapes": ((32, 32), (4, 4)),
        "output_shape": (3, 128, 128),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": n_classes,
        "fc_input_type": "vqvec",
    }
    train_args = {
        "lr": args.lr,
        "max_epoch": args.max_epoch,
        "reducelr_patience": 3,
        "reducelr_increment": 0.1,
        "earlystop_patience": 6,
    }

    print(f"Training CytoselfFull: {n_classes} classes, {args.max_epoch} epochs")
    trainer = CytoselfFullTrainer(
        train_args=train_args,
        homepath=str(out_dir),
        model_args=model_args,
        device=args.device,
    )
    trainer.fit(datamanager, tensorboard_path="tb_logs")

    # ── Export embeddings ─────────────────────────────────────────────────
    print("Exporting embeddings...")
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    export_embeddings(
        trainer=trainer,
        export_loaders=export_loaders,
        split_dfs=split_dfs,
        out_dir=out_dir / "embeddings_export",
        layers=export_layers,
    )

    # ── Save run summary ──────────────────────────────────────────────────
    summary = {
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_classes": int(n_classes),
        "channels": "pro+nuc+nucdist (upstream 3ch, 128x128)",
        "intensity_adjustment": ManifestNpyDataset.INTENSITY_ADJ,
        "input_shape": list(model_args["input_shape"]),
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(trainer.device),
        "export_layers": export_layers,
        "split_strategy": "tech_replicate (T1+T2=train, T3=val, T4=test)",
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete. Output: {out_dir}")
    print(trainer.history.tail(1).to_string(index=False))

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[out_dir])


if __name__ == "__main__":
    main()
