"""MisLocus dataset for SubCell fine-tuning with HPA-scale preprocessing.

Memory-mapped design (lazy, OS-page-cache-shared across DDP ranks):
  1. __init__ opens one mmap handle per (allele, channel) .npy file — zero bulk copy.
  2. __getitem__ slices the memory-mapped arrays on demand.
  3. GPU-side HPA-scale resize runs in on_after_batch_transfer.

Prior design pre-allocated a (N_cells, 4, 128, 128) uint16 array per rank,
costing ~218 GB (train) + ~114 GB (val) per rank as non-reclaimable anon memory.
With mmap the same data lives in kernel page cache, shared across ranks and
reclaimable by the kernel — fitting the spirit-server 500 GiB cgroup quota.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from prot_loc_benchmark.config import SUBCELL_CHANNEL_FILES


class MisLocusSubCellDataset(Dataset):
    """Memory-mapped MisLocus dataset for SubCell fine-tuning.

    On construction, opens one mmap handle per (allele, channel) .npy file —
    ~zero RAM cost per rank. Data pages materialize in kernel page cache on
    first __getitem__ access and are reused across DDP ranks via shared inodes.

    The HPA-scale preprocessing (128→953→448 + normalize) is NOT done here.
    It runs on GPU via on_after_batch_transfer in the Lightning module.

    Parameters
    ----------
    manifest_csv : Path
        CSV with columns: base_path, cell_idx, gene, variant, split, ...
    split : str
        One of "train", "val", "test".
    n_cells : int
        Number of cells to sample per protein group per item.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str = "train",
        n_cells: int = 4,
        mask_prob: float = 0.0,
        color_channels: Optional[list[str]] = None,
        normalize: str = "min_max",
        return_cell_mask: bool = False,
        ssl_transform: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.split = split
        self.n_cells = n_cells
        self.return_cell_mask = return_cell_mask

        # Load manifest
        full_df = pd.read_csv(manifest_csv)
        all_proteins = sorted(full_df["gene"].unique())
        self.protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
        self.num_classes = len(all_proteins)
        self.unique_cats = all_proteins
        self.color_channels = color_channels or ["red", "yellow", "blue", "green"]

        df = full_df[full_df["split"] == split].reset_index(drop=True)

        # --- Phase 1: Index allele mmap handles (no bulk copy) ---
        print(f"MisLocusSubCellDataset[{split}]: indexing {len(df):,} cells (mmap)...")

        grouped = df.groupby("base_path")
        allele_groups = sorted(grouped.groups.keys())

        n_total = len(df)
        # Per-cell metadata (flat): allele_id (int32) → lookup in _allele_mmaps list
        self._allele_mmaps: list[list[np.ndarray]] = []
        self._flat_allele_id = np.empty(n_total, dtype=np.int32)
        self._flat_local_idx = np.empty(n_total, dtype=np.int32)
        self._protein_ids = np.empty(n_total, dtype=np.int64)

        cursor = 0
        for allele_id, base_path in enumerate(tqdm(allele_groups, desc=f"  Indexing [{split}]")):
            group = grouped.get_group(base_path)
            cell_indices = group["cell_idx"].values.astype(np.int32)
            gene = group["gene"].iloc[0]
            protein_idx = self.protein_to_idx[gene]
            n = len(cell_indices)

            # Lazy mmap: zero bytes loaded, just header parse + kernel map
            self._allele_mmaps.append([
                np.load(f"{base_path}/{ch_file}", mmap_mode="r")
                for ch_file in SUBCELL_CHANNEL_FILES
            ])
            self._flat_allele_id[cursor:cursor + n] = allele_id
            self._flat_local_idx[cursor:cursor + n] = cell_indices
            self._protein_ids[cursor:cursor + n] = protein_idx
            cursor += n

        assert cursor == n_total
        data_bytes = n_total * 4 * 128 * 128 * 2  # uint16
        print(
            f"  Indexed: {n_total:,} cells × 4ch × 128×128 "
            f"({data_bytes / 1e9:.1f} GB on disk, mmap — ~0 RAM per rank), "
            f"{self.num_classes} protein classes, {len(self._allele_mmaps)} alleles"
        )

        # Build per-protein index groups for SupCon sampling
        self._protein_groups: list[np.ndarray] = []
        for pid in range(self.num_classes):
            indices = np.where(self._protein_ids == pid)[0]
            if len(indices) > 0:
                self._protein_groups.append(indices)

        # Each item = one protein group (sample n_cells from it)
        self._n_items = len(self._protein_groups)
        print(f"  {self._n_items} protein groups, n_cells={n_cells}/item")

    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, idx: int):
        indices = self._protein_groups[idx]
        protein_idx = self._protein_ids[indices[0]]

        # Sample n_cells from this protein group
        if self.n_cells > 0 and len(indices) > self.n_cells:
            selected = np.random.choice(indices, self.n_cells, replace=False)
        else:
            selected = indices
        n = len(selected)

        # Assemble (n, 4, 128, 128) from mmap slices (kernel page cache hit after 1st pass)
        images_u16 = np.empty((n, 4, 128, 128), dtype=np.uint16)
        for k, flat_idx in enumerate(selected):
            mmaps = self._allele_mmaps[self._flat_allele_id[flat_idx]]
            local = self._flat_local_idx[flat_idx]
            for ch in range(4):
                images_u16[k, ch] = mmaps[ch][local]
        images_tensor = torch.from_numpy(images_u16.astype(np.float32))

        # Dual view (augmentation + GPU resize applied after batch transfer)
        x_i = images_tensor
        x_j = images_tensor.clone()

        # Protein labels
        protein_id_tensor = torch.full((n,), protein_idx, dtype=torch.long)

        # One-hot targets
        targets = torch.zeros(n, self.num_classes, dtype=torch.float32)
        targets[:, protein_idx] = 1.0

        # Cell mask (computed after GPU preprocessing if needed)
        mask = None

        return x_i, x_j, protein_id_tensor, targets, mask
