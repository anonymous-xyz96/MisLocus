#!/usr/bin/env python3
"""Convert existing crop manifests into a CSV for Cytoself training.

This is a lightweight format converter — the cell set is already defined by
the cell-crop dataset bundle. We just reshape
``crop_manifest/{batch}/manifest.parquet`` into the format
``ManifestNpyDataset`` expects, with allele-identity labels (the
``variant`` column) used as the classification target.

Usage:
    pixi run -e cytoself preprocess-cytoself
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (hardcoded since this env can't import prot_loc_benchmark)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CROPS_DIR = DATA_DIR / "interim" / "single_cell_crops"
MANIFEST_OUT_DIR = DATA_DIR / "interim" / "cytoself" / "manifests"

FOCUS_BATCHES = [
    "2025_01_27_Batch_13",
    "2025_01_28_Batch_14",
    "2025_03_17_Batch_15",
    "2025_03_17_Batch_16",
]

# Technical replicate split: T1+T2=train, T3=val, T4=test
TECH_REP_SPLIT = {1: "train", 2: "train", 3: "val", 4: "test"}


def get_template_number(plate_name: str) -> int | None:
    """Extract template number (1-4) from plate barcode.

    Returns None for plates without T-numbering.
    """
    m = re.search(r"T(\d+)", plate_name)
    if not m:
        return None
    return int(m.group(1))


def build_manifest(batches: list[str], crops_dir: Path) -> pd.DataFrame:
    """Build per-cell manifest from existing crop metadata."""
    rows = []
    for batch in batches:
        batch_dir = crops_dir / batch
        if not batch_dir.exists():
            print(f"  SKIP {batch}: directory not found")
            continue

        allele_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
        print(f"  {batch}: {len(allele_dirs)} allele directories")

        for allele_dir in allele_dirs:
            meta_path = allele_dir / "metadata.parquet"
            if not meta_path.exists():
                continue

            meta = pd.read_parquet(meta_path)
            n_cells = len(meta)
            if n_cells == 0:
                continue

            variant = allele_dir.name
            gene = variant.split("_")[0] if "_" in variant else variant

            for idx in range(n_cells):
                row = meta.iloc[idx]
                plate = str(row["Metadata_Plate"])
                template_num = get_template_number(plate)
                # Plates without T-numbering → assign all to "test"
                split = TECH_REP_SPLIT[template_num] if template_num is not None else "test"

                rows.append({
                    "base_path": str(allele_dir.resolve()),
                    "cell_idx": idx,
                    "batch_id": batch,
                    "gene": gene,
                    "variant": variant,
                    "metadata_cell_id": str(row.get("Metadata_CellID", "")),
                    "plate": plate,
                    "template_num": template_num if template_num is not None else 0,
                    "split": split,
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No cells found. Check --crops-dir and --batches.")
    print(f"  Total cells: {len(df):,}")

    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        print(f"  {s}: {n:,} cells ({n * 100 // len(df)}%)")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Cytoself training manifest from existing crop data.")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=MANIFEST_OUT_DIR / "manifest_b13_b16.csv",
    )
    p.add_argument("--crops-dir", type=Path, default=CROPS_DIR)
    p.add_argument(
        "--batches",
        type=str,
        default=",".join(FOCUS_BATCHES),
        help="Comma-separated batch IDs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    batches = [b.strip() for b in args.batches.split(",") if b.strip()]

    print(f"Building Cytoself manifest for {len(batches)} batches")
    df = build_manifest(batches, args.crops_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df):,} rows to {args.output_csv}")

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[args.output_csv.parent])


if __name__ == "__main__":
    main()
