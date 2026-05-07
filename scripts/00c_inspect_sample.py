#!/usr/bin/env python3
"""Print a summary of the downloaded dataset sample under ``data/sample/``.

Companion to ``scripts/00_download_dataset.py --sample``. Walks the
``data/sample/`` tree, lists alleles per batch and on-disk sizes, then
shows the schema and per-allele cell counts of
``manifest_sample.parquet`` if present.

Usage:
    pixi run python scripts/00c_inspect_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prot_loc_benchmark.config import DATA_DIR


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    sample_dir = DATA_DIR / "sample"
    if not sample_dir.is_dir():
        print(
            f"data/sample/ not found at {sample_dir}.\n"
            "Run `just download-sample` (or "
            "`pixi run python scripts/00_download_dataset.py --sample`) first.",
            file=sys.stderr,
        )
        return 1

    batches = sorted(p for p in sample_dir.iterdir() if p.is_dir())
    total = _dir_bytes(sample_dir) / 1e6
    print(f"=== {sample_dir} — {len(batches)} batches, {total:.1f} MB total ===")
    for b in batches:
        alleles = sorted(p for p in b.iterdir() if p.is_dir())
        size_mb = _dir_bytes(b) / 1e6
        names = ", ".join(a.name for a in alleles)
        print(f"  {b.name}: {len(alleles)} alleles ({size_mb:.1f} MB)")
        print(f"    {names}")

    mp = sample_dir / "manifest_sample.parquet"
    if not mp.is_file():
        print(f"\n(no manifest_sample.parquet at {mp})")
        return 0

    import polars as pl  # imported lazily so missing data doesn't pull polars

    m = pl.read_parquet(mp)
    print(f"\n=== {mp.name} — {m.height} cells × {len(m.columns)} columns ===")
    print(f"first 8 columns: {m.columns[:8]}")

    if "Metadata_gene_allele" in m.columns:
        per_allele = (
            m.group_by("Metadata_gene_allele")
            .agg(pl.len().alias("n_cells"))
            .sort("Metadata_gene_allele")
        )
        print(f"\ncells per allele ({per_allele.height} alleles):")
        for row in per_allele.iter_rows(named=True):
            print(f"  {row['Metadata_gene_allele']:24s} {row['n_cells']:>6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
