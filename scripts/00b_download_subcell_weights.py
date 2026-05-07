#!/usr/bin/env python3
"""Download SubCell pretrained encoder checkpoints from CZI's public S3 bucket.

Six ``encoder.pth`` checkpoints (~2 GB total), required only if you intend
to run ``scripts/08a`` (extract SubCell embeddings) or ``scripts/08c``
(fine-tune SubCell). Skip this step if you're only using shipped features.

Re-runs are idempotent (skips destinations that already exist); pass
``--force`` to re-fetch. Hashes aren't pinned because the upstream files
live in a third-party bucket we don't control.

Usage:
    pixi run python scripts/00b_download_subcell_weights.py
    pixi run python scripts/00b_download_subcell_weights.py --force
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pooch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prot_loc_benchmark.config import SUBCELL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

# CZI's public bucket — anonymous HTTPS access (no AWS credentials).
SUBCELL_WEIGHT_BASE_URL = "https://czi-subcell-public.s3.amazonaws.com/models/"

# Map (channels, model_type) → S3 filename.
SUBCELL_WEIGHT_URLS: dict[tuple[str, str], str] = {
    ("bg",   "mae_contrast_supcon_model"): "DNA-Protein_MAE-CellS-ProtS-Pool.pth",
    ("bg",   "vit_supcon_model"):          "DNA-Protein_ViT-ProtS-Pool.pth",
    ("rbg",  "mae_contrast_supcon_model"): "MT-DNA-Protein_MAE-CellS-ProtS-Pool.pth",
    ("rbg",  "vit_supcon_model"):          "MT-DNA-Protein_ViT-ProtS-Pool.pth",
    ("rybg", "mae_contrast_supcon_model"): "all_channels_MAE-CellS-ProtS-Pool.pth",
    ("rybg", "vit_supcon_model"):          "all_channels_ViT-ProtS-Pool.pth",
}

# TODO: pin SHA-256 hashes once verified.
SUBCELL_WEIGHT_HASHES: dict[str, str | None] = {
    fn: None for fn in SUBCELL_WEIGHT_URLS.values()
}


def download_subcell_weights(force: bool = False) -> None:
    """Download SubCell pretrained encoders from CZI's public S3 bucket."""
    SUBCELL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = SUBCELL_WEIGHTS_DIR / "_cache"

    fetcher = pooch.create(
        path=cache_dir,
        base_url=SUBCELL_WEIGHT_BASE_URL,
        registry=SUBCELL_WEIGHT_HASHES,
    )

    n_skipped, n_downloaded = 0, 0
    for (channels, model_type), filename in SUBCELL_WEIGHT_URLS.items():
        dst_dir = SUBCELL_WEIGHTS_DIR / channels / model_type
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "encoder.pth"

        if dst.exists() and not force:
            logger.info("[skip] %s/%s/encoder.pth already present", channels, model_type)
            n_skipped += 1
            continue

        logger.info("Downloading %s → %s", filename, dst)
        cached = Path(fetcher.fetch(filename, progressbar=True))
        # Move from pooch's cache into the canonical layout (encoder.pth).
        if dst.exists():
            dst.unlink()
        shutil.move(str(cached), str(dst))
        n_downloaded += 1

    if cache_dir.exists() and not any(cache_dir.iterdir()):
        cache_dir.rmdir()

    logger.info(
        "SubCell weights: %d downloaded, %d already present (in %s)",
        n_downloaded, n_skipped, SUBCELL_WEIGHTS_DIR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if destination already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    download_subcell_weights(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
