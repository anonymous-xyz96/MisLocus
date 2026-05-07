#!/usr/bin/env python3
"""Extract SubCell embeddings from MisLocus single-cell crops (no fine-tuning).

Approach 08a: Direct embedding extraction using pre-trained SubCell checkpoints
from https://virtualcellmodels.cziscience.com/ (s3://czi-subcell-public/models/).
No training or fine-tuning — we use the frozen encoder + attention pooling to
produce 1536-dim embeddings, then feed them into the same XGBoost classification
pipeline (09_classify.py) used for CellProfiler and Cytoself representations.

Vendored source: vendor/subcellportable/ (cloned from github.com/czi-ai/SubCellPortable)
  - NO modifications to vendored code; all files are identical to upstream.
  - Requires transformers==4.45.* (ViTSdpaAttention removed in later versions).

Checkpoint source (downloaded via `aws s3 cp --no-sign-request`):
  s3://czi-subcell-public/models/DNA-Protein_MAE-CellS-ProtS-Pool.pth        → weights/bg/mae_.../encoder.pth
  s3://czi-subcell-public/models/DNA-Protein_ViT-ProtS-Pool.pth              → weights/bg/vit_.../encoder.pth
  s3://czi-subcell-public/models/MT-DNA-Protein_MAE-CellS-ProtS-Pool.pth     → weights/rbg/mae_.../encoder.pth
  s3://czi-subcell-public/models/MT-DNA-Protein_ViT-ProtS-Pool.pth           → weights/rbg/vit_.../encoder.pth
  s3://czi-subcell-public/models/all_channels_MAE-CellS-ProtS-Pool.pth       → weights/rybg/mae_.../encoder.pth
  s3://czi-subcell-public/models/all_channels_ViT-ProtS-Pool.pth             → weights/rybg/vit_.../encoder.pth

Physical-scale rescaling rationale:
  SubCell was trained on HPA 63x confocal (0.0801 µm/px). Our MisLocus data
  is 20x spinning disk (0.598 µm/px, Opera Phenix, Andor Zyla, 2×2 binning).
  We upscale crops by 7.47× to match the physical pixel size, then center-crop
  to 448×448 (matches training resolution). This ensures each 16×16 ViT patch
  covers 1.28 µm — the same as in HPA training — so subcellular structures
  (mitochondria, ER, nuclear features) occupy the correct number of patches.
  Fine intra-patch texture is lost (0.25 µm optical vs HPA's 0.18 µm), but
  JUMP1 Cell Painting evaluation (~8× upscale) achieved SOTA with this approach.

Channel mapping (MisLocus → SubCell slot):
  DNA  (Hoechst, 405/456nm)         → B (Nucleus)   — direct match
  GFP  (Alexa 488, 488/522nm)       → G (Protein)   — direct match
  AGP  (Alexa 568/Phalloidin, 561nm)→ R (MT)        — actin ↔ microtubules (both cytoskeletal)
  Mito (MitoTracker Deep Red, 640nm)→ Y (ER)        — mitochondria ↔ ER (both organellar networks)

Performance (H100 NVL, FP16, 448×448 crop, batch_size=128):
  ~1.0 ms/cell → ~12 min per batch (724K cells) per config
  6 configs × 2 batches = 12 runs → ~36 min total with 4 GPUs

Usage:
    pixi run -e subcell python scripts/08a_extract_subcell_embeddings.py \\
        --batch 2025_01_27_Batch_13 --channels bg --gpu 0

    # All 6 configurations (3 channels × 2 model types) across 4 GPUs:
    for ch in bg rbg rybg; do
      for mt in mae_contrast_supcon_model vit_supcon_model; do
        pixi run -e subcell python scripts/08a_extract_subcell_embeddings.py \\
          --batch 2025_01_27_Batch_13 --channels $ch --model-type $mt --gpu 0
      done
    done

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Vendor imports (SubCellPortable)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SUBCELL = str(REPO_ROOT / "vendor" / "subcellportable")
if VENDOR_SUBCELL not in sys.path:
    sys.path.insert(0, VENDOR_SUBCELL)

try:
    from vit_model import ViTPoolClassifier  # noqa: E402
except ImportError as e:
    raise ImportError(
        "SubCellPortable requires transformers==4.45.* "
        "(ViTSdpaAttention was removed in later versions). "
        "Run this script inside the 'subcell' pixi environment: "
        "pixi run -e subcell python scripts/08a_extract_subcell_embeddings.py"
    ) from e

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from prot_loc_benchmark.config import (  # noqa: E402
    CELLPROFILER_DIR,
    FOCUS_BATCHES,
    INTERIM_DIR,
    SINGLE_CELL_CROPS_DIR,
    SUBCELL_CHANNEL_CONFIGS,
    SUBCELL_EMBED_DIM,
    SUBCELL_MODEL_TYPES,
    SUBCELL_PIXEL_SIZE_HPA,
    SUBCELL_PIXEL_SIZE_MISLOCUS,
    SUBCELL_SCALE_FACTOR,
    SUBCELL_WEIGHTS_DIR,
)

# ---------------------------------------------------------------------------
# Metadata columns required by classification pipeline
# ---------------------------------------------------------------------------
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


# ============================================================================
# Model loading
# ============================================================================


def load_model(
    channels: str,
    model_type: str,
    device: torch.device,
) -> ViTPoolClassifier:
    """Load a pre-trained SubCell model from vendored configs + local weights.

    Only the encoder weights (encoder.pth) are required. Classifier weights
    are NOT needed — we extract embeddings via encoder + attention pooling,
    bypassing the classification head entirely.

    Weight files must be present in SUBCELL_WEIGHTS_DIR / channels / model_type /.
    Download from https://virtualcellmodels.cziscience.com/ before running.
    """
    # Read model config from vendored YAML
    config_path = (
        Path(VENDOR_SUBCELL) / "models" / channels / model_type / "model_config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model_config"]
    model = ViTPoolClassifier(model_config)

    # Load encoder + pool_model weights only (skip classifiers)
    weights_dir = SUBCELL_WEIGHTS_DIR / channels / model_type
    encoder_path = weights_dir / "encoder.pth"

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Missing encoder weights: {encoder_path}\n"
            f"Download from https://virtualcellmodels.cziscience.com/"
        )

    checkpoint = torch.load(str(encoder_path), map_location=device, weights_only=True)

    # Extract encoder weights (prefixed with "encoder.")
    encoder_ckpt = {
        k[len("encoder."):]: v for k, v in checkpoint.items() if k.startswith("encoder.")
    }
    status = model.encoder.load_state_dict(encoder_ckpt)
    print(f"  Encoder: {status}")

    # Extract pool_model weights (prefixed with "pool_model.")
    pool_ckpt = {
        k.replace("pool_model.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("pool_model.")
    }
    # SubCellPortable applies key remapping: "1." → "0."
    pool_ckpt = {k.replace("1.", "0."): v for k, v in pool_ckpt.items()}
    if pool_ckpt and model.pool_model:
        status = model.pool_model.load_state_dict(pool_ckpt)
        print(f"  Pool model: {status}")

    model.to(device)
    model.eval()
    return model


# ============================================================================
# Preprocessing
# ============================================================================


def min_max_standardize(im: torch.Tensor) -> torch.Tensor:
    """Per-cell global min-max normalization to [0, 1].

    Matches SubCellPortable's inference.py:min_max_standardize exactly:
    global across all channels and spatial dims per cell.
    """
    min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)
    return (im - min_val) / (max_val - min_val + 1e-6)



# ============================================================================
# Metadata join + output
# ============================================================================


def export_batch(
    batch_id: str,
    all_embeddings: np.ndarray,
    all_metadata: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Join embeddings with features.parquet metadata, write embeddings.parquet."""
    features_path = CELLPROFILER_DIR / batch_id / "features.parquet"
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found — cannot get metadata columns")
        return

    # Build embedding DataFrame with CellID
    feat_cols = [f"SubCell_{i}" for i in range(all_embeddings.shape[1])]
    emb_df = pl.DataFrame(
        {col: all_embeddings[:, i].astype(np.float32) for i, col in enumerate(feat_cols)}
    )

    # Add CellID from crop metadata
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
    available_meta = [c for c in REQUIRED_META_COLS if c in joined.columns]
    result = joined.select(available_meta + feat_cols)

    # Drop nulls
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


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract SubCell embeddings from MisLocus single-cell crops."
    )
    p.add_argument(
        "--batch",
        nargs="+",
        default=[b for b in FOCUS_BATCHES],
        help="Batch ID(s) to process (default: all focus batches)",
    )
    p.add_argument(
        "--channels",
        choices=list(SUBCELL_CHANNEL_CONFIGS.keys()),
        default="rybg",
        help="Channel configuration: bg (DNA+GFP), rbg (+AGP), rybg (+Mito)",
    )
    p.add_argument(
        "--model-type",
        choices=list(SUBCELL_MODEL_TYPES.keys()),
        default="mae",
        help="Model variant: mae (MAE-CellS-ProtS-Pool) or vit (ViT-ProtS-Pool)",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size (default: 128)",
    )
    p.add_argument(
        "--preprocess-chunk",
        type=int,
        default=256,
        help="Cells per GPU preprocessing chunk (default: 256)",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        default=448,
        choices=[448, 640],
        help="Inference crop size in pixels (default: 448, matches training; "
        "640 matches HPA inference but is 2× slower)",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16 (slower but higher precision)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve short model name to full directory name
    mt_short = args.model_type  # "mae" or "vit"
    model_type = SUBCELL_MODEL_TYPES[mt_short]
    rep_name = f"subcell_portable_{args.channels}_{mt_short}"
    channel_files = SUBCELL_CHANNEL_CONFIGS[args.channels]

    print(f"SubCell embedding extraction")
    print(f"  Representation: {rep_name}")
    print(f"  Channels: {args.channels} → {channel_files}")
    print(f"  Model: {model_type}")
    use_fp16 = not args.fp32
    print(f"  Scale factor: {SUBCELL_SCALE_FACTOR:.3f}× "
          f"({SUBCELL_PIXEL_SIZE_MISLOCUS} µm/px → {SUBCELL_PIXEL_SIZE_HPA} µm/px)")
    print(f"  Inference crop: {args.crop_size}×{args.crop_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
    print()

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading model {args.channels}/{model_type}...")
    model = load_model(args.channels, model_type, device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Process each batch
    output_dirs = []
    for batch_id in args.batch:
        print(f"=== {batch_id} ===")
        crops_dir = SINGLE_CELL_CROPS_DIR / batch_id
        if not crops_dir.exists():
            print(f"  SKIP: {crops_dir} not found")
            continue

        # Discover alleles
        allele_dirs = sorted(
            d for d in crops_dir.iterdir()
            if d.is_dir() and (d / "metadata.parquet").exists()
        )
        print(f"  Found {len(allele_dirs)} alleles")

        # --- Phase 1: Load all crops + metadata into memory ---
        # This eliminates per-allele I/O overhead during GPU inference.
        print("  Loading crops...")
        all_arrays = {ch: [] for ch in channel_files}
        all_metadata = []
        n_skipped = 0

        for allele_dir in tqdm(allele_dirs, desc="  Load"):
            missing = [ch for ch in channel_files if not (allele_dir / f"{ch}.npy").exists()]
            if missing:
                n_skipped += 1
                continue

            meta = pl.read_parquet(str(allele_dir / "metadata.parquet"))
            if meta.height == 0:
                continue

            for ch in channel_files:
                all_arrays[ch].append(np.load(allele_dir / f"{ch}.npy"))
            all_metadata.append(meta)

        if not all_metadata:
            print(f"  No cells found for {batch_id}")
            continue
        if n_skipped:
            print(f"  Skipped {n_skipped} alleles with missing channels")

        # Concatenate all alleles into single arrays
        concat_arrays = {ch: np.concatenate(all_arrays[ch], axis=0) for ch in channel_files}
        all_meta = pl.concat(all_metadata)
        n_total = all_meta.height
        del all_arrays  # free memory
        print(f"  Loaded: {n_total:,} cells × {len(channel_files)} channels")

        # --- Phase 2: Preprocess + inference in a single streaming pass ---
        # Process in GPU chunks: rescale → crop → normalize → inference.
        # No CPU↔GPU round-trips between alleles.
        print("  Extracting embeddings...")
        torch.set_grad_enabled(False)
        chunk_size = args.preprocess_chunk
        rescaled_size = int(128 * SUBCELL_SCALE_FACTOR)
        crop_size = args.crop_size
        # Pre-compute constant crop offsets (rescaled_size is fixed)
        crop_top = (rescaled_size - crop_size) // 2
        crop_left = (rescaled_size - crop_size) // 2
        # Pre-allocate embedding array to avoid doubling memory at concat
        all_emb = np.empty((n_total, SUBCELL_EMBED_DIM), dtype=np.float32)
        cursor = 0

        for start in tqdm(range(0, n_total, chunk_size), desc="  Infer"):
            end = min(start + chunk_size, n_total)

            # Stack channels and transfer to GPU (cast during transfer, no CPU copy)
            stacked = np.stack(
                [concat_arrays[ch][start:end] for ch in channel_files], axis=1
            )
            tensor = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)

            # Rescale to HPA pixel size
            tensor = F.interpolate(
                tensor, size=rescaled_size, mode="bilinear", align_corners=False
            )

            # Center-crop
            tensor = tensor[:, :, crop_top : crop_top + crop_size, crop_left : crop_left + crop_size]

            # Normalize
            tensor = min_max_standardize(tensor)

            # Inference (sub-batch if chunk > batch_size)
            for i in range(0, tensor.shape[0], args.batch_size):
                batch = tensor[i : i + args.batch_size]
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    encoder_out = model.encoder(
                        batch, output_attentions=False, interpolate_pos_encoding=True
                    )
                    pool_op, _ = model.pool_model(encoder_out.last_hidden_state)
                n_batch = pool_op.shape[0]
                all_emb[cursor : cursor + n_batch] = pool_op.float().cpu().numpy()
                cursor += n_batch
        del concat_arrays  # free memory
        assert all_emb.shape == (n_total, SUBCELL_EMBED_DIM), (
            f"Shape mismatch: {all_emb.shape} vs ({n_total}, {SUBCELL_EMBED_DIM})"
        )
        print(f"  Total: {all_emb.shape[0]:,} cells × {all_emb.shape[1]} dims")

        # --- Phase 3: Metadata join + export ---
        output_dir = INTERIM_DIR / rep_name / batch_id
        export_batch(batch_id, all_emb, all_meta, output_dir)
        output_dirs.append(output_dir)
        print()

    if output_dirs:
        from prot_loc_benchmark.provenance import record
        record(output_dirs=output_dirs)


if __name__ == "__main__":
    main()
