# SubCell-embed Vendored Code

## Source

- **Repository**: https://github.com/CellProfiling/subcell-embed
- **Commit**: `a71236f35d75b849cb7435b38f4f0e0b28c97a62` (2026-04-20)
- **Vendored date**: 2026-04-20

## Purpose

Training infrastructure for fine-tuning SubCell ViT-B/16 models on MisLocus
single-cell crops. Two model variants:

1. **MAE-CellS-ProtS-Pool** (`ContrastMAE`): Masked Autoencoder reconstruction
   + cell-level contrastive + protein-level contrastive
2. **ViT-ProtS-Pool** (`BaseSSL`): Protein-supervised contrastive only

## Modifications

**None** — all files identical to upstream commit. Custom MisLocus dataset class
and training script live in `src/prot_loc_benchmark/` and `scripts/`, not here.

## Key Architecture

- Encoder: ViT-B/16 (768 hidden, 12 layers, 12 heads, patch_size=16, image_size=448)
- Pooler: GatedAttentionPooler (768 → 1536, 2 heads, 512 int_dim)
- Projection: 1536 → 8192 → 8192 → 512 (for contrastive loss)
- Decoder (MAE only): 512 hidden, 8 layers, 16 heads

## Dependencies

Requires `transformers==4.45.*` (later versions removed `ViTSdpaAttention`).
Also needs: lightning, torchvision, omegaconf, scipy, timm.

## Paper

Gupta et al., "SubCell: Vision foundation models for microscopy capture
single-cell biology", bioRxiv 2024. DOI: 10.1101/2024.12.06.627299
