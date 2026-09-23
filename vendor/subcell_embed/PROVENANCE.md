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

- Protocol v2: `models/object_aware_mae.py:ViTMAEEmbeddings.random_masking`
  returns original-order tokens, a zero mask and identity restore indices when
  `mask_ratio=0` **in eval mode only**. This avoids numerical permutation noise
  and RNG consumption during validation/extraction. Training, including the
  zero-mask second view's shuffle and RNG consumption, retains upstream behavior.
  Nonzero random masking is unchanged in both modes.
- The allele-specific dataset/training loop lives in `src/prot_loc_benchmark/`
  and `scripts/`. The historical vendor localization training modules are not
  the v2 MisLocus entry point.

## Key Architecture

- Encoder: ViT-B/16 (768 hidden, 12 layers, 12 heads, patch_size=16, image_size=448)
- Pooler: GatedAttentionPooler (768 → 1536, 2 heads, 512 int_dim)
- Projection: 1536 → 8192 → 8192 → 512 (for contrastive loss)
- Decoder (MAE only): 512 hidden, 8 layers, 16 heads

## Dependencies

The validated companion stack pins PyTorch2.4.1, Transformers4.45.2,
Torchvision0.19.1, Lightning2.6.1 and timm1.0.26; use the `subcell` environment
and committed `pixi.lock`, not an unconstrained upgrade. Transformer internals
such as `ViTSdpaAttention` are version-sensitive. Other dependencies include
omegaconf and scipy.

## Execution scope

Completed allele-v2 training used companion commit `03b1961`; seed42 extraction
used `e306037`. Their immutable source archives are authoritative, not this later
documentation revision. See [protocol and limitations](../../docs/subcell_allele_v2.md)
and [completion evidence](../../docs/evidence/README.md).

The local loop retains same-run cross-rank RNG correlation. Different run seeds
are not identical experiments, but no claim of independent per-rank streams or
single-seed downstream robustness is made. Later runtime-metadata hardening does
not reconstruct unrecorded historical determinism flags.

## Paper

Gupta et al., "SubCell: Vision foundation models for microscopy capture
single-cell biology", bioRxiv 2024. DOI: 10.1101/2024.12.06.627299
