# Model Checkpoints & Weights

All files in this directory are gitignored (except this README). This document
explains the two-tier storage system and where to obtain each file.

## Weights vs. Checkpoints

This project distinguishes between two types of saved model files:

| | Weights (`best.pth`) | Full Checkpoint (`best_checkpoint.pth`) |
|---|---|---|
| **Contains** | Model parameters only | Model params + optimizer state + scheduler + epoch + RNG |
| **Size** | ~300 MB (ViT-B) | ~900 MB (ViT-B) |
| **Used for** | Inference / embedding extraction | Resume training / audit / reproduce exact training state |
| **Who needs it** | Everyone | Training owner + team reviewers |

**Rule of thumb**: If you're running `scripts/03_extract_{model}.py`, you need
weights. If you're running `scripts/train_{model}.py --resume`, you need the full
checkpoint.

## Two-Tier Storage

### Tier 1: Internal (team S3, full fidelity)

Everything needed to audit and reproduce training. Stored on team S3, never in git.

```
checkpoints/
├── README.md                          # This file (git-tracked)
│
├── {model}/pretrained/                # Downloaded from upstream repos
│   └── {arch}.pth                     #   Weights-only
│
└── {model}/{variant}/                 # Fine-tuned on MisLocus
    ├── best.pth                       #   Weights-only at best val epoch (for inference)
    ├── best_checkpoint.pth            #   Full checkpoint at best val epoch (for audit/resume)
    ├── last_checkpoint.pth            #   Full checkpoint at final epoch (for resume)
    ├── config.yaml                    #   Frozen copy of training config
    ├── training_log.json              #   Per-epoch metrics (loss, lr, time, GPU mem)
    └── training_summary.json          #   Final summary (best epoch, total time, hardware)
```

**Why three saved files, not one per epoch?** Per-epoch metrics live in
`training_log.json` (a few KB). Full checkpoints are only saved at the *best* and
*last* epoch. To verify any intermediate epoch, retrain from scratch with the frozen
config + deterministic seed — that's cheaper than storing 100 x 900 MB.

### Tier 2: Public release (weights-only)

When publishing results, release only `best.pth` files — external users only need
inference capability, not optimizer state.

```
releases/                              # Created at publication time
└── {model}/
    ├── best.pth                       #   Weights-only
    └── config.yaml                    #   Training config (for reference)
```

## Pretrained Weights (from upstream)

| Model | File | Source | SHA256 |
|-------|------|--------|--------|
| SubCell ViT-B/16 | `subcell/pretrained/vit_b_16.pth` | [SubCellPortable](https://github.com/czi-ai/SubCellPortable) | TBD |
| Cytoself VQ-VAE-2 | `cytoself/pretrained/opencell_vqvae2.pth` | [royerlab/cytoself](https://github.com/royerlab/cytoself) | TBD |

Download pretrained weights:

```bash
just download-weights-subcell
just download-weights-cytoself
```

## Fine-tuned Checkpoints (team internal)

| Model | Variant | Training Issue | Date | Notes |
|-------|---------|---------------|------|-------|
| — | — | — | — | — |

Upload / download fine-tuned checkpoints:

```bash
# Upload your training results to team S3
just put-checkpoints-for subcell/finetune_v1

# Download a teammate's checkpoints
just get-checkpoints-for subcell/finetune_v1
```

## Adding New Checkpoints

1. Train your model (see `docs/plans/benchmarking_multiple_method_with_collab.md`,
   Section 8 — Training Protocol)
2. Your training script must save all 5 files:
   - `best.pth` — weights-only (`torch.save(model.state_dict(), ...)`)
   - `best_checkpoint.pth` — full checkpoint at best epoch
   - `last_checkpoint.pth` — full checkpoint at final epoch
   - `config.yaml` — frozen copy of the config used (not a symlink)
   - `training_log.json` + `training_summary.json`
3. Upload to team S3: `just put-checkpoints-for {model}/{variant}`
4. Update the fine-tuned checkpoints table above
5. Link the GitHub issue where training results are documented
