"""Representation extraction modules.

Lazy imports — submodules pull in heavy deps (torch, transformers,
SubCellPortable) on access:

- ``subcell_io``       SubCell frozen-ViT extraction helpers (used by 08a)
- ``subcell_finetune`` SubCell fine-tuned model wrapper (used by 08c/08d)
- ``vit``              MorphEm frozen ViT extractor (used by 08b)
"""
