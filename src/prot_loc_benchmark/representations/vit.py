"""ViT feature extraction using HuggingFace models (default: MorphEm).

Implements Bag-of-Channels (BoC) extraction: each imaging channel is processed
independently through the ViT, producing a CLS token embedding per channel.
Per-channel embeddings are then concatenated to form the final feature vector.

Reference: CaicedoLab/MorphEm — ViT-S/16 trained with DINO on CHAMMI-75 (2.8M images)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── MorphEm preprocessing (batched, from HuggingFace model card) ─────────────


def preprocess_batch(crops: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Vectorized MorphEm preprocessing: uint16 (N, H, W) → float32 (N, 1, 224, 224).

    Steps (matching MorphEm model card):
    1. Scale uint16 → [0, 255] float32
    2. SaturationNoiseInjector: replace saturated pixels (255) with noise in [200, 255]
    3. PerImageNormalize: instance normalization (zero-mean, unit-variance per image)
    4. Resize to target_size × target_size
    """
    # uint16 → float32 in [0, 255]
    max_val = crops.max()
    if max_val > 0:
        x = torch.from_numpy(crops.astype(np.float32) * (255.0 / max_val))
    else:
        x = torch.from_numpy(crops.astype(np.float32))

    # (N, H, W) → (N, 1, H, W)
    x = x.unsqueeze(1)

    # Saturation noise: replace 255 with uniform noise in [200, 255]
    mask = x == 255.0
    if mask.any():
        noise = torch.empty_like(x).uniform_(200, 255)
        x = torch.where(mask, noise, x)

    # Per-image instance normalization
    instance_norm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False, eps=1e-7)
    x = instance_norm(x)

    # Resize to target
    if x.shape[-2:] != (target_size, target_size):
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

    return x  # (N, 1, 224, 224)


# ── ViT Extractor ───────────────────────────────────────────────────────────


class ViTExtractor:
    """Extract CLS token embeddings from a HuggingFace ViT model.

    Default model is MorphEm (CaicedoLab/MorphEm), a ViT-S/16 trained
    with DINO on microscopy images. Accepts single-channel (1, H, W) input.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    device : str
        Torch device (cuda or cpu).
    """

    def __init__(self, model_name: str = "CaicedoLab/MorphEm", device: str = "cuda"):
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        logger.info("Loading ViT model: %s", model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Load model class directly to avoid AutoModel registration bug
        # with trust_remote_code (duplicate class identity issue)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], model_name
        )
        self.model = model_class.from_pretrained(model_name, config=config)
        self.model.to(device).eval()
        self.device = device
        self.model_name = model_name

    @torch.no_grad()
    def extract_channel(
        self,
        crops: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Extract CLS token embeddings for one channel.

        Parameters
        ----------
        crops : np.ndarray
            Shape (N, H, W), dtype uint16.
        batch_size : int
            Inference batch size.

        Returns
        -------
        np.ndarray
            Shape (N, hidden_dim), float32. CLS token embeddings.
        """
        n = len(crops)
        embeddings = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = preprocess_batch(crops[start:end]).to(self.device)
            output = self.model.forward_features(batch)
            emb = output["x_norm_clstoken"].cpu().numpy()
            embeddings.append(emb)

        return np.concatenate(embeddings, axis=0)

    def extract_boc(
        self,
        channel_arrays: dict[str, np.ndarray],
        batch_size: int = 256,
    ) -> tuple[np.ndarray, list[str]]:
        """Bag of Channels: extract per-channel embeddings and concatenate.

        Parameters
        ----------
        channel_arrays : dict[str, np.ndarray]
            Mapping channel name → (N, H, W) uint16 array. All arrays must
            have the same N (number of cells).
        batch_size : int
            Inference batch size.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            - Concatenated embeddings (N, hidden_dim * n_channels)
            - Column names: ["ViT_gfp_0", ..., "ViT_dna_0", ...]
        """
        all_embeddings = []
        all_columns = []

        for channel_name, crops in channel_arrays.items():
            logger.info("  Extracting channel: %s (%d cells)", channel_name, len(crops))
            emb = self.extract_channel(crops, batch_size=batch_size)
            hidden_dim = emb.shape[1]

            col_names = [f"ViT_{channel_name}_{i}" for i in range(hidden_dim)]
            all_embeddings.append(emb)
            all_columns.extend(col_names)

        concatenated = np.concatenate(all_embeddings, axis=1)
        return concatenated, all_columns
