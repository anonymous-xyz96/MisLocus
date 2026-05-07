"""XGBoost training, prediction, and device selection."""

from __future__ import annotations

import logging
import os
import subprocess

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from prot_loc_benchmark.config import XGBOOST_PARAMS

logger = logging.getLogger(__name__)


def _count_gpus() -> int:
    """Count available NVIDIA GPUs via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return len(out.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


# Optional fallback search paths for NVIDIA CUDA runtime libraries (NVIDIA pip
# wheels under .../site-packages/nvidia/<sublib>/lib). Set the
# PROT_LOC_BENCHMARK_CUDA_LIB_PATH environment variable (colon-separated) to
# add site-specific locations; the gpu pixi env normally provides everything
# needed without this fallback.
_CUDA_LIB_SEARCH_PATHS = [
    p for p in os.environ.get("PROT_LOC_BENCHMARK_CUDA_LIB_PATH", "").split(":") if p
]

_CUDA_SUBLIBS = [
    "cuda_runtime", "cublas", "cusolver", "cusparse",
    "curand", "cufft", "cuda_nvrtc", "nvjitlink",
]


def _ensure_cuda_libs() -> None:
    """Add NVIDIA CUDA libraries to LD_LIBRARY_PATH if not already present."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if "nvidia" in current and "cuda_runtime" in current:
        return  # Already configured

    for base in _CUDA_LIB_SEARCH_PATHS:
        lib_dirs = []
        for sublib in _CUDA_SUBLIBS:
            d = os.path.join(base, sublib, "lib")
            if os.path.isdir(d):
                lib_dirs.append(d)

        if lib_dirs:
            new_path = ":".join(lib_dirs)
            if current:
                new_path = f"{new_path}:{current}"
            os.environ["LD_LIBRARY_PATH"] = new_path
            logger.info("Added %d CUDA lib dirs from %s", len(lib_dirs), base)
            return

    logger.warning("No CUDA libraries found in known locations")


def select_device() -> str:
    """Select compute device based on environment variable.

    Reads ``MISLOCUS_CLASSIFIER_BACKEND``:
    - "cpu" → "cpu"
    - "gpu" → pick first available GPU (falls back to CPU)
    - "auto" (default) → GPU if available, else CPU

    Returns "cpu" or "cuda:N".
    """
    backend = os.environ.get("MISLOCUS_CLASSIFIER_BACKEND", "auto").strip().lower()

    if backend == "cpu":
        return "cpu"

    n_gpus = _count_gpus()
    if n_gpus == 0:
        if backend == "gpu":
            logger.warning("GPU requested but no CUDA devices found, falling back to CPU")
        return "cpu"

    # Ensure CUDA runtime libraries are on LD_LIBRARY_PATH
    _ensure_cuda_libs()

    # Try CuPy for smart GPU selection (least memory usage)
    try:
        import cupy as cp

        best_gpu = 0
        min_used = float("inf")
        for i in range(n_gpus):
            mem_free, mem_total = cp.cuda.Device(i).mem_info
            mem_used = mem_total - mem_free
            if mem_used < min_used:
                min_used = mem_used
                best_gpu = i

        device = f"cuda:{best_gpu}"
    except ImportError:
        device = "cuda:0"
    except Exception:
        logger.warning("CuPy GPU detection failed, defaulting to cuda:0", exc_info=True)
        device = "cuda:0"

    logger.info("Selected GPU device: %s (%d GPUs available)", device, n_gpus)
    return device


def train_and_predict(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str = "Label",
    device: str = "cpu",
    xgb_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    """Train XGBoost and return test predictions.

    Returns (predictions, true_labels, feature_importances) or None if
    the training data has extreme class imbalance (>100:1).
    """
    params = {**XGBOOST_PARAMS, **(xgb_params or {})}

    train_labels = train_df[label_col].to_numpy()
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        logger.warning("Single-class training data, skipping")
        return None

    imbalance = max(n_pos, n_neg) / min(n_pos, n_neg)
    if imbalance > 100:
        logger.warning("Extreme class imbalance %.0f:1, skipping", imbalance)
        return None

    params["scale_pos_weight"] = n_neg / n_pos

    if device != "cpu":
        params["device"] = device
        params.pop("n_jobs", None)

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_labels
    y_test = test_df[label_col].to_numpy()

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)[:, 1]
    importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))

    return preds, y_test, importances
