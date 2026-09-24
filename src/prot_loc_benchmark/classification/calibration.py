"""Persist the existing NC+PC well-position null for controls-first execution."""

from __future__ import annotations

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import polars as pl

from prot_loc_benchmark.config import (
    BATCH_CONTROLS,
    MAX_IMBALANCE_RATIO,
    MIN_CELL_COUNT,
    NULL_PERCENTILE,
)
from prot_loc_benchmark.provenance import code_fingerprint, save_json, sha256

from .metrics import compute_null_threshold, validate_thresholds


def calibration_context(
    features: Path,
    representation: str,
    batch: str,
    channel_features: dict[str, list[str]],
    protocol: str,
    xgb_params: dict,
    device: str,
    features_sha256: str,
) -> dict:
    """Bind calibration to the exact processed input, code and evaluator settings."""
    from prot_loc_benchmark.stages import xgboost_identity

    from .train import gpu_identity

    return {
        "allocated_gpu": gpu_identity() if device != "cpu" else None,
        "xgboost_build": xgboost_identity(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if device != "cpu" else None,
        "features_path": str(features.resolve()),
        "features_sha256": features_sha256,
        "representation": representation,
        "batch": batch,
        "protocol": protocol,
        "channel_features": channel_features,
        "xgb_params": xgb_params,
        "device": device,
        "control_alleles": BATCH_CONTROLS[batch],
        "min_cells": MIN_CELL_COUNT,
        "max_imbalance": MAX_IMBALANCE_RATIO,
        "percentile": NULL_PERCENTILE,
        "interpolation": "nearest",
        "code_sha256": code_fingerprint(),
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("polars", "numpy", "xgboost", "scikit-learn")},
        },
    }


def save_calibration(directory: Path, context: dict) -> dict[str, float]:
    """Publish last, only after control outputs have been completely written."""
    receipt_path = directory / "calibration.json"
    if receipt_path.exists():
        raise FileExistsError(f"Calibration already exists: {receipt_path}")
    metrics = pl.read_csv(directory / "metrics.csv")
    if not metrics["category"].is_in(["NC", "PC"]).all():
        raise ValueError("Calibration must contain only NC+PC control classifiers")
    thresholds = compute_null_threshold(metrics)
    validate_thresholds(thresholds, list(context["channel_features"]))
    counts = metrics.filter(pl.col("auroc").is_finite()).group_by("channel").len()
    save_json(
        receipt_path,
        {
            "schema_version": 1,
            "status": "complete",
            "context": context,
            "thresholds": thresholds,
            "n_finite_controls": dict(counts.iter_rows()),
            "outputs": {
                p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "calibration.json"
            },
        },
    )
    return thresholds


def load_calibration(directory: Path, context: dict) -> tuple[dict[str, float], pl.DataFrame]:
    """Reject missing, partial, changed or mismatched control runs before training."""
    path = directory / "calibration.json"
    if not path.is_file():
        raise ValueError(f"Missing completed control calibration: {path}; run --scope control first")
    from prot_loc_benchmark.stages import require_stage

    require_stage(directory, "calibration.json")
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != 1 or receipt.get("status") != "complete":
        raise ValueError("Incomplete control calibration")
    saved = receipt["context"]
    for key, value in context.items():
        if key == "channel_features":
            if any(saved[key].get(ch) != columns for ch, columns in value.items()):
                raise ValueError("Control calibration mismatch: ordered channel features")
        elif saved.get(key) != value:
            raise ValueError(f"Control calibration mismatch: {key}")
    for name, digest in receipt["outputs"].items():
        if Path(name).name != name or not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError(f"Changed or missing control calibration output: {name}")
    metrics = pl.read_csv(directory / "metrics.csv")
    thresholds = receipt["thresholds"]
    validate_thresholds(thresholds, list(context["channel_features"]))
    if thresholds != compute_null_threshold(metrics):
        raise ValueError("Saved calibration thresholds do not match control metrics")
    return thresholds, metrics
