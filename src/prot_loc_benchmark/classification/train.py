"""XGBoost training with explicit device allocation and no silent CPU fallback."""

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import xgboost
from xgboost import XGBClassifier

from prot_loc_benchmark.config import XGBOOST_PARAMS

logger = logging.getLogger(__name__)


def select_device() -> str:
    backend = os.environ.get("MISLOCUS_CLASSIFIER_BACKEND", "cpu").strip().lower()
    if backend in ("cpu", "auto"):
        return "cpu"  # Auto never discovers/allocates somebody else's GPU.
    if backend != "gpu":
        raise ValueError(f"Unknown classifier backend: {backend}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "-1" or "," in visible:
        raise ValueError("GPU execution requires exactly one explicit CUDA_VISIBLE_DEVICES allocation")
    if not xgboost.build_info().get("USE_CUDA", False):
        raise ValueError("Installed XGBoost has no CUDA support; use the locked gpu environment")
    return "cuda:0"  # Logical index within the one-device allocation, never physical discovery.


def gpu_identity():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()


@contextmanager
def allocated_gpu(device):
    """Reserve one GPU among these runners; refuse other active compute clients.

    This is an advisory per-user lease, not a reservation against external schedulers.
    """
    if device == "cpu":
        yield None
        return
    import fcntl

    gpu = gpu_identity()
    uuid = gpu.split(",", 1)[0].strip()
    runtime = Path(os.environ["XDG_RUNTIME_DIR"])
    if runtime.stat().st_uid != os.getuid() or runtime.stat().st_mode & 0o077:
        raise ValueError("GPU leases require the private user XDG_RUNTIME_DIR")
    with (runtime / f"mislocus-{uuid}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clients = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True, timeout=10
        )
        if any(line.split(",", 1)[0].strip() == uuid for line in clients.splitlines()):
            raise RuntimeError(f"Allocated GPU {uuid} already has compute clients; retry admission later")
        yield gpu


def train_and_predict(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str = "Label",
    device: str = "cpu",
    xgb_params: dict | None = None,
    model_path=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
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
    params["device"] = device
    # n_jobs bounds host-side work even when tree construction runs on a GPU.
    params.setdefault("n_jobs", 1)
    params.setdefault("random_state", 0)
    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    clf = XGBClassifier(**params)
    clf.fit(X_train, train_labels)
    actual_device = json.loads(clf.get_booster().save_config())["learner"]["generic_param"]["device"]
    if actual_device != device:
        raise RuntimeError(f"XGBoost device fallback: requested {device}, used {actual_device}")
    if model_path is not None:
        from prot_loc_benchmark.identity import ordered_id_hash

        clf.get_booster().set_attr(
            feature_columns=json.dumps(feature_cols),
            ordered_training_cell_ids_sha256=ordered_id_hash(train_df),
            ordered_test_cell_ids_sha256=ordered_id_hash(test_df),
            positive_label="reference",
        )
        clf.save_model(model_path)
    if device == "cpu":
        preds = clf.predict_proba(X_test)[:, 1]
    else:
        # Explicit host-to-device DMatrix path; avoids sklearn's implicit device-mismatch fallback.
        preds = clf.get_booster().predict(xgboost.DMatrix(X_test, nthread=params["n_jobs"]))
    return preds, test_df[label_col].to_numpy(), dict(zip(feature_cols, clf.feature_importances_.tolist()))
