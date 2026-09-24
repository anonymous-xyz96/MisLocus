"""Parallel classifier execution loop extracted from scripts/09_classify.py.

Encapsulates the task → thread-pool → ClassificationWriter pipeline so
multiple classification scripts can share one implementation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from pathlib import Path

from prot_loc_benchmark.identity import CELL_ID, ordered_id_hash

from .io import ClassificationWriter
from .metrics import compute_classifier_metrics
from .train import train_and_predict

logger = logging.getLogger(__name__)


def _run_classifier(task: dict, task_device: str, model_dir: Path) -> dict:
    """Train one declared eligible task; never silently discard a failed fit."""
    train_ids, test_ids = task["train_df"][CELL_ID], task["test_df"][CELL_ID]
    if (
        train_ids.null_count()
        or test_ids.null_count()
        or train_ids.n_unique() != len(train_ids)
        or test_ids.n_unique() != len(test_ids)
        or set(train_ids) & set(test_ids)
    ):
        raise ValueError("Duplicate cell identity or train/test overlap")
    classifier_id = f"{task['pair'].pair_id}__{task['channel']}__fold{task['fold'].fold_id}"
    model_path = model_dir / (hashlib.sha256(classifier_id.encode()).hexdigest() + ".ubj")
    result = train_and_predict(
        task["train_df"],
        task["test_df"],
        task["ch_features"],
        device=task_device,
        xgb_params=task.get("xgb_params"),
        model_path=model_path,
    )
    if result is None:
        raise ValueError(f"Declared eligible classifier produced no fit: {classifier_id}")

    preds, labels, importances = result
    pair = task["pair"]
    channel = task["channel"]
    fold = task["fold"]
    train_df = task["train_df"]
    test_df = task["test_df"]

    m = compute_classifier_metrics(preds, labels)
    n_train_pos = int((train_df["Label"] == 1).sum())
    n_train_neg = int((train_df["Label"] == 0).sum())
    imbalance = max(n_train_pos, n_train_neg) / max(min(n_train_pos, n_train_neg), 1)

    return {
        "classifier_id": classifier_id,
        "model_path": str(model_path.relative_to(model_dir.parent)),
        "pair": pair,
        "channel": channel,
        "fold": fold,
        "preds": preds,
        "labels": labels,
        "importances": importances,
        "metrics": m,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "imbalance": imbalance,
        "test_df": test_df,
        "train_df": train_df,
        "train_height": train_df.height,
    }


def run_classifier_tasks(
    tasks: Iterable[dict],
    device: str,
    output_dir: Path,
    max_workers: int = 1,
    *,
    stage_id: str,
    representation: str,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Run a batch of classifier tasks in parallel and stream predictions.

    Parameters
    ----------
    tasks
        Iterable of dicts with ``pair``, ``channel``, ``ch_features``, ``fold``,
        ``train_df``, ``test_df`` and optional ``xgb_params``.
        Only max_workers datasets are submitted at once; tasks are not mutated.
    device
        Explicit device for every task; never fan out onto unallocated GPUs.
    output_dir
        Directory where ``predictions.parquet`` will be written.

    Returns
    -------
    (metrics_rows, importance_rows, info_rows, n_classifiers_run)
        Three lists of dicts ready to feed into ``pl.DataFrame`` for CSV output,
        plus a count of completed classifiers. Failed eligible tasks abort the stage.
    """
    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []
    info_rows: list[dict] = []
    n_classifiers = 0

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    logger.info("Running classifiers (max_workers=%d, device=%s)", max_workers, device)

    t_class = time.time()
    predictions_path = output_dir / "predictions.parquet"
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=False)
    with ClassificationWriter(predictions_path, stage_id=stage_id, representation=representation) as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = (
                result
                for chunk in batched(tasks, max_workers)
                for result in executor.map(partial(_run_classifier, task_device=device, model_dir=model_dir), chunk)
            )
            for r in results:
                n_classifiers += 1
                pair = r["pair"]
                channel = r["channel"]
                fold = r["fold"]
                test_df = r["test_df"]

                writer.write_predictions(
                    classifier_id=r["classifier_id"],
                    labels=r["labels"],
                    predictions=r["preds"],
                    channel=channel,
                    is_control=pair.is_control,
                    identities=test_df,
                )
                for role, frame in (("train", r["train_df"]), ("test", test_df)):
                    writer.write_membership(r["classifier_id"], pair.pair_id, fold.fold_id, frame, role)

                metrics_rows.append(
                    {
                        "analysis_stage_id": stage_id,
                        "representation": representation,
                        "classifier_id": r["classifier_id"],
                        "pair_id": pair.pair_id,
                        "gene": pair.gene,
                        "allele_ref": pair.allele_ref,
                        "allele_var": pair.allele_var,
                        "channel": channel,
                        "fold_id": fold.fold_id,
                        "is_control": pair.is_control,
                        "category": pair.category,
                        "n_train": r["train_height"],
                        "n_test": test_df.height,
                        "imbalance_ratio": r["imbalance"],
                        **r["metrics"],
                    }
                )

                top_feats = sorted(r["importances"].items(), key=lambda x: x[1], reverse=True)[:50]
                for feat_name, feat_imp in top_feats:
                    importance_rows.append(
                        {
                            "classifier_id": r["classifier_id"],
                            "feature": feat_name,
                            "importance": feat_imp,
                        }
                    )

                info_rows.append(
                    {
                        "analysis_stage_id": stage_id,
                        "representation": representation,
                        "model_path": r["model_path"],
                        "classifier_id": r["classifier_id"],
                        "pair_id": pair.pair_id,
                        "gene": pair.gene,
                        "allele_ref": pair.allele_ref,
                        "allele_var": pair.allele_var,
                        "channel": channel,
                        "fold_id": fold.fold_id,
                        "is_control": pair.is_control,
                        "category": pair.category,
                        "n_train_ref": r["n_train_pos"],
                        "n_train_var": r["n_train_neg"],
                        "n_test": test_df.height,
                        "ordered_train_cell_ids_sha256": ordered_id_hash(r["train_df"]),
                        "ordered_test_cell_ids_sha256": ordered_id_hash(test_df),
                        "train_plates": ",".join(fold.train_plates),
                        "test_plates": ",".join(fold.test_plates),
                        "test_wells": ",".join(fold.test_wells),
                    }
                )

    t_class_elapsed = time.time() - t_class
    logger.info(
        "Classifier phase: %d classifiers in %.1fs (%.2fs/classifier)",
        n_classifiers,
        t_class_elapsed,
        t_class_elapsed / max(n_classifiers, 1),
    )

    return metrics_rows, importance_rows, info_rows, n_classifiers
