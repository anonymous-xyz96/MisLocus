#!/usr/bin/env python3
"""Audit a finished downstream campaign, not just process exit codes.

Requires every requested preprocessing/control/XGBoost/copairs stage, verifies
all artifacts and distinct input hashes once, and compares retained cell cohorts.
Writes a new acceptance.json only after all checks pass. Does not run analyses.
"""

import argparse
import json
from pathlib import Path

from prot_loc_benchmark.stages import bound_environment

bound_environment()
# ruff: noqa: E402 -- native thread bounds must precede numerical-library imports.

import polars as pl
import xgboost

from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES
from prot_loc_benchmark.identity import CELL_ID, check_cell_subset, ordered_id_hash
from prot_loc_benchmark.provenance import code_fingerprint, save_json, sha256
from prot_loc_benchmark.stages import require_bounded_execution, require_stage


def verify_predictions(directory, cells):
    receipt = json.loads((directory / "stage.json").read_text())
    context = receipt["parameters"]["context"]
    predictions = pl.read_parquet(directory / "predictions.parquet")
    members = pl.read_parquet(directory / "fold_membership.parquet")
    info = pl.read_csv(directory / "classifier_info.csv")
    keys = ["Classifier_ID", CELL_ID]
    if (
        predictions.is_empty()
        or info["classifier_id"].n_unique() != info.height
        or predictions.select(keys).unique().height != predictions.height
        or members.select(keys).unique().height != members.height
        or set(predictions["Classifier_ID"]) != set(info["classifier_id"])
        or set(members["role"]) != {"train", "test"}
    ):
        raise ValueError("Duplicate, missing or overlapping classifier/cell membership")
    for frame in (predictions, members):
        if (
            not (frame["AnalysisStageID"] == receipt["stage_id"]).fill_null(False).all()
            or not (frame["Representation"] == context["representation"]).fill_null(False).all()
            or not frame["Label"].is_in([0, 1]).fill_null(False).all()
        ):
            raise ValueError("Wrong prediction/membership stage identity")
    check_cell_subset(cells, members.select([c for c in members.columns if c.startswith("Metadata_")]).unique())
    test_members = members.filter(pl.col("role") == "test")
    columns = (
        keys
        + ["Label"]
        + [c for c in predictions.columns if c.startswith("Metadata_") and c != CELL_ID and c in test_members.columns]
    )
    left = predictions.select(columns).cast(pl.String).sort(keys)
    right = test_members.select(columns).cast(pl.String).sort(keys)
    if not left.equals(right):
        raise ValueError("Predictions disagree with exact held-out membership/labels")
    if (
        predictions["Prediction"].null_count()
        or not predictions["Prediction"].is_finite().all()
        or not predictions["Prediction"].is_between(0, 1).all()
        or not predictions["Metadata_Plate"].str.ends_with("T4").all()
        or not members.filter(pl.col("role") == "train")["Metadata_Plate"].str.contains(r"T[123]$").all()
    ):
        raise ValueError("Invalid probabilities or train/test plate roles")
    hashes = {
        (classifier, role): ordered_id_hash(pl.DataFrame({CELL_ID: ids}))
        for classifier, role, ids in members.group_by("Classifier_ID", "role", maintain_order=True)
        .agg(pl.col(CELL_ID))
        .iter_rows()
    }
    for fitted in info.iter_rows(named=True):
        name = fitted["model_path"]
        if not name.startswith("models/") or name not in receipt["outputs"]:
            raise ValueError("Missing fitted model binding")
        model = xgboost.Booster(params={"device": "cpu", "nthread": 1})
        model.load_model(directory / name)
        feature_columns = context["channel_features"][fitted["channel"]]
        if json.loads(model.attr("feature_columns")) != feature_columns or model.num_features() != len(feature_columns):
            raise ValueError("Fitted model feature order mismatch")
        for role, field in [("train", "training"), ("test", "test")]:
            expected = fitted[f"ordered_{role}_cell_ids_sha256"]
            if (
                hashes.get((fitted["classifier_id"], role)) != expected
                or model.attr(f"ordered_{field}_cell_ids_sha256") != expected
            ):
                raise ValueError("Fitted model/fold membership hash mismatch")


def verify_campaign(root, representations, batches, *, reused_stages=None):
    """Reused stages require explicit canonical receipt paths and exact SHA256 pins.

    All other stages must use the current frozen source when reuse is declared.
    Reuse never bypasses artifact, source-archive, parent or scientific checks.
    """
    root = Path(root).resolve()
    if (root / "acceptance.json").exists():
        raise FileExistsError("Campaign acceptance already exists")
    current_code = code_fingerprint() if reused_stages is not None else None
    reused_stages = dict(reused_stages or {})
    if any(str(Path(path).resolve()) != path for path in reused_stages):
        raise ValueError("Reused stage receipt paths must be absolute and canonical")
    bindings, receipts, cohorts, codes = {}, {}, {}, set()
    used_reuse, all_codes = {}, set()
    for rep in representations:
        require_bounded_execution(rep)
        for batch in batches:
            preprocessing = root / "interim" / rep / batch
            xgb = root / "processed/classification" / f"{rep}_t4" / batch
            copairs = root / "processed/classification_PA" / f"{rep}_t4" / batch
            expected = {
                preprocessing: (
                    "features.parquet",
                    "features_cells.parquet",
                    "features_schema.json",
                    "plate_stats.parquet",
                ),
                xgb / "controls": (
                    "calibration.json",
                    "predictions.parquet",
                    "fold_membership.parquet",
                    "classifier_info.csv",
                    "allele_inventory.parquet",
                ),
                xgb: (
                    "completion.json",
                    "metrics_summary.csv",
                    "predictions.parquet",
                    "fold_membership.parquet",
                    "classifier_info.csv",
                    "allele_inventory.parquet",
                ),
                copairs: ("mAP_control.parquet", "mAP_results.parquet", "allele_inventory.parquet"),
            }
            for directory, files in expected.items():
                path = require_stage(directory, representation=rep, batch=batch).resolve()
                receipt = json.loads(path.read_text())
                if set(files) - set(receipt["outputs"]):
                    raise ValueError(f"Missing required stage artifacts: {directory}")
                digest = sha256(path)
                all_codes.add(receipt["code_sha256"])
                if str(path) in reused_stages:
                    if digest != reused_stages[str(path)]:
                        raise ValueError(f"Reused stage receipt checksum mismatch: {path}")
                    used_reuse[str(path)] = digest
                else:
                    if current_code is not None and receipt["code_sha256"] != current_code:
                        raise ValueError(f"Unadmitted stage source version: {path}")
                    codes.add(receipt["code_sha256"])
                receipts[str(path)] = digest
                for input_path, digest in {**receipt["inputs"], **receipt["parents"]}.items():
                    if input_path in bindings and bindings[input_path] != digest:
                        raise ValueError(f"Input changed between campaign stages: {input_path}")
                    bindings[input_path] = digest
            cells = pl.read_parquet(preprocessing / "features_cells.parquet")
            verify_predictions(xgb / "controls", cells)
            verify_predictions(xgb, cells)
            # Require the evaluator stages to descend from this exact preprocessing stage.
            prep_receipt = str((preprocessing / "stage.json").resolve())
            for directory in (xgb / "controls", xgb, copairs):
                receipt = json.loads((directory / "stage.json").read_text())
                if receipt["parents"].get(prep_receipt) != receipts[prep_receipt]:
                    raise ValueError(f"Wrong preprocessing parent: {directory}")
            xgb_receipt = json.loads((xgb / "stage.json").read_text())
            controls_receipt = json.loads((xgb / "controls/stage.json").read_text())
            if xgb_receipt["parameters"]["scope"] != "all" or controls_receipt["parameters"]["scope"] != "control":
                raise ValueError("Campaign requires complete controls and Exp+cPC scopes")
            feature_path = str((preprocessing / "features.parquet").resolve())
            feature_digest = json.loads((preprocessing / "stage.json").read_text())["outputs"]["features.parquet"]
            for receipt in (xgb_receipt, controls_receipt):
                context = receipt["parameters"]["context"]
                if context["features_path"] != feature_path or context["features_sha256"] != feature_digest:
                    raise ValueError("Wrong processed feature artifact")
            if json.loads((copairs / "stage.json").read_text())["inputs"].get(feature_path) != feature_digest:
                raise ValueError("Copairs used a different processed feature artifact")
            control_parent = str((xgb / "controls/stage.json").resolve())
            if xgb_receipt["parents"].get(control_parent) != receipts[control_parent]:
                raise ValueError("Wrong XGBoost control parent")
            if xgb_receipt["parameters"]["context"]["protocol"] != "t1-t3_train_t4_test":
                raise ValueError("Campaign contains a non-T4 XGBoost evaluation")
            pa_params = json.loads((copairs / "stage.json").read_text())["parameters"]
            if (
                pa_params["scope"] != "all"
                or pa_params["test_split"] != "t4"
                or not pa_params["control_null"]
                or not pa_params["aggregate"]
                or pa_params["sample_level"] != "site"
                or pa_params["neg_per_plate"] != 0
                or pa_params["null_percentile"] != 95
                or pa_params["null_size"] != 10000
                or pa_params["ctrl_null_size"] != 1000
            ):
                raise ValueError("Campaign copairs settings differ from the approved reference protocol")
            schema = json.loads((preprocessing / "features_schema.json").read_text())
            cohorts.setdefault(batch, {})[rep] = {
                "cells": schema["rows"],
                "cohort_cell_ids_sha256": schema["cohort_cell_ids_sha256"],
                "features": len(schema["features"]),
            }
    if used_reuse != reused_stages:
        raise ValueError("Reuse manifest contains unconsumed stage receipts")
    if len(codes) != 1:
        raise ValueError("Campaign must have one frozen source for non-reused stages")
    if current_code is not None and code_fingerprint() != current_code:
        raise ValueError("Source changed during campaign verification")
    for batch, models in cohorts.items():
        if len({v["cohort_cell_ids_sha256"] for v in models.values()}) != 1:
            raise ValueError(f"Unmatched retained cell cohorts in {batch}; do not silently intersect")
    for path, digest in bindings.items():
        if sha256(path) != digest:
            raise ValueError(f"Changed campaign input: {path}")
    result = {
        "status": "complete",
        "representations": representations,
        "batches": batches,
        "stage_receipts": receipts,
        "inputs": bindings,
        "cohorts": cohorts,
        "code_sha256": codes.pop(),
        "source_code_sha256s": sorted(all_codes),
        "reused_stage_receipts": used_reuse,
    }
    save_json(root / "acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--representations", nargs="+", required=True)
    parser.add_argument("--batches", nargs="+", default=list(ALL_PUBLIC_BATCHES))
    parser.add_argument("--reuse-manifest", type=Path, help="JSON mapping canonical stage.json paths to SHA256 pins")
    args = parser.parse_args()
    verify_campaign(
        args.root,
        args.representations,
        args.batches,
        reused_stages=json.loads(args.reuse_manifest.read_text()) if args.reuse_manifest else None,
    )
