"""Synthetic safety/lineage checks. No production data, checkpoints or GPU jobs."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark import downstream_inputs, provenance, stages
from prot_loc_benchmark.classification.train import allocated_gpu, select_device, train_and_predict
from prot_loc_benchmark.config import REPO_ROOT
from prot_loc_benchmark.downstream_inputs import verify_export
from prot_loc_benchmark.identity import CELL_ID, identify_cells, ordered_id_hash
from prot_loc_benchmark.provenance import capture_source, code_fingerprint, save_json, sha256


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Safeguards(unittest.TestCase):
    def test_identity_repeated_object_numbers_missing_and_conflicting_ids(self):
        frame = pl.DataFrame(
            {
                "Metadata_Plate": ["P_T4"] * 2,
                "Metadata_Well": ["A01"] * 2,
                "Metadata_ImageNumber": [1, 2],
                "Metadata_ObjectNumber": [7, 7],
            }
        )
        result = identify_cells(frame, "batch")
        self.assertEqual(result[CELL_ID].n_unique(), 2)
        sites = frame.with_columns(pl.lit(1).alias("Metadata_ImageNumber"), pl.Series("Metadata_Site", [1, 2]))
        self.assertEqual(identify_cells(sites, "batch")[CELL_ID].n_unique(), 2)
        self.assertEqual(
            identify_cells(frame.reverse(), "batch").sort(CELL_ID).to_dicts(), result.sort(CELL_ID).to_dicts()
        )
        for invalid in (
            frame.drop("Metadata_ImageNumber"),
            frame.with_columns(pl.lit(1).alias("Metadata_ImageNumber")),
            result.with_columns(pl.lit("wrong").alias(CELL_ID)),
        ):
            with self.assertRaises(ValueError):
                identify_cells(invalid, "batch")
        with self.assertRaisesRegex(ValueError, "identity"):
            identify_cells(frame, "batch", canonical=True)

    def test_stage_interruption_tampering_parent_binding_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.bin"
            raw.write_bytes(b"original")
            with (
                patch.object(stages, "DATA_DIR", root),
                patch.object(provenance, "PROVENANCE_LOG", root / "ledger.json"),
                patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(root)}),
            ):
                parent = root / "parent"
                with stages.stage(parent, [raw], {"kind": "fixture"}):
                    (parent / "features.bin").write_bytes(b"features")
                parent_receipt = stages.require_stage(parent)
                with self.assertRaisesRegex(ValueError, "representation/batch"):
                    stages.require_stage(parent, representation="wrong-model")
                child = root / "child"
                with stages.stage(child, [parent / "features.bin"], {}, parents=[parent_receipt]):
                    (child / "result.bin").write_bytes(b"result")
                stages.require_stage(child)
                entries = json.loads((root / "ledger.json").read_text())["runs"]
                self.assertEqual(len({entry["id"] for entry in entries}), 2)
                for target in (parent, child):
                    with self.assertRaises(FileExistsError):
                        with stages.stage(target, [raw], {}):
                            pass
                partial = root / "interrupted"
                with self.assertRaises(KeyboardInterrupt):
                    with stages.stage(partial, [raw], {}):
                        save_json(partial / "completion.json", {"status": "complete"})
                        raise KeyboardInterrupt()
                self.assertFalse((partial / "stage.json").exists())
                self.assertFalse((partial / "completion.json").exists())
                self.assertTrue((partial / "failed.json").exists())
                with self.assertRaisesRegex(ValueError, "changed"):
                    with stages.stage(root / "mutated-input", [raw], {}):
                        raw.write_bytes(b"changed")
                original = parent_receipt.read_bytes()
                parent_receipt.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "parent"):
                    stages.require_stage(child)
                parent_receipt.write_bytes(original)
                (child / "result.bin").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "output"):
                    stages.require_stage(child)
                outside = root.parent / f"{root.name}-outside"
                (root / "escape").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    with stages.stage(root / "escape", [raw], {}):
                        pass

    def test_explicit_gpu_allocation_and_no_cpu_fallback(self):
        with patch.dict(os.environ, {"MISLOCUS_CLASSIFIER_BACKEND": "gpu"}, clear=False):
            for visible in ("", "-1", "0,1"):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), self.assertRaises(ValueError):
                    select_device()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-assigned"}):
                with (
                    patch("xgboost.build_info", return_value={"USE_CUDA": False}),
                    self.assertRaisesRegex(ValueError, "CUDA"),
                ):
                    select_device()
                with patch("xgboost.build_info", return_value={"USE_CUDA": True}):
                    self.assertEqual(select_device(), "cuda:0")
        with patch.dict(os.environ, {"MISLOCUS_CLASSIFIER_BACKEND": "auto"}):
            self.assertEqual(select_device(), "cpu")
        frame = pl.DataFrame({"f": [0.0, 1.0, 2.0, 3.0], "Label": [0, 1, 0, 1]})
        with patch("prot_loc_benchmark.classification.train.XGBClassifier") as classifier:
            classifier.return_value.get_booster.return_value.save_config.return_value = json.dumps(
                {"learner": {"generic_param": {"device": "cpu"}}}
            )
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                train_and_predict(frame, frame, ["f"], device="cuda:0", xgb_params={"n_jobs": 3})
            self.assertEqual(classifier.call_args.kwargs["n_jobs"], 3)

    def test_gpu_admission_is_exclusive_and_rejects_existing_clients(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"XDG_RUNTIME_DIR": directory, "CUDA_VISIBLE_DEVICES": "GPU-fixture"}),
        ):

            def query(args, **kwargs):
                return "" if "--query-compute-apps=gpu_uuid,pid" in args else "GPU-fixture, H100, driver, 95830 MiB"

            with patch("prot_loc_benchmark.classification.train.subprocess.check_output", side_effect=query):
                with allocated_gpu("cuda:0"):
                    with self.assertRaises(BlockingIOError):
                        with allocated_gpu("cuda:0"):
                            pass
                with allocated_gpu("cuda:0"):  # Previous context released the lease.
                    pass
            with patch(
                "prot_loc_benchmark.classification.train.subprocess.check_output", return_value="GPU-fixture, 123"
            ):
                with self.assertRaisesRegex(RuntimeError, "compute clients"):
                    with allocated_gpu("cuda:0"):
                        pass

    def test_unbounded_production_is_rejected_before_data_work(self):
        with patch.object(
            stages,
            "cgroup_limits",
            return_value={"/scope": {"cpu.max": "max 100000", "memory.max": "max", "pids.max": "max"}},
        ):
            with self.assertRaisesRegex(ValueError, "cgroup"):
                stages.require_bounded_execution("subcell_allele_rybg_v2_mae_s42")
        name = "mislocus-downstream-test.slice"
        limits = {
            "cpu.max": "400000 100000",
            "memory.max": "34359738368",
            "memory.high": "25769803776",
            "pids.max": "128",
        }
        with (
            patch.dict(os.environ, {"MISLOCUS_CAMPAIGN_SLICE": name}),
            patch.object(stages, "cgroup_limits", return_value={"/" + name: limits}),
        ):
            self.assertTrue(stages.require_bounded_execution("subcell_allele_rybg_v2_mae_s42"))
            limits["cpu.max"] = "6500000 100000"
            with self.assertRaisesRegex(ValueError, "ceilings"):
                stages.require_bounded_execution("subcell_allele_rybg_v2_mae_s42")

    @patch.dict(os.environ, {"MISLOCUS_CAMPAIGN_SLICE": "mislocus-downstream-test.slice"})
    @patch.object(
        stages,
        "cgroup_limits",
        return_value={
            "/mislocus-downstream-test.slice": {
                "cpu.max": "200000 100000",
                "memory.high": "3221225472",
                "memory.max": "4294967296",
                "pids.max": "128",
            }
        },
    )
    def test_verified_export_contract_rejects_partial_wrong_checkpoint_and_tampering(self, _limits):
        # A small producer-shaped fixture, not a trusted substitute for production verification.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "interim" / "subcell_allele_rybg_v2_mae_s42"
            export.mkdir(parents=True)
            capture_source(export)
            control = root / "control"
            control.mkdir()
            batch = "fixture_batch"
            rep = export.name
            frame = pl.DataFrame(
                {
                    "Metadata_Plate": [f"P_T{i}" for i in range(1, 5)],
                    "Metadata_Well": ["A01"] * 4,
                    "Metadata_Site": [1] * 4,
                    "Metadata_ImageNumber": [1] * 4,
                    "Metadata_ObjectNumber": [1] * 4,
                    "Metadata_CellID": [f"cell{i}" for i in range(4)],
                    "Metadata_Split": ["train", "train", "val", "test"],
                    **{f"SubCell_{i}": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32) for i in range(1536)},
                }
            )
            frame = identify_cells(frame, batch)
            path = export / batch / "embeddings.parquet"
            path.parent.mkdir()
            frame.write_parquet(path)
            counts = {"train": 2, "val": 1, "test": 1}
            output = {
                "cells": 4,
                "sha256": sha256(path),
                "split_counts": counts,
                "ordered_cell_ids_sha256": ordered_id_hash(frame),
            }
            checkpoint, selection, preflight, crops = [
                root / n for n in ("checkpoint", "selection.json", "preflight.json", "crops.json")
            ]
            for dependency in (checkpoint, selection, preflight, crops):
                dependency.write_text("{}")
            spec = {
                "models": {
                    "mae": {
                        "representation": rep,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "selected_pass": 100,
                        "selection": str(selection),
                        "selection_sha256": sha256(selection),
                    }
                },
                "expected_batch_split_counts": {batch: counts},
                "control": str(control),
                "export_root": str(root / "interim"),
                "training_code_sha256": "training",
                "code_sha256": code_fingerprint(),
                "source_commit": json.loads((export / "source.json").read_text())["git_head"],
                "preflight": str(root),
                "preflight_sha256": sha256(preflight),
                "crop_verification": str(crops),
                "crop_verification_sha256": sha256(crops),
            }
            spec_path = root / "run-spec.json"
            save_json(spec_path, spec)
            spec_path.with_suffix(".sha256").write_text(sha256(spec_path))
            receipt = {
                "status": "complete",
                "artifact_kind": "raw_embeddings",
                "split": "all",
                "family": "mae",
                "checkpoint_sha256": sha256(checkpoint),
                "selected_pass": 100,
                "outputs": {batch: output},
                "training": {"code_sha256": "training"},
                "invocation": {"code_sha256": code_fingerprint()},
                "source_archive_sha256": sha256(export / "source.tar.gz"),
                "feature_columns": [f"SubCell_{i}" for i in range(1536)],
            }

            def publish(value, status="complete"):
                save_json(export / "extraction.json", value)
                verification = control / "verified-production-mae.json"
                save_json(
                    verification, {"receipt_sha256": sha256(export / "extraction.json"), "outputs": value["outputs"]}
                )
                save_json(
                    control / "production-status.json",
                    {
                        "status": status,
                        "spec_sha256": sha256(spec_path),
                        "jobs": {"mae": {"verified_receipt_sha256": sha256(verification)}},
                    },
                )

            publish(receipt)
            with (
                patch.object(downstream_inputs, "DATA_DIR", export),
                self.assertRaisesRegex(ValueError, "producer exports"),
            ):
                verify_export(spec_path, rep, batch, path)
            preprocess = load_script(REPO_ROOT / "scripts/06_preprocess_profiles.py", "preprocess_isolation")
            frozen = {str(p.relative_to(export)): sha256(p) for p in export.rglob("*") if p.is_file()}
            alias = root / "linked-interim"
            alias.symlink_to(export.parent, target_is_directory=True)
            with (
                patch.object(
                    sys,
                    "argv",
                    ["preprocess", "--batch", batch, "--representation", rep, "--extraction-spec", str(spec_path)],
                ),
                patch.object(preprocess, "preprocess_embedding_batch") as compute,
                patch.object(provenance, "PROVENANCE_LOG", root / "ledger.json"),
            ):
                for interim in (export.parent, alias):
                    with (
                        patch.object(preprocess, "INTERIM_DIR", interim),
                        patch.object(stages, "DATA_DIR", root),
                        patch.object(downstream_inputs, "DATA_DIR", root),
                        patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(root)}),
                        self.assertRaisesRegex(ValueError, "producer exports"),
                    ):
                        try:
                            preprocess.main()
                        finally:
                            self.assertEqual(
                                {str(p.relative_to(export)): sha256(p) for p in export.rglob("*") if p.is_file()},
                                frozen,
                                "Preprocessing wrote into frozen producer exports before rejecting the output",
                            )
                compute.assert_not_called()
                # A separate output with a raw-file symlink is the supported layout.
                analysis = root / "analysis"
                output_dir = analysis / "interim" / rep / batch
                output_dir.mkdir(parents=True)
                (output_dir / "embeddings.parquet").symlink_to(path)
                with (
                    patch.object(preprocess, "INTERIM_DIR", analysis / "interim"),
                    patch.object(stages, "DATA_DIR", analysis),
                    patch.object(downstream_inputs, "DATA_DIR", analysis),
                    patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(analysis)}),
                ):
                    preprocess.main()
                compute.assert_called_once_with(batch, rep, normalized_only=False)
                stages.require_stage(output_dir)
                self.assertEqual(
                    {str(p.relative_to(export)): sha256(p) for p in export.rglob("*") if p.is_file()}, frozen
                )
            inputs = verify_export(spec_path, rep, batch, path)
            self.assertIn(checkpoint, inputs)
            for changed in (
                {**receipt, "split": "test"},
                {**receipt, "checkpoint_sha256": "wrong"},
                {**receipt, "artifact_kind": "diagnostic_embeddings"},
            ):
                publish(changed)
                with self.assertRaisesRegex(ValueError, "Wrong"):
                    verify_export(spec_path, rep, batch, path)
            publish(receipt, status="running")
            with self.assertRaisesRegex(ValueError, "completed"):
                verify_export(spec_path, rep, batch, path)
            publish(receipt)
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(Exception):
                verify_export(spec_path, rep, batch, path)
            path.write_bytes(original)
            frame.with_columns(pl.lit(float("nan"), dtype=pl.Float32).alias("SubCell_0")).write_parquet(path)
            publish({**receipt, "outputs": {batch: {**output, "sha256": sha256(path)}}})
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                verify_export(spec_path, rep, batch, path)
            path.write_bytes(original)
            publish(receipt)
            checkpoint.write_text("changed")
            with self.assertRaisesRegex(ValueError, "dependency"):
                verify_export(spec_path, rep, batch, path)


if __name__ == "__main__":
    unittest.main()
