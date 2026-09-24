"""Synthetic safety/lineage checks. No production data, checkpoints or GPU jobs."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
from copairs.matching import UnpairedException

from prot_loc_benchmark import provenance, stages
from prot_loc_benchmark.classification.train import allocated_gpu, select_device, train_and_predict
from prot_loc_benchmark.config import REPO_ROOT
from prot_loc_benchmark.copairs_runtime import bounded_copairs
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

    def test_unexpected_skipped_classifier_aborts_instead_of_publishing_partial_results(self):
        from prot_loc_benchmark.classification.executor import _run_classifier

        train = pl.DataFrame({CELL_ID: ["train/1", "train/2"], "f": [1.0, 2.0], "Label": [1, 1]})
        test = pl.DataFrame({CELL_ID: ["test/1", "test/2"], "f": [1.0, 2.0], "Label": [0, 1]})
        task = {
            "train_df": train,
            "test_df": test,
            "ch_features": ["f"],
            "channel": "EMBED",
            "pair": SimpleNamespace(pair_id="p"),
            "fold": SimpleNamespace(fold_id="t4"),
        }
        with self.assertRaisesRegex(ValueError, "produced no fit"):
            _run_classifier(task, "cpu", Path("unused-model-directory"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            _run_classifier({**task, "test_df": train}, "cpu", Path("unused-model-directory"))

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

    def test_all_copairs_pools_are_bounded_without_changing_results(self):
        from copairs import compute

        lock = threading.Lock()
        active = peak = 0

        def work(_):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(0.005)
            with lock:
                active -= 1

        original = compute.ThreadPool
        with bounded_copairs(2):
            compute.parallel_map(work, np.arange(24), progress_bar=False)
        self.assertLessEqual(peak, 2)
        self.assertIs(compute.ThreadPool, original)
        with tempfile.TemporaryDirectory() as directory:
            arrays = []
            for workers in (1, 2):
                with bounded_copairs(workers):
                    arrays.append(
                        compute.get_null_dists(
                            np.array([[2, 8], [3, 9]]),
                            32,
                            seed=42,
                            cache_dir=Path(directory) / str(workers),
                            progress_bar=False,
                        )
                    )
            np.testing.assert_array_equal(*arrays)

    def test_copairs_reference_parity_membership_and_missing_negative_support(self):
        current = load_script(REPO_ROOT / "scripts/09c_classify_PA.py", "pa_current")
        reference_path = REPO_ROOT.parent / "prot-loc-publish-readiness-reference/scripts/09c_classify_PA.py"
        if not reference_path.exists():
            self.skipTest("Pinned read-only publication checkout is required for parity check")
        reference = load_script(reference_path, "pa_reference")
        rng = np.random.default_rng(72)
        rows = [
            dict(
                Metadata_Plate=f"P_T{t}",
                Metadata_Well=well,
                Metadata_ImageNumber=site,
                Metadata_ObjectNumber=cell,
                Metadata_CellID=f"{t}:{well}:{site}:{cell}",
                Metadata_gene_allele=allele,
                Metadata_symbol="G",
                Metadata_node_type=node,
                Metadata_Control="Exp",
                f=float(rng.normal()),
                g=float(rng.normal()),
            )
            for t in range(1, 5)
            for allele, well, node in [("G", "A01", "disease_wt"), ("G_v", "B01", "allele")]
            for site in (1, 2)
            for cell in range(3)
        ]
        frame = identify_cells(pl.DataFrame(rows), "fixture")
        kwargs = dict(
            alleles=["G_v"],
            feat_cols=["f", "g"],
            cells_per_site=20,
            neg_per_plate=0,
            sample_level="site",
            aggregate=True,
            null_size=32,
            threshold=0.05,
            seed=42,
            max_workers=1,
            test_split="t4",
        )
        with tempfile.TemporaryDirectory() as directory, bounded_copairs(1):
            root = Path(directory)
            # The original library's default home cache must never be touched by a test.
            with patch("copairs.compute.Path.home", return_value=root / "reference-home"):
                expected = reference._compute_map_vs_ref(frame, **kwargs)
            current._TRACE_DIR = root / "trace"
            actual = current._compute_map_vs_ref(frame.reverse(), **kwargs)
            self.assertEqual(actual.to_dicts(), expected.to_dicts())
            members = pl.read_parquet(root / "trace/00000/members.parquet")
            self.assertEqual(set(members[CELL_ID]), set(frame[CELL_ID]))
            queries = pl.read_parquet(root / "trace/00000/queries.parquet")
            self.assertTrue(queries["Metadata_Plate"].str.ends_with("T4").all())
            self.assertTrue((queries["n_total_pairs"] > queries["n_pos_pairs"]).all())
            broken = frame.filter(
                ~((pl.col("Metadata_Plate") == "P_T4") & (pl.col("Metadata_node_type") == "disease_wt"))
            )
            with patch.object(current, "average_precision", side_effect=AssertionError("No eligible queries")):
                skipped = current._compute_map_vs_ref(broken, **kwargs)
            self.assertTrue(skipped.is_empty())
            trace = root / "trace/00001"
            self.assertEqual(
                json.loads((trace / "status.json").read_text()),
                {"status": "not_estimable", "reason": "missing_same_plate_reference"},
            )
            excluded = pl.read_parquet(trace / "excluded_queries.parquet")
            self.assertEqual(excluded.height, 2)
            self.assertEqual(set(excluded["Metadata_gene_allele"]), {"G_v"})
            self.assertEqual(set(excluded["Metadata_Plate"]), {"P_T4"})
            self.assertEqual(set(excluded["exclusion_reason"]), {"missing_same_plate_reference"})
            self.assertTrue(pl.read_parquet(trace / "queries.parquet").is_empty())
            # Other plates of this allele remain eligible in all-query mode.
            actual = current._compute_map_vs_ref(broken, **{**kwargs, "test_split": None})
            self.assertEqual(actual["Metadata_gene_allele"].to_list(), ["G_v"])
            trace = root / "trace/00002"
            queries = pl.read_parquet(trace / "queries.parquet")
            self.assertEqual(set(queries["Metadata_Plate"]), {"P_T1", "P_T2", "P_T3"})
            # Unsupported queries still serve as positives: the pool is unchanged.
            self.assertEqual(set(queries["n_pos_pairs"]), {6})
            self.assertEqual(pl.read_parquet(trace / "profiles.parquet").height, 14)
            # Missing positive partners are still a hard failure, not another skip.
            no_positives = frame.filter(pl.col("Metadata_Plate") == "P_T4")
            with self.assertRaisesRegex(UnpairedException, "positive pairs"):
                current._compute_map_vs_ref(no_positives, **kwargs)
            self.assertFalse((root / "trace/00003/status.json").exists())
            # A missing reference for G must not remove supported H queries.
            supported = frame.with_columns(
                pl.lit("H").alias("Metadata_symbol"),
                pl.col("Metadata_gene_allele").str.replace("G", "H"),
                pl.col("Metadata_Plate").str.replace("P", "Q"),
                (pl.col(CELL_ID) + ":H").alias(CELL_ID),
            )
            mixed = current._compute_map_vs_ref(pl.concat([broken, supported]), **{**kwargs, "alleles": ["G_v", "H_v"]})
            self.assertEqual(mixed["Metadata_gene_allele"].to_list(), ["H_v"])
            self.assertEqual(
                mixed.drop("Metadata_gene_allele").to_dicts(), expected.drop("Metadata_gene_allele").to_dicts()
            )
            trace = root / "trace/00004"
            self.assertEqual(json.loads((trace / "status.json").read_text())["status"], "complete")
            self.assertEqual(pl.read_parquet(trace / "excluded_queries.parquet").height, 2)

    def test_missing_copairs_controls_cannot_publish_a_completed_stage(self):
        rng = np.random.default_rng(14)
        batch, rep = "2024_01_23_Batch_7", "vit"
        rows = [
            dict(
                Metadata_Plate=f"P_T{t}",
                Metadata_Well=well,
                Metadata_ImageNumber=site,
                Metadata_ObjectNumber=cell,
                Metadata_CellID=f"{t}:{well}:{site}:{cell}",
                Metadata_plate_map_name="P",
                Metadata_gene_allele=allele,
                Metadata_symbol="G",
                f=float(rng.normal()),
                g=float(rng.normal()),
            )
            for t in range(1, 5)
            for allele, well in [("G", "A01"), ("G_v", "B01")]
            for site in (1, 2)
            for cell in range(15)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "interim" / rep / batch
            inputs.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(inputs / "embeddings.parquet")
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "MISLOCUS_DATA_ROOT": str(root)}
            commands = [
                ("06_preprocess_profiles.py", []),
                ("09c_classify_PA.py", ["--test-split", "t4", "--null-size", "32", "--ctrl-null-size", "32"]),
            ]
            for script, flags in commands:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / script),
                        "--batch",
                        batch,
                        "--representation",
                        rep,
                        *flags,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode == 0, script.startswith("06"), result.stdout + result.stderr)
            output = root / "processed/classification_PA" / f"{rep}_t4" / batch
            self.assertFalse((output / "stage.json").exists())
            self.assertTrue((output / "failed.json").exists())

    def test_verified_export_contract_rejects_partial_wrong_checkpoint_and_tampering(self):
        # A small producer-shaped fixture, not a trusted substitute for production verification.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "exports" / "subcell_allele_rybg_v2_mae_s42"
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
                "export_root": str(root / "exports"),
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
