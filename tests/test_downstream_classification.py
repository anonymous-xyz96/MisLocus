"""Controls-first T4 contract; tiny CPU fixtures, never production inputs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from prot_loc_benchmark.classification.cv import generate_folds, split_fold
from prot_loc_benchmark.classification.metrics import (
    aggregate_allele_metrics,
    compute_null_threshold,
    load_single_fold_metrics,
)
from prot_loc_benchmark.classification.pairs import build_control_pairs, get_pair_data
from prot_loc_benchmark.classification.reporting import plot_auroc_distributions
from prot_loc_benchmark.config import REPO_ROOT
from prot_loc_benchmark.provenance import sha256


class CalibrationChecks(unittest.TestCase):
    def test_missing_calibration_cannot_call_hits(self):
        metrics = pl.DataFrame(
            {
                "pair_id": ["p"],
                "gene": ["G"],
                "allele_var": ["G_v"],
                "channel": ["EMBED"],
                "imbalance_ratio": [1.0],
                "auroc": [0.8],
                "auprc": [0.8],
                "balanced_accuracy": [0.8],
            }
        )
        for thresholds in ({}, {"EMBED": float("nan")}, {"EMBED": None}):
            with self.subTest(thresholds=thresholds), self.assertRaisesRegex(ValueError, "calibration"):
                aggregate_allele_metrics(metrics, thresholds, min_classifiers=1)
        with self.assertRaisesRegex(ValueError, "control"):
            compute_null_threshold(pl.DataFrame())

    def test_both_alk_alleles_supply_six_same_allele_well_pairs(self):
        rows = [
            dict(
                Metadata_gene_allele=allele,
                Metadata_plate_map_name="P",
                Metadata_Plate=f"P_T{t}",
                Metadata_well_position=well,
                Metadata_Control="PC",
            )
            for allele in ("ALK", "ALK_Arg1275Gln")
            for t in range(1, 5)
            for well in ("A01", "A02", "B01", "B02")
            for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        pairs = build_control_pairs(df.lazy())
        self.assertEqual(len(pairs), 12)
        for allele in ("ALK", "ALK_Arg1275Gln"):
            self.assertEqual(sum(p.gene == allele for p in pairs), 6)
        for pair in pairs:
            selected = get_pair_data(df, pair)
            self.assertEqual(selected["Metadata_gene_allele"].unique().to_list(), [pair.gene])
            self.assertEqual(selected["Metadata_well_position"].n_unique(), 2)
            self.assertEqual(selected["Label"].n_unique(), 2)

    def test_holdout_uses_only_t1_t2_t3_to_predict_t4(self):
        frame = pl.DataFrame(
            {
                "Metadata_Plate": [f"P_T{i}" for i in range(1, 5)],
                "Metadata_plate_map_name": ["P"] * 4,
                "Metadata_well_position": ["A01"] * 4,
            }
        )
        folds = generate_folds(frame, "single_rep", test_split="t4")
        self.assertEqual(len(folds), 1)
        train, test = split_fold(frame, folds[0], "single_rep")
        self.assertEqual(set(train["Metadata_Plate"]), {"P_T1", "P_T2", "P_T3"})
        self.assertEqual(test["Metadata_Plate"].to_list(), ["P_T4"])
        self.assertEqual(generate_folds(frame.tail(1), "single_rep", test_split="t4"), [])
        with self.assertRaises(ValueError):
            generate_folds(frame, "multi_rep", test_split="t4")

    def test_plot_uses_supplied_threshold_not_another_quantile(self):
        import matplotlib.pyplot  # noqa: F401 — initialize pyplot before patching an Axes method.
        from matplotlib.axes import Axes

        metrics = pl.DataFrame({"channel": ["EMBED"] * 4, "auroc": [0.5, 0.6, 0.7, 0.9]})
        self.assertEqual(compute_null_threshold(metrics), {"EMBED": 0.9})
        with tempfile.TemporaryDirectory() as directory, patch.object(Axes, "axvline") as line:
            plot_auroc_distributions(metrics, metrics, Path(directory), "fixture", {"EMBED": 0.83})
            self.assertEqual(line.call_args.args[0], 0.83)
            self.assertTrue((Path(directory) / "auroc_distribution.png").exists())


class DownstreamCLIChecks(unittest.TestCase):
    def test_imported_t4_summarizer_uses_only_t4_control_scores(self):
        batch, rep = "2024_01_23_Batch_7", "fixture"
        rows = [
            dict(
                classifier_id=f"c{i}",
                pair_id=f"p{i}",
                gene="G",
                allele_var="G_v",
                channel="EMBED",
                category=category,
                imbalance_ratio=1.0,
                auroc=score,
                auprc=score,
                balanced_accuracy=score,
            )
            for i, (category, score) in enumerate([("NC", 1.0), ("PC", 0.6), ("PC", 0.9), ("Exp", 0.8)])
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "processed/classification" / rep / batch
            src.mkdir(parents=True)
            pl.DataFrame(rows).write_csv(src / "metrics.csv")
            pl.DataFrame(
                {"classifier_id": ["c0", "c1", "c2", "c3"], "test_plates": ["P_T3", "P_T4", "P_T4", "P_T4"]}
            ).write_csv(src / "classifier_info.csv")
            classification_root = root / "processed/classification"
            info = pl.read_csv(src / "classifier_info.csv")
            clean = load_single_fold_metrics(rep, batch, classification_dir=classification_root)
            self.assertEqual(clean["null_threshold"].to_list(), [0.9])
            for invalid in (info.head(3), pl.concat([info, info.head(1)])):
                invalid.write_csv(src / "classifier_info.csv")
                with self.assertRaisesRegex(ValueError, "classifier"):
                    load_single_fold_metrics(rep, batch, classification_dir=classification_root)
            info.write_csv(src / "classifier_info.csv")
            pl.DataFrame(rows + [rows[-1]]).write_csv(src / "metrics.csv")
            with self.assertRaisesRegex(ValueError, "classifier"):
                load_single_fold_metrics(rep, batch, classification_dir=classification_root)
            pl.DataFrame(rows).write_csv(src / "metrics.csv")
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "MISLOCUS_DATA_ROOT": str(root)}
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/09e_summarize_t4.py"),
                "--representation",
                rep,
                "--batches",
                batch,
            ]
            result = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            out = root / "processed/classification" / f"{rep}_t4" / batch
            summary = pl.read_csv(out / "metrics_summary.csv")
            self.assertEqual(summary["null_threshold"].to_list(), [0.9])
            self.assertEqual(summary["is_hit"].to_list(), [False])
            self.assertEqual(summary["auroc_std"].null_count(), 1)
            repeat = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0)

    def test_preprocess_controls_then_experiments_and_reference_copairs(self):
        batch, rep = (
            "2024_01_23_Batch_7",
            "vit",
        )  # Small legacy-format numerical fixture; producer gate tested separately.
        rng = np.random.default_rng(42)
        rows = []
        alleles = [
            "RHEB",
            "MAPK9",
            "PRKACB",
            "SLIRP",
            "ALK",
            "ALK_Arg1275Gln",
            "GENE",
            "GENE_v1",
            "GENE_v2",
            "ORPHAN_v1",
        ]
        for t in range(1, 5):
            well_index = 0
            for a, allele in enumerate(alleles):
                n_wells = 4 if a < 6 else 1
                for w in range(n_wells):
                    well_index += 1
                    well = f"A{well_index:02}"
                    for site in range(2):
                        for cell in range(15):
                            rows.append(
                                {
                                    "Metadata_CellID": f"{batch}:P_T{t}:{well}:{site}:{cell}",
                                    "Metadata_Plate": f"P_T{t}",
                                    "Metadata_plate_map_name": "P",
                                    "Metadata_Well": well,
                                    "Metadata_ImageNumber": well_index * 2 + site,
                                    "Metadata_ObjectNumber": cell,
                                    "Metadata_Site": site,
                                    "Metadata_gene_allele": allele,
                                    "Metadata_symbol": allele
                                    if a < 6
                                    else ("ORPHAN" if allele == "ORPHAN_v1" else "GENE"),
                                    **{
                                        f"SubCell_{k}": float(rng.normal() + 0.15 * w + (a == 8) * (k - 2))
                                        for k in range(6)
                                    },
                                }
                            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "interim" / rep / batch
            folder.mkdir(parents=True)
            raw = folder / "embeddings.parquet"
            pl.DataFrame(rows).write_parquet(raw)
            raw_hash = sha256(raw)
            env = {
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT / "src"),
                "PYTHONNOUSERSITE": "1",
                "MISLOCUS_DATA_ROOT": str(root),
                "CUDA_VISIBLE_DEVICES": "",
                "MISLOCUS_CLASSIFIER_BACKEND": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "2",
            }

            def run(script, *args, success=True):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / script),
                        "--batch",
                        batch,
                        "--representation",
                        rep,
                        *args,
                    ],
                    env=env,
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout + result.stderr

            run("06_preprocess_profiles.py")
            self.assertEqual(sha256(raw), raw_hash)
            features = folder / "features.parquet"
            processed = pl.read_parquet(features)
            self.assertEqual(processed["Metadata_CellID"].n_unique(), len(rows))
            for allele in ("ALK", "ALK_Arg1275Gln"):
                self.assertEqual(
                    processed.filter(pl.col("Metadata_gene_allele") == allele)["Metadata_Control"].unique().to_list(),
                    ["PC"],
                )
            base = root / "processed/classification" / f"{rep}_t4" / batch
            self.assertIn("run --scope control first", run("09_classify.py", "--scope", "exp", success=False))
            self.assertFalse(base.exists())
            run("09_classify.py", "--scope", "control")
            control_dir = base / "controls"
            receipt_path = control_dir / "calibration.json"
            receipt = json.loads(receipt_path.read_text())
            ctrl = pl.read_csv(control_dir / "metrics.csv")
            for allele in ("ALK", "ALK_Arg1275Gln"):
                self.assertEqual(ctrl.filter(pl.col("gene") == allele).height, 6)
            self.assertEqual(ctrl.height, 36)
            self.assertFalse((base / "completion.json").exists())
            control_hash = sha256(receipt_path)
            # Protocol / threads are bound to calibration, not just channel names.
            self.assertIn("calibration mismatch", run("09_classify.py", "--threads", "2", success=False))
            self.assertIn("calibration mismatch", run("09_classify.py", "--workers", "2", success=False))
            original = features.read_bytes()
            features.write_bytes(original + b"changed")
            # Check the loader directly because tampering can invalidate the Parquet footer first.
            from prot_loc_benchmark.classification.calibration import load_calibration

            changed = {**receipt["context"], "features_sha256": sha256(features)}
            with self.assertRaisesRegex(ValueError, "calibration mismatch"):
                load_calibration(control_dir, changed)
            features.write_bytes(original)
            saved_metrics = (control_dir / "metrics.csv").read_bytes()
            (control_dir / "metrics.csv").write_bytes(saved_metrics + b"\n")
            self.assertIn("Changed or missing control", run("09_classify.py", success=False))
            (control_dir / "metrics.csv").write_bytes(saved_metrics)
            run("09_classify.py")
            self.assertEqual(sha256(receipt_path), control_hash)
            self.assertEqual(json.loads((base / "completion.json").read_text())["status"], "complete")
            summary = pl.read_csv(base / "metrics_summary.csv")
            self.assertEqual(summary.height, 2)
            self.assertEqual(set(summary["allele_var"]), {"GENE_v1", "GENE_v2"})
            inventory = pl.read_parquet(base / "allele_inventory.parquet")
            orphan = inventory.filter(pl.col("Metadata_gene_allele") == "ORPHAN_v1")
            self.assertEqual(orphan["pair_status"].to_list(), ["insufficient_matching_reference_cells"])
            self.assertEqual(inventory["cells"].sum(), len(rows))
            self.assertEqual(summary["n_classifiers"].to_list(), [1, 1])
            self.assertEqual(summary["auroc_std"].null_count(), 2)
            self.assertEqual(summary["null_threshold"].unique().to_list(), [receipt["thresholds"]["EMBED"]])
            predictions = pl.read_parquet(base / "predictions.parquet")
            memberships = pl.read_parquet(base / "fold_membership.parquet")
            key = ["Classifier_ID", "Metadata_BatchQualifiedCellID"]
            self.assertEqual(predictions.select(key).unique().height, predictions.height)
            self.assertEqual(
                predictions["AnalysisStageID"].unique().to_list(),
                [json.loads((base / "stage.json").read_text())["stage_id"]],
            )
            self.assertEqual(predictions["Representation"].unique().to_list(), [rep])
            cells = processed.select("Metadata_BatchQualifiedCellID", "Metadata_ImageNumber", "Metadata_ObjectNumber")
            joined = predictions.join(cells, on="Metadata_BatchQualifiedCellID", how="left", suffix="_source")
            self.assertEqual(joined.height, predictions.height)
            self.assertTrue((joined["Metadata_ImageNumber"] == joined["Metadata_ImageNumber_source"]).all())
            self.assertTrue((joined["Metadata_ObjectNumber"] == joined["Metadata_ObjectNumber_source"]).all())
            self.assertTrue(
                memberships.filter(pl.col("role") == "train")
                .join(memberships.filter(pl.col("role") == "test"), on=key, how="inner")
                .is_empty()
            )
            self.assertEqual(memberships.filter(pl.col("role") == "test").height, predictions.height)
            self.assertTrue((base / "stage.json").exists())
            info = pl.read_csv(base / "classifier_info.csv")
            import xgboost

            fitted = info.row(0, named=True)
            booster = xgboost.Booster(params={"device": "cpu", "nthread": 1})
            booster.load_model(base / fitted["model_path"])
            self.assertEqual(booster.attr("ordered_training_cell_ids_sha256"), fitted["ordered_train_cell_ids_sha256"])
            columns = json.loads(booster.attr("feature_columns"))
            saved = predictions.filter(pl.col("Classifier_ID") == fitted["classifier_id"])
            replay = saved.select("Metadata_BatchQualifiedCellID").join(
                processed, on="Metadata_BatchQualifiedCellID", how="left", maintain_order="left"
            )
            scores = booster.predict(xgboost.DMatrix(replay.select(columns).to_numpy().astype(np.float32)))
            np.testing.assert_allclose(scores, saved["Prediction"].to_numpy(), atol=1e-7, rtol=0)
            self.assertEqual(info["test_plates"].unique().to_list(), ["P_T4"])
            self.assertEqual(info["train_plates"].unique().to_list(), ["P_T1,P_T2,P_T3"])

            loaded = load_single_fold_metrics(rep, batch, classification_dir=root / "processed/classification")
            self.assertTrue(loaded.equals(summary.with_columns(pl.col("auroc_std").cast(pl.Float64))))
            from prot_loc_benchmark.benchmark.clinvar import average_across_bioreps

            replicated = pl.concat(
                [
                    loaded.with_columns(
                        representation=pl.lit(rep), pair_name=pl.lit("fixture"), batch=pl.lit(batch_name)
                    )
                    for batch_name in ("a", "b")
                ]
            )
            self.assertEqual(average_across_bioreps(replicated).height, summary.height)
            self.assertIn("already exist", run("09_classify.py", success=False))
            # Tiny cohort, but retain the actual reference permutation budgets.
            run("09c_classify_PA.py", "--test-split", "t4", "--max-workers", "1")
            pa = root / "processed/classification_PA" / f"{rep}_t4" / batch
            controls = pl.read_parquet(pa / "mAP_control.parquet")
            results = pl.read_parquet(pa / "mAP_results.parquet")
            self.assertEqual(results.height, 2)
            self.assertTrue(results["mAP_vs_ref_norm"].is_finite().all())
            for allele in ("ALK", "ALK_Arg1275Gln"):
                self.assertEqual(controls.filter(pl.col("allele") == allele).height, 4)
            self.assertTrue(results["null_threshold_p95"].is_finite().all())
            self.assertEqual(
                results["is_hit"].to_list(), (results["mAP_vs_ref_norm"] > results["null_threshold_p95"]).to_list()
            )
            self.assertTrue((pa / "stage.json").exists())
            for path in (pa / "trace").glob("*/members.parquet"):
                members = pl.read_parquet(path)
                self.assertTrue(
                    members.join(
                        processed.select("Metadata_BatchQualifiedCellID"),
                        on="Metadata_BatchQualifiedCellID",
                        how="anti",
                    ).is_empty()
                )
            self.assertIn("already exist", run("09c_classify_PA.py", "--test-split", "t4", success=False))
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/09f_verify_downstream.py"),
                "--root",
                str(root),
                "--representations",
                rep,
                "--batches",
                batch,
            ]
            model_path = base / fitted["model_path"]
            model_bytes = model_path.read_bytes()
            model_path.write_bytes(model_bytes + b"changed")
            rejected = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "acceptance.json").exists())
            model_path.write_bytes(model_bytes)
            from test_downstream_safeguards import load_script

            verifier = load_script(REPO_ROOT / "scripts/09f_verify_downstream.py", "reuse_verifier")
            prep_receipt = (features.parent / "stage.json").resolve()
            reused = {str(prep_receipt): sha256(prep_receipt)}
            with self.assertRaisesRegex(ValueError, "receipt checksum mismatch"):
                verifier.verify_campaign(root, [rep], [batch], reused_stages={str(prep_receipt): "wrong"})
            with patch.object(verifier, "code_fingerprint", return_value="unadmitted-source"):
                with self.assertRaisesRegex(ValueError, "Unadmitted stage source"):
                    verifier.verify_campaign(root, [rep], [batch], reused_stages=reused)
            with self.assertRaisesRegex(ValueError, "unconsumed"):
                verifier.verify_campaign(
                    root, [rep], [batch], reused_stages={**reused, str(root / "unused/stage.json"): "unused"}
                )
            self.assertFalse((root / "acceptance.json").exists())
            accepted = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            acceptance = json.loads((root / "acceptance.json").read_text())
            self.assertEqual(acceptance["cohorts"][batch][rep]["cells"], len(rows))
            self.assertEqual(len(acceptance["stage_receipts"]), 4)
            self.assertEqual(acceptance["reused_stage_receipts"], {})
            (root / "acceptance.json").unlink()  # Synthetic sandbox only: exercise explicit reuse admission.
            reuse_manifest = root / "reuse.json"
            reuse_manifest.write_text(json.dumps(reused))
            accepted = subprocess.run(
                [*command, "--reuse-manifest", str(reuse_manifest)], env=env, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            acceptance = json.loads((root / "acceptance.json").read_text())
            self.assertEqual(acceptance["reused_stage_receipts"], reused)
            repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
