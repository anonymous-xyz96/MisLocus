"""Regression cases found during the pre-commit standards/spec review."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl
from test_downstream_safeguards import load_script

from prot_loc_benchmark import provenance, stages
from prot_loc_benchmark.classification.metrics import load_single_fold_metrics
from prot_loc_benchmark.config import REPO_ROOT


class ReviewRegressions(unittest.TestCase):
    def test_parent_output_changed_before_child_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(stages, "DATA_DIR", root),
                patch.object(provenance, "PROVENANCE_LOG", root / "ledger.json"),
                patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(root)}),
            ):
                parent = root / "parent"
                with stages.stage(parent, [], {}):
                    (parent / "features").write_text("verified features")
                receipt = stages.require_stage(parent)
                (parent / "features").write_text("changed after parent verification")
                with self.assertRaisesRegex(ValueError, "parent.*input|input.*parent"):
                    with stages.stage(root / "child", [parent / "features"], {}, parents=[receipt]):
                        pass
                self.assertFalse((root / "child/stage.json").exists())

    def test_cellprofiler_pa_entrypoint_reaches_the_correct_parent_gate(self):
        pa = load_script(REPO_ROOT / "scripts/09c_classify_PA.py", "review_pa_cp")
        batch = "2025_03_17_Batch_15"
        for source in ("normalized", "features"):
            with (
                patch(
                    "sys.argv",
                    ["09c", "--batch", batch, "--representation", "cellprofiler", "--cp-feature-file", source],
                ),
                patch.object(pa, "require_stage", side_effect=RuntimeError("parent gate")) as gate,
            ):
                with self.assertRaisesRegex(RuntimeError, "parent gate"):
                    pa.main()
                gate.assert_called_once_with(
                    pa.CELLPROFILER_DIR / batch, f"{source}.parquet", representation="cellprofiler", batch=batch
                )

    def test_copairs_pre_trace_control_failure_is_not_silently_skipped(self):
        pa = load_script(REPO_ROOT / "scripts/09c_classify_PA.py", "review_pa_controls")
        rows = [
            dict(
                Metadata_gene_allele="C",
                Metadata_symbol="C",
                Metadata_Plate=f"P_T{t}",
                Metadata_plate_map_name="P",
                Metadata_well_position=w,
                Metadata_Well=w,
                Metadata_Control="NC",
                Metadata_node_type="NC",
                f=1.0,
            )
            for t in range(1, 5)
            for w in ["A01", "A02", "A03"]
        ]
        calls = 0

        def fail_before_trace(pool, alleles, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected pre-trace failure")
            return pl.DataFrame({"Metadata_gene_allele": alleles, "mAP_vs_ref_norm": [0.4]})

        with patch.object(pa, "_compute_map_vs_ref", side_effect=fail_before_trace):
            with self.assertRaisesRegex(RuntimeError, "pre-trace failure"):
                pa._compute_control_null(
                    pl.DataFrame(rows), {"EMBED": ["f"]}, 20, 0, "site", True, 32, 0.05, 42, 1, "t4", min_cells=1
                )

    def test_zero_norm_profiles_cannot_produce_finite_but_meaningless_ap(self):
        pa = load_script(REPO_ROOT / "scripts/09c_classify_PA.py", "review_pa_zero")
        rows = [
            dict(
                Metadata_gene_allele=allele,
                Metadata_node_type=node,
                Metadata_symbol="G",
                Metadata_Plate=f"P_T{t}",
                f=0.0,
                g=0.0,
            )
            for t in range(1, 5)
            for allele, node in [("G", "disease_wt"), ("G_v", "allele")]
        ]
        with self.assertRaisesRegex(ValueError, "norm|cosine"):
            pa._run_map(
                pl.DataFrame(rows),
                ["f", "g"],
                "Metadata_node_type == 'disease_wt'",
                32,
                0.05,
                42,
                "vs_ref",
                max_workers=1,
                neg_sameby=["Metadata_Plate", "Metadata_symbol"],
                test_split="t4",
            )

    def test_empty_t4_queries_are_explicitly_not_estimable_before_pairing(self):
        pa = load_script(REPO_ROOT / "scripts/09c_classify_PA.py", "review_pa_empty")
        frame = pl.DataFrame(
            {
                "Metadata_gene_allele": ["G", "G_v"],
                "Metadata_node_type": ["disease_wt", "allele"],
                "Metadata_symbol": ["G", "G"],
                "Metadata_Plate": ["P_T1", "P_T1"],
                "f": [1.0, 2.0],
            }
        )
        with patch.object(pa, "average_precision", side_effect=AssertionError("No pairing without queries")):
            result = pa._run_map(
                frame,
                ["f"],
                "Metadata_node_type == 'disease_wt'",
                32,
                0.05,
                42,
                "vs_ref",
                max_workers=1,
                test_split="t4",
            )
        self.assertTrue(result.empty)

    def test_summary_consumer_refuses_an_omitted_requested_representation(self):
        module = load_script(REPO_ROOT / "scripts/11_summarize_across_reps.py", "review_summary_consumer")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing requested"):
                module.load_per_rep_summaries(Path(directory), ["missing"])

    def test_new_t4_controls_only_run_cannot_fall_back_to_legacy_or_empty_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = root / "vit_t4/batch/controls"
            controls.mkdir(parents=True)
            (controls / "started.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Incomplete|Missing"):
                load_single_fold_metrics("vit", "batch", classification_dir=root)
            with self.assertRaisesRegex(ValueError, "Incomplete|Missing"):
                load_single_fold_metrics("subcell_allele_rybg_v2_mae_s42", "absent", classification_dir=root)

    def test_undefined_single_fold_sd_keeps_a_numeric_schema_for_consumers(self):
        # Receipt validation is covered end-to-end elsewhere; isolate CSV inference here.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "vit_t4/batch"
            (output / "controls").mkdir(parents=True)
            calibration = output / "controls/calibration.json"
            calibration.write_text("{}")
            (output / "completion.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "context": {"protocol": "t1-t3_train_t4_test"},
                        "calibration_sha256": provenance.sha256(calibration),
                    }
                )
            )
            (output / "metrics_summary.csv").write_text("auroc_mean,auroc_std\n0.8,\n")
            with (
                patch.object(stages, "require_stage") as gate,
                patch("prot_loc_benchmark.classification.calibration.load_calibration"),
            ):
                result = load_single_fold_metrics("vit", "batch", classification_dir=root)
                gate.assert_called_once_with(output, representation="vit", batch="batch")
                receipt = json.loads((output / "completion.json").read_text())
                receipt["context"]["protocol"] = "lopo"
                (output / "completion.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "T4 protocol"):
                    load_single_fold_metrics("vit", "batch", classification_dir=root)
            self.assertEqual(result["auroc_std"].dtype, pl.Float64)
            self.assertIsNone(result["auroc_std"].mean())

    def test_pa_consumer_selects_verified_t4_outputs_and_rejects_missing_batches(self):
        module = load_script(REPO_ROOT / "scripts/10_benchmark_clinvar.py", "review_pa_consumer")
        rep = "subcell_allele_rybg_v2_mae_s42"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Test the consumer, not a production resource allocation or real producer.
            with (
                patch.object(stages, "DATA_DIR", root),
                patch.object(stages, "require_bounded_execution", return_value={}),
                patch.object(provenance, "PROVENANCE_LOG", root / "ledger.json"),
                patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(root)}),
                patch.object(module, "CLASSIFICATION_PA_DIR", root / "classification_PA"),
            ):
                for batch in ("a", "b", "full"):
                    path = root / "classification_PA" / (rep + "_t4") / batch
                    # A complete all-query run filed under _t4 must not be relabeled.
                    split = "none" if batch == "full" else "t4"
                    with stages.stage(path, [], {"representation": rep, "batch": batch, "test_split": split}):
                        pl.DataFrame(
                            {"Metadata_gene_allele": ["G_v"], "channel": ["EMBED"], "mAP_vs_ref_norm": [0.7]}
                        ).write_parquet(path / "mAP_results.parquet")
                loaded = module.load_pa_metrics([rep], {"pair": ("a", "b")}, fold_mode="t4-only")
                self.assertEqual(loaded["channel"].to_list(), ["EMBED_vs_ref"] * 2)
                self.assertEqual(loaded["representation"].unique().to_list(), [rep])
                with self.assertRaisesRegex(ValueError, "T4-query protocol"):
                    module.load_pa_metrics([rep], {"pair": ("a", "full")}, fold_mode="t4-only")
                with self.assertRaisesRegex(ValueError, "Missing"):
                    module.load_pa_metrics([rep], {"pair": ("a", "missing")}, fold_mode="t4-only")
                (root / "classification_PA" / (rep + "_t4") / "b/stage.json").unlink()
                with self.assertRaisesRegex(ValueError, "Missing"):
                    module.load_pa_metrics([rep], {"pair": ("a", "b")}, fold_mode="t4-only")


if __name__ == "__main__":
    unittest.main()
