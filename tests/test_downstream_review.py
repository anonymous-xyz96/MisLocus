"""Regression cases found during the pre-commit standards/spec review."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark import provenance, stages
from prot_loc_benchmark.classification.metrics import load_single_fold_metrics


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


if __name__ == "__main__":
    unittest.main()
