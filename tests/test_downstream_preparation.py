"""CPU-only cleanup and external-root checks; never consume production embeddings."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark.config import REPO_ROOT
from prot_loc_benchmark.preprocessing.clean import drop_nan_features
from prot_loc_benchmark.preprocessing.normalize import compute_plate_stats, robustmad, select_variant_features
from prot_loc_benchmark.provenance import sha256


class DownstreamPreparationChecks(unittest.TestCase):
    def test_nan_inf_null_threshold_and_metadata(self):
        frame = pl.DataFrame(
            {
                "Metadata_CellID": ["a", "b", "c", "d", "e", "f"],
                "Metadata_optional": [None] * 6,
                "many_missing": [float("nan"), float("inf"), -float("inf"), None, 1.0, 2.0],
                "few_missing": [1.0, float("nan"), 3.0, 4.0, 5.0, 6.0],
                "integer": [1, 2, 3, 4, 5, 6],
            }
        )
        result = drop_nan_features(frame.lazy(), cell_threshold=1).collect()
        self.assertEqual(result.columns, ["Metadata_CellID", "Metadata_optional", "few_missing", "integer"])
        self.assertEqual(result["Metadata_CellID"].to_list(), ["a", "c", "d", "e", "f"])
        self.assertEqual(result["Metadata_optional"].null_count(), 5)
        for missing in (float("nan"), float("inf"), -float("inf"), None):
            with self.subTest(missing=missing):
                data = pl.DataFrame({"Metadata_CellID": ["a", "b", "c"], "value": [1.0, missing, 2.0]})
                self.assertEqual(drop_nan_features(data.lazy()).collect()["Metadata_CellID"].to_list(), ["a", "c"])

    def test_feature_selection_includes_entirely_failing_plates(self):
        frame = pl.DataFrame({"Metadata_Plate": ["A", "A", "B", "B"], "feature": [1.0, 3.0, 5.0, 5.0]})
        stats = pl.DataFrame(
            {
                "Metadata_Plate": ["A", "B"],
                "feature": ["feature", "feature"],
                "mad": [1.0, 0.0],
                "abs_coef_var": [0.5, 0.0],
            }
        )
        self.assertEqual(select_variant_features(frame.lazy(), stats).collect().columns, ["Metadata_Plate"])
        for invalid in (stats.head(1), pl.concat([stats, stats.head(1)])):
            with self.assertRaisesRegex(ValueError, "statistics"):
                select_variant_features(frame.lazy(), invalid)

    def test_normalization_requires_complete_finite_statistics_and_preserves_cells(self):
        frame = pl.DataFrame(
            {
                "Metadata_Plate": ["B", "A"] * 4,
                "Metadata_CellID": list("abcdefgh"),
                "Metadata_optional": [None] * 8,
                "f": [5.0, 1.0, 5.0, 3.0, 7.0, 5.0, 5.0, 7.0],
                "g": [float(i) for i in range(10, 18)],
            }
        )
        stats = compute_plate_stats(frame.lazy())
        for invalid in (
            stats.filter(pl.col("Metadata_Plate") == "A"),
            stats.slice(1),
            pl.concat([stats, stats.head(1)]),
            stats.with_columns(pl.lit(None, dtype=pl.Float32).alias("median")),
            stats.with_columns(pl.lit(float("nan")).alias("mad")),
            stats.with_columns(pl.lit(float("inf")).alias("median")),
            stats.with_columns(pl.lit(-1.0).alias("mad")),
        ):
            with self.subTest(stats=invalid), self.assertRaisesRegex(ValueError, "statistics"):
                robustmad(frame.lazy(), invalid).collect()
        # Extra features in saved stats are normal after CP feature selection.
        extra = stats.with_columns(pl.lit("removed_feature").alias("feature"))
        valid = pl.concat([stats.with_columns(pl.col("feature").cast(pl.String)), extra])
        normalized = robustmad(frame.lazy(), valid).collect()
        self.assertEqual(normalized["Metadata_CellID"].to_list(), list("abcdefgh"))
        self.assertEqual(normalized["Metadata_optional"].null_count(), 8)
        self.assertEqual(normalized["f"].to_list(), [5.0, -1.5, 5.0, -0.5, 7.0, 0.5, 5.0, 1.5])
        self.assertEqual(normalized["f"].dtype, pl.Float32)
        # Reordering cells or statistics changes neither fitted values nor identities.
        permuted = robustmad(frame.reverse().lazy(), valid.reverse()).collect()
        self.assertTrue(permuted.sort("Metadata_CellID").equals(normalized.sort("Metadata_CellID")))
        with self.assertRaisesRegex(ValueError, "plate"):
            robustmad(frame.with_columns(pl.lit(None, dtype=pl.String).alias("Metadata_Plate")).lazy(), stats).collect()

    def test_mad_audit_distinguishes_sparse_from_constant_dimensions(self):
        spec = importlib.util.spec_from_file_location("mad_audit", REPO_ROOT / "docs/checks/audit_plate_mad.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        stats = pl.DataFrame(
            {
                "Metadata_Plate": ["A", "B", "C", "D", "E"],
                "feature": ["f"] * 5,
                "median": [0.0, 0.0, 5.0, 5.0, float("nan")],
                "mad": [1.0, 0.0, 0.0, 0.0, 1.0],
                "min": [-2.0, 0.0, 5.0, 5.0, 0.0],
                "max": [101.0, 2.0, 5.0, 7.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plate_stats.parquet"
            stats.write_parquet(path)
            before = sha256(path)
            result = audit.audit(path)
            for key, value in {
                "plate_feature_pairs": 5,
                "invalid_stat_pairs": 1,
                "zero_mad_pairs": 3,
                "zero_mad_constant_pairs": 1,
                "zero_mad_nonconstant_pairs": 2,
                "zero_mad_nonconstant_nonzero_median_pairs": 1,
                "pairs_with_extrema_outside_100_after_normalization": 1,
                "largest_abs_normalized_extremum": 101.0,
            }.items():
                self.assertEqual(result[key], value, key)
            self.assertEqual(sha256(path), before)

    def test_chunked_audit_matches_batch_statistics(self):
        folder = REPO_ROOT / "docs/checks"
        spec = importlib.util.spec_from_file_location("export_audit", folder / "audit_export_mad.py")
        audit = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(folder), *sys.path]):
            spec.loader.exec_module(audit)
        frame = pl.DataFrame(
            {
                "Metadata_Plate": ["B", "A"] * 24 + ["C"] * 3,
                "Metadata_Well": ["A01"] * 51,
                "f": [float(i) for i in range(51)],
                "g": [5.0] * 51,
            }
        )
        frame = frame.with_columns([pl.col("f").alias(f"feature_{i}") for i in range(63)])
        features = [c for c in frame.columns if not c.startswith("Metadata_")]
        expected = compute_plate_stats(audit.drop_low_cell_count_wells(frame.lazy()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            frame.write_parquet(path)
            before = sha256(path)
            stats = audit.compute_export_stats(path, features, frame.height)
            keys = ["Metadata_Plate", "feature"]
            self.assertEqual(stats.sort(keys).to_dicts(), expected.sort(keys).to_dicts())
            self.assertEqual(sha256(path), before)
            with self.assertRaisesRegex(ValueError, "rows"):
                audit.compute_export_stats(path, features, frame.height + 1)

    def test_data_root_default_override_and_validation(self):
        code = """
import json
from prot_loc_benchmark.config import DATA_DIR, INTERIM_DIR, CLASSIFICATION_OUTPUT_DIR, CLASSIFICATION_PA_DIR, ANNOTATIONS_DIR
from prot_loc_benchmark.provenance import PROVENANCE_LOG
print(json.dumps([str(p) for p in (DATA_DIR, INTERIM_DIR, CLASSIFICATION_OUTPUT_DIR, CLASSIFICATION_PA_DIR, PROVENANCE_LOG, ANNOTATIONS_DIR)]))
"""
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        env.pop("MISLOCUS_DATA_ROOT", None)
        with tempfile.TemporaryDirectory() as directory:
            for override in (None, directory, "~/derived-fixture"):
                configured = {**env, "HOME": directory}
                if override is not None:
                    configured["MISLOCUS_DATA_ROOT"] = override
                expected = REPO_ROOT / "data" if override is None else Path(directory)
                if override == "~/derived-fixture":
                    expected /= "derived-fixture"
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code], env=configured, capture_output=True, text=True, check=True
                )
                self.assertEqual(
                    json.loads(result.stdout),
                    [
                        str(p)
                        for p in (
                            expected,
                            expected / "interim",
                            expected / "processed/classification",
                            expected / "processed/classification_PA",
                            expected / "provenance_log.json",
                            REPO_ROOT / "annotations",
                        )
                    ],
                )
            self.assertFalse((Path(directory) / "derived-fixture").exists())  # Import does not create data.
            for invalid in ("", "relative/data"):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code],
                    env={**env, "MISLOCUS_DATA_ROOT": invalid},
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("absolute path", result.stderr)


if __name__ == "__main__":
    unittest.main()
