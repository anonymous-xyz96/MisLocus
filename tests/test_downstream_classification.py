"""Controls-first T4 contract; tiny CPU fixtures, never production inputs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark.classification.cv import generate_folds, split_fold
from prot_loc_benchmark.classification.metrics import (
    aggregate_allele_metrics,
    compute_null_threshold,
)
from prot_loc_benchmark.classification.pairs import build_control_pairs, get_pair_data
from prot_loc_benchmark.classification.reporting import plot_auroc_distributions


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


if __name__ == "__main__":
    unittest.main()
