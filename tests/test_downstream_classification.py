"""Controls-first T4 contract; tiny CPU fixtures, never production inputs."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
