"""Synthetic safety/lineage checks. No production data, checkpoints or GPU jobs."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark import provenance, stages
from prot_loc_benchmark.identity import CELL_ID, identify_cells
from prot_loc_benchmark.provenance import save_json


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


if __name__ == "__main__":
    unittest.main()
