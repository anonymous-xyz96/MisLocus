"""Regression cases found during the pre-commit standards/spec review."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prot_loc_benchmark import provenance, stages


class ReviewRegressions(unittest.TestCase):
    def test_output_symlinks_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis"
            external = root / "external"
            external.mkdir()
            raw = external / "raw"
            raw.write_text("unchanged input")
            with (
                patch.object(stages, "DATA_DIR", analysis),
                patch.object(provenance, "PROVENANCE_LOG", analysis / "ledger.json"),
                patch.dict(os.environ, {"MISLOCUS_DATA_ROOT": str(analysis)}),
            ):
                for kind in ("external-directory", "internal-directory", "file", "dangling"):
                    with self.subTest(kind=kind):
                        output = analysis / kind
                        with self.assertRaisesRegex(ValueError, "Output symlink/escape"):
                            with stages.stage(output, [], {}):
                                nested = output / "nested"
                                nested.mkdir()
                                (nested / "result").write_text("result")
                                target = {
                                    "external-directory": external,
                                    "internal-directory": nested,
                                    "file": raw,
                                    "dangling": external / "missing",
                                }[kind]
                                (output / "link").symlink_to(target, target_is_directory=kind.endswith("directory"))
                                provenance.save_json(output / "completion.json", {"status": "complete"})
                        self.assertFalse((output / "stage.json").exists())
                        self.assertFalse((output / "completion.json").exists())
                        self.assertTrue((output / "failed.json").exists())
                        with self.assertRaisesRegex(ValueError, "Missing completed"):
                            stages.require_stage(output)
                # Ordinary nested outputs and explicitly allowed raw-input links still work.
                output = analysis / "valid"
                output.mkdir()
                (output / "raw").symlink_to(raw)
                with stages.stage(output, [output / "raw"], {}, allowed=("raw",)):
                    (output / "nested").mkdir()
                    (output / "nested/result").write_text("result")
                receipt, _ = provenance.read_json_with_hash(stages.require_stage(output))
                self.assertIn("nested/result", receipt["outputs"])
                self.assertNotIn("raw", receipt["outputs"])
                self.assertEqual(raw.read_text(), "unchanged input")

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


if __name__ == "__main__":
    unittest.main()
