"""Regression cases found during the pre-commit standards/spec review."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prot_loc_benchmark import provenance, stages


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


if __name__ == "__main__":
    unittest.main()
