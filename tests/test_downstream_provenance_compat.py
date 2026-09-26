"""Shared provenance must remain usable by Cytoself's Python 3.10 callers."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prot_loc_benchmark.config import REPO_ROOT


class ProvenanceCompatibilityChecks(unittest.TestCase):
    def test_record_without_datetime_utc_alias(self):
        # Also exercises the missing 3.11 alias when the main suite runs on 3.12.
        code = """
import datetime
if hasattr(datetime, 'UTC'):
    del datetime.UTC
from prot_loc_benchmark.config import DATA_DIR
from prot_loc_benchmark.provenance import record
output = DATA_DIR / 'output'
output.mkdir()
record([output], stage_id='compatibility-check')
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "MISLOCUS_DATA_ROOT": directory},
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sidecar = json.loads((root / "output/_provenance.json").read_text())
            entries = json.loads((root / "provenance_log.json").read_text())["runs"]
            self.assertEqual(sidecar["stage_id"], "compatibility-check")
            self.assertTrue(sidecar["timestamp"].endswith("+00:00"))
            self.assertEqual([entry["id"] for entry in entries], ["compatibility-check"])
            self.assertEqual(entries[0]["timestamp"], sidecar["timestamp"])


if __name__ == "__main__":
    unittest.main()
