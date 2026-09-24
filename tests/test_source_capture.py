"""Source snapshots must describe archived bytes, not later edits of the checkout."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from prot_loc_benchmark import provenance


class SourceCaptureChecks(unittest.TestCase):
    def test_edit_during_capture_keeps_archive_consistent_and_blocks_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'snapshot'
            (source / 'src').mkdir(parents=True)
            output.mkdir()
            for name in ('pyproject.toml', 'pixi.lock'):
                (source / name).write_text('fixture')
            changing = source / 'src/example.py'
            changing.write_text('VERSION = 1\n')
            addfile = tarfile.TarFile.addfile

            def archive_then_edit(archive, member, stream=None):
                addfile(archive, member, stream)
                if member.name == 'src/example.py':
                    changing.write_text('VERSION = 2\n')

            with patch.object(provenance, 'REPO_ROOT', source), patch.object(
                    provenance.subprocess, 'check_output', return_value='fixture'), patch.object(
                    tarfile.TarFile, 'addfile', archive_then_edit):
                original = provenance.code_fingerprint()
                provenance.capture_source(output)
                receipt = json.loads((output / 'source.json').read_text())
                self.assertEqual(receipt['code_sha256'], original)
                provenance.verify_source(output, original)
                self.assertNotEqual(provenance.code_fingerprint(), original)
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    provenance.invocation(output)
                changing.write_text('VERSION = 1\n')
                self.assertEqual(provenance.invocation(output)['code_sha256'], original)


if __name__ == '__main__':
    unittest.main()
