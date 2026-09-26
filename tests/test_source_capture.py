"""Source snapshots must describe archived bytes, not later edits of the checkout."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from prot_loc_benchmark import provenance


class SourceCaptureChecks(unittest.TestCase):
    def test_json_hash_is_bound_to_parsed_bytes_during_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            original = b'{"version": 1}\n'
            path.write_bytes(original)
            read = Path.read_bytes

            def read_then_replace(target):
                content = read(target)
                target.write_text('{"version": 2}')
                return content

            with patch.object(Path, 'read_bytes', read_then_replace):
                receipt, digest = provenance.read_json_with_hash(path)
            self.assertEqual(receipt, {'version': 1})
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())
            self.assertNotEqual(digest, provenance.sha256(path))

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
