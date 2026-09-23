"""Common crop staging needs only stdlib/git, never a model environment."""
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prot_loc_benchmark import cell_crops, provenance
from prot_loc_benchmark.config import REPO_ROOT, CELL_CROP_CHANNEL_FILES


class CellCropPreparationChecks(unittest.TestCase):
    def test_cli_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, '-S', str(REPO_ROOT / 'scripts/01_prepare_cell_crops.py'), '--help'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('extract', result.stdout)

    def test_raw_bytes_receipt_and_safe_extraction(self):
        for unsafe in (None, '../outside.npy', '/outside.npy', 'ALLELE/link.npy'):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                release = root / 'release'
                relative = 'single_cell_crops/example_Batch_99/shard-00.tar.gz'
                archive_path = release / relative
                archive_path.parent.mkdir(parents=True)
                payload = b'native crop bytes; no resize, normalization or label policy'
                with tarfile.open(archive_path, 'w:gz') as archive:
                    for name in (*CELL_CROP_CHANNEL_FILES, 'metadata.parquet'):
                        member = tarfile.TarInfo(f'ALLELE/{name}')
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                    if unsafe:
                        member = tarfile.TarInfo(unsafe)
                        member.size = 0
                        if unsafe.endswith('link.npy'):
                            member.type = tarfile.SYMTYPE
                            member.linkname = '/outside'
                        archive.addfile(member, io.BytesIO())
                digest = provenance.sha256(archive_path)
                inventory = {'remote': 'synthetic', 'revision': 'fixture',
                             'files': {relative: {'sha256': digest, 'size': archive_path.stat().st_size}}}
                output = root / 'crops'
                with patch.object(cell_crops, 'release_inventory', return_value=inventory), patch.object(
                        provenance, 'PROVENANCE_LOG', root / 'ledger.json'):
                    if unsafe:
                        with self.assertRaises(ValueError):
                            cell_crops.extract(release, output)
                        self.assertFalse((output / 'extraction.json').exists())
                    else:
                        cell_crops.extract(release, output)
                        receipt = json.loads((output / 'extraction.json').read_text())
                        self.assertEqual(receipt['release'], inventory)
                        self.assertEqual(len(receipt['files']), 5)
                        for name, info in receipt['files'].items():
                            self.assertEqual((output / name).read_bytes(), payload)
                            self.assertEqual(provenance.sha256(output / name), info['sha256'])
                        with self.assertRaises(FileExistsError):
                            cell_crops.extract(release, output)
                        with self.assertRaisesRegex(ValueError, 'mirror'):
                            cell_crops.extract(release, release / 'forbidden')
                self.assertEqual(provenance.sha256(archive_path), digest)
                self.assertFalse((root / 'outside.npy').exists())


if __name__ == '__main__':
    unittest.main()
