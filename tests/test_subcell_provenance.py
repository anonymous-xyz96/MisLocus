"""Staging/selection/provenance regression checks, using tiny disposable releases."""
import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from test_subcell_allele_v2 import make_cohort
from prot_loc_benchmark import cell_crops, provenance
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, REPO_ROOT, SUBCELL_CHANNEL_FILES
from prot_loc_benchmark.representations.subcell_manifest import (
    build_manifest, load_preflight, save_json, sha256,
)
from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint, capture_source, verify_selection, verify_source


def stage_fixture(root):
    """Exercise real tar extraction + metadata/header preflight, mock only git inventory."""
    payload, release, crops, cohort = [root / name for name in ('payload', 'release', 'crops', 'cohort')]
    payload.mkdir()
    frame, _ = make_cohort(payload, count=2)
    frame.loc[frame.cell_idx == 11, 'Metadata_Plate'] = frame.loc[frame.cell_idx == 11, 'Metadata_Plate'].str.replace('T3', 'T4')
    frame.loc[frame.cell_idx == 11, 'Metadata_CellID'] = frame.loc[frame.cell_idx == 11, 'Metadata_CellID'].str.replace('T3', 'T4')
    for allele, group in frame.groupby('Metadata_gene_allele'):
        meta = pd.read_parquet(payload / allele / 'metadata.parquet')
        meta['Metadata_Plate'] = group.Metadata_Plate.to_numpy()
        meta['Metadata_CellID'] = group.Metadata_CellID.to_numpy()
        meta.to_parquet(payload / allele / 'metadata.parquet', index=False)
    (release / 'manifest').mkdir(parents=True)
    inventory = {'remote': 'synthetic-only', 'revision': 'test-fixture', 'files': {}}
    for batch in ALL_PUBLIC_BATCHES:
        manifest = release / f'manifest/manifest_Batch_{batch.rsplit("_", 1)[1]}.parquet'
        frame[[c for c in frame if c.startswith('Metadata_')]].iloc[::-1].to_parquet(manifest, index=False)
        shard = release / 'single_cell_crops' / batch / 'shard-00.tar.gz'
        shard.parent.mkdir(parents=True)
        with tarfile.open(shard, 'w:gz') as archive:
            for allele in sorted(frame.Metadata_gene_allele.unique()):
                for name in [*SUBCELL_CHANNEL_FILES, 'metadata.parquet']:
                    archive.add(payload / allele / name, arcname=f'{allele}/{name}')
        plate_metadata = release / 'representations/cellprofiler' / batch / 'features.parquet'
        plate_metadata.parent.mkdir(parents=True)
        plates = frame[['Metadata_Plate']].drop_duplicates().copy()
        plates['Metadata_plate_map_name'] = 'synthetic_P1'
        plates.to_parquet(plate_metadata, index=False)
        for path in (manifest, shard, plate_metadata):
            inventory['files'][str(path.relative_to(release))] = {'sha256': sha256(path), 'size': path.stat().st_size}
    raw_inventory = {**inventory, 'files': {p: info for p, info in inventory['files'].items()
                                           if p.startswith(('manifest/', 'single_cell_crops/'))}}
    before = {p: sha256(release / p) for p in inventory['files']}
    with patch.object(cell_crops, 'release_inventory', return_value=raw_inventory), patch(
            'prot_loc_benchmark.representations.subcell_manifest.release_inventory', return_value=inventory), patch.object(
            provenance, 'PROVENANCE_LOG', root / 'ledger.json'):
        cell_crops.extract(release, crops)
        build_manifest(release, crops, cohort)
        # The earlier full-inventory receipt remains reusable without re-extraction.
        receipt_path = crops / 'extraction.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['release'] = inventory
        save_json(receipt_path, receipt)
        legacy = root / 'legacy-cohort'
        build_manifest(release, crops, legacy)
        for name in ('manifest.parquet', 'class_index.json', 'validation_ids.json'):
            assert sha256(cohort / name) == sha256(legacy / name)
        receipt['release'] = raw_inventory
        save_json(receipt_path, receipt)
    assert before == {p: sha256(release / p) for p in inventory['files']}, 'Release checkout was modified'
    return release, crops, cohort


class ProvenanceChecks(unittest.TestCase):
    def test_snapshot_does_not_require_uncommitted_plan(self):
        from prot_loc_benchmark import provenance as snapshots
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('pyproject.toml', 'pixi.lock'):
                (root / name).write_text('test')
            with patch.object(snapshots, 'REPO_ROOT', root):
                without_plan = snapshots.code_fingerprint()
                plan = root / 'docs/plans/subcell-allele-rybg-finetuning.md'
                plan.parent.mkdir(parents=True)
                plan.write_text('locally retained contract')
                self.assertNotEqual(snapshots.code_fingerprint(), without_plan)
                self.assertIn(plan, snapshots.source_files())

    def test_frozen_and_adapted_cli_are_separate(self):
        from prot_loc_benchmark.representations.subcell_extract import main
        common = ['extract', '--preflight', 'unused', '--family', 'mae', '--output', 'unused']
        for frozen, wrong_source in ((True, ['--checkpoint', 'unused']),
                                     (False, ['--frozen-weights', 'unused', '--weights-sha256', 'unused'])):
            with patch('sys.argv', common + wrong_source), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main(frozen=frozen)
            self.assertEqual(failure.exception.code, 2)

    def test_separate_extraction_preflight_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, crops, cohort = stage_fixture(root)
            frame, classes, validation, evidence = load_preflight(cohort)
            self.assertEqual(len(frame), 144)
            self.assertEqual(len(classes), 2)
            self.assertEqual(set(frame.Metadata_plate_map_name), {'synthetic_P1'})
            self.assertEqual(set(frame.batch_id), set(ALL_PUBLIC_BATCHES))
            self.assertEqual(len(validation), 24)
            self.assertEqual(set(frame.loc[frame.cell_id.isin(validation), 'split']), {'val'})
            self.assertEqual(sha256(cohort / 'source.tar.gz'), evidence['source_archive_sha256'])
            self.assertFalse((release / 'extraction.json').exists())
            with self.assertRaisesRegex(ValueError, 'Hugging Face mirror'):
                build_manifest(release, crops, release / 'forbidden-output')
            path = Path(frame.base_path.iloc[0]) / 'dna.npy'
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with self.assertRaisesRegex(ValueError, 'changed after preflight'):
                load_preflight(cohort)

    def test_source_snapshot_and_authoritative_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_source(root)
            source = json.loads((root / 'source.json').read_text())
            self.assertEqual(source['archive_sha256'], sha256(root / 'source.tar.gz'))
            verify_source(root, source['code_sha256'])
            with self.assertRaisesRegex(ValueError, 'recorded code identity'):
                verify_source(root, '0' * 64)
            with tarfile.open(root / 'source.tar.gz') as archive:
                names = archive.getnames()
                self.assertIn('scripts/08d_extract_subcell_finetune_embeddings.py', names)
                self.assertIn('src/prot_loc_benchmark/representations/subcell_training.py', names)
                self.assertNotIn('data', names)
            identity = {'config': {'output': str(root)}}
            checkpoint = {'global_step': 970, 'epoch': 9, 'allele_v2': {'identity': identity}}
            path = root / 'best.ckpt'
            torch.save(checkpoint, path)
            save_json(root / 'selection.json', {'identity': identity, 'sha256': sha256(path), 'pass': 10,
                                              'global_step': 970, 'macro_ap': .123456789})
            verify_selection(path, checkpoint)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'authoritative'):
                verify_selection(path, checkpoint)

    def test_native_selector_ties_small_improvements_and_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = AlleleCheckpoint(directory)
            trainer = SimpleNamespace(strategy=SimpleNamespace(reduce_boolean_decision=lambda value: value))
            checkpoint.best_k_models = {'best': torch.tensor(.123456789, dtype=torch.float64)}
            checkpoint.kth_value = checkpoint.best_k_models['best']
            checkpoint.kth_best_model_path = 'best'
            self.assertFalse(checkpoint.check_monitor_top_k(trainer, torch.tensor(.123456789, dtype=torch.float64)))
            self.assertTrue(checkpoint.check_monitor_top_k(trainer, torch.tensor(.123456790, dtype=torch.float64)))
            stopping = EarlyStopping('val/macro_ap', mode='max', min_delta=.001, patience=5)
            self.assertFalse(stopping._evaluate_stopping_criteria(torch.tensor(.5, dtype=torch.float64))[0])
            for value in (.5001, .5002, .5003):
                self.assertFalse(stopping._evaluate_stopping_criteria(torch.tensor(value, dtype=torch.float64))[0])
            resumed = EarlyStopping('val/macro_ap', mode='max', min_delta=.001, patience=5)
            resumed.load_state_dict(stopping.state_dict())
            self.assertFalse(resumed._evaluate_stopping_criteria(torch.tensor(.5004, dtype=torch.float64))[0])
            self.assertTrue(resumed._evaluate_stopping_criteria(torch.tensor(.5005, dtype=torch.float64))[0])


if __name__ == '__main__':
    unittest.main()
