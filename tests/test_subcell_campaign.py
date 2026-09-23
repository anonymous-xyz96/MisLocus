"""Fail-closed campaign resource guard, without launching processes or using GPUs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/08e_run_subcell_campaign.py'
spec = importlib.util.spec_from_file_location('campaign', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignChecks(unittest.TestCase):
    def test_live_job_status(self):
        baseline = {'ActiveState': 'active', 'SubState': 'running', 'Result': 'success', 'ExecMainStatus': '0'}
        for change, expected in (({}, 'running'), ({'SubState': 'start', 'ActiveState': 'activating'}, 'starting'),
                                 ({'SubState': 'exited'}, 'success'),
                                 ({'SubState': 'exited', 'ExecMainStatus': '1'}, 'failed'),
                                 ({'ActiveState': 'failed', 'Result': 'oom-kill'}, 'failed'),
                                 ({'ActiveState': 'inactive'}, 'failed')):
            with self.subTest(change=change):
                self.assertEqual(campaign.job_status({**baseline, **change}), expected)

    def test_completion_binds_native_score_and_finished_attempt(self):
        import torch
        from prot_loc_benchmark.provenance import capture_source, save_json, sha256
        from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            capture_source(run)
            fingerprint = json.loads((run / 'source.json').read_text())['code_sha256']
            config = {'output': str(run), 'seed': 42}
            identity = {'kind': 'production', 'code_sha256': fingerprint, 'config': config}
            save_json(run / 'run.json', identity)
            (run / 'models').mkdir()
            selector = AlleleCheckpoint(run)
            selector.best_model_score = torch.tensor(.25, dtype=torch.float64)
            checkpoint = {'epoch': 99, 'global_step': 9700, 'allele_v2': {'identity': identity},
                          'callbacks': {selector.state_key: selector.state_dict()}}
            best = run / 'models/best_model_ap.ckpt'
            torch.save(checkpoint, best)
            selection = {'identity': identity, 'checkpoint': 'models/best_model_ap.ckpt', 'sha256': sha256(best),
                         'pass': 100, 'global_step': 9700, 'macro_ap': .25, 'metric_dtype': 'float64'}
            save_json(run / 'selection.json', selection)
            marker = run / 'attempts/first/completed.json'
            marker.parent.mkdir(parents=True)
            complete = {'identity': identity, 'selection': selection, 'global_step': 9700, 'status': 'fit_completed'}
            with self.assertRaisesRegex(RuntimeError, 'completion/selection binding'):
                campaign.verify_completed_run(run, config, fingerprint)
            save_json(marker, complete)
            self.assertEqual(campaign.verify_completed_run(run, config, fingerprint), selection)
            for change in ({'status': 'failed'}, {'global_step': 0}, {'identity': {}}, {'selection': {}}):
                save_json(marker, {**complete, **change})
                with self.assertRaisesRegex(RuntimeError, 'completion/selection binding'):
                    campaign.verify_completed_run(run, config, fingerprint)
            save_json(marker, complete)
            save_json(run / 'selection.json', {**selection, 'macro_ap': .9})
            with self.assertRaisesRegex(ValueError, 'native checkpoint selector'):
                campaign.verify_completed_run(run, config, fingerprint)

    def test_resource_guards(self):
        baseline = {'available_memory_bytes': 700 * campaign.GIB, 'free_disk_bytes': 200 * campaign.GIB,
                    'gpu': '\n'.join(f'{i}, UUID{i}, 1, 95830, 30, 0' for i in range(4))}
        campaign.require_resources(baseline, starting=True)
        for change, starting in [({'available_memory_bytes': 639 * campaign.GIB}, True),
                                 ({'available_memory_bytes': 127 * campaign.GIB}, False),
                                 ({'free_disk_bytes': 99 * campaign.GIB}, True),
                                 ({'free_disk_bytes': 31 * campaign.GIB}, False),
                                 ({'gpu': baseline['gpu'].replace(', 1,', ', 2000,')}, True),
                                 ({'gpu': baseline['gpu'].replace(', 30,', ', 85,')}, False),
                                 ({'gpu': baseline['gpu'].splitlines()[0]}, True)]:
            with self.subTest(change=change, starting=starting), self.assertRaises(RuntimeError):
                campaign.require_resources({**baseline, **change}, starting=starting)
        campaign.require_resources({**baseline, 'gpu': baseline['gpu'].replace(', 1,', ', 15000,')})


if __name__ == '__main__':
    unittest.main()
