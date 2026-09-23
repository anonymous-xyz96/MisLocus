"""Fail-closed campaign resource guard, without launching processes or using GPUs."""
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/08e_run_subcell_campaign.py'
spec = importlib.util.spec_from_file_location('campaign', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignChecks(unittest.TestCase):
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
