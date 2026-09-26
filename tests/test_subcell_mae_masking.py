"""Standalone regression for the local patch to upstream SubCell a71236f masking.

Run: PYTHONPATH=vendor/subcell_embed python -m unittest discover -s tests -p test_subcell_mae_masking.py
"""
import unittest

import torch

from models.object_aware_mae import ViTMAEEmbeddings, ViTMAEMaskAwareConfig


class MAEMaskingChecks(unittest.TestCase):
    def test_only_eval_zero_mask_bypasses_upstream_shuffle(self):
        embeddings = ViTMAEEmbeddings(ViTMAEMaskAwareConfig(
            image_size=448, patch_size=16, num_channels=4, hidden_size=16,
            mask_ratio=.25, object_mask_ratio=0,
        ))
        tokens = torch.arange(2 * 784 * 16, dtype=torch.float32).reshape(2, 784, 16)
        identity = torch.arange(784).expand(2, -1)
        for training in (True, False):
            embeddings.train(training)
            for ratio in (0., .25):
                with self.subTest(training=training, ratio=ratio):
                    torch.manual_seed(2026)
                    before = torch.get_rng_state().clone()
                    # no_grad alone must NOT change the train-mode masking behavior.
                    with torch.no_grad():
                        actual, mask, restore = embeddings.random_masking(tokens, mask_ratio=ratio)
                    after = torch.get_rng_state().clone()
                    if not training and ratio == 0:
                        self.assertTrue(torch.equal(actual, tokens))
                        self.assertTrue(torch.equal(restore, identity))
                        self.assertEqual(mask.count_nonzero().item(), 0)
                        self.assertTrue(torch.equal(before, after))
                        continue
                    # Original a71236f algorithm for object_mask=None: draw, sort,
                    # keep the prefix, then restore the binary mask's spatial order.
                    torch.set_rng_state(before)
                    order = torch.rand(2, 784).argsort(dim=1)
                    expected_restore = order.argsort(dim=1)
                    kept = int(784 * (1 - ratio))
                    expected = tokens.gather(1, order[:, :kept, None].expand(-1, -1, 16))
                    expected_mask = torch.ones(2, 784)
                    expected_mask[:, :kept] = 0
                    expected_mask = expected_mask.gather(1, expected_restore)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(mask, expected_mask))
                    self.assertTrue(torch.equal(restore, expected_restore))
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    self.assertFalse(torch.equal(before, after))
                    if ratio == 0:
                        self.assertFalse(torch.equal(actual, tokens))
                        self.assertTrue(torch.equal(actual.gather(1, restore[..., None].expand_as(tokens)), tokens))
        embeddings.eval()
        embeddings.config.mask_ratio = 0
        before = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(embeddings.random_masking(tokens)[0], tokens))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
