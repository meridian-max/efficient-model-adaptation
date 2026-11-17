import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from model_adaptation.lora import LoRALinear, parameter_counts
from model_adaptation.experiment import make_data, fit, run_experiment


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4)
        self.base = nn.Linear(32, 6)
        self.adapter = LoRALinear(self.base, rank=2)
        self.x = torch.randn(12, 32)

    def test_initial_adapter_preserves_base_outputs(self):
        torch.testing.assert_close(self.base(self.x), self.adapter(self.x))

    def test_base_has_no_gradient(self):
        self.adapter(self.x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in self.adapter.base.parameters()))
        self.assertIsNotNone(self.adapter.b.grad)

    def test_optimization_leaves_base_unchanged(self):
        old = copy.deepcopy(self.adapter.base.state_dict())
        labels = torch.arange(12) % 6
        fit(self.adapter, (self.x, labels), (self.x, labels), steps=20)
        for name, value in old.items():
            torch.testing.assert_close(value, self.adapter.base.state_dict()[name], rtol=0, atol=0)

    def test_merge_matches_nonzero_adapter(self):
        with torch.no_grad():
            self.adapter.b.normal_()
        torch.testing.assert_close(self.adapter(self.x), self.adapter.merged()(self.x), rtol=1e-5, atol=1e-5)

    def test_merge_does_not_mutate_adapter(self):
        before = self.adapter(self.x).detach().clone()
        merged = self.adapter.merged()
        with torch.no_grad():
            merged.weight.zero_()
        torch.testing.assert_close(before, self.adapter(self.x))

    def test_parameter_reduction(self):
        self.assertEqual(parameter_counts(self.adapter)["trainable"], 76)
        self.assertEqual(parameter_counts(self.base)["trainable"], 198)

    def test_biasless_layer_supported(self):
        adapter = LoRALinear(nn.Linear(32, 6, bias=False))
        self.assertIsNone(adapter.merged().bias)

    def test_invalid_rank_rejected(self):
        for rank in [0, -1, 7, True]:
            with self.assertRaises(ValueError):
                LoRALinear(self.base, rank)

    def test_invalid_alpha_rejected(self):
        for alpha in [0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                LoRALinear(self.base, alpha=alpha)

    def test_data_manifest_is_reproducible_and_splits_differ(self):
        _, first = make_data(42)
        _, second = make_data(42)
        self.assertEqual(first, second)
        self.assertEqual(len({s["sha256"] for s in first.values()}), 4)

    def test_report_is_written_from_executed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            result = run_experiment(path, seeds=[7], steps=20)
            self.assertTrue(path.exists())
            self.assertEqual(len(result["summary"]), 3)
            self.assertLess(result["runs"][0]["adapter_merge_max_absolute_error"], 1e-4)


if __name__ == "__main__":
    unittest.main()
