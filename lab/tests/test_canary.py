import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.export.kaggle_canary_notebook import SAMPLE, canary_weights, read_safetensors_header  # noqa: E402


@unittest.skipUnless((SAMPLE / "adapter_model.safetensors").exists(), "needs the competition sample adapter")
class CanaryTest(unittest.TestCase):
    def test_same_layout_as_the_sample_with_nonzero_lora_b(self):
        sample = (SAMPLE / "adapter_model.safetensors").read_bytes()
        canary = canary_weights(sample)
        layout = lambda h: {k: (v["dtype"], v["shape"]) for k, v in h.items()}  # noqa: E731
        self.assertEqual(layout(read_safetensors_header(sample)), layout(read_safetensors_header(canary)))
        self.assertEqual(len(canary), len(sample))
        self.assertEqual(canary, canary_weights(sample))  # fixed seed: the notebook embeds a known sha
        self.assertNotEqual(canary, sample)


if __name__ == "__main__":
    unittest.main()
