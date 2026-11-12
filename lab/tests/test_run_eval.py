import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

HAS_HARNESS = all(importlib.util.find_spec(m) for m in ("adk_submission", "swegemma", "yaml"))


@unittest.skipUnless(HAS_HARNESS, "adk_submission/swegemma not installed")
class OllamaThinkingOffTest(unittest.TestCase):
    def test_reasoning_effort_none_only_when_thinking_is_off(self):
        from adk_submission import compile_submission
        from swegemma.config import build_submission_limits

        from lab.eval.run_eval import model_registry, patch_thinking_off_for_ollama
        from lab.export.package import HARNESS_TOOLS
        from lab.export.variants import build_variant

        patch_thinking_off_for_ollama()
        limits, gen = build_submission_limits()
        with tempfile.TemporaryDirectory() as d:
            for name, effort in (("b1-no-skill", None), ("b1-think-off", "none")):
                agent = compile_submission(
                    submission_dir=build_variant(name, dest_root=Path(d)),
                    tool_registry={n: (lambda **_: "") for n in HARNESS_TOOLS},
                    model_registry=model_registry("http://127.0.0.1:1/v1", "qwen3:8b", "openai/", "EMPTY"),
                    limits=limits, generation_constraints=gen)
                self.assertEqual(agent.model._additional_args.get("reasoning_effort"), effort, name)



@unittest.skipUnless(HAS_HARNESS, "adk_submission/swegemma not installed")
class HarnessVersionsTest(unittest.TestCase):
    def test_versions_recorded_for_the_harness_packages(self):
        from lab.eval.run_eval import harness_versions

        v = harness_versions()
        self.assertEqual(set(v), {"swegemma", "adk-submission", "adk-eval-core", "google-adk", "vllm"})
        self.assertTrue(v["swegemma"])


if __name__ == "__main__":
    unittest.main()
