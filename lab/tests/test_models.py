import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.export import models  # noqa: E402
from lab.export.variants import build_variant  # noqa: E402

HAS_HARNESS = all(importlib.util.find_spec(m) for m in ("adk_submission", "swegemma", "yaml"))

# Pinned by handoff O2: the default 31B build must keep these exact archives.
PINNED = {
    "top-mix-1k-an-off-tb0": "981a1e386896eed8dd3dd64babd0b21c5400a8bda45e0fd360edd884ea49b6a8",
    "top-mix-1k": "a80c27f46ccb19b5b857724ecd15211391cd599bb3063c19e210e5c7e7005310",
}
ENV = {"PYTHONPATH": str(ROOT / "lab/src")}


def package(src: Path, out: Path, model: str) -> Path:
    subprocess.run([sys.executable, "-m", "lab.export.package", "--src", str(src), "--model", model,
                    "--build", str(out / "submission"), "--zip", str(out / "submission.zip")],
                   check=True, capture_output=True, env=ENV)
    return out / "submission.zip"


def extract(archive: Path, dest: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    return {str(p.relative_to(dest)): p.read_bytes() for p in dest.rglob("*") if p.is_file()}


class ModelMapTest(unittest.TestCase):
    def test_names_sources_and_suffixes(self):
        self.assertEqual(models.name("31b"), "gemma-4-31b-it-qat-w4a16-ct")
        self.assertEqual(models.name(models.name("12b")), "gemma-4-12b-it-qat-w4a16-ct")  # a full name is accepted
        self.assertEqual(models.source("12b"), f"google/gemma-4/Other/{models.name('12b')}/2")
        self.assertEqual(models.path("12b"), f"/kaggle/input/models/google/gemma-4/other/{models.name('12b')}/2")
        self.assertEqual(models.bf16_source("12b"), "google/gemma-4/Transformers/gemma-4-12b-it-qat-q4_0-unquantized/2")
        self.assertEqual(models.variant_suffix("31b"), "")
        self.assertEqual(models.variant_suffix("12b"), "-12b")

    def test_unknown_model_fails_loudly(self):
        with self.assertRaises(SystemExit):
            models.name("99b")

    def test_run_eval_registers_every_package_model_name(self):
        # A 12B or E4B package declares its own checkpoint name (test above); run_eval must resolve it without
        # importing lab.*, so its alias list is a copy of models.NAMES. CI has no yaml (run_eval imports it to
        # read the sampling configs), so read the list from the source there and import it only when possible.
        source = (ROOT / "lab/src/lab/eval/run_eval.py").read_text()
        self.assertIn("CHECKPOINTS", source)
        for full in models.NAMES.values():
            self.assertIn(full, source, full)
        try:
            from lab.eval import run_eval
        except ImportError:
            return
        self.assertEqual(set(run_eval.CHECKPOINTS), set(models.NAMES.values()))
        self.assertEqual(run_eval.DECLARED_MODEL, models.name("31b"))


@unittest.skipUnless(HAS_HARNESS, "needs adk_submission and swegemma")
class PackageModelTest(unittest.TestCase):
    def test_default_builds_keep_their_pinned_sha_and_12b_differs_only_in_the_model(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, sha in PINNED.items():
                src = build_variant(name, dest_root=d / "variants-31b")
                self.assertEqual(hashlib.sha256(package(src, d / name, "31b").read_bytes()).hexdigest(), sha, name)

            src31 = build_variant("top-mix-1k-an-off-tb0", dest_root=d / "variants-31b-a")
            src12 = build_variant("top-mix-1k-an-off-tb0", dest_root=d / "variants-12b")
            files31 = extract(package(src31, d / "b31", "31b"), d / "out31")
            files12 = extract(package(src12, d / "b12", "12b"), d / "out12")

            self.assertEqual(sorted(files31), sorted(files12))
            changed = []
            for rel, blob31 in files31.items():
                if blob31 != files12[rel]:
                    changed.append(rel)
                    self.assertEqual(files12[rel], blob31.replace(models.name("31b").encode(), models.name("12b").encode()), rel)
            # agent.yaml plus the analyzer sub-agent carry the model line; nothing else changes
            self.assertEqual(sorted(changed), ["agent.yaml", "sub_agents/code_analyzer.yaml"])
            self.assertIn(models.name("12b").encode(), files12["agent.yaml"])

    def test_e4b_probe_differs_from_the_base_only_in_the_checkpoint(self):
        # doc 03 §14 item 41: bf-gov-to-e4b is the base package with the E4B checkpoint and nothing else
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = build_variant("bf-gov-to", dest_root=d / "variants")
            base = package(src, d / "b31", "31b")
            self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest()[:12], "1bab17a745ac")
            files31 = extract(base, d / "out31")
            files4 = extract(package(src, d / "be4b", "e4b"), d / "oute4b")
            changed = [rel for rel, blob in files31.items() if blob != files4[rel]]
            self.assertEqual(sorted(files31), sorted(files4))
            self.assertEqual(changed, ["agent.yaml"])
            self.assertEqual(files4["agent.yaml"],
                             files31["agent.yaml"].replace(models.name("31b").encode(), models.name("e4b").encode()))


if __name__ == "__main__":
    unittest.main()
