import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

CELL_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # nbformat 4.5 schema
HAS_HARNESS = all(importlib.util.find_spec(m) for m in ("adk_submission", "swegemma", "yaml"))


def package(src: Path, out: Path, model: str) -> Path:
    """Build and zip a package the way `make package` does (same helper as test_models.py)."""
    subprocess.run([sys.executable, "-m", "lab.export.package", "--src", str(src), "--model", model,
                    "--build", str(out / "submission"), "--zip", str(out / "submission.zip")],
                   check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
    return out / "submission.zip"


def build_submission_notebook(out: Path) -> dict:
    archive = out / "submission.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("agent.yaml", "name: a\n")
    subprocess.run([sys.executable, "-m", "lab.export.kaggle_notebook", "--zip", str(archive), "--out", str(out)],
                   check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
    return json.loads((out / "submission.ipynb").read_text())


class NotebookCellIdsTest(unittest.TestCase):
    def test_every_cell_has_a_unique_valid_id(self):
        with tempfile.TemporaryDirectory() as d:
            nb = build_submission_notebook(Path(d))
        ids = [c.get("id") for c in nb["cells"]]
        self.assertTrue(all(i and CELL_ID.match(i) for i in ids), ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual((nb["nbformat"], nb["nbformat_minor"]), (4, 5))

    def test_ids_are_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = build_submission_notebook(Path(a)), build_submission_notebook(Path(b))
        self.assertEqual([c["id"] for c in first["cells"]], [c["id"] for c in second["cells"]])


class EvalNotebookArmsTest(unittest.TestCase):
    def test_arms_are_embedded_verified_and_run_in_order(self):
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            zips = {}
            for name in ("b2", "b1"):
                zips[name] = Path(d, f"{name}.zip")
                with zipfile.ZipFile(zips[name], "w") as zf:
                    zf.writestr("agent.yaml", f"name: {name}\n")
            out = Path(d, "nb")
            subprocess.run([sys.executable, "-m", "lab.export.kaggle_eval_notebook", "--out", str(out),
                            "--arms", ",".join(f"{n}={p}" for n, p in zips.items()), "--task-ids", "rich_1,rich_2"],
                           check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
            nb = json.loads((out / "eval.ipynb").read_text())
            sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
            for src in sources:
                compile(src, "cell", "exec")
            materialize, run = sources[1], sources[3]
            order = [materialize.index(hashlib.sha256(p.read_bytes()).hexdigest()) for p in zips.values()]
            self.assertEqual(order, sorted(order))
            self.assertIn("RESULTS / name", run)
            self.assertIn("f'-{name}'", run)


FAKE_RUN_EVAL = '''import argparse, json, pathlib
ap = argparse.ArgumentParser()
for a in ("--submission", "--tasks", "--snapshots", "--graphs", "--embeddings", "--wheels", "--results", "--task-ids",
          "--sandbox", "--api-base", "--served-model", "--model-prefix", "--label", "--hardware"):
    ap.add_argument(a)
args = ap.parse_args()
out = pathlib.Path(args.results); (out / "traces").mkdir(parents=True); (out / "patches").mkdir()
(out / "submission").mkdir()
(out / "submission" / "agent.yaml").write_text("x")
task = args.task_ids
(out / "task_results.jsonl").write_text(json.dumps({"instance_id": task, "resolved": task == "rich_1"}) + "\\n")
(out / "traces" / f"trace_{task}.json").write_text("{}")
(out / "patches" / f"{task}.patch").write_text("diff")
(out / "run_meta.json").write_text(json.dumps({"label": args.label, "task_ids": [task], "wall_seconds": 1.0}))
'''


class EvalNotebookInterleaveTest(unittest.TestCase):
    def _build(self, d: Path) -> list[str]:
        zips = {}
        for name in ("bf", "h24", "h24-long"):
            zips[name] = d / f"{name}.zip"
            with zipfile.ZipFile(zips[name], "w") as zf:
                zf.writestr("agent.yaml", f"name: {name}\n")
        subprocess.run([sys.executable, "-m", "lab.export.kaggle_eval_notebook", "--out", str(d / "nb"), "--interleave",
                        "--arms", ",".join(f"{n}={p}" for n, p in zips.items()), "--task-ids", "rich_1,rich_2,rich_3"],
                       check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
        nb = json.loads((d / "nb" / "eval.ipynb").read_text())
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def test_server_waits_40_minutes_to_start(self):
        with tempfile.TemporaryDirectory() as d:
            text = "\n".join(self._build(Path(d)))
        self.assertIn("VLLM_ENGINE_READY_TIMEOUT_S'] = '2400'", text)
        self.assertIn("startup_timeout=60 * 40", text)

    def _run(self, d: Path, sources: list[str], budget_min: float, round_min: float) -> Path:
        """Execute the merge and run cells against a fake runner, as the notebook would after serving."""
        work = d / "work"
        work.mkdir()
        (work / "run_eval.py").write_text(FAKE_RUN_EVAL)
        server = type("S", (), {"base_url": "http://x", "stop": lambda self: None})()
        ns = {"WORK": work, "subs": {n: work / "submissions" / n for n in ("bf", "h24", "h24-long")},
              "TASK_IDS": ["rich_1", "rich_2", "rich_3"], "BUDGET_MIN": budget_min, "ROUND_MIN": round_min,
              "server": server, "SERVED_MODEL": "m", "gpus": 4, "Path": Path}
        for src in sources[-2:]:
            exec(compile(src, "cell", "exec"), ns)
        return work

    def test_every_arm_gets_every_task_in_the_sequential_layout(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sources = self._build(d)
            self.assertEqual(len(sources), 5)
            work = self._run(d, sources, budget_min=600, round_min=30)
            done = json.loads((work / "progress.json").read_text())["done"]
            self.assertEqual([x["arm"] for x in done[:3]], ["bf", "h24", "h24-long"])
            self.assertEqual([x["arm"] for x in done[3:6]], ["h24", "h24-long", "bf"])  # rotated per task
            for arm in ("bf", "h24", "h24-long"):
                res = work / "results" / arm
                rows = [json.loads(l) for l in (res / "task_results.jsonl").read_text().splitlines()]
                self.assertEqual([r["instance_id"] for r in rows], ["rich_1", "rich_2", "rich_3"])
                self.assertEqual(json.loads((res / "run_meta.json").read_text())["task_ids"], ["rich_1", "rich_2", "rich_3"])
                self.assertEqual(json.loads((res / "summary.json").read_text())["resolved"], 1)
                self.assertEqual(len(list((res / "traces").glob("trace_*.json"))), 3)
                self.assertFalse((res / "submission").exists())
            import tarfile
            names = tarfile.open(work / "results.tar.gz").getnames()
            self.assertIn("results/h24-long/task_results.jsonl", names)
            self.assertFalse([n for n in names if "_parts" in n or "/submission" in n])

    def test_budget_stops_between_tasks_never_inside_one(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            work = self._run(d, self._build(d), budget_min=10, round_min=30)
            self.assertEqual(json.loads((work / "progress.json").read_text())["done"], [])

    def test_interleave_refuses_a_global_cap(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-m", "lab.export.kaggle_eval_notebook", "--out", str(Path(d, "nb")),
                                "--interleave", "--max-minutes", "5", "--task-ids", "rich_1"],
                               capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
            self.assertNotEqual(r.returncode, 0)


class NotebookModelTest(unittest.TestCase):
    """--model swaps the checkpoint everywhere and leaves no unsubstituted placeholder."""

    PLACEHOLDER = re.compile(r"__[A-Z_]+__")
    MODEL_31 = "gemma-4-31b-it-qat-w4a16-ct"
    MODEL_12 = "gemma-4-12b-it-qat-w4a16-ct"

    @staticmethod
    def _stub(d: str) -> Path:
        pkg = Path(d, "top-mix-1k", "submission.zip")
        pkg.parent.mkdir()
        with zipfile.ZipFile(pkg, "w") as zf:
            zf.writestr("agent.yaml", "model: gemma-4-31b-it-qat-w4a16-ct\n")
        return pkg

    def _notebook(self, module: str, out: Path, *extra: str) -> dict:
        subprocess.run([sys.executable, "-m", module, "--out", str(out), *extra],
                       check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
        return json.loads(next(out.glob("*.ipynb")).read_text())

    def _check(self, nb: dict):
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        joined = "\n".join(sources)
        self.assertIsNone(self.PLACEHOLDER.search(joined), self.PLACEHOLDER.findall(joined))
        self.assertIn(self.MODEL_12, joined)
        self.assertNotIn(self.MODEL_31, joined)
        for src in sources:
            compile(src, "cell", "exec")

    def test_eval_and_lora_notebooks_take_the_12b(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pkg = self._stub(d)
            modules = {
                "lab.export.kaggle_eval_notebook": ("--arms", f"main={pkg}", "--task-ids", "rich_1"),
                "lab.export.kaggle_lora_effect_notebook": ("--package", str(pkg)),
                "lab.export.kaggle_lora_rehearsal_notebook": ("--package", str(pkg)),
                "lab.export.kaggle_canary2_notebook": (),
            }
            for module, extra in modules.items():
                nb = self._notebook(module, d / module.rsplit(".", 1)[1], "--model", "12b", *extra)
                self._check(nb)

    def test_only_the_12b_serves_with_hf_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pkg = self._stub(d)
            args = ("--arms", f"main={pkg}", "--task-ids", "rich_1")
            nb12 = self._notebook("lab.export.kaggle_eval_notebook", d / "nb12", "--model", "12b", *args)
            nb31 = self._notebook("lab.export.kaggle_eval_notebook", d / "nb31", *args)
            src = lambda nb: "\n".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn("'--hf-overrides'", src(nb12))
            self.assertIn("Gemma4ForCausalLM", src(nb12))
            self.assertNotIn("--hf-overrides", src(nb31))


class G1NotebookTest(unittest.TestCase):
    """The G1 notebook embeds its three arms, the runner, the patch and both models, and every cell compiles."""

    PLACEHOLDER = re.compile(r"__[A-Z_]+__")

    @staticmethod
    def _stub(d: str, name: str) -> Path:
        pkg = Path(d, name, "submission.zip")
        pkg.parent.mkdir(parents=True, exist_ok=True)
        info = zipfile.ZipInfo("agent.yaml", date_time=(1980, 1, 1, 0, 0, 0))  # fixed stamp: the sha is asserted below
        with zipfile.ZipFile(pkg, "w") as zf:
            zf.writestr(info, f"name: {name}\n")
        return pkg

    def _notebook(self, d: Path, arms: str | None = None) -> tuple[dict, list[Path]]:
        packages = [self._stub(str(d), n) for n in
                    ("top-mix-1k-an-off-tb0-12b", "top-mix-1k-12b", "top-mix-1k-an-off-tb0")]
        cmd = [sys.executable, "-m", "lab.export.kaggle_g1_notebook", "--out", str(d / "nb"),
               "--a12", str(packages[0]), "--b12", str(packages[1]), "--anc", str(packages[2]),
               "--pool-tasks", "fastapi_1,rich_2", "--budget-12b-min", "5", "--budget-31b-min", "3"]
        if arms is not None:
            cmd += ["--arms", arms]
        subprocess.run(cmd, check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "lab/src")})
        return json.loads((d / "nb/g1.ipynb").read_text()), packages

    def test_g1_embeds_three_arms_and_every_cell_compiles(self):
        import hashlib
        from lab.export import models

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            nb, packages = self._notebook(d)
            sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
            joined = "\n".join(sources)
            self.assertIsNone(self.PLACEHOLDER.search(joined), self.PLACEHOLDER.findall(joined))
            for src in sources:
                compile(src, "cell", "exec")
            for pkg in packages:
                self.assertIn(hashlib.sha256(pkg.read_bytes()).hexdigest(), joined)
            self.assertIn(hashlib.sha256((ROOT / "lab/src/lab/eval/run_eval.py").read_bytes()).hexdigest(), joined)
            self.assertIn(hashlib.sha256((ROOT / "lab/src/lab/export/vllm_patch.py").read_bytes()).hexdigest(), joined)
            self.assertIn("gemma-4-12b-it-qat-w4a16-ct", joined)
            self.assertIn("gemma-4-31b-it-qat-w4a16-ct", joined)
            self.assertIn("'--hf-overrides'", joined)
            self.assertIn("vision_embedder", joined)
            self.assertIn("run_analyzer", joined)
            # The kernel's analyzer_config and the module-level one must stay identical (one is tested, one runs).
            import inspect
            from lab.export.kaggle_g1_notebook import analyzer_config
            self.assertIn(inspect.getsource(analyzer_config).rstrip(), joined)
            meta = json.loads((d / "nb/kernel-metadata.json").read_text())
            self.assertEqual(sorted(meta["model_sources"]),
                             [models.source("12b"), models.source("31b")])

    def test_g1_reads_only_from_attached_sources(self):
        # v1 died after loading the 12B: the parity cell read plans.json from the competition data, where it is not.
        from lab.export.kaggle_lora_rehearsal_notebook import LORA_DATA

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            nb, _ = self._notebook(d)
            joined = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
            meta = json.loads((d / "nb/kernel-metadata.json").read_text())
        competition_files = {"tasks.jsonl", "snapshots", "graphs", "embeddings", "wheels"}
        self.assertLessEqual(set(re.findall(r"DATA_DIR / '([^']+)'", joined)), competition_files)
        self.assertNotIn("coder_turns(DATA_DIR)", joined)
        self.assertIn("coder_turns(TRACE_DIR)", joined)
        self.assertIn(LORA_DATA, meta["dataset_sources"])

    def test_parity_helpers_detect_a_difference(self):
        from lab.export.kaggle_g1_notebook import first_difference, parity_verdict
        self.assertIsNone(first_difference([1, 2, 3], [1, 2, 3]))
        self.assertEqual(first_difference([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(first_difference([1, 2], [1, 2, 3]), 2)
        self.assertEqual(first_difference([1, 2, 3], [1, 2]), 2)
        equal = {"prompt": {"equal": True}, "full": {"equal": True}}
        other = {"prompt": {"equal": True}, "full": {"equal": False}}
        self.assertEqual(parity_verdict([equal, equal]), "parity")
        self.assertEqual(parity_verdict([equal, other]), "difference")
        self.assertEqual(parity_verdict([{"error": "x"}]), "not measured")

    def test_anc_only_skips_the_12b_cells_and_seeds_the_summary_state(self):
        # With `--arms ANC` the 12B cells (load, parity, analyzer) are absent, so SUMMARY must read state seeded by a
        # cell that always runs, and LOAD31B must not call boot_server/litellm defined in the skipped LOAD12B (O10).
        with tempfile.TemporaryDirectory() as d:
            nb, _ = self._notebook(Path(d), arms="ANC")
        sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        joined = "\n".join(sources)
        for src in sources:
            compile(src, "cell", "exec")
        for gone in ("boot_server(SERVED_12B", "run_analyzer('12b'", "def first_difference", "coder_turns(TRACE_DIR)"):
            self.assertNotIn(gone, joined)
        summary_at = next(i for i, s in enumerate(sources) if "'g1.json'" in s)
        seeded_at = next(i for i, s in enumerate(sources) if re.search(r"(?m)^done12b, done31b = ", s))
        self.assertLess(seeded_at, summary_at)
        # Every 12B-only name SUMMARY reads is bound before SUMMARY by the always-run seeding cell.
        for name in ("done12b", "ANALYZER_12B", "PARITY_VERDICT", "PATCH_APPLIED", "DEADLINE_12B", "server"):
            self.assertTrue(any(re.search(rf"(?m)(?:^|,)\s*{name}\b[^=\n]*=", s) for s in sources[:summary_at]),
                            name)


class VllmPatchTest(unittest.TestCase):
    """The vision_embedder patch targets Gemma4ForCausalLM.load_weights and fails loudly if the anchor moves."""

    SOURCE = (
        "class Gemma4Model:\n"
        "    def load_weights(self, weights):\n"
        "        skip = [\n"
        '            "vision_tower.",\n'
        "        ]\n"
        "        return weights\n"
        "\n"
        "class Gemma4ForCausalLM:\n"
        "    def load_weights(self, weights):\n"
        "        skip = [\n"
        '            "audio_tower.",\n'
        '            "vision_tower.",\n'
        '            "embed_vision.",\n'
        "        ]\n"
        "        return skip, weights\n"
    )

    def test_inserts_into_the_causal_lm_list_only(self):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "lab/src"))
        from lab.export import vllm_patch

        out = vllm_patch.patch_source(self.SOURCE)
        self.assertEqual(out.count('"vision_embedder.",'), 1)
        self.assertEqual(vllm_patch.patch_source(out), out)  # idempotent
        causal = out.split("class Gemma4ForCausalLM", 1)[1]
        self.assertIn('"vision_embedder.",', causal)
        self.assertNotIn('"vision_embedder.",', out.split("class Gemma4ForCausalLM", 1)[0])
        compile(out, "patched", "exec")

    def test_fails_loudly_when_the_anchor_moved(self):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "lab/src"))
        from lab.export import vllm_patch

        for bad in ("", "class Other:\n    pass\n", "class Gemma4ForCausalLM:\n    def load_weights(self, weights):\n        return weights\n"):
            with self.assertRaises(vllm_patch.PatchError):
                vllm_patch.patch_source(bad)


class LoraNotebookPackageTest(unittest.TestCase):
    """The lora notebooks take --package and derive the arm names and the embedded sha from it."""

    @staticmethod
    def _package(d: str, name: str) -> Path:
        pkg = Path(d, name, "submission.zip")
        pkg.parent.mkdir()
        with zipfile.ZipFile(pkg, "w") as zf:
            zf.writestr("agent.yaml", "name: a\n")
        return pkg

    @staticmethod
    def _run(module: str, out: Path, *extra: str, env: dict | None = None) -> dict:
        subprocess.run([sys.executable, "-m", module, "--out", str(out), *extra],
                       check=True, capture_output=True, env=env or {"PYTHONPATH": str(ROOT / "lab/src")})
        return json.loads(next(out.glob("*.ipynb")).read_text())

    def _sources(self, nb: dict) -> list[str]:
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def test_rehearsal_31b_arms_and_sha_come_from_the_package(self):
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            pkg = self._package(d, "top-mix-9k-an-off")
            nb = self._run("lab.export.kaggle_lora_rehearsal_notebook", Path(d, "nb"),
                           "--package", str(pkg), "--model", "31b")
            sources = self._sources(nb)
            for src in sources:
                compile(src, "cell", "exec")
            joined = "\n".join(sources)
            self.assertIn(hashlib.sha256(pkg.read_bytes()).hexdigest(), joined)
            self.assertIn("'top-mix-9k-an-off-lora'", sources[-1])
            self.assertIn("'top-mix-9k-an-off'", sources[-1])
            self.assertNotIn("__ARM__", joined)

    def test_effect_arms_and_sha_come_from_the_package(self):
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            pkg = self._package(d, "top-mix-9k-an-off")
            nb = self._run("lab.export.kaggle_lora_effect_notebook", Path(d, "nb"), "--package", str(pkg))
            sources = self._sources(nb)
            for src in sources:
                compile(src, "cell", "exec")
            joined = "\n".join(sources)
            self.assertIn(hashlib.sha256(pkg.read_bytes()).hexdigest(), joined)
            arms = next(src for src in sources if "ARMS = {" in src)
            self.assertIn("'top-mix-9k-an-off-lora'", arms)
            self.assertIn("'top-mix-9k-an-off'", arms)
            self.assertNotIn("__ARM__", joined)


class G1NotebookIncludeResolutionTest(unittest.TestCase):
    """The analyzer config resolves `!include` the way the harness does, on the real A12/B12/ANC packages."""

    def test_analyzer_config_from_real_packages_resolves_includes(self):
        if not HAS_HARNESS:
            self.skipTest("needs adk_submission, swegemma and yaml")
        sys.path.insert(0, str(ROOT / "lab/src"))
        from lab.export import kaggle_g1_notebook
        from lab.export.variants import build_variant
        from adk_submission import yaml_loader

        # (arm, variant source, model) as the three packages were built for the G1 session.
        specs = [
            ('A12', 'top-mix-1k-an-off-tb0', '12b'),
            ('B12', 'top-mix-1k', '12b'),
            ('ANC', 'top-mix-1k-an-off-tb0', '31b'),
        ]
        expected = {
            'A12': (False, None),      # analyzer: thinking_budget 0 -> off
            'B12': (True, 1024),       # analyzer: thinking on, budget 1024
            'ANC': (False, None),      # analyzer off, same as A12
        }

        with tempfile.TemporaryDirectory() as d:
            # resolve(): the harness loader rejects a path with a symlink component, and macOS' /var is one
            d = Path(d).resolve()
            for name, variant, model in specs:
                src = build_variant(variant, dest_root=d / f"variants-{variant}-{model}")
                zip_path = package(src, d / name, model)
                extract = d / f"extract-{name}"
                extract.mkdir()
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(extract)

                y = yaml_loader.load_yaml(extract / 'sub_agents' / 'code_analyzer.yaml', root_dir=extract)
                self.assertIn('instruction', y)
                self.assertIn('generate_content_config', y)
                # The include resolved to the file verbatim, not left as a marker.
                self.assertEqual(y['instruction'], (extract / 'prompts' / 'analyzer.md').read_text())
                self.assertNotIn('!include', y['instruction'])

                cfg = kaggle_g1_notebook.analyzer_config(extract)
                self.assertIn('system', cfg)
                self.assertIn('Read-only code navigator', cfg['system'])
                # The system prompt the harness sends is the instruction plus its identity footer, nothing else.
                footer = (f'You are an agent. Your internal name is "{y["name"]}". '
                          f'The description about you is "{y["description"]}".')
                self.assertEqual(cfg['system'], y['instruction'] + '\n\n' + footer)
                self.assertEqual((cfg['enable_thinking'], cfg['thinking_token_budget']), expected[name], name)

    def test_analyzer_config_resolves_includes_on_a_synthetic_package(self):
        """Runs everywhere: a hand-made package proves the include resolution and is the control for the old way."""
        if not HAS_HARNESS:
            self.skipTest("needs adk_submission, swegemma and yaml")
        sys.path.insert(0, str(ROOT / "lab/src"))
        import yaml
        from lab.export import kaggle_g1_notebook

        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d).resolve() / "pkg"  # resolve(): no symlink component for the harness loader
            (pkg / "prompts").mkdir(parents=True)
            (pkg / "configs").mkdir()
            (pkg / "prompts" / "analyzer.md").write_text("SYNTHETIC ANALYZER PROMPT\n")
            (pkg / "configs" / "sampling.yaml").write_text(
                "temperature: 0.1\ntop_p: 0.9\nmax_output_tokens: 4096\n"
                "thinking_config:\n  thinking_budget: 0\n  include_thoughts: false\n")
            (pkg / "sub_agents").mkdir()
            (pkg / "sub_agents" / "code_analyzer.yaml").write_text(
                "name: code_analyzer\nmodel: m\ndescription: d\n"
                "instruction: !include ../prompts/analyzer.md\n"
                "generate_content_config: !include ../configs/sampling.yaml\n")

            cfg = kaggle_g1_notebook.analyzer_config(pkg)
            self.assertIn("SYNTHETIC ANALYZER PROMPT", cfg['system'])
            self.assertEqual(cfg['max_tokens'], 4096)
            self.assertEqual((cfg['enable_thinking'], cfg['thinking_token_budget']), (False, None))

            # Control: the old approach (plain yaml.safe_load) does not understand the `!include` tag at all.
            with self.assertRaises(yaml.constructor.ConstructorError):
                yaml.safe_load((pkg / "sub_agents" / "code_analyzer.yaml").read_text())


if __name__ == "__main__":
    unittest.main()


class TpuSmokeKernelTest(unittest.TestCase):
    def test_kernel_embeds_the_sources_verbatim_and_mounts_its_inputs(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-m", "lab.export.kaggle_tpu_smoke", "--out", d], check=True,
                           env={"PYTHONPATH": str(root / "lab/src"), "PATH": "/usr/bin:/bin"}, capture_output=True)
            script = (Path(d) / "gemma4-tpu-smoke.py").read_text()
            meta = json.loads((Path(d) / "kernel-metadata.json").read_text())
        compile(script, "kernel", "exec")
        sources = eval(re.search(r"^SOURCES = (\{.*\})$", script, re.M).group(1))  # noqa: S307 (our own repr)
        for name in ("train.py", "tpu_smoke.py"):
            self.assertEqual(sources[name], (root / "lab/src/lab/lora" / name).read_text(), name)
        self.assertTrue(meta["enable_tpu"])
        self.assertFalse(meta["enable_gpu"])
        self.assertEqual(meta["model_sources"], ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"])
        self.assertEqual(sorted(meta["dataset_sources"]), ["gorigamidev/gemma4-lora-data", "gorigamidev/gemma4-lora-wheels"])
        self.assertIn("31b-it-qat-q4_0-unquantized", script)


class TpuServeProbeKernelTest(unittest.TestCase):
    def test_kernel_embeds_the_zip_and_runner_with_their_hashes(self):
        import base64
        import hashlib
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "submission.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("agent.yaml", "name: swe_agent\n")
            subprocess.run([sys.executable, "-m", "lab.export.kaggle_tpu_serve_probe", "--zip", str(zpath), "--out", d],
                           check=True, env={"PYTHONPATH": str(root / "lab/src"), "PATH": "/usr/bin:/bin"},
                           capture_output=True)
            script = (Path(d) / "gemma4-tpu-serve-probe.py").read_text()
            meta = json.loads((Path(d) / "kernel-metadata.json").read_text())
            zip_bytes = zpath.read_bytes()
        compile(script, "kernel", "exec")
        runner = (root / "lab/src/lab/eval/run_eval.py").read_bytes()
        self.assertIn(repr(base64.b64encode(runner).decode()), script)
        self.assertIn(repr(hashlib.sha256(runner).hexdigest()), script)
        self.assertIn(repr(hashlib.sha256(zip_bytes).hexdigest()), script)
        self.assertTrue(meta["enable_tpu"] and meta["enable_internet"])
        self.assertFalse(meta["enable_gpu"])
        self.assertEqual(meta["model_sources"], ["google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"])

    def test_arms_are_embedded_and_interleaved(self):
        import hashlib
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            zips = {}
            for name in ("a", "b"):
                zips[name] = Path(d) / f"{name}.zip"
                with zipfile.ZipFile(zips[name], "w") as z:
                    z.writestr("agent.yaml", f"name: {name}\n")
            subprocess.run([sys.executable, "-m", "lab.export.kaggle_tpu_serve_probe", "--slug", "x-pool",
                            "--arms", ",".join(f"{n}={p}" for n, p in zips.items()), "--task-ids", "t1,t2",
                            "--budget-min", "330", "--out", d],
                           check=True, env={"PYTHONPATH": str(root / "lab/src"), "PATH": "/usr/bin:/bin"},
                           capture_output=True)
            script = (Path(d) / "x-pool.py").read_text()
            meta = json.loads((Path(d) / "kernel-metadata.json").read_text())
            shas = [hashlib.sha256(p.read_bytes()).hexdigest() for p in zips.values()]
        compile(script, "kernel", "exec")
        self.assertEqual(meta["id"], "gorigamidev/x-pool")
        self.assertIn("TASKS = ['t1', 't2']", script)
        self.assertIn("BUDGET_MIN, ROUND_MIN = 330, 15", script)
        self.assertIn("def merge_part(", script)
        self.assertNotIn("# __MERGE__", script)
        self.assertIn("HARNESS_PYTHON, GOLD_TASKS = '3.13', ['rich_3006', 'rich_3521']", script)
        self.assertIn("_deduplicate_wheels", script)  # the task wheels go into `harness`
        self.assertIn('"--python-preference", "only-managed"', script)  # system 3.13 on TPU has no ensurepip
        self.assertIn('sandbox="subprocess"', script)  # the gold check uses the agent's sandbox
        for sha in shas:
            self.assertIn(repr(sha), script)

    def test_gold_check_reports_a_crash_as_a_failed_row(self):
        root = Path(__file__).resolve().parents[2]
        stubs = {
            "adk_submission/__init__.py": "", "swegemma/__init__.py": "", "swegemma/harness/__init__.py": "",
            "adk_submission/registry.py": "class ModelRegistry: pass\n",
            "swegemma/config.py": "class EvalConfig:\n    def __init__(self, **kw): self.snapshots_dir = kw['snapshots_dir']\n"
                                  "def build_submission_limits(): return None, None\n",
            "swegemma/deduplication.py": "def resolve_task_snapshot_paths(*a): return None, None, None\n",
            "swegemma/evaluate.py": "class Evaluator:\n    def __init__(self, cfg): self.docker = None\n",
            "swegemma/harness/verification.py": "async def verify_task(*a, **kw):\n    print('noise on stdout')\n"
                                                "    raise RuntimeError('sandbox broke')\n",
            "swegemma/models.py": "class T:\n    instance_id, repo, patch = 't1', 'r', ''\n"
                                  "def load_tasks(p): return [T()]\n",
        }
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-m", "lab.export.kaggle_tpu_serve_probe", "--zip", str(self._zip(d)),
                            "--out", d], check=True, env={"PYTHONPATH": str(root / "lab/src"), "PATH": "/usr/bin:/bin"},
                           capture_output=True)
            script = (Path(d) / "gemma4-tpu-serve-probe.py").read_text()
            ns = {}
            exec(re.search(r'^GOLD_CHECK = """.*?^"""', script, re.S | re.M).group(0), ns)
            for rel, body in stubs.items():
                (Path(d) / "stubs" / rel).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / "stubs" / rel).write_text(body)
            (Path(d) / "gold_check.py").write_text(ns["GOLD_CHECK"])
            g = subprocess.run([sys.executable, str(Path(d) / "gold_check.py"), d, d, d, "t1"], capture_output=True,
                               text=True, env={"PYTHONPATH": str(Path(d) / "stubs")})
        self.assertEqual(g.returncode, 1)
        self.assertIn("GOLD [", g.stdout, g.stderr)
        line = next(x for x in g.stdout.splitlines() if x.startswith("GOLD ["))
        row = json.loads(line[5:])[0]
        self.assertFalse(row["resolved"])
        self.assertIn("sandbox broke", row["error"])

    @staticmethod
    def _zip(d):
        zpath = Path(d) / "submission.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("agent.yaml", "name: swe_agent\n")
        return zpath

    def test_harness_check_covers_every_runner_import(self):
        import ast
        from lab.export.kaggle_tpu_serve_probe import HARNESS_IMPORTS
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "lab/src/lab/eval/run_eval.py").read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module != "__future__":
                mods.add(node.module)
        third_party = {m for m in mods if m.split(".")[0] not in sys.stdlib_module_names}
        self.assertEqual(third_party, set(HARNESS_IMPORTS))
