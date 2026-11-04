import filecmp
import importlib.util
import json
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.export.variants import EXPLICIT_NO_GRAPH_LINE, GRAPH_TOOLS, PROBE_ADAPTER, REREAD_BULLET, build_variant  # noqa: E402

HAS_HARNESS = all(importlib.util.find_spec(m) for m in ("adk_submission", "swegemma", "yaml"))

SRC = ROOT / "submission"


class ThinkingOffTest(unittest.TestCase):
    def test_only_include_thoughts_changes(self):
        with tempfile.TemporaryDirectory() as d:
            for name, base in (("b2-think-off", None), ("b1-think-off", "b1-no-skill")):
                out = build_variant(name, SRC, Path(d))
                ref = build_variant(base, SRC, Path(d, "ref")) if base else SRC
                sampling = (out / "configs/sampling.yaml").read_text()
                self.assertIn("include_thoughts: false", sampling)
                self.assertEqual((ref / "configs/sampling.yaml").read_text().replace("include_thoughts: true",
                                                                                   "include_thoughts: false"), sampling)
                for same in ("eval_config.yaml", "agent.yaml", "prompts/system.md"):
                    self.assertTrue(filecmp.cmp(ref / same, out / same, shallow=False), (name, same))


class B1NoSkillTest(unittest.TestCase):
    def test_only_the_skill_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("b1-no-skill", SRC, Path(d))
            self.assertFalse((out / "skills").exists())
            for same in ("eval_config.yaml", "configs/sampling.yaml"):
                self.assertTrue(filecmp.cmp(SRC / same, out / same, shallow=False), same)
            original = (SRC / "agent.yaml").read_text().splitlines()
            variant = (out / "agent.yaml").read_text().splitlines()
            self.assertEqual([ln for ln in original if ln not in variant], ["skills:", "  - skills/mini-palace"])
            self.assertIn("generate_content_config: !include configs/sampling.yaml", variant)
            prompt = (out / "prompts/system.md").read_text()
            self.assertNotIn("mini-palace", prompt)
            self.assertNotIn("run_skill_script", prompt)
            self.assertIn("### 1. Locate the code", prompt)

    def test_unexpected_prompt_aborts_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "src")
            shutil.copytree(SRC, src)
            (src / "prompts/system.md").write_text("A prompt without the expected sections.\n")
            with self.assertRaises(SystemExit):
                build_variant("b1-no-skill", src, Path(d, "out"))
            self.assertEqual(list(Path(d, "out").iterdir()) if Path(d, "out").exists() else [], [])


class StructureVariantsTest(unittest.TestCase):
    def test_graph_tools_are_the_only_tools_removed(self):
        with tempfile.TemporaryDirectory() as d:
            ref = build_variant("b2-think-off", SRC, Path(d, "ref"))
            out = build_variant("b2-no-graph", SRC, Path(d))
            original = (ref / "agent.yaml").read_text().splitlines()
            variant = (out / "agent.yaml").read_text().splitlines()
            self.assertEqual([ln for ln in original if ln not in variant], [f"  - {t}" for t in GRAPH_TOOLS])
            self.assertTrue(filecmp.cmp(ref / "configs/sampling.yaml", out / "configs/sampling.yaml", shallow=False))
            self.assertNotIn("search_similar_code", (out / "prompts/system.md").read_text())

    def test_phases_split_tools_and_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            ref = build_variant("b2-think-off", SRC, Path(d, "ref"))
            out = build_variant("b2-phases", SRC, Path(d))
            self.assertIn("agent_class: SequentialAgent", (out / "agent.yaml").read_text())
            locator = (out / "agents/locator.yaml").read_text()
            self.assertIn("tools:\n  - read_file\n  - get_status\nskills:", locator)
            self.assertIn("output_key: plan", locator)
            ref_tools = re.search(r"^tools:\n((?:  - [^\n]*\n)+)", (ref / "agent.yaml").read_text(), re.M).group(1)
            self.assertIn(ref_tools, (out / "agents/editor.yaml").read_text())
            self.assertIn("{plan?}", (out / "prompts/edit.md").read_text())
            self.assertIn("{plan?}", (out / "prompts/locate.md").read_text())
            self.assertFalse((out / "prompts/system.md").exists())

    def test_locator_tool_keeps_the_editor_as_root(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("b2-locator-tool", SRC, Path(d))
            agent = (out / "agent.yaml").read_text()
            self.assertNotIn("agent_class", agent)
            self.assertIn("  - agent_tool:\n      config_path: agents/locator.yaml\n", agent)
            locate = (out / "prompts/locate.md").read_text()
            self.assertIn("{problem_description}", locate)
            self.assertNotIn("{plan?}", locate)
            self.assertNotIn("output_key", (out / "agents/locator.yaml").read_text())
            self.assertIn("call the `locator` tool", (out / "prompts/system.md").read_text())

    def test_b2_3_adds_only_the_reread_line(self):
        with tempfile.TemporaryDirectory() as d:
            ref = build_variant("b2-think-off", SRC, Path(d, "ref"))
            out = build_variant("b2.3", SRC, Path(d))
            for same in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                self.assertTrue(filecmp.cmp(ref / same, out / same, shallow=False), same)
            original = (ref / "prompts/system.md").read_text().splitlines()
            variant = (out / "prompts/system.md").read_text().splitlines()
            self.assertEqual([ln for ln in variant if ln not in original], [REREAD_BULLET.rstrip("\n")])
            self.assertEqual([ln for ln in original if ln not in variant], [])

    def test_b2_explicit_changes_only_the_system_prompt(self):
        for name in ("b2-explicit", "b2-explicit-short"):
            self._only_system_prompt(name)

    def _only_system_prompt(self, name):
        with tempfile.TemporaryDirectory() as d:
            ref = build_variant("b2-think-off", SRC, Path(d, "ref"))
            out = build_variant(name, SRC, Path(d))
            changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                             and not filecmp.cmp(p, ref / p.relative_to(out), shallow=False))
            self.assertEqual(changed, ["prompts/system.md"])
            prompt = (out / "prompts/system.md").read_text()
            self.assertIn('args={"action": "locate", "text": "TEXT"}', prompt)
            self.assertIn('"status": "discarded"', prompt)
            self.assertNotIn("docs_src", prompt)

    def test_b2_explicit_no_graph_is_no_graph_plus_the_explicit_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            ref = build_variant("b2-no-graph", SRC, Path(d, "ref"))
            explicit = build_variant("b2-explicit", SRC, Path(d, "explicit"))
            out = build_variant("b2-explicit-no-graph", SRC, Path(d))
            changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                             and not filecmp.cmp(p, ref / p.relative_to(out), shallow=False))
            self.assertEqual(changed, ["prompts/system.md"])
            original = (explicit / "prompts/system.md").read_text().splitlines()
            prompt = (out / "prompts/system.md").read_text().splitlines()
            self.assertEqual([ln for ln in prompt if ln not in original], [EXPLICIT_NO_GRAPH_LINE.rstrip("\n")])
            self.assertEqual(len([ln for ln in original if ln not in prompt]), 1)
            self.assertNotIn("search_similar_code", "\n".join(prompt))

    def test_top_check_adds_only_the_skill_and_one_prompt_line(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("top-check", dest_root=Path(d))
            top = ROOT / "external/top-bundle"
            changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                             and not p.relative_to(out).parts[0] == "skills"
                             and not filecmp.cmp(p, top / p.relative_to(out), shallow=False))
            self.assertEqual(changed, ["agent.yaml", "prompts/system.md"])
            self.assertFalse((out / "eval_config.yaml").exists())
            original = (top / "prompts/system.md").read_text().splitlines()
            added = [ln for ln in (out / "prompts/system.md").read_text().splitlines() if ln not in original]
            self.assertEqual(len(added), 1)
            self.assertIn('{"action": "check"}', added[0])
            self.assertTrue((out / "skills/mini-palace/scripts/check.py").exists())
            self.assertIn('"check": check.action_check', (out / "skills/mini-palace/scripts/entrypoint.py").read_text())

    def test_top_is_the_public_bundle_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("top", dest_root=Path(d))
            top = ROOT / "external/top-bundle"
            files = sorted(str(p.relative_to(top)) for p in top.rglob("*") if p.is_file())
            self.assertEqual(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()), files)
            for f in files:
                self.assertTrue(filecmp.cmp(top / f, out / f, shallow=False), f)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "needs PyYAML")
    def test_top_mix_changes_thinking_stop_step_and_budgets(self):
        import yaml
        top = ROOT / "external/top-bundle"
        for name, budget in (("top-mix-1k", 1024), ("top-mix-2k", 2048)):
            with tempfile.TemporaryDirectory() as d:
                out = build_variant(name, dest_root=Path(d))
                changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                                 and not ((top / p.relative_to(out)).exists()
                                          and filecmp.cmp(p, top / p.relative_to(out), shallow=False)))
                self.assertEqual(changed, ["configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"])
                sampling = yaml.safe_load((out / "configs/sampling.yaml").read_text())["thinking_config"]
                self.assertEqual(sampling, {"thinking_budget": budget, "include_thoughts": True})
                ev = yaml.safe_load((out / "eval_config.yaml").read_text())["evaluation"]
                self.assertEqual(ev, {"max_time_minutes": 4.5, "timeout_seconds": 300, "max_tool_calls": 100, "max_turns": 150})
                original = (top / "prompts/system.md").read_text().splitlines()
                new = (out / "prompts/system.md").read_text().splitlines()
                self.assertEqual(len(new), len(original))
                (added,) = [ln for ln in new if ln not in original]
                self.assertTrue(added.startswith("6. **Submit**:") and "DONE" in added and "git restore" in added)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "needs PyYAML")
    def test_top_mix_analyzer_off_changes_only_the_analyzer_sampling(self):
        import yaml
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            mix = build_variant("top-mix-1k", dest_root=Path(d1))
            off = build_variant("top-mix-1k-an-off", dest_root=Path(d2))
            changed = sorted(str(p.relative_to(off)) for p in off.rglob("*") if p.is_file()
                             and not ((mix / p.relative_to(off)).exists()
                                      and filecmp.cmp(p, mix / p.relative_to(off), shallow=False)))
            self.assertEqual(changed, ["configs/sampling_analyzer.yaml", "sub_agents/code_analyzer.yaml"])
            an = yaml.safe_load((off / "configs/sampling_analyzer.yaml").read_text())["thinking_config"]
            self.assertEqual(an["include_thoughts"], False)
            self.assertIn("!include ../configs/sampling_analyzer.yaml", (off / "sub_agents/code_analyzer.yaml").read_text())
            self.assertTrue(filecmp.cmp(off / "configs/sampling.yaml", mix / "configs/sampling.yaml", shallow=False))

    def test_top_think_changes_only_include_thoughts(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("top-think", dest_root=Path(d))
            top = ROOT / "external/top-bundle"
            changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                             and not filecmp.cmp(p, top / p.relative_to(out), shallow=False))
            self.assertEqual(changed, ["configs/sampling.yaml"])
            original = (top / "configs/sampling.yaml").read_text().splitlines()
            new = (out / "configs/sampling.yaml").read_text().splitlines()
            self.assertEqual([ln for ln in new if ln not in original], ["  include_thoughts: true"])

    def test_top_directed_changes_only_the_analyzer(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("top-directed", dest_root=Path(d))
            top = ROOT / "external/top-bundle"
            changed = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                             and p.relative_to(out).parts[0] != "skills"
                             and (not (top / p.relative_to(out)).exists()
                                  or not filecmp.cmp(p, top / p.relative_to(out), shallow=False)))
            self.assertEqual(changed, ["agent.yaml", "prompts/analyzer.md", "prompts/system.md",
                                       "sub_agents/code_analyzer.yaml"])
            self.assertIn("skip_summarization: false", (out / "agent.yaml").read_text())
            original = (top / "prompts/system.md").read_text().splitlines()
            added = [ln for ln in (out / "prompts/system.md").read_text().splitlines() if ln not in original]
            self.assertEqual(len(added), 1)
            self.assertIn("exactly once", added[0])
            analyzer = (out / "sub_agents/code_analyzer.yaml").read_text()
            self.assertIn("include_contents: none", analyzer)
            self.assertNotIn("run_command", analyzer)
            self.assertIn("REASONING:", (out / "prompts/analyzer.md").read_text())

    def test_check_variants_document_no_changes(self):
        # The top-check preflight coder read an empty check as a green light and submitted an empty patch.
        with tempfile.TemporaryDirectory() as d:
            for name, src in (("top-check", None), ("b2-check", SRC)):
                out = build_variant(name, src, Path(d))
                self.assertIn("no_changes", (out / "skills/mini-palace/SKILL.md").read_text(), name)

    def test_probe_adapter_is_valid_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("b2-phases-adapter", SRC, Path(d))
            blob = (out / "adapters" / PROBE_ADAPTER / "adapter_model.safetensors").read_bytes()
            size = struct.unpack("<Q", blob[:8])[0]
            header = json.loads(blob[8:8 + size])
            self.assertEqual(len(blob), 8 + size + max(v["data_offsets"][1] for v in header.values()))
            self.assertIn(f"adapter: {PROBE_ADAPTER}", (out / "agents/editor.yaml").read_text())
            self.assertNotIn("adapter:", (out / "agents/locator.yaml").read_text())


@unittest.skipUnless(HAS_HARNESS, "needs adk_submission and swegemma")
class CompiledRoutingTest(unittest.TestCase):
    def test_adapter_routes_only_the_editor_and_keeps_thinking_off(self):
        from adk_submission import compile_submission, discover_adapters
        from swegemma.config import build_submission_limits
        from swegemma.models.registry import resolve_swegemma_adapter

        from lab.eval.run_eval import model_registry
        from lab.export.package import HARNESS_TOOLS

        limits, gen = build_submission_limits()
        models = model_registry("http://127.0.0.1:1/v1", "base", "openai/", "EMPTY")
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("b2-phases-adapter", SRC, Path(d))
            self.assertEqual(list(discover_adapters(out, limits.adapter_extensions).adapters), [PROBE_ADAPTER])
            agent = compile_submission(submission_dir=out, tool_registry={n: (lambda **_: "") for n in HARNESS_TOOLS},
                                       model_registry=models, limits=limits, generation_constraints=gen,
                                       adapter_resolver_fn=lambda b, i: resolve_swegemma_adapter(b, i, models))
        seen = {a.name: a.model for a in agent.sub_agents}
        self.assertEqual(seen["locator"].model, "openai/base")
        self.assertEqual(seen["editor"].model, f"openai/{PROBE_ADAPTER}")
        for model in seen.values():
            self.assertIs(model._additional_args["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()


class BudgetFitTest(unittest.TestCase):
    def test_bf_is_the_public_bundle_byte_for_byte(self):
        src = ROOT / "external/budget-fit"
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("bf", dest_root=Path(d))
            files = sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())
            self.assertEqual(files, ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"])
            self.assertEqual(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()), files)
            for f in files:
                self.assertTrue(filecmp.cmp(src / f, out / f, shallow=False), f)
            self.assertIn("max_tool_calls: 28", (out / "eval_config.yaml").read_text())


class BfHabitsTest(unittest.TestCase):
    def test_bf_habits_changes_only_two_prompt_rules(self):
        src = ROOT / "external/budget-fit"
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("bf-habits", dest_root=Path(d))
            for f in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                self.assertTrue(filecmp.cmp(src / f, out / f, shallow=False), f)
            before = (src / "prompts/system.md").read_text().splitlines()
            after = (out / "prompts/system.md").read_text().splitlines()
            added = [ln for ln in after if ln not in before]
            removed = [ln for ln in before if ln not in after]
            self.assertEqual(len(added), 5)
            self.assertTrue(added[0].startswith("- If `edit_file` returns an error"))
            self.assertIn("with `rm`", added[1])
            self.assertEqual(added[2], "  2. Call `submit_patch`.")
            self.assertEqual(removed, ["  1. Call `submit_patch` immediately.",
                                       "  2. Verify `patch_size > 0` and `files_changed > 0`.",
                                       "  3. Output a short summary of the fix to end the session."])
            self.assertIn("max_output_tokens: 8192", (out / "configs/sampling.yaml").read_text())


class BfGovTest(unittest.TestCase):
    def test_bf_gov_adds_two_rules_and_keeps_the_submission_steps(self):
        src = ROOT / "external/budget-fit"
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("bf-gov", dest_root=Path(d))
            for f in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                self.assertTrue(filecmp.cmp(src / f, out / f, shallow=False), f)
            before = (src / "prompts/system.md").read_text().splitlines()
            after = (out / "prompts/system.md").read_text().splitlines()
            self.assertEqual([ln for ln in before if ln not in after], [])
            added = [ln for ln in after if ln not in before]
            self.assertEqual(len(added), 2)
            self.assertTrue(added[0].startswith("- If `edit_file` returns an error"))
            self.assertTrue(added[1].startswith("- Never create scratch or reproduction files"))
            self.assertNotIn("/tmp", added[1])
            self.assertIn("  1. Call `submit_patch` immediately.", after)


class BfGovToTest(unittest.TestCase):
    def test_bf_gov_to_adds_only_the_timeout_rule_to_bf_gov(self):
        with tempfile.TemporaryDirectory() as d:
            gov = build_variant("bf-gov", dest_root=Path(d))
            to = build_variant("bf-gov-to", dest_root=Path(d))
            for f in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                self.assertTrue(filecmp.cmp(gov / f, to / f, shallow=False), f)
            before = (gov / "prompts/system.md").read_text().splitlines()
            after = (to / "prompts/system.md").read_text().splitlines()
            self.assertEqual([ln for ln in before if ln not in after], [])
            added = [ln for ln in after if ln not in before]
            self.assertEqual(len(added), 1)
            self.assertIn("`timeout 90`", added[0])
            self.assertEqual(after.index(added[0]) - 1,
                             next(i for i, ln in enumerate(after) if ln.startswith("- Never create scratch")))


class BfGovTbTest(unittest.TestCase):
    def test_bf_gov_tb_only_halves_the_thinking_budget_of_bf_gov_to(self):
        with tempfile.TemporaryDirectory() as d:
            to = build_variant("bf-gov-to", dest_root=Path(d))
            tb = build_variant("bf-gov-tb", dest_root=Path(d))
            for f in ("agent.yaml", "eval_config.yaml", "prompts/system.md"):
                self.assertTrue(filecmp.cmp(to / f, tb / f, shallow=False), f)
            before = (to / "configs/sampling.yaml").read_text().splitlines()
            after = (tb / "configs/sampling.yaml").read_text().splitlines()
            self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                             [("  thinking_budget: 4096", "  thinking_budget: 2048")])
            self.assertEqual(len(before), len(after))


class BfGovEdTest(unittest.TestCase):
    def test_bf_gov_ed_changes_only_the_budget_step_1_and_adds_the_locator(self):
        with tempfile.TemporaryDirectory() as d:
            to = build_variant("bf-gov-to", dest_root=Path(d))
            ed = build_variant("bf-gov-ed", dest_root=Path(d))
            self.assertTrue(filecmp.cmp(to / "configs/sampling.yaml", ed / "configs/sampling.yaml", shallow=False))
            before = (to / "eval_config.yaml").read_text().splitlines()
            after = (ed / "eval_config.yaml").read_text().splitlines()
            self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                             [("  max_tool_calls: 28", "  max_tool_calls: 20")])
            agent_to, agent_ed = (to / "agent.yaml").read_text(), (ed / "agent.yaml").read_text()
            self.assertEqual(agent_ed.replace("  - agent_tool:\n      config_path: agents/locator.yaml\n"
                                              "      skip_summarization: false\n", ""), agent_to)
            locator = (ed / "agents/locator.yaml").read_text()
            self.assertIn("  - read_file\n  - run_command\n  - get_status\n", locator)
            self.assertNotIn("edit_file", locator)
            self.assertNotIn("skills", locator)
            self.assertIn("{problem_description}", (ed / "prompts/locate.md").read_text())
            p_to, p_ed = (to / "prompts/system.md").read_text(), (ed / "prompts/system.md").read_text()
            self.assertIn("call the `locator` tool", p_ed)
            self.assertIn("## Budget: 20 tool calls, 8 minutes", p_ed)
            self.assertNotIn("28 tool calls", p_ed)
            keep = p_to.split("### 2. Implement the Solution Directly")[1].split("## Budget:")[0]
            self.assertIn(keep, p_ed)
            self.assertEqual(p_to.split("## Anti-Patterns")[1], p_ed.split("## Anti-Patterns")[1])


class BfGovLeversTest(unittest.TestCase):
    """doc 03 §14.40: each lever is bf-gov-to plus one change."""

    def _pair(self, name, d):
        return build_variant("bf-gov-to", dest_root=Path(d)), build_variant(name, dest_root=Path(d))

    def _prompt_diff(self, to, out):
        before = (to / "prompts/system.md").read_text().splitlines()
        after = (out / "prompts/system.md").read_text().splitlines()
        return before, after, [ln for ln in after if ln not in before], [ln for ln in before if ln not in after]

    def test_prompt_levers_add_one_line_in_place(self):
        cases = {"bf-gov-ct": ("- Count your tool calls.", "- By your 12th tool call", 1),
                 "bf-gov-ti": ("- If the problem statement has no code", "- If the problem statement does not provide", 1),
                 "bf-gov-ec": ("- Editing is cheap and reversible.", "- Apply the minimal necessary fix", -1)}
        for name, (start, neighbour, offset) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                to, out = self._pair(name, d)
                for f in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                    self.assertTrue(filecmp.cmp(to / f, out / f, shallow=False), f)
                _, after, added, removed = self._prompt_diff(to, out)
                self.assertEqual(removed, [])
                self.assertEqual(len(added), 1)
                self.assertTrue(added[0].startswith(start))
                at = after.index(added[0])
                self.assertTrue(after[at - offset].startswith(neighbour))

    def test_bf_gov_22_changes_only_the_cap_and_its_stated_limits(self):
        with tempfile.TemporaryDirectory() as d:
            to, out = self._pair("bf-gov-22", d)
            for f in ("agent.yaml", "configs/sampling.yaml"):
                self.assertTrue(filecmp.cmp(to / f, out / f, shallow=False), f)
            before = (to / "eval_config.yaml").read_text().splitlines()
            after = (out / "eval_config.yaml").read_text().splitlines()
            self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                             [("  max_tool_calls: 28", "  max_tool_calls: 22")])
            _, after, added, removed = self._prompt_diff(to, out)
            self.assertEqual(len(added), 3)
            self.assertEqual(len(removed), 3)
            self.assertIn("## Budget: 22 tool calls, 8 minutes", after)
            self.assertNotIn("28 tool calls", "\n".join(after))
            self.assertIn("By your 12th tool call", "\n".join(after))
            self.assertIn("after your 16th tool call", "\n".join(after))


    def test_levers_on_tb_differ_from_levers_on_to_only_in_the_thinking_budget(self):
        for k in ("ct", "ti", "ec", "22"):
            with self.subTest(k), tempfile.TemporaryDirectory() as d:
                on_to = build_variant(f"bf-gov-{k}", dest_root=Path(d))
                on_tb = build_variant(f"bf-gov-tb-{k}", dest_root=Path(d))
                files = sorted(p.relative_to(on_to) for p in on_to.rglob("*") if p.is_file())
                self.assertEqual(files, sorted(p.relative_to(on_tb) for p in on_tb.rglob("*") if p.is_file()))
                for f in files:
                    if f != Path("configs/sampling.yaml"):
                        self.assertTrue(filecmp.cmp(on_to / f, on_tb / f, shallow=False), f)
                before = (on_to / "configs/sampling.yaml").read_text().splitlines()
                after = (on_tb / "configs/sampling.yaml").read_text().splitlines()
                self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                                 [("  thinking_budget: 4096", "  thinking_budget: 2048")])


class MemoryForE4BTest(unittest.TestCase):
    def test_mp_adds_only_the_skill_its_line_and_the_locate_bullets(self):
        with tempfile.TemporaryDirectory() as d:
            tb = build_variant("bf-gov-tb", dest_root=Path(d))
            mp = build_variant("bf-gov-tb-mp", dest_root=Path(d))
            nohab = build_variant("bf-gov-tb-mp-nohab", dest_root=Path(d))
            tb_files = sorted(p.relative_to(tb) for p in tb.rglob("*") if p.is_file())
            mp_files = sorted(p.relative_to(mp) for p in mp.rglob("*") if p.is_file())
            skill = SRC / "skills/mini-palace"
            skill_files = sorted(Path("skills/mini-palace") / p.relative_to(skill) for p in skill.rglob("*")
                                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                                 and p.name != ".DS_Store")
            self.assertEqual(mp_files, sorted(tb_files + skill_files))
            for f in skill_files:
                self.assertTrue(filecmp.cmp(skill / f.relative_to("skills/mini-palace"), mp / f, shallow=False), f)
            for f in tb_files:
                if str(f) not in ("agent.yaml", "prompts/system.md"):
                    self.assertTrue(filecmp.cmp(tb / f, mp / f, shallow=False), f)
            for f, added in (("agent.yaml", 2), ("prompts/system.md", 2)):
                before, after = (tb / f).read_text().splitlines(), (mp / f).read_text().splitlines()
                self.assertEqual([ln for ln in after if ln not in before][:added], [ln for ln in after if ln not in before])
                self.assertEqual(len(after), len(before) + added)
            self.assertIn("  - skills/mini-palace", (mp / "agent.yaml").read_text())
            # The local arm differs only in the emptied habits file.
            nohab_files = sorted(p.relative_to(nohab) for p in nohab.rglob("*") if p.is_file())
            self.assertEqual(nohab_files, mp_files)
            seeds = Path("skills/mini-palace/references/seed_patterns.txt")
            for f in mp_files:
                if f != seeds:
                    self.assertTrue(filecmp.cmp(mp / f, nohab / f, shallow=False), f)
            self.assertEqual((nohab / seeds).read_text(), "")
            self.assertTrue((mp / seeds).read_text().strip())


class H24Test(unittest.TestCase):
    def test_h24_is_the_public_bundle_and_only_its_prompt_differs_from_bf(self):
        src, bf = ROOT / "external/h24", ROOT / "external/budget-fit"
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("h24", dest_root=Path(d))
            files = sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())
            self.assertEqual(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()), files)
            for f in files:
                self.assertTrue(filecmp.cmp(src / f, out / f, shallow=False), f)
            for f in ("agent.yaml", "configs/sampling.yaml", "eval_config.yaml"):
                self.assertTrue(filecmp.cmp(bf / f, out / f, shallow=False), f)
            self.assertFalse(filecmp.cmp(bf / "prompts/system.md", out / "prompts/system.md", shallow=False))

    def test_h24_long_changes_only_the_budget_and_its_line_in_the_prompt(self):
        src = ROOT / "external/h24"
        with tempfile.TemporaryDirectory() as d:
            out = build_variant("h24-long", dest_root=Path(d))
            for f in ("agent.yaml", "configs/sampling.yaml"):
                self.assertTrue(filecmp.cmp(src / f, out / f, shallow=False), f)
            config = (out / "eval_config.yaml").read_text()
            self.assertIn("max_tool_calls: 36", config)
            self.assertIn("max_time_minutes: 9", config)
            self.assertIn("timeout_seconds: 240", config)
            self.assertIn("max_turns: 80", config)
            before = (src / "prompts/system.md").read_text().splitlines()
            after = (out / "prompts/system.md").read_text().splitlines()
            self.assertEqual(len(before), len(after))
            changed = [(a, b) for a, b in zip(before, after) if a != b]
            self.assertEqual(len(changed), 1)
            self.assertIn("36 tool calls and 9 minutes", changed[0][1])
            self.assertIn("Calls 9-12: edit", (out / "prompts/system.md").read_text())
