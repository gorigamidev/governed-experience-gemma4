import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab/src"))

from lab.lora import scripted as S  # noqa: E402
from lab.lora.scripted_server import respond  # noqa: E402

PATCH = """--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -2,3 +2,3 @@ def f():
     x = 1
-    return x
+    return x + 1

@@ -8,2 +8,3 @@ def g():
     y = 2
+    y += 1
     return y
"""
BASE = "def f():\n    x = 1\n    return x\n\n\ndef g():\n    pass\n    y = 2\n    return y\n"


class ParsePatchTest(unittest.TestCase):
    def test_hunks_keep_old_and_new_sides(self):
        h1, h2 = S.parse_patch(PATCH)
        self.assertEqual((h1.path, h1.old_start), ("pkg/mod.py", 2))
        self.assertEqual(h1.old_lines, ["    x = 1", "    return x", ""])
        self.assertEqual(h1.new_lines, ["    x = 1", "    return x + 1", ""])
        self.assertEqual(h2.old_lines, ["    y = 2", "    return y"])
        self.assertEqual(h2.new_lines, ["    y = 2", "    y += 1", "    return y"])

    def test_old_string_is_widened_until_unique(self):
        src = "a\nb\nx\nc\nx\nd".split("\n")
        old, new = S._unique_old_string(src, S.Hunk("f", 3, ["x"], ["y"]))
        self.assertEqual("\n".join(src).count(old), 1)
        self.assertEqual((old, new), ("b\nx\nc", "b\ny\nc"))

    def test_hunk_that_does_not_match_the_base_fails_loudly(self):
        with self.assertRaises(AssertionError):
            S._unique_old_string(["a", "b"], S.Hunk("f", 1, ["z"], ["y"]))

    def test_plan_edits_reproduce_the_patch(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pkg").mkdir()
            (Path(d) / "pkg/mod.py").write_text(BASE)
            task = {"instance_id": "x_1", "patch": PATCH, "problem_statement": "## f is off by one\nbody"}
            steps = S.plan(task, Path(d), None)
            self.assertEqual([s.tool for s in steps], ["code_analyzer", "read_file", "edit_file", "edit_file",
                                                       "run_command", "run_command", "submit_patch", None])
            self.assertEqual(steps[0].args, {"request": task["problem_statement"]})
            self.assertEqual(steps[-1].content, "DONE")
            S.apply_edits(Path(d), steps)
            self.assertEqual((Path(d) / "pkg/mod.py").read_text(),
                             "def f():\n    x = 1\n    return x + 1\n\n\ndef g():\n    pass\n    y = 2\n    y += 1\n    return y\n")


class EligibleTest(unittest.TestCase):
    def test_filters(self):
        self.assertEqual(S.eligible({"instance_id": "rich_3043", "patch": PATCH}), "reserve")
        self.assertEqual(S.eligible({"instance_id": "rich_3930", "patch": PATCH}), "impossible")
        self.assertEqual(S.eligible({"instance_id": "a_1", "patch": "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x\n"}), "new_file")
        big = "--- a/m.py\n+++ b/m.py\n@@ -1,61 +1,0 @@\n" + "-x\n" * 61
        self.assertEqual(S.eligible({"instance_id": "a_1", "patch": big}), "big_patch")
        self.assertIsNone(S.eligible({"instance_id": "a_1", "patch": PATCH}))


class ScriptedServerTest(unittest.TestCase):
    PLAN = {"x_1": {"variant": "x_1-0", "problem_head": "f is off by one", "analyzer_text": "LOCATION: pkg/mod.py:3",
                    "steps": [{"thought": "ask", "tool": "code_analyzer", "args": {"request": "r"}, "content": None},
                              {"thought": "", "tool": None, "args": {}, "content": "DONE"}]}}

    def body(self, tools, n_assistant):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Problem Statement:\nf is off by one"}]
        msgs += [{"role": "assistant", "content": None}] * n_assistant
        return {"messages": msgs, "tools": [{"type": "function", "function": {"name": t}} for t in tools]}

    def test_coder_gets_the_step_of_its_turn(self):
        msg, meta = respond(self.body(["submit_patch", "read_file"], 0), self.PLAN)
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"request": "r"})
        self.assertEqual((msg["reasoning_content"], meta["turn"]), ("ask", 0))
        msg, _ = respond(self.body(["submit_patch"], 1), self.PLAN)
        self.assertEqual((msg["content"], msg.get("tool_calls")), ("DONE", None))

    def test_analyzer_gets_the_recorded_answer(self):
        msg, meta = respond(self.body(["read_file", "search_similar_code"], 0), self.PLAN)
        self.assertEqual((msg["content"], meta["role"]), ("LOCATION: pkg/mod.py:3", "analyzer"))

    def test_unknown_task_is_reported(self):
        body = self.body(["submit_patch"], 0)
        body["messages"][1]["content"] = "another issue"
        self.assertEqual(respond(body, self.PLAN)[1]["error"], "unknown task")


@unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
class RenderTest(unittest.TestCase):
    def test_as_vllm_parses_arguments(self):
        sys.path.insert(0, str(ROOT / "lab/src/lab/lora"))
        from train import as_vllm
        out = as_vllm([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})


if __name__ == "__main__":
    unittest.main()


class LabeledPositionsTest(unittest.TestCase):
    def test_every_labeled_segment_is_kept(self):
        sys.path.insert(0, str(ROOT / "lab/src/lab/lora"))
        from train import labeled_positions
        labels = [-100, -100, 7, 8, -100, -100, 9, -100, 10]
        keep, tokens = labeled_positions(labels)
        self.assertEqual(tokens, [2, 3, 6, 8])
        self.assertEqual(keep, [1, 2, 5, 7])


CONFIG = ROOT / "data/models/e4b-bf16/config.json"


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers") and CONFIG.exists(),
                     "needs torch, transformers and the E4B config")
class LabeledLossTest(unittest.TestCase):
    def test_matches_the_standard_masked_loss_on_a_merged_example(self):
        import torch
        sys.path.insert(0, str(ROOT / "lab/src/lab/lora"))
        from train import labeled_loss, tiny_model
        torch.manual_seed(0)
        model = tiny_model(CONFIG).eval()
        ids = torch.randint(10, 1000, (40,)).tolist()
        labels = [-100] * 40
        for i in list(range(12, 18)) + list(range(30, 34)):  # two segments, like a fused trajectory
            labels[i] = ids[i]
        with torch.no_grad():
            ours = labeled_loss(model, {"input_ids": ids, "labels": labels}, "cpu").item()
            ref = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss.item()
            n = sum(x != -100 for x in labels)  # the first version: last n tokens
            old = torch.nn.functional.cross_entropy(model(input_ids=torch.tensor([ids]), logits_to_keep=n + 1).logits[0, :n].float(),
                                                    torch.tensor(ids[-n:])).item()
        self.assertAlmostEqual(ours, ref, places=4)
        self.assertNotAlmostEqual(old, ref, places=2)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers") and CONFIG.exists(),
                     "needs torch, transformers and the E4B config")
class TpuSmokeLossTest(unittest.TestCase):
    def test_padded_batch_loss_matches_labeled_loss_per_row(self):
        import torch
        sys.path.insert(0, str(ROOT / "lab/src/lab/lora"))
        from tpu_smoke import batch_loss, make_batch, select_labeled
        from train import labeled_loss, tiny_model
        torch.manual_seed(0)
        model = tiny_model(CONFIG).eval()
        rows = []
        for n, segs in ((40, [(12, 18), (30, 34)]), (25, [(5, 9)])):
            ids = torch.randint(10, 1000, (n,)).tolist()
            labels = [-100] * n
            for lo, hi in segs:
                for i in range(lo, hi):
                    labels[i] = ids[i]
            rows.append({"input_ids": ids, "labels": labels})
        with torch.no_grad():
            for r in rows:
                (ids, mask, pos, tgt, w), dropped = make_batch([r], 48, 16, 0)
                self.assertEqual(dropped, 0)
                self.assertAlmostEqual(batch_loss(model, ids, mask, pos, tgt, w).item(),
                                       labeled_loss(model, r, "cpu").item(), places=4)
        pos, tgt, w, dropped = select_labeled([-100, 7, -100, 9, 11], 2)
        self.assertEqual((pos, tgt, w, dropped), ([0, 2], [7, 9], [1.0, 1.0], 1))


class FailureSplitTest(unittest.TestCase):
    def test_buckets(self):
        from lab.analysis.failure_split import bucket
        gold = {"pkg/a.py"}
        patch_a = "--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        patch_b = "--- a/pkg/b.py\n+++ b/pkg/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        self.assertEqual(bucket(True, "", patch_a, gold, ""), "resolved")
        self.assertEqual(bucket(False, "Agent exceeded session timeout (4.5 min)", patch_a, gold, ""), "timeout, with patch")
        self.assertEqual(bucket(False, "Agent exceeded session timeout", "", gold, ""), "timeout, no patch")
        self.assertEqual(bucket(False, "litellm.ContextWindowExceededError", patch_a, gold, ""), "context overflow")
        self.assertEqual(bucket(False, "", "", gold, ""), "finished, empty patch")
        self.assertEqual(bucket(False, "", patch_b, gold, ""), "patch misses the gold file")
        self.assertEqual(bucket(False, "", patch_a, gold, "E   ModuleNotFoundError: x"), "right file, import/collection error in tests")
        self.assertEqual(bucket(False, "", patch_a, gold, "1 failed, 3 passed"), "right file, tests fail")


class LossMaskTokenTest(unittest.TestCase):
    TOK = ROOT / "data/models/gemma4-31b-tok"
    R1 = ROOT / "data/lora/r1"

    @unittest.skipUnless(importlib.util.find_spec("transformers") and (TOK / "tokenizer.json").exists()
                         and (R1 / "plans.json").exists() and (R1 / "requests.jsonl").exists(),
                         "needs transformers, the 31B tokenizer and the generated R1 data")
    def test_loss_false_turn_tokens_are_fully_masked(self):
        sys.path.insert(0, str(ROOT / "lab/src/lab/lora"))
        from train import coder_turns, merge, render

        from transformers import AutoTokenizer

        plans = json.loads((self.R1 / "plans.json").read_text())
        target_variant = None
        for v in plans:
            rows = [json.loads(ln) for ln in (self.R1 / "requests.jsonl").read_text().splitlines()
                    if json.loads(ln)["variant"] == v]
            if any(r.get("loss") is False for r in rows):
                target_variant = v
                break
        self.assertIsNotNone(target_variant)

        td = Path(tempfile.mkdtemp())
        data = td / "data"
        data.mkdir()
        (data / "plans.json").write_text(json.dumps({target_variant: plans[target_variant]}))
        rows = [json.loads(ln) for ln in (self.R1 / "requests.jsonl").read_text().splitlines()
                if json.loads(ln)["variant"] == target_variant]
        with (data / "requests.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        tok = AutoTokenizer.from_pretrained(str(self.TOK))
        turns = coder_turns(data)
        by_variant: dict[str, list] = {}
        for t in turns:
            by_variant.setdefault(t["variant"], []).append(t)
        merged = merge(tok, by_variant[target_variant], 16384)
        labeled = sum(1 for ex in merged for x in ex["labels"] if x != -100)
        recovery_target_tokens = sum(
            len(tok(render(tok, t)[1], add_special_tokens=False)["input_ids"]) for t in turns
        )
        self.assertEqual(labeled, recovery_target_tokens)
        self.assertGreater(recovery_target_tokens, 0)

        # Negative control: without the `r.get("loss", True)` filter, the loss=False turn's
        # target tokens are labeled, so the count differs.
        kept = set(json.loads((data / "plans.json").read_text()))
        no_filter = []
        for line in (data / "requests.jsonl").read_text().splitlines():
            r = json.loads(line)
            if (r.get("role") == "coder" and r.get("variant") in kept
                    and not r.get("overrun") and not r.get("error")):
                no_filter.append(r)
        merged_nf = merge(tok, [t for t in no_filter if t["variant"] == target_variant], 16384)
        labeled_nf = sum(1 for ex in merged_nf for x in ex["labels"] if x != -100)
        self.assertNotEqual(labeled_nf, labeled)
