import unittest

from lab.analysis.hidden_thinking import is_hidden, summarize, turns


def step(tokens, message="", args=None):
    s = {"source": "agent", "metrics": {"completion_tokens": tokens}, "tool_calls": [{"arguments": args or {}}]}
    if message:
        s["message"] = message
    return s


class HiddenThinkingTest(unittest.TestCase):
    def test_long_turn_with_short_visible_output_is_hidden(self):
        self.assertTrue(is_hidden(3000, 120))
        self.assertFalse(is_hidden(3000, 7000))  # thinking shown in the message
        self.assertFalse(is_hidden(700, 10))  # short turn

    def test_turns_count_message_and_arguments(self):
        t = {"steps": [step(1000, "x" * 50, {"command": "ls"}), {"source": "user"}, {"source": "agent"}]}
        self.assertEqual(turns(t), [(1000, 50 + len('{"command": "ls"}'))])

    def test_summary_share(self):
        s = summarize([(3000, 100), (1000, 5000), (500, 10)])
        self.assertEqual((s["hidden_turns"], s["hidden_tokens"], s["tokens"], s["hidden_share"]), (1, 3000, 4500, 0.667))


if __name__ == "__main__":
    unittest.main()
