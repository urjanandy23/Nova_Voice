"""Validates the eval case sets and the provider-specific history builder, so a
malformed multi-turn case is caught here instead of as a confusing API error
(or a silently wrong score) during a paid eval run."""
import os
import sys
import unittest

import _stubs  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

import cases  # noqa: E402
import tool_calling_eval as ev  # noqa: E402
from src.functions import TOOLS  # noqa: E402

TOOL_NAMES = {t["name"] for t in TOOLS}
LABELS = TOOL_NAMES | {cases.NONE}


class TestCaseSets(unittest.TestCase):
    def test_labels_are_real_tools_or_none(self):
        for prompt, expected in cases.SINGLE_TURN_CASES:
            self.assertIn(expected, LABELS, prompt)
        for c in cases.MULTI_TURN_CASES:
            self.assertIn(c.expected, LABELS, c.name)

    def test_multi_turn_cases_are_really_multi_turn(self):
        self.assertGreaterEqual(len(cases.MULTI_TURN_CASES), 15)
        for c in cases.MULTI_TURN_CASES:
            self.assertGreaterEqual(len(c.history), 1, c.name)

    def test_names_are_unique(self):
        names = [c.name for c in cases.MULTI_TURN_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_tool_and_the_no_tool_label_is_covered_by_multi_turn(self):
        covered = {c.expected for c in cases.MULTI_TURN_CASES}
        self.assertEqual(covered, LABELS)

    def test_scripted_tool_calls_use_real_tools_with_required_args(self):
        schema = {t["name"]: t["input_schema"] for t in TOOLS}
        for c in cases.MULTI_TURN_CASES:
            for turn in c.history:
                if turn.tool:
                    name, args, _ = turn.tool
                    self.assertIn(name, TOOL_NAMES, c.name)
                    for req in schema[name].get("required", []):
                        self.assertIn(req, args, f"{c.name}: {name} missing {req}")


class TestBuildMessages(unittest.TestCase):

    def test_ollama_history_pairs_tool_calls_with_tool_messages(self):
        for c in cases.MULTI_TURN_CASES:
            msgs = cases.build_messages(c, "ollama")
            self.assertEqual(msgs[-1], {"role": "user", "content": c.user}, c.name)
            ids = [call["id"] for m in msgs for call in m.get("tool_calls", [])]
            tool_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
            self.assertEqual(ids, tool_ids, c.name)

        def test_tool_case_renders_tool_messages(self):
         c = next(c for c in cases.MULTI_TURN_CASES if c.name == "cancel 'that' reminder")
         self.assertTrue(any(m["role"] == "tool" for m in cases.build_messages(c, "ollama")))
    def test_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            cases.build_messages(cases.MULTI_TURN_CASES[0], "nope")


class TestEvalRunner(unittest.TestCase):
    def test_build_jobs_respects_suite(self):
        n_single, n_multi = len(cases.SINGLE_TURN_CASES), len(cases.MULTI_TURN_CASES)
        self.assertEqual(len(ev.build_jobs("single", "ollama")), n_single)
        self.assertEqual(len(ev.build_jobs("multi", "ollama")), n_multi)
        self.assertEqual(len(ev.build_jobs("all", "ollama")), n_single + n_multi)

    def test_report_has_per_suite_accuracy_and_failures(self):
        rows = [
            {"suite": "single", "name": "a", "expected": "search_web", "predicted": "search_web"},
            {"suite": "multi", "name": "b", "expected": cases.NONE, "predicted": "search_web"},
            {"suite": "multi", "name": "c", "expected": cases.NONE, "predicted": cases.NONE},
        ]
        meta = {"date": "2026-01-01", "provider": "ollama", "model": "m",
                "prompt_hash": "h", "repeats": 1}
        md = ev.render_report(rows, meta)
        self.assertIn("**Overall accuracy: 2/3 (66.7%)**", md)
        self.assertIn("| multi | 1/2 (50.0%) |", md)
        self.assertIn("| single | 1/1 (100.0%) |", md)
        self.assertIn("Failures (1)", md)


if __name__ == "__main__":
    unittest.main()
