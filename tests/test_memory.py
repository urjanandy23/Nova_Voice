import unittest

import _stubs  # noqa: F401  (must come before src imports)
from src.memory import AnthropicMemory, ConversationMemory, OllamaMemory


def tool_turn(mem, n):
    mem.add_user(f"question {n}")
    mem.add_assistant([{"type": "tool_use", "id": f"t{n}", "name": "x", "input": {}}])
    mem.add_tool_results([(f"t{n}", "result")])
    mem.add_assistant_text(f"answer {n}")


def ollama_tool_turn(mem, n):
    mem.add_user(f"question {n}")
    mem.add_assistant_tool_calls([{"id": f"t{n}", "function": {"name": "x", "arguments": {}}}])
    mem.add_tool_message(f"t{n}", "x", "result")
    mem.add_assistant_text(f"answer {n}")


class TestMemory(unittest.TestCase):
    def test_trim_keeps_whole_turns_and_never_orphans_tool_results(self):
        mem = AnthropicMemory(max_turns=2)
        for n in range(5):
            tool_turn(mem, n)
        msgs = mem.as_list()
        self.assertEqual(msgs[0], {"role": "user", "content": "question 3"})
        self.assertEqual(sum(1 for m in msgs if isinstance(m["content"], str)
                             and m["role"] == "user"), 2)
        # every tool_result still has its tool_use
        ids_used = {b["id"] for m in msgs if isinstance(m["content"], list)
                    for b in m["content"] if b.get("type") == "tool_use"}
        ids_resulted = {b["tool_use_id"] for m in msgs if isinstance(m["content"], list)
                        for b in m["content"] if b.get("type") == "tool_result"}
        self.assertEqual(ids_used, ids_resulted)

    def test_first_message_is_always_a_user_turn(self):
        mem = AnthropicMemory(max_turns=1)
        for n in range(3):
            tool_turn(mem, n)
        self.assertEqual(mem.as_list()[0]["role"], "user")
        self.assertIsInstance(mem.as_list()[0]["content"], str)

    def test_multiple_tool_results_share_one_message(self):
        mem = AnthropicMemory()
        mem.add_user("hi")
        mem.add_tool_results([("a", "1"), ("b", "2")])
        self.assertEqual(len(mem.as_list()[-1]["content"]), 2)

    def test_pop_incomplete_turn(self):
        mem = AnthropicMemory()
        tool_turn(mem, 0)
        mem.add_user("failing question")
        mem.add_assistant([{"type": "tool_use", "id": "z", "name": "x", "input": {}}])
        mem.pop_incomplete_turn()
        self.assertEqual(mem.as_list()[-1], {"role": "assistant", "content": "answer 0"})


class TestOllamaMemory(unittest.TestCase):
    def test_trim_never_orphans_tool_messages(self):
        mem = OllamaMemory(max_turns=2)
        for n in range(5):
            ollama_tool_turn(mem, n)
        msgs = mem.as_list()
        self.assertEqual(msgs[0], {"role": "user", "content": "question 3"})
        self.assertEqual([m["role"] for m in msgs],
                         ["user", "assistant", "tool", "assistant"] * 2)

    def test_pop_incomplete_turn_drops_tool_messages_too(self):
        mem = OllamaMemory()
        ollama_tool_turn(mem, 0)
        mem.add_user("failing")
        mem.add_assistant_tool_calls([{"id": "z", "function": {"name": "x", "arguments": {}}}])
        mem.pop_incomplete_turn()
        self.assertEqual(mem.as_list()[-1], {"role": "assistant", "content": "answer 0"})


class TestFormatsAreSeparate(unittest.TestCase):
    def test_base_has_no_provider_specific_api(self):
        base = ConversationMemory()
        for name in ("add_assistant", "add_tool_results",
                     "add_assistant_tool_calls", "add_tool_message"):
            self.assertFalse(hasattr(base, name), name)

    def test_each_backend_memory_has_only_its_own_format(self):
        self.assertFalse(hasattr(AnthropicMemory(), "add_tool_message"))
        self.assertFalse(hasattr(OllamaMemory(), "add_tool_results"))


if __name__ == "__main__":
    unittest.main()
