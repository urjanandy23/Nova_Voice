import unittest

import _stubs  # noqa: F401  (must come before src imports)
from src.memory import ConversationMemory, OllamaMemory


def ollama_tool_turn(mem, n):
    mem.add_user(f"question {n}")
    mem.add_assistant_tool_calls([{"id": f"t{n}", "function": {"name": "x", "arguments": {}}}])
    mem.add_tool_message(f"t{n}", "x", "result")
    mem.add_assistant_text(f"answer {n}")


class TestMemory(unittest.TestCase):
    def test_first_message_is_always_a_user_turn(self):
        mem = OllamaMemory(max_turns=1)
        for n in range(3):
            ollama_tool_turn(mem, n)
        self.assertEqual(mem.as_list()[0]["role"], "user")
        self.assertIsInstance(mem.as_list()[0]["content"], str)

    def test_reset_clears_history(self):
        mem = OllamaMemory()
        ollama_tool_turn(mem, 0)
        mem.reset()
        self.assertEqual(mem.as_list(), [])


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


class TestBaseMemory(unittest.TestCase):
    def test_base_has_no_provider_specific_api(self):
        base = ConversationMemory()
        for name in ("add_assistant_tool_calls", "add_tool_message"):
            self.assertFalse(hasattr(base, name), name)


if __name__ == "__main__":
    unittest.main()