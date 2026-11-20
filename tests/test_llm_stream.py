"""Tests the shared streaming / tool-loop / cancel logic (src/llm_base.py) against a fake Ollama client (no network, no running Ollama server)."""
import os
import tempfile
import threading
import unittest

import _stubs  # noqa: F401  (must come before src imports)
from src import llm_base
from src import llm_ollama as ollama_module
from src.config import Config
from src.llm_base import BaseVoiceLLM
from src.memory import OllamaMemory
from src.metrics import Timeline


class Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class FakeStream:
    def __init__(self, pieces, final):
        self._pieces, self._final = pieces, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        yield from self._pieces

    def get_final_message(self):
        return self._final



class FakeOllamaClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        return iter(self.scripted.pop(0))


# ------------------------------------------------------------------ shared scenarios
class SharedLoopTests:
    """Mixed into one TestCase per backend. Subclasses provide the builders."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (Config.FIRST_CHUNK_MIN_CHARS, Config.REMINDERS_FILE)
        Config.FIRST_CHUNK_MIN_CHARS = 0
        Config.REMINDERS_FILE = os.path.join(self._tmp.name, "reminders.json")

    def tearDown(self):
        Config.FIRST_CHUNK_MIN_CHARS, Config.REMINDERS_FILE = self._saved
        self._tmp.cleanup()

    def test_backend_uses_the_shared_loop(self):
        self.assertTrue(issubclass(self.backend_class, BaseVoiceLLM))
        self.assertIs(self.backend_class.stream_response, BaseVoiceLLM.stream_response)

    def test_chunks_arrive_as_sentences_complete(self):
        llm = self.make_llm([self.text_round(["Hello th", "ere. How are", " you? Fine"])])
        mem = llm.new_memory()
        mem.add_user("hi")
        got = []
        llm.stream_response(mem, got.append)
        self.assertEqual(got, ["Hello there.", "How are you?", "Fine"])

    def test_tool_round_trip_counts_call_and_speaks_followup(self):
        llm = self.make_llm([
            self.tool_round("list_reminders", {}, "t1"),
            self.text_round(["You have nothing."]),
        ])
        mem = llm.new_memory()
        mem.add_user("what reminders?")
        got, tl = [], Timeline()
        llm.stream_response(mem, got.append, timeline=tl)
        self.assertEqual(got, ["You have nothing."])
        self.assertEqual(tl.tool_calls, 1)
        self.assertIn("first_sentence", tl.marks)

    def test_cancel_stops_generation_and_keeps_spoken_text(self):
        cancel = threading.Event()
        spoken = []

        def on_sentence(s):
            spoken.append(s)
            cancel.set()  # user interrupts after the first sentence

        llm = self.make_llm([self.text_round(["One. ", "Two. ", "Three."])])
        mem = llm.new_memory()
        mem.add_user("count")
        self.assertIsNone(llm.stream_response(mem, on_sentence, cancel=cancel))
        self.assertEqual(spoken, ["One."])
        self.assertEqual(mem.as_list()[-1], {"role": "assistant", "content": "One."})

    def test_runaway_tool_loop_is_bounded(self):
        llm = self.make_llm([self.tool_round("list_reminders", {}, f"t{i}")
                             for i in range(llm_base.MAX_TOOL_ROUNDS)])
        mem = llm.new_memory()
        mem.add_user("loop")
        got = []
        self.assertIsNone(llm.stream_response(mem, got.append))
        self.assertIn("Sorry", got[-1])

    def test_bad_tool_arguments_do_not_crash_the_turn(self):
        llm = self.make_llm([
            self.tool_round("set_reminder", {"text": "x"}, "t1"),  # minutes missing
            self.text_round(["Which time?"]),
        ])
        mem = llm.new_memory()
        mem.add_user("remind me")
        got = []
        llm.stream_response(mem, got.append)
        self.assertEqual(got, ["Which time?"])

    def test_new_memory_matches_backend_format(self):
        self.assertIsInstance(self.make_llm([]).new_memory(), self.memory_class)



class TestOllamaBackend(SharedLoopTests, unittest.TestCase):
    backend_class = ollama_module.OllamaVoiceLLM
    memory_class = OllamaMemory

    def make_llm(self, scripted):
        llm = ollama_module.OllamaVoiceLLM.__new__(ollama_module.OllamaVoiceLLM)
        llm.model = "fake"
        llm.tools = []
        llm.client = FakeOllamaClient(scripted)
        return llm

    def text_round(self, pieces):
        return [{"message": {"content": p}} for p in pieces]

    def tool_round(self, name, args, call_id):
        return [{"message": {"content": "", "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": args}}]}}]

    def test_tool_messages_use_ollama_format_and_system_prompt_is_sent(self):
        llm = self.make_llm([
            self.tool_round("list_reminders", {}, "t1"),
            self.text_round(["Done."]),
        ])
        mem = llm.new_memory()
        mem.add_user("q")
        llm.stream_response(mem, lambda s: None)
        roles = [m["role"] for m in mem.as_list()]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertEqual(mem.as_list()[2]["tool_call_id"], "t1")
        self.assertEqual(llm.client.requests[0]["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
