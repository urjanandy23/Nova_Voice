"""Orchestration logic of VoiceAssistant with fake STT / TTS / LLM / wake word:
turn flow, failure recovery, degrade-to-working-mode, and timing marks."""
import contextlib
import io
import sys
import types
import unittest
from unittest import mock

import _stubs  # noqa: F401
from src import assistant as asst
from src import stt as stt_module  # noqa: F401  (so it can be patched)
from src import wake_word as ww_module  # noqa: F401
from src.config import Config
from src.errors import MicrophoneError, TranscriptionError, AudioInputError
from src.memory import ConversationMemory


class FakeTTS:
    def __init__(self):
        self.said, self.now, self.events = [], [], []

    def start(self, timeline=None):
        self.events.append("start")

    def say(self, s):
        self.said.append(s)

    def finish(self):
        self.events.append("finish")

    def stop(self):
        self.events.append("stop")

    def speak_now(self, text):
        self.now.append(text)


class FakeLLM:
    def __init__(self, chunks=("Hi there.",), error=None):
        self.chunks, self.error = chunks, error
        self.calls = 0
        self.last_timeline = None

    def new_memory(self):
        return ConversationMemory()

    def stream_response(self, memory, on_sentence, cancel=None, timeline=None):
        self.calls += 1
        self.last_timeline = timeline
        if self.error:
            raise self.error
        for c in self.chunks:
            on_sentence(c)
        memory.add_assistant_text(" ".join(self.chunks))


class FakeSTT:
    def __init__(self, script=()):
        self.script = list(script)

    def listen_and_transcribe(self, preroll=None, timeline=None):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeWake:
    def __init__(self, script=(), stats=None):
        self.script = list(script)
        self._stats = stats or {"frames": 5, "mean_ms": 3.0, "max_ms": 9.0, "budget_ms": 80.0}

    def wait_for_wake_word(self):
        item = self.script.pop(0) if self.script else KeyboardInterrupt()
        if isinstance(item, BaseException):
            raise item
        return True

    def stats(self):
        return self._stats


def make_assistant(mode="push_to_talk", stt=None, wake=None, llm=None, barge_in=False):
    a = asst.VoiceAssistant.__new__(asst.VoiceAssistant)
    a.mode, a.barge_in, a._audio_failures = mode, barge_in, 0
    a.stt, a.wake_word = stt, wake
    a.tts, a.llm = FakeTTS(), llm or FakeLLM()
    a.memory = a.llm.new_memory()
    return a


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        try:
            result = fn(*a, **kw)
        except KeyboardInterrupt:
            result = "interrupted"
    return result, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        sleep = mock.patch("time.sleep")
        sleep.start()
        self.addCleanup(sleep.stop)
        self._timings = Config.SHOW_TIMINGS
        Config.SHOW_TIMINGS = True
        self.addCleanup(setattr, Config, "SHOW_TIMINGS", self._timings)


class TestTurnFlow(Base):
    def test_text_turn_speaks_reply_and_updates_memory(self):
        a = make_assistant("text_only", llm=FakeLLM(("One.", "Two.")))
        _, out = quiet(a.handle_one_exchange, user_text="hello")
        self.assertEqual(a.tts.said, ["One.", "Two."])
        self.assertEqual(a.tts.events, ["start", "finish"])
        self.assertEqual(a.memory.as_list()[0], {"role": "user", "content": "hello"})
        self.assertIn("[timing]", out)

    def test_spoken_turn_goes_through_stt(self):
        a = make_assistant(stt=FakeSTT(["what time is it"]))
        _, out = quiet(a.handle_one_exchange)
        self.assertIn("You said: what time is it", out)
        self.assertEqual(a.llm.calls, 1)

    def test_empty_transcript_skips_llm(self):
        a = make_assistant(stt=FakeSTT([""]))
        quiet(a.handle_one_exchange)
        self.assertEqual(a.llm.calls, 0)
        self.assertEqual(a.memory.as_list(), [])

    def test_llm_failure_rolls_back_history_and_apologises(self):
        a = make_assistant("text_only", llm=FakeLLM(error=RuntimeError("api down")))
        _, out = quiet(a.handle_one_exchange, user_text="hello")
        self.assertEqual(a.memory.as_list(), [])  # incomplete turn removed
        self.assertEqual(a.tts.said, [asst.APOLOGY])
        self.assertEqual(a.tts.events[-1], "finish")  # TTS always finished

    def test_ctrl_c_during_reply_stops_tts_and_keeps_running(self):
        a = make_assistant("text_only", llm=FakeLLM(error=KeyboardInterrupt()))
        _, out = quiet(a.handle_one_exchange, user_text="hello")
        self.assertIn("stop", a.tts.events)
        self.assertIn("(interrupted)", out)

    def test_wake_timing_is_recorded_and_printed(self):
        a = make_assistant(stt=FakeSTT(["hi"]))
        _, out = quiet(a.handle_one_exchange, wake_at=123.0,
                       wake_stats={"frames": 5, "mean_ms": 3.0, "max_ms": 9.0, "budget_ms": 80.0})
        self.assertEqual(a.llm.last_timeline.marks["wake_detected"], 123.0)
        self.assertIn("wake_infer_mean=3.0ms", out)


class TestAudioFailures(Base):
    def test_microphone_error_abandons_turn_without_crashing(self):
        a = make_assistant(stt=FakeSTT([MicrophoneError("no mic")]))
        result, out = quiet(a.handle_one_exchange)
        self.assertIsNone(result)
        self.assertEqual(a.llm.calls, 0)
        self.assertEqual(a.memory.as_list(), [])
        self.assertIn("no mic", out)
        self.assertEqual(a._audio_failures, 1)

    def test_transcription_error_is_spoken_once(self):
        a = make_assistant(stt=FakeSTT([TranscriptionError("whisper"),
                                        TranscriptionError("whisper")]))
        quiet(a.handle_one_exchange)
        quiet(a.handle_one_exchange)
        self.assertEqual(len(a.tts.now), 1)

    def test_success_resets_failure_count(self):
        a = make_assistant(stt=FakeSTT([MicrophoneError("x"), "hello"]))
        quiet(a.handle_one_exchange)
        quiet(a.handle_one_exchange)
        self.assertEqual(a._audio_failures, 0)

    def test_repeated_mic_failures_fall_back_to_text_only(self):
        a = make_assistant(stt=FakeSTT([MicrophoneError("x")] * 3), barge_in=True)
        for _ in range(3):
            quiet(a.handle_one_exchange)
        self.assertEqual(a.mode, "text_only")
        self.assertFalse(a.barge_in)

    def test_wake_word_loop_survives_failures_then_degrades(self):
        wake = FakeWake([MicrophoneError("a"), MicrophoneError("b"), MicrophoneError("c")])
        a = make_assistant("wake_word", wake=wake)
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            result, out = quiet(a.run_forever)
        self.assertEqual(result, "interrupted")  # reached the text prompt, did not crash earlier
        self.assertEqual(a.mode, "text_only")
        self.assertIn("keeps failing", out)

    def test_broken_wake_model_falls_back_to_push_to_talk(self):
        wake = FakeWake([AudioInputError("model")] * 3)
        a = make_assistant("wake_word", wake=wake)
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            quiet(a.run_forever)
        self.assertEqual(a.mode, "push_to_talk")

    def test_wake_word_success_starts_a_timed_turn(self):
        wake = FakeWake([True])
        a = make_assistant("wake_word", wake=wake, stt=FakeSTT(["hello"]))
        result, out = quiet(a.run_forever)  # next wait -> KeyboardInterrupt ends the loop
        self.assertEqual(result, "interrupted")
        self.assertEqual(a.llm.calls, 1)
        self.assertIn("wake_detected", a.llm.last_timeline.marks)


class TestConstruction(Base):
    def build(self, mode, **patches):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(asst, "StreamingTTS", FakeTTS))
        stack.enter_context(mock.patch.object(asst, "_build_llm", lambda: FakeLLM()))
        stack.enter_context(mock.patch.object(asst, "start_reminder_watcher", lambda **kw: None))
        stack.enter_context(mock.patch.object(stt_module, "SpeechToText", lambda: FakeSTT()))
        wake_cls = patches.get("wake", lambda: FakeWake())
        stack.enter_context(mock.patch.object(ww_module, "WakeWordDetector", wake_cls))
        with stack:
            a, out = quiet(asst.VoiceAssistant, mode=mode)
        return a, out

    def test_normal_wake_word_construction(self):
        a, _ = self.build("wake_word")
        self.assertEqual(a.mode, "wake_word")
        self.assertIsNotNone(a.wake_word)
        self.assertIsInstance(a.memory, ConversationMemory)

    def test_wake_word_model_failure_falls_back_to_push_to_talk(self):
        def boom():
            raise RuntimeError("model download failed")
        a, out = self.build("wake_word", wake=boom)
        self.assertEqual(a.mode, "push_to_talk")
        self.assertIsNone(a.wake_word)
        self.assertIn("push-to-talk", out)

    def test_missing_audio_libs_fall_back_to_text_only(self):
        with mock.patch.dict(sys.modules, {"src.stt": types.ModuleType("src.stt")}):
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(asst, "StreamingTTS", FakeTTS))
                stack.enter_context(mock.patch.object(asst, "_build_llm", lambda: FakeLLM()))
                stack.enter_context(mock.patch.object(asst, "start_reminder_watcher",
                                                      lambda **kw: None))
                a, out = quiet(asst.VoiceAssistant, mode="hybrid", barge_in=True)
        self.assertEqual(a.mode, "text_only")
        self.assertIsNone(a.stt)
        self.assertFalse(a.barge_in)


if __name__ == "__main__":
    unittest.main()
