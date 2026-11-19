"""StreamingTTS worker logic with a fake synthesizer and fake output stream:
failure isolation, ordering, interruption and the first_audio mark. Real
edge-tts, ffmpeg decoding and speakers are not exercised."""
import threading
import unittest
from unittest import mock

import numpy as np

import _stubs  # noqa: F401
from src import tts as tts_module
from src.metrics import Timeline

FRAMES = tts_module.PLAY_BLOCK_FRAMES


class FakeOut:
    instances = []

    def __init__(self, **kw):
        self.writes = []
        self.aborted = False
        FakeOut.instances.append(self)

    def start(self): pass
    def stop(self): pass
    def close(self): pass

    def abort(self):
        self.aborted = True

    def write(self, arr):
        self.writes.append(arr.copy())


def run_with_timeout(fn, seconds=5):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(seconds)
    return not t.is_alive()


class TestStreamingTTS(unittest.TestCase):
    def setUp(self):
        FakeOut.instances = []
        self.spoken = []

        def to_pcm(audio):
            # each "audio" is the sentence text; encode its length as the level
            return np.full((FRAMES * 2, 1), len(audio), dtype=np.int16), 24000

        patches = [mock.patch.object(tts_module, "_to_pcm", to_pcm),
                   mock.patch.object(tts_module.sd, "OutputStream", FakeOut)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def make(self, fail_on=()):
        tts = tts_module.StreamingTTS(voice="fake")

        async def synth(text):
            if text in fail_on:
                raise RuntimeError("network down")
            self.spoken.append(text)
            return text

        tts._synthesize = synth
        return tts

    def played_levels(self):
        out = []
        for inst in FakeOut.instances:
            out += [int(w[0, 0]) for w in inst.writes]
        return out

    def test_plays_all_sentences_in_order_and_marks_first_audio(self):
        tts = self.make()
        tl = Timeline()
        tts.start(tl)
        for s in ("one", "three", "fifteen"):
            tts.say(s)
        self.assertTrue(run_with_timeout(tts.finish))
        self.assertEqual(self.spoken, ["one", "three", "fifteen"])
        levels = self.played_levels()
        self.assertEqual(sorted(set(levels), key=levels.index), [3, 5, 7])
        self.assertIn("first_audio", tl.marks)

    def test_one_failed_sentence_does_not_stall_the_rest(self):
        tts = self.make(fail_on={"bad"})
        tts.start()
        for s in ("ok", "bad", "fine"):
            tts.say(s)
        self.assertTrue(run_with_timeout(tts.finish))
        self.assertEqual(self.spoken, ["ok", "fine"])

    def test_markdown_is_stripped_before_synthesis(self):
        tts = self.make()
        tts.start()
        tts.say("**Hello** see https://example.com now")
        self.assertTrue(run_with_timeout(tts.finish))
        self.assertEqual(self.spoken, ["Hello see now"])

    def test_stop_discards_queue_and_finish_still_returns(self):
        tts = self.make()
        tts.start()
        tts.stop()
        tts.say("never spoken")  # ignored once stopped
        self.assertTrue(tts.interrupted)
        self.assertTrue(run_with_timeout(tts.finish))
        self.assertEqual(self.spoken, [])

    def test_stop_mid_playback_aborts_the_stream(self):
        tts = self.make()
        started = threading.Event()
        orig_write = FakeOut.write

        def slow_write(self_, arr):
            orig_write(self_, arr)
            started.set()
            tts.stop()  # barge-in lands while audio is playing

        with mock.patch.object(FakeOut, "write", slow_write):
            tts.start()
            tts.say("long sentence")
            self.assertTrue(run_with_timeout(tts.finish))
        self.assertTrue(started.is_set())
        self.assertTrue(any(i.aborted for i in FakeOut.instances))

    def test_start_resets_state_between_turns(self):
        tts = self.make()
        tts.start()
        tts.stop()
        run_with_timeout(tts.finish)
        tts.start()
        self.assertFalse(tts.interrupted)
        tts.say("again")
        self.assertTrue(run_with_timeout(tts.finish))
        self.assertEqual(self.spoken, ["again"])

    def test_speak_now_never_raises(self):
        tts = self.make(fail_on={"Reminder: x"})
        tts.speak_now("Reminder: x")  # synthesis fails -> logged, not raised


if __name__ == "__main__":
    unittest.main()
