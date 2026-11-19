import sys
import unittest
from unittest import mock

import numpy as np

import _stubs
from src import wake_word as ww
from src.errors import AudioInputError, MicrophoneError


class FakeStream:
    def __init__(self, chunks=None, fail_open=False, fail_read_after=None):
        self.fail_open = fail_open
        self.fail_read_after = fail_read_after
        self.reads = 0

    def __call__(self, **kw):
        if self.fail_open:
            raise ww.sd.PortAudioError("no default input device")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.reads += 1
        if self.fail_read_after is not None and self.reads > self.fail_read_after:
            raise OSError("device unplugged")
        return np.zeros((n, 1), dtype=np.int16), False


class FakeModel:
    def __init__(self, scores, raise_on=None):
        self.scores = list(scores)
        self.raise_on = raise_on
        self.calls = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict(self, frame):
        self.calls += 1
        if self.raise_on and self.calls == self.raise_on:
            raise ValueError("bad frame")
        return {"hey_jarvis": self.scores.pop(0) if self.scores else 0.0}


def detector(model):
    d = ww.WakeWordDetector.__new__(ww.WakeWordDetector)
    d.wake_word = "hey_jarvis"
    d.model = model
    d.sample_rate = ww.SAMPLE_RATE
    d.chunk_size = ww.CHUNK_SIZE
    d._infer_ms = []
    d.last_score = 0.0
    return d


class TestWakeWord(unittest.TestCase):
    def wait(self, d, stream, threshold=0.5):
        with mock.patch.object(ww.sd, "InputStream", stream):
            return d.wait_for_wake_word(threshold)

    def test_returns_when_score_crosses_threshold(self):
        model = FakeModel([0.0, 0.1, 0.2, 0.9])
        d = detector(model)
        self.assertTrue(self.wait(d, FakeStream()))
        self.assertEqual(model.calls, 4)
        self.assertEqual(model.reset_calls, 1)  # history cleared first
        self.assertAlmostEqual(d.last_score, 0.9)

    def test_inference_timing_stats_cover_every_frame(self):
        d = detector(FakeModel([0.0, 0.0, 0.8]))
        self.wait(d, FakeStream())
        s = d.stats()
        self.assertEqual(s["frames"], 3)
        self.assertAlmostEqual(s["budget_ms"], 80.0)
        self.assertGreaterEqual(s["max_ms"], s["mean_ms"])
        self.assertEqual(s["frames_over_budget"], 0)

    def test_stats_empty_before_first_wait(self):
        self.assertEqual(detector(FakeModel([])).stats(), {"frames": 0})

    def test_missing_device_raises_microphone_error(self):
        with self.assertRaises(MicrophoneError):
            self.wait(detector(FakeModel([])), FakeStream(fail_open=True))

    def test_device_lost_mid_wait_raises_microphone_error(self):
        with self.assertRaises(MicrophoneError):
            self.wait(detector(FakeModel([0.0] * 50)), FakeStream(fail_read_after=3))

    def test_model_failure_raises_audio_input_error_not_microphone_error(self):
        with self.assertRaises(AudioInputError) as cm:
            self.wait(detector(FakeModel([0.0] * 10, raise_on=2)), FakeStream())
        self.assertNotIsInstance(cm.exception, MicrophoneError)

    def test_load_failure_gives_actionable_message(self):
        oww = sys.modules["openwakeword.model"]
        with mock.patch.object(oww, "Model", side_effect=ValueError("unknown model")):
            with self.assertRaises(RuntimeError) as cm:
                ww.WakeWordDetector("nope")
        self.assertIn("WAKE_WORD", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
