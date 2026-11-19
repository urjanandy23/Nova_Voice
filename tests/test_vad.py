import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vad import Endpointer, SustainedSpeech, calibrate_threshold


class TestCalibration(unittest.TestCase):
    def test_noisy_room_raises_threshold(self):
        self.assertAlmostEqual(calibrate_threshold([400, 500, 450], floor=300), 1350)

    def test_quiet_room_uses_floor(self):
        self.assertEqual(calibrate_threshold([10, 20, 15], floor=300), 300)

    def test_empty_uses_floor(self):
        self.assertEqual(calibrate_threshold([], floor=300), 300)


def run(endpointer, levels):
    for i, rms in enumerate(levels):
        reason = endpointer.update(rms)
        if reason:
            return reason, i
    return None, len(levels)


class TestEndpointer(unittest.TestCase):
    def make(self, **kw):
        base = dict(threshold=100, silence_duration=0.8, block_duration=0.1,
                    min_speech_duration=0.3, max_duration=20.0, no_speech_timeout=3.0)
        base.update(kw)
        return Endpointer(**base)

    def test_ends_after_trailing_silence(self):
        reason, idx = run(self.make(), [500] * 10 + [0] * 20)
        self.assertEqual(reason, "end")
        self.assertEqual(idx, 10 + 8 - 1)  # 8 silent blocks = 0.8 s

    def test_single_click_does_not_start_speech(self):
        reason, _ = run(self.make(), [0, 500, 0, 0, 0, 0] + [0] * 30)
        self.assertEqual(reason, "no_speech")

    def test_pause_shorter_than_silence_does_not_end(self):
        levels = [500] * 5 + [0] * 5 + [500] * 5 + [0] * 20
        reason, idx = run(self.make(), levels)
        self.assertEqual(reason, "end")
        self.assertGreater(idx, 14)

    def test_no_speech_timeout(self):
        reason, idx = run(self.make(), [0] * 100)
        self.assertEqual(reason, "no_speech")
        self.assertEqual(idx, 29)

    def test_max_duration(self):
        reason, _ = run(self.make(max_duration=2.0), [500] * 100)
        self.assertEqual(reason, "max")

    def test_already_speaking_ends_without_new_speech(self):
        reason, _ = run(self.make(already_speaking=True), [0] * 20)
        self.assertEqual(reason, "end")


class TestSustainedSpeech(unittest.TestCase):
    def test_needs_sustained_loudness(self):
        s = SustainedSpeech(threshold=1000, sustain=0.4, block_duration=0.1)
        results = [s.update(r) for r in [2000, 2000, 2000, 2000]]
        self.assertEqual(results, [False, False, False, True])

    def test_brief_noise_is_ignored(self):
        s = SustainedSpeech(threshold=1000, sustain=0.4, block_duration=0.1)
        results = [s.update(r) for r in [2000, 2000, 0, 2000, 2000, 0]]
        self.assertFalse(any(results))


if __name__ == "__main__":
    unittest.main()
