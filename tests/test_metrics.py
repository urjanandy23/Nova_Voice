import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import Timeline, percentile


class TestMetrics(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        data = list(range(1, 101))
        self.assertEqual(percentile(data, 50), 50)
        self.assertEqual(percentile(data, 95), 95)
        self.assertEqual(percentile([7], 95), 7)
        self.assertIsNone(percentile([], 50))

    def test_timeline_first_mark_wins(self):
        t = Timeline()
        t.mark("a")
        first = t.marks["a"]
        t.mark("a")
        self.assertEqual(t.marks["a"], first)

    def test_between_and_missing(self):
        t = Timeline()
        t.marks = {"llm_start": 1.0, "first_token": 1.5}
        self.assertAlmostEqual(t.between("llm_start", "first_token"), 0.5)
        self.assertIsNone(t.between("llm_start", "done"))
        self.assertIn("llm_first_token=0.50s", t.format())


if __name__ == "__main__":
    unittest.main()


class TestTimingStages(unittest.TestCase):
    """End-of-speech wait, wake->listen gap and perceived latency."""

    def timeline(self):
        from src.metrics import Timeline
        tl = Timeline()
        tl.mark("wake_detected", at=10.00)
        tl.mark("listening", at=10.12)
        tl.mark("speech_last", at=12.00)
        tl.mark("speech_end", at=12.80)
        tl.mark("transcribed", at=13.10)
        tl.mark("llm_start", at=13.10)
        tl.mark("first_sentence", at=13.60)
        tl.mark("first_audio", at=14.00)
        tl.mark("done", at=15.00)
        return tl

    def test_new_stages(self):
        s = self.timeline().stages()
        self.assertAlmostEqual(s["wake_to_listen_s"], 0.12)
        self.assertAlmostEqual(s["endpoint_wait_s"], 0.80)
        self.assertAlmostEqual(s["perceived_s"], 2.00)  # includes the silence wait
        self.assertAlmostEqual(s["heard_to_first_audio_s"], 1.20)

    def test_perceived_latency_exceeds_heard_to_first_audio_by_the_wait(self):
        s = self.timeline().stages()
        self.assertAlmostEqual(s["perceived_s"] - s["heard_to_first_audio_s"], s["endpoint_wait_s"])

    def test_mark_with_explicit_time_keeps_first_value(self):
        from src.metrics import Timeline
        tl = Timeline()
        tl.mark("x", at=1.0)
        tl.mark("x", at=2.0)
        self.assertEqual(tl.marks["x"], 1.0)

    def test_text_turns_report_no_audio_stages(self):
        from src.metrics import Timeline
        tl = Timeline()
        tl.mark("llm_start", at=1.0)
        tl.mark("done", at=2.0)
        text = tl.format()
        self.assertNotIn("endpoint_wait", text)
        self.assertNotIn("wake_to_listen", text)
        self.assertIn("total=1.00s", text)

    def test_format_shows_wake_inference_stats(self):
        tl = self.timeline()
        tl.wake_stats = {"frames": 10, "mean_ms": 4.0, "max_ms": 12.5, "budget_ms": 80.0}
        text = tl.format()
        self.assertIn("endpoint_wait=0.80s", text)
        self.assertIn("wake_infer_max=12.5ms", text)
