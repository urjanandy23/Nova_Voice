"""STT recording + transcription against a fake mic and fake Whisper model.
Covers the device/model error paths and the timing marks; real audio hardware
and real Whisper are not exercised here."""
import sys
import unittest
from unittest import mock

import numpy as np

import _stubs
from src import stt as stt_module
from src.config import Config
from src.errors import MicrophoneError, TranscriptionError
from src.metrics import Timeline

BLOCK = int(Config.SAMPLE_RATE * Config.BLOCK_SECONDS)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeInput:
    """Stands in for sd.InputStream: yields int16 blocks of constant level,
    advancing a fake clock by one block per read()."""

    def __init__(self, levels, clock, fail_on_read=None):
        self.levels = list(levels)
        self.clock = clock
        self.fail_on_read = fail_on_read
        self.reads = 0

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.reads += 1
        if self.fail_on_read and self.reads >= self.fail_on_read:
            raise OSError("device unplugged")
        self.clock.t += Config.BLOCK_SECONDS
        level = self.levels.pop(0) if self.levels else 0
        return np.full((n, 1), level, dtype=np.int16), False


def make_stt():
    return stt_module.SpeechToText.__new__(stt_module.SpeechToText)


class Seg:
    def __init__(self, text):
        self.text = text


class TestRecording(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        patches = [
            mock.patch("time.perf_counter", self.clock),
            mock.patch.multiple(Config, SILENCE_DURATION=0.8, MIN_SPEECH_RMS=300,
                                NOISE_MULTIPLIER=3.0, MIN_SPEECH_SECONDS=0.25,
                                NO_SPEECH_TIMEOUT=8.0, MAX_RECORD_SECONDS=20.0),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def record(self, levels, **kw):
        fake = FakeInput(levels, self.clock, kw.pop("fail_on_read", None))
        with mock.patch.object(stt_module.sd, "InputStream", fake):
            return make_stt().record_utterance(**kw)

    def test_records_speech_and_trims_trailing_silence(self):
        levels = [10] * 3 + [2000] * 5 + [10] * 12
        tl = Timeline()
        audio = self.record(levels, timeline=tl)
        self.assertGreater(audio.size, 0)
        self.assertEqual(audio.dtype, np.int16)
        self.assertLess(audio.size, 16 * BLOCK)  # silence tail was cut

    def test_end_of_speech_wait_is_measured(self):
        levels = [10] * 3 + [2000] * 5 + [10] * 12
        tl = Timeline()
        self.record(levels, timeline=tl)
        self.assertIn("listening", tl.marks)
        wait = tl.stages()["endpoint_wait_s"]
        self.assertIsNotNone(wait)
        self.assertAlmostEqual(wait, Config.SILENCE_DURATION, delta=0.15)

    def test_listening_mark_is_set_on_first_block(self):
        tl = Timeline()
        self.record([10] * 3 + [2000] * 5 + [10] * 12, timeline=tl)
        self.assertLessEqual(tl.marks["listening"], tl.marks["speech_last"])
        self.assertLessEqual(tl.marks["speech_last"], tl.marks["speech_end"])

    def test_silence_returns_empty_audio(self):
        audio = self.record([], timeline=Timeline())
        self.assertEqual(audio.size, 0)

    def test_preroll_starts_in_speaking_state(self):
        pre = np.full(BLOCK * 3, 2000, dtype=np.int16)
        audio = self.record([2000] * 3 + [10] * 12, preroll=pre)
        self.assertGreater(audio.size, pre.size)

    def test_missing_device_raises_microphone_error(self):
        def boom(**kw):
            raise stt_module.sd.PortAudioError("no default input device")
        with mock.patch.object(stt_module.sd, "InputStream", boom):
            with self.assertRaises(MicrophoneError) as cm:
                make_stt().record_utterance()
        self.assertIn("Microphone", str(cm.exception))

    def test_device_lost_mid_recording_raises_microphone_error(self):
        with self.assertRaises(MicrophoneError):
            self.record([10] * 3 + [2000] * 5, fail_on_read=6)


class TestTranscribe(unittest.TestCase):
    def audio(self):
        return np.full(BLOCK * 5, 1000, dtype=np.int16)

    def test_joins_segments(self):
        s = make_stt()
        s.model = mock.Mock()
        s.model.transcribe.return_value = (iter([Seg(" hello "), Seg("world ")]), None)
        self.assertEqual(s.transcribe(self.audio()), "hello world")

    def test_empty_audio_skips_model(self):
        s = make_stt()
        s.model = mock.Mock()
        self.assertEqual(s.transcribe(np.array([], dtype=np.int16)), "")
        s.model.transcribe.assert_not_called()

    def test_model_failure_raises_transcription_error(self):
        s = make_stt()
        s.model = mock.Mock()
        s.model.transcribe.side_effect = RuntimeError("cuda out of memory")
        with self.assertRaises(TranscriptionError):
            s.transcribe(self.audio())

    def test_failure_during_lazy_decoding_is_caught_too(self):
        def gen():
            yield Seg("partial")
            raise RuntimeError("decode error")
        s = make_stt()
        s.model = mock.Mock()
        s.model.transcribe.return_value = (gen(), None)
        with self.assertRaises(TranscriptionError):
            s.transcribe(self.audio())

    def test_listen_and_transcribe_marks_transcribed(self):
        s = make_stt()
        s.model = mock.Mock()
        s.model.transcribe.return_value = (iter([Seg("hi")]), None)
        tl = Timeline()
        with mock.patch.object(s, "record_utterance", return_value=self.audio()):
            self.assertEqual(s.listen_and_transcribe(timeline=tl), "hi")
        self.assertIn("transcribed", tl.marks)


class TestModelLoad(unittest.TestCase):
    def test_load_failure_gives_actionable_message(self):
        fw = sys.modules["faster_whisper"]
        with mock.patch.object(fw, "WhisperModel", side_effect=OSError("offline")):
            with self.assertRaises(RuntimeError) as cm:
                stt_module.SpeechToText()
        self.assertIn("internet", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
