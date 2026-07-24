"""Speech-to-text: records the user's utterance from the mic (auto-stopping
on silence) and transcribes it with a local faster-whisper model."""
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from .config import Config


class SpeechToText:
    def __init__(self):
        # compute_type="int8" keeps this fast enough to run on CPU-only laptops
        self.model = WhisperModel(Config.WHISPER_MODEL_SIZE, compute_type="int8")

    def record_utterance(self) -> np.ndarray:
        """Records from the default mic until the user stops talking
        (silence for SILENCE_DURATION seconds), or MAX_RECORD_SECONDS is hit."""
        sr = Config.SAMPLE_RATE
        block_duration = 0.1
        block_size = int(sr * block_duration)

        frames = []
        silence_run = 0.0
        started_talking = False
        start_time = time.time()

        with sd.InputStream(samplerate=sr, channels=Config.CHANNELS, dtype="int16") as stream:
            while True:
                block, _ = stream.read(block_size)
                frames.append(block.copy())

                rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
                if rms > Config.SILENCE_THRESHOLD:
                    started_talking = True
                    silence_run = 0.0
                else:
                    silence_run += block_duration

                elapsed = time.time() - start_time
                if started_talking and silence_run >= Config.SILENCE_DURATION:
                    break
                if elapsed >= Config.MAX_RECORD_SECONDS:
                    break

        audio = np.concatenate(frames, axis=0).flatten()
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        # faster-whisper expects float32 in [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_f32, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def listen_and_transcribe(self) -> str:
        audio = self.record_utterance()
        return self.transcribe(audio)
