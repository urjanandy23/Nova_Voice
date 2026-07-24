"""Hands-free activation: listens continuously on the mic for a wake word
(e.g. 'hey jarvis') using openwakeword, and returns as soon as it's heard."""
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from .config import Config


class WakeWordDetector:
    def __init__(self, wake_word: str = None):
        self.wake_word = wake_word or Config.WAKE_WORD
        # openwakeword ships pretrained models; passing the name downloads/loads it
        self.model = Model(wakeword_models=[self.wake_word])
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms chunks, openwakeword's expected frame size

    def wait_for_wake_word(self, threshold: float = 0.5):
        """Blocks until the wake word is detected."""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(self.chunk_size)
                audio_chunk = audio_chunk.flatten()
                predictions = self.model.predict(audio_chunk)
                score = predictions.get(self.wake_word, 0.0)
                if score >= threshold:
                    return True
