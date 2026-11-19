"""Hands-free activation: listens on the mic for a wake word (e.g. 'hey jarvis')
using openwakeword and returns as soon as it is heard.

Also keeps per-frame inference timings of the last wait so you can check the
model keeps up with real time (one 80 ms frame must be processed in < 80 ms)."""
from __future__ import annotations

import logging
import time

import sounddevice as sd

from .config import Config
from .errors import AudioInputError, MicrophoneError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80 ms, openwakeword's expected frame size
FRAME_BUDGET_MS = CHUNK_SIZE / SAMPLE_RATE * 1000.0


class WakeWordDetector:
    def __init__(self, wake_word: str | None = None) -> None:
        self.wake_word = wake_word or Config.WAKE_WORD
        try:
            from openwakeword.model import Model
            # openwakeword ships pretrained models; passing the name downloads/loads it
            self.model = Model(wakeword_models=[self.wake_word])
        except Exception as e:
            raise RuntimeError(
                f"Could not load the wake-word model '{self.wake_word}' "
                f"({type(e).__name__}: {e}). Check WAKE_WORD, and note the first run "
                "downloads the model so it needs internet access."
            ) from e
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self._infer_ms: list[float] = []
        self.last_score = 0.0

    def stats(self) -> dict:
        """Inference timing over the frames of the last wait_for_wake_word()."""
        times = self._infer_ms
        if not times:
            return {"frames": 0}
        return {
            "frames": len(times),
            "mean_ms": sum(times) / len(times),
            "max_ms": max(times),
            "budget_ms": FRAME_BUDGET_MS,
            "frames_over_budget": sum(1 for t in times if t > FRAME_BUDGET_MS),
        }

    def wait_for_wake_word(self, threshold: float | None = None) -> bool:
        """Blocks until the wake word is detected.

        Raises MicrophoneError if the input device cannot be opened or read,
        or AudioInputError if the model fails while scoring audio."""
        threshold = Config.WAKE_WORD_THRESHOLD if threshold is None else threshold
        # Clear audio history so the previous detection cannot retrigger.
        if hasattr(self.model, "reset"):
            self.model.reset()
        self._infer_ms = []
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=self.chunk_size
            ) as stream:
                while True:
                    audio_chunk, _ = stream.read(self.chunk_size)
                    t0 = time.perf_counter()
                    try:
                        predictions = self.model.predict(audio_chunk.flatten())
                    except Exception as e:
                        logger.exception("wake-word model failed")
                        raise AudioInputError(
                            f"Wake-word model failed ({type(e).__name__})."
                        ) from e
                    self._infer_ms.append((time.perf_counter() - t0) * 1000.0)
                    score = predictions.get(self.wake_word, 0.0)
                    if score >= threshold:
                        self.last_score = float(score)
                        return True
        except AudioInputError:
            raise
        except (sd.PortAudioError, OSError, ValueError) as e:
            raise MicrophoneError(
                f"Microphone unavailable ({type(e).__name__}: {e}). Check that a mic is "
                "connected, not in use by another app, and allowed in system privacy settings."
            ) from e
