"""Barge-in: lets the user interrupt the assistant by speaking.

While the assistant talks, a thread watches the mic. When speech stays above
BARGE_IN_RMS for BARGE_IN_SUSTAIN_SECONDS it silences the TTS, tells the LLM to
stop generating, and hands the audio it already heard to the recorder so the
first words of the interruption are not lost.

Limitation: there is no acoustic echo cancellation. With loudspeakers the
assistant's own voice can trigger this, so headphones are recommended, or raise
BARGE_IN_RMS. It is opt-in (--barge-in) for that reason.
"""
from __future__ import annotations
import collections
import logging
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from .config import Config
from .vad import SustainedSpeech

logger = logging.getLogger(__name__)

PREROLL_SECONDS = 1.0


class BargeInMonitor:
    def __init__(self, tts: Any, cancel: threading.Event) -> None:
        self.tts = tts
        self.cancel = cancel
        self.triggered = False
        self.captured: np.ndarray | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        sr = Config.SAMPLE_RATE
        block_s = Config.BLOCK_SECONDS
        block_size = int(sr * block_s)
        detector = SustainedSpeech(Config.BARGE_IN_RMS, Config.BARGE_IN_SUSTAIN_SECONDS, block_s)
        recent = collections.deque(maxlen=int(PREROLL_SECONDS / block_s))
        try:
            with sd.InputStream(samplerate=sr, channels=Config.CHANNELS, dtype="int16") as stream:
                while not self._done.is_set():
                    block, _ = stream.read(block_size)
                    block = block.flatten()
                    recent.append(block.copy())
                    rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                    if detector.update(rms):
                        self.captured = np.concatenate(list(recent))
                        self.triggered = True
                        self.cancel.set()
                        self.tts.stop()
                        return
        except Exception as e:
            # No mic / device busy: barge-in simply stays unavailable this turn.
            logger.warning("barge-in unavailable this turn: %s: %s", type(e).__name__, e)
            return
