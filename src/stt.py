"""Speech-to-text: records one utterance (ending on trailing silence, with a
noise threshold measured per recording) and transcribes it locally with
faster-whisper.

Device and model failures are raised as MicrophoneError / TranscriptionError
(see errors.py) so the assistant loop can recover instead of crashing."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import sounddevice as sd

from .config import Config
from .errors import MicrophoneError, TranscriptionError
from .vad import Endpointer, calibrate_threshold

logger = logging.getLogger(__name__)

# Silence kept after the last speech so Whisper does not clip the final word.
TRAILING_KEEP_SECONDS = 0.3


def block_rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


class SpeechToText:
    def __init__(self) -> None:
        try:
            from faster_whisper import WhisperModel
            # int8 keeps this fast enough for CPU-only laptops
            self.model = WhisperModel(Config.WHISPER_MODEL_SIZE, compute_type="int8")
        except Exception as e:
            raise RuntimeError(
                f"Could not load the Whisper model '{Config.WHISPER_MODEL_SIZE}' "
                f"({type(e).__name__}: {e}). The first run downloads it, so it needs "
                "internet access; check the connection or try WHISPER_MODEL_SIZE=tiny."
            ) from e

    def record_utterance(self, preroll: np.ndarray | None = None,
                         timeline: Any = None) -> np.ndarray:
        """Records from the default mic until the user stops talking.

        preroll: audio already captured (e.g. by barge-in) that is prepended,
        and recording starts in the "already speaking" state.
        Returns int16 samples, or an empty array if nothing was heard.
        Raises MicrophoneError if the input device cannot be opened or read.
        """
        sr = Config.SAMPLE_RATE
        block_s = Config.BLOCK_SECONDS
        block_size = int(sr * block_s)

        frames = [] if preroll is None else [preroll.astype(np.int16).flatten()]
        endpointer = None
        ambient: list[float] = []
        reason = None
        last_loud_t = None  # perf_counter() of the last block above threshold

        try:
            with sd.InputStream(samplerate=sr, channels=Config.CHANNELS, dtype="int16") as stream:
                while reason is None:
                    block, _ = stream.read(block_size)
                    now = time.perf_counter()
                    if timeline:
                        timeline.mark("listening")  # first block read = mic is live
                    block = block.flatten()
                    frames.append(block.copy())
                    rms = block_rms(block)

                    if endpointer is None:
                        if preroll is None:
                            ambient.append(rms)
                            if len(ambient) < Config.CALIBRATION_BLOCKS:
                                continue
                            threshold = calibrate_threshold(
                                ambient, Config.MIN_SPEECH_RMS, Config.NOISE_MULTIPLIER
                            )
                        else:
                            threshold = Config.MIN_SPEECH_RMS
                        endpointer = Endpointer(
                            threshold,
                            Config.SILENCE_DURATION,
                            block_duration=block_s,
                            min_speech_duration=Config.MIN_SPEECH_SECONDS,
                            max_duration=Config.MAX_RECORD_SECONDS,
                            no_speech_timeout=Config.NO_SPEECH_TIMEOUT,
                            already_speaking=preroll is not None,
                        )
                        # the calibration blocks were consumed as noise samples
                        continue
                    reason = endpointer.update(rms)
                    if endpointer.started and rms > endpointer.threshold:
                        last_loud_t = now
        except (sd.PortAudioError, OSError, ValueError) as e:
            raise MicrophoneError(
                f"Microphone unavailable ({type(e).__name__}: {e}). Check that a mic is "
                "connected, not in use by another app, and allowed in system privacy settings."
            ) from e

        if timeline:
            if last_loud_t is not None:
                timeline.mark("speech_last", at=last_loud_t)
            timeline.mark("speech_end")
        if reason == "no_speech" or not frames:
            return np.array([], dtype=np.int16)

        # Drop most of the trailing silence: Whisper time scales with audio length.
        extra_blocks = int((endpointer.silence_run - TRAILING_KEEP_SECONDS) / block_s)
        audio = np.concatenate(frames)
        if extra_blocks > 0:
            audio = audio[: len(audio) - extra_blocks * block_size]
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Returns the transcript ('' for empty audio). Raises TranscriptionError
        if Whisper fails."""
        if audio.size == 0:
            return ""
        audio_f32 = audio.astype(np.float32) / 32768.0  # faster-whisper wants [-1, 1]
        try:
            segments, _ = self.model.transcribe(
                audio_f32,
                language=Config.WHISPER_LANGUAGE,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
            )
            # `segments` is a lazy generator: decoding happens while iterating.
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            logger.exception("Whisper transcription failed")
            raise TranscriptionError(f"Speech recognition failed ({type(e).__name__}).") from e

    def listen_and_transcribe(self, preroll: np.ndarray | None = None,
                              timeline: Any = None) -> str:
        audio = self.record_utterance(preroll=preroll, timeline=timeline)
        text = self.transcribe(audio)
        if timeline:
            timeline.mark("transcribed")
        return text
