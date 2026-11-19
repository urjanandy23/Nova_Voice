"""Energy-based voice activity logic, written as plain state machines so
it can be unit tested without a microphone."""
from __future__ import annotations
import statistics
from typing import Sequence


def calibrate_threshold(ambient_rms: Sequence[float], floor: float,
                        multiplier: float = 3.0) -> float:
    """Speech threshold = multiplier x median ambient noise, never below floor.

    Adapts to noisy rooms and hot/quiet microphones instead of a fixed
    magic number."""
    if not ambient_rms:
        return floor
    return max(floor, multiplier * statistics.median(ambient_rms))


class Endpointer:
    """Decides when the user has finished speaking.

    Feed one RMS value per audio block. update() returns None while
    recording should continue, else a reason: "end" (trailing silence after
    speech), "max" (hit max_duration) or "no_speech" (nothing heard).
    Speech only counts as started after min_speech_duration of continuous
    loud blocks, so a single click or cough does not start a recording.
    """

    EPS = 1e-9

    def __init__(self, threshold: float, silence_duration: float,
                 block_duration: float = 0.1, min_speech_duration: float = 0.25,
                 max_duration: float = 20.0, no_speech_timeout: float = 8.0,
                 already_speaking: bool = False) -> None:
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.block = block_duration
        self.min_speech = min_speech_duration
        self.max_duration = max_duration
        self.no_speech_timeout = no_speech_timeout
        self.started = already_speaking
        self.loud_run = 0.0
        self.silence_run = 0.0
        self.elapsed = 0.0

    def update(self, rms: float) -> str | None:
        self.elapsed += self.block
        if rms > self.threshold:
            self.loud_run += self.block
            if self.started:
                self.silence_run = 0.0
            elif self.loud_run >= self.min_speech - self.EPS:
                self.started = True
                self.silence_run = 0.0
        else:
            self.loud_run = 0.0
            if self.started:
                self.silence_run += self.block

        if self.started and self.silence_run >= self.silence_duration - self.EPS:
            return "end"
        if self.elapsed >= self.max_duration - self.EPS:
            return "max"
        if not self.started and self.elapsed >= self.no_speech_timeout - self.EPS:
            return "no_speech"
        return None


class SustainedSpeech:
    """True once the signal has stayed above threshold for `sustain`
    seconds. Used for barge-in so a brief noise does not cut the assistant off."""

    EPS = 1e-9

    def __init__(self, threshold: float, sustain: float, block_duration: float = 0.1) -> None:
        self.threshold = threshold
        self.sustain = sustain
        self.block = block_duration
        self.run = 0.0

    def update(self, rms: float) -> bool:
        self.run = self.run + self.block if rms > self.threshold else 0.0
        return self.run >= self.sustain - self.EPS
