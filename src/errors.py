"""Exceptions for the audio-input side of the pipeline.

The assistant loop catches these (and only these) around the microphone, the
wake-word listener and speech-to-text, so a missing mic or a Whisper failure
turns into a clear message and a retry instead of a crash.
"""
from __future__ import annotations


class AudioInputError(Exception):
    """Base class for failures while capturing or interpreting audio."""


class MicrophoneError(AudioInputError):
    """No usable input device, device busy/unplugged, or PortAudio missing."""


class TranscriptionError(AudioInputError):
    """The speech-to-text model failed on a recording."""
