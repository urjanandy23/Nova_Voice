"""Fake hardware / provider modules so src.* imports and runs anywhere (CI,
a laptop with no PortAudio, no API key, no network).

Import this module BEFORE importing anything from src. Tests then patch the
fake modules' attributes (e.g. src.stt.sd.InputStream) per test.
"""
import logging
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)  # expected-failure tests would otherwise print tracebacks


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod._nova_stub = True
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


class PortAudioError(Exception):
    pass


def install():
    # Always replace: the real sounddevice raises OSError at import when
    # PortAudio is missing, and real models must never load in unit tests.
    _module("sounddevice", PortAudioError=PortAudioError, InputStream=None, OutputStream=None)
    _module("faster_whisper", WhisperModel=None)
    oww_model = _module("openwakeword.model", Model=None)
    oww = _module("openwakeword", model=oww_model)
    oww.model = oww_model
    _module("edge_tts", Communicate=None)
    _module("pydub", AudioSegment=type("AudioSegment", (), {}))
    _module("anthropic", Anthropic=object)
    _module("ollama", Client=None)


if not getattr(sys.modules.get("sounddevice"), "_nova_stub", False):
    install()
