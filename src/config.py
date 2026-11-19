"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for tests / CI
    pass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")
    WAKE_WORD_THRESHOLD = _float("WAKE_WORD_THRESHOLD", 0.5)

    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
    TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewNeural")

    # --- audio capture / endpointing -------------------------------------
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SECONDS = 0.1
    # Speech threshold = NOISE_MULTIPLIER x ambient noise, but never below
    # MIN_SPEECH_RMS (int16 scale). Measured fresh at the start of each recording.
    MIN_SPEECH_RMS = _float("MIN_SPEECH_RMS", 300)
    NOISE_MULTIPLIER = _float("NOISE_MULTIPLIER", 3.0)
    CALIBRATION_BLOCKS = 3
    MIN_SPEECH_SECONDS = _float("MIN_SPEECH_SECONDS", 0.25)
    # Trailing silence that ends an utterance. Every turn waits this long
    # before the assistant even starts thinking, so keep it small.
    SILENCE_DURATION = _float("SILENCE_DURATION", 0.8)
    NO_SPEECH_TIMEOUT = _float("NO_SPEECH_TIMEOUT", 8.0)
    MAX_RECORD_SECONDS = _float("MAX_RECORD_SECONDS", 20)

    # --- latency ----------------------------------------------------------
    # If > 0, the first spoken chunk of a reply may be cut at a comma once it
    # has this many characters, instead of waiting for the first full sentence.
    FIRST_CHUNK_MIN_CHARS = _int("FIRST_CHUNK_MIN_CHARS", 40)
    SHOW_TIMINGS = _bool("SHOW_TIMINGS", True)

    # --- barge-in (interrupt the assistant by speaking) ---------------------
    # Fixed RMS threshold: the assistant's own voice leaks into the mic through
    # speakers, so use headphones, or raise this if it interrupts itself.
    BARGE_IN_RMS = _float("BARGE_IN_RMS", 1500)
    BARGE_IN_SUSTAIN_SECONDS = _float("BARGE_IN_SUSTAIN_SECONDS", 0.4)

    MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 12)

    REMINDERS_FILE = "reminders.json"
    SEARCH_MAX_RESULTS = _int("SEARCH_MAX_RESULTS", 3)
    SEARCH_CACHE_SECONDS = _int("SEARCH_CACHE_SECONDS", 300)


def require_api_key() -> None:
    if Config.LLM_PROVIDER != "anthropic":
        return
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama in .env to use a free local model instead."
        )
