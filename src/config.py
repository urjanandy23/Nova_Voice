"""Central configuration, loaded from environment / .env file."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewNeural")

    SAMPLE_RATE = 16000
    CHANNELS = 1
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 1.2
    MAX_RECORD_SECONDS = 20

    MAX_HISTORY_TURNS = 12

    REMINDERS_FILE = "reminders.json"


def require_api_key():
    if Config.LLM_PROVIDER != "anthropic":
        return
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama in .env to use a free local model instead."
        )