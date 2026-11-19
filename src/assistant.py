"""Ties wake word -> STT -> LLM (streaming + tools) -> streaming TTS into the
end-to-end voice assistant loop, with per-turn timing, error recovery and
optional barge-in.

Audio-input failures (no mic, Whisper error, wake-word model error) are caught
here: the turn is abandoned with a clear message, the loop retries, and after
MAX_AUDIO_FAILURES in a row the assistant degrades to a mode that still works
instead of crashing."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import Config, require_api_key
from .errors import AudioInputError, MicrophoneError, TranscriptionError
from .functions import start_reminder_watcher
from .metrics import Timeline
from .tts import StreamingTTS

logger = logging.getLogger(__name__)
MODES = ("wake_word", "push_to_talk", "hybrid", "text_only")
APOLOGY = "Sorry, something went wrong. Please try again."
MAX_AUDIO_FAILURES = 3
AUDIO_RETRY_SECONDS = 1.0


def _build_llm() -> Any:
    if Config.LLM_PROVIDER == "ollama":
        from .llm_ollama import OllamaVoiceLLM
        return OllamaVoiceLLM()
    from .llm import ClaudeVoiceLLM
    return ClaudeVoiceLLM()


class VoiceAssistant:
    def __init__(self, mode: str = "wake_word", barge_in: bool = False) -> None:
        require_api_key()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.barge_in = barge_in and mode != "text_only"
        if barge_in and mode == "text_only":
            print("(barge-in needs a microphone; ignored in --text-only mode)")
        self._audio_failures = 0

        self.stt = None
        if self.mode != "text_only":
            try:
                from .stt import SpeechToText
            except (ImportError, OSError) as e:  # e.g. PortAudio not installed
                print(f"Microphone support is unavailable ({type(e).__name__}: {e}).")
                self._fall_back("text_only")
            else:
                print("Loading speech-to-text model...")
                self.stt = SpeechToText()

        print("Loading text-to-speech...")
        self.tts = StreamingTTS()

        print(f"Connecting to LLM ({Config.LLM_PROVIDER})...")
        self.llm = _build_llm()
        self.memory = self.llm.new_memory()

        self.wake_word = None
        if self.mode == "wake_word":
            try:
                from .wake_word import WakeWordDetector
                print(f"Loading wake word model ('{Config.WAKE_WORD}')...")
                self.wake_word = WakeWordDetector()
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Wake word unavailable ({e}).")
                self._fall_back("push_to_talk")

        start_reminder_watcher(on_fire=self._speak_reminder)

    def _speak_reminder(self, text: str) -> None:
        print(f"\n[reminder] {text}")
        self.tts.speak_now(f"Reminder: {text}")

    # -- audio failure handling ---------------------------------------------
    def _fall_back(self, mode: str) -> None:
        print(f"Switching to {mode.replace('_', '-')} mode.")
        self.mode = mode
        if mode == "text_only":
            self.barge_in = False

    def _audio_failed(self, error: AudioInputError) -> None:
        """Report one audio-input failure; degrade after repeated ones."""
        self._audio_failures += 1
        print(f"\n[audio] {error}")
        if self._audio_failures == 1 and isinstance(error, TranscriptionError):
            self.tts.speak_now("Sorry, I couldn't make out what you said. Please try again.")
        if self._audio_failures >= MAX_AUDIO_FAILURES:
            self._audio_failures = 0
            if self.mode == "wake_word" and not isinstance(error, MicrophoneError):
                self._fall_back("push_to_talk")  # only the wake-word model is broken
            elif self.mode != "text_only":
                print("The microphone keeps failing.")
                self._fall_back("text_only")
        else:
            time.sleep(AUDIO_RETRY_SECONDS)

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        print("\nVoice assistant ready. Press Ctrl+C while it is talking to interrupt it.")
        preroll = None  # audio captured by barge-in, to start the next recording
        while True:
            if preroll is not None:
                print("\n(interrupted - listening...)")
                preroll = self.handle_one_exchange(preroll=preroll)
                continue

            if self.mode == "wake_word":
                print(f"\nSay the wake word ('{Config.WAKE_WORD.replace('_', ' ')}')...")
                try:
                    self.wake_word.wait_for_wake_word()
                except AudioInputError as e:
                    self._audio_failed(e)
                    continue
                wake_at = time.perf_counter()
                print("Wake word detected. Listening...")
                preroll = self.handle_one_exchange(
                    wake_at=wake_at, wake_stats=self.wake_word.stats())

            elif self.mode == "push_to_talk":
                input("\nPress Enter, then speak...")
                print("Listening...")
                preroll = self.handle_one_exchange()

            elif self.mode == "text_only":
                typed = input("\nYou (typed): ").strip()
                if typed:
                    self.handle_one_exchange(user_text=typed)

            elif self.mode == "hybrid":
                typed = input(
                    "\nPress Enter to speak, or type a message and press Enter: "
                ).strip()
                if typed:
                    self.handle_one_exchange(user_text=typed)
                else:
                    print("Listening...")
                    preroll = self.handle_one_exchange()

    def handle_one_exchange(self, user_text: str | None = None, preroll: Any = None,
                            wake_at: float | None = None,
                            wake_stats: dict | None = None) -> Any:
        """Runs one turn. Returns audio captured by barge-in if the user
        interrupted by voice (so the caller can record the interruption), else None."""
        timeline = Timeline()
        if wake_at is not None:
            timeline.mark("wake_detected", at=wake_at)
            timeline.wake_stats = wake_stats
        spoken_input = user_text is None
        if spoken_input:
            try:
                user_text = self.stt.listen_and_transcribe(preroll=preroll, timeline=timeline)
            except AudioInputError as e:
                self._audio_failed(e)
                return None
            self._audio_failures = 0
        if not user_text:
            print("(didn't catch anything)")
            return None

        if spoken_input:
            print(f"You said: {user_text}")
        self.memory.add_user(user_text)

        label = "Claude" if Config.LLM_PROVIDER == "anthropic" else "Assistant"
        print(f"{label}: ", end="", flush=True)

        cancel = threading.Event()
        self.tts.start(timeline)
        monitor = None
        if self.barge_in:
            from .barge_in import BargeInMonitor
            monitor = BargeInMonitor(self.tts, cancel)
            monitor.start()

        def on_sentence(sentence: str) -> None:
            if cancel.is_set():
                return
            print(sentence, end=" ", flush=True)
            self.tts.say(sentence)

        timeline.mark("llm_start")
        try:
            self.llm.stream_response(self.memory, on_sentence, cancel=cancel, timeline=timeline)
        except KeyboardInterrupt:
            cancel.set()
            self.tts.stop()
            print("\n(interrupted)")
        except Exception:
            logger.exception("turn failed")
            self.memory.pop_incomplete_turn()  # keep history valid for the next turn
            print(f"\n{APOLOGY}")
            self.tts.say(APOLOGY)
        finally:
            try:
                self.tts.finish()
            except KeyboardInterrupt:
                self.tts.stop()
            if monitor:
                monitor.stop()
            timeline.mark("done")

        print()
        if Config.SHOW_TIMINGS:
            print(timeline.format())
        return monitor.captured if monitor and monitor.triggered else None
