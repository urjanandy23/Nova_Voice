"""Ties wake word -> STT -> LLM (streaming + tools) -> streaming TTS
together into the end-to-end voice assistant loop.
"""
from .config import Config, require_api_key
from .functions import start_reminder_watcher
from .memory import ConversationMemory
from .stt import SpeechToText
from .tts import StreamingTTS
from .wake_word import WakeWordDetector


def _build_llm():
    if Config.LLM_PROVIDER == "ollama":
        from .llm_ollama import OllamaVoiceLLM
        return OllamaVoiceLLM()
    from .llm import ClaudeVoiceLLM
    return ClaudeVoiceLLM()


class VoiceAssistant:
    def __init__(self, mode: str = "wake_word"):
        require_api_key()
        assert mode in ("wake_word", "push_to_talk", "hybrid", "text_only")
        self.mode = mode

        if self.mode != "text_only":
            print("Loading speech-to-text model...")
            self.stt = SpeechToText()
        else:
            self.stt = None

        print("Loading text-to-speech...")
        self.tts = StreamingTTS()

        print(f"Connecting to LLM ({Config.LLM_PROVIDER})...")
        self.llm = _build_llm()

        self.memory = ConversationMemory()

        if self.mode == "wake_word":
            print(f"Loading wake word model ('{Config.WAKE_WORD}')...")
            self.wake_word = WakeWordDetector()
        else:
            self.wake_word = None

        start_reminder_watcher(on_fire=self._speak_reminder)

    def _speak_reminder(self, text: str):
        print(f"\n[reminder] {text}")
        self.tts.speak_now(f"Reminder: {text}")

    def run_forever(self):
        print("\nVoice assistant ready.")
        while True:
            if self.mode == "wake_word":
                print(f"\nSay the wake word ('{Config.WAKE_WORD.replace('_', ' ')}')...")
                self.wake_word.wait_for_wake_word()
                print("Wake word detected. Listening...")
                self.handle_one_exchange()

            elif self.mode == "push_to_talk":
                input("\nPress Enter, then speak...")
                print("Listening...")
                self.handle_one_exchange()

            elif self.mode == "text_only":
                typed = input("\nYou (typed): ").strip()
                if not typed:
                    continue
                self.handle_one_exchange(user_text=typed)

            elif self.mode == "hybrid":
                typed = input(
                    "\nPress Enter to speak, or type a message and press Enter: "
                ).strip()
                if typed:
                    self.handle_one_exchange(user_text=typed)
                else:
                    print("Listening...")
                    self.handle_one_exchange()

    def handle_one_exchange(self, user_text: str = None):
        spoken_input = user_text is None
        if spoken_input:
            user_text = self.stt.listen_and_transcribe()
        if not user_text:
            print("(didn't catch anything)")
            return

        if spoken_input:
            print(f"You said: {user_text}")
        self.memory.add_user(user_text)

        label = "Claude" if Config.LLM_PROVIDER == "anthropic" else "Assistant"
        print(f"{label}: ", end="", flush=True)
        self.tts.start()

        def on_sentence(sentence: str):
            print(sentence, end=" ", flush=True)
            self.tts.say(sentence)

        self.llm.stream_response(self.memory, on_sentence)
        self.tts.finish()
        print()