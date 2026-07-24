"""Entry point.

Usage:
    python main.py                 # wake-word mode (hands-free, voice only)
    python main.py --push-to-talk  # press Enter to talk (voice only)
    python main.py --hybrid        # EVERY turn: press Enter to speak, OR
                                    # type a message instead — your choice
    python main.py --text-only     # no microphone touched at all, type only
                                    # (Claude's reply is still spoken aloud)
"""
import argparse

from src.assistant import VoiceAssistant


def main():
    parser = argparse.ArgumentParser(description="Voice AI Assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Skip wake-word detection; press Enter to start listening instead.",
    )
    group.add_argument(
        "--hybrid",
        action="store_true",
        help="Each turn, either press Enter to speak or type a message instead.",
    )
    group.add_argument(
        "--text-only",
        action="store_true",
        help="No microphone needed: type your messages instead of speaking. "
             "Claude's reply is still streamed out loud through your speakers.",
    )
    args = parser.parse_args()

    if args.hybrid:
        mode = "hybrid"
    elif args.text_only:
        mode = "text_only"
    elif args.push_to_talk:
        mode = "push_to_talk"
    else:
        mode = "wake_word"

    assistant = VoiceAssistant(mode=mode)
    try:
        assistant.run_forever()
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
