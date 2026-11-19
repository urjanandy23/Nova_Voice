"""Entry point.

Usage:
    python main.py                 # wake-word mode (hands-free, voice only)
    python main.py --push-to-talk  # press Enter to talk (voice only)
    python main.py --hybrid        # each turn: press Enter to speak OR type
    python main.py --text-only     # no microphone; type, reply is still spoken
    add --barge-in                 # interrupt the assistant by speaking
                                   # (use headphones; see README)
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova Voice assistant")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--push-to-talk", action="store_true",
                       help="Skip wake-word detection; press Enter to start listening.")
    group.add_argument("--hybrid", action="store_true",
                       help="Each turn, press Enter to speak or type a message instead.")
    group.add_argument("--text-only", action="store_true",
                       help="No microphone: type your messages. Replies are still spoken.")
    parser.add_argument("--barge-in", action="store_true",
                        help="Let your voice interrupt the assistant mid-reply "
                             "(headphones recommended: no echo cancellation).")
    args = parser.parse_args()

    if args.hybrid:
        mode = "hybrid"
    elif args.text_only:
        mode = "text_only"
    elif args.push_to_talk:
        mode = "push_to_talk"
    else:
        mode = "wake_word"

    try:
        from src.assistant import VoiceAssistant
    except OSError as e:  # sounddevice raises this when PortAudio is missing
        sys.exit(f"Audio libraries could not be loaded ({e}).\n"
                 "Install PortAudio and ffmpeg first (see README > Setup).")

    try:
        assistant = VoiceAssistant(mode=mode, barge_in=args.barge_in)
        assistant.run_forever()
    except (KeyboardInterrupt, EOFError):  # Ctrl+C, or stdin closed (Ctrl+D)
        print("\nGoodbye!")
    except RuntimeError as e:  # missing API key, model download failed, ...
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
