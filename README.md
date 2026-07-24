# Nova Voice

A hands-free voice assistant: wake word → speech-to-text → Claude (with
tool use) → streaming text-to-speech. Built to minimize perceived latency —
the assistant starts speaking the first sentence of its answer while later
sentences are still being generated and synthesized.

## How it works

```
  mic audio                                              speaker audio
      │                                                        ▲
      ▼                                                        │
┌─────────────┐    ┌───────────┐    ┌────────────────┐   ┌──────────┐
│ Wake word   │───▶│ Whisper   │───▶│ Claude          │──▶│ Edge-TTS │
│ (openwake-  │    │ (STT,     │    │ (streaming +    │   │ (stream, │
│  word)      │    │ local)    │    │  tool use)      │   │  queued) │
└─────────────┘    └───────────┘    └────────┬────────┘   └──────────┘
                                              │
                                     ┌────────▼────────┐
                                     │ Tools:          │
                                     │ set_reminder    │
                                     │ list_reminders  │
                                     │ search_web      │
                                     └─────────────────┘
```

**The latency trick:** as Claude streams tokens, `src/llm.py` cuts the
running text into complete sentences the moment it sees terminal
punctuation (`.`/`!`/`?`). Each sentence is immediately handed to
`src/tts.py`, which runs a synth thread and a playback thread over a
queue — so sentence 1 is already playing through the speakers while
sentence 2 is being synthesized and sentence 3 is still being generated
by the model. The user hears a response in well under a second instead
of waiting for the entire answer to finish generating first.

**Tool use / function calling:** the assistant can set reminders (checked
by a background thread and spoken aloud when due, even mid-conversation)
and search the web via DuckDuckGo, using Claude's native tool-use API.

**Context:** `src/memory.py` keeps a rolling window of the last N turns
so you can have a real back-and-forth ("remind me to call mom in 20
minutes" → "actually make it 30").

## Project structure

```
voice-ai-assistant/
├── main.py                # entry point / CLI
├── src/
│   ├── config.py           # env vars & tunables
│   ├── wake_word.py        # openwakeword listener
│   ├── stt.py               # mic recording + faster-whisper transcription
│   ├── llm.py                # Claude streaming + sentence chunking + tool loop
│   ├── tts.py                # edge-tts streaming synth/playback queue
│   ├── functions.py         # tool schemas + implementations (reminders, search)
│   ├── memory.py             # conversation history window
│   └── assistant.py          # orchestrates everything above
├── tests/
│   └── test_functions.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

1. **Clone and enter the repo, create a virtualenv:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **System audio dependencies** (needed by `sounddevice` / `pydub`):
   - macOS: `brew install portaudio ffmpeg`
   - Ubuntu/Debian: `sudo apt install portaudio19-dev ffmpeg`
   - Windows: install [ffmpeg](https://ffmpeg.org/download.html) and add it to PATH; PortAudio ships with the `sounddevice` wheel.

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
   ```
   Get a key at https://console.anthropic.com.

5. **First run** will download the faster-whisper model (~150MB for
   `base`) and the openwakeword model — this only happens once.

## Running it

**Hands-free, wake-word mode** (say "hey jarvis" to activate):
```bash
python main.py
```

**Push-to-talk mode** (no wake word needed — good for noisy environments
or if your mic doesn't reliably pick up the wake word):
```bash
python main.py --push-to-talk
```

Then just talk naturally:
- *"What's the capital of Portugal?"*
- *"Remind me to take the laundry out in 15 minutes."*
- *"What are the pending reminders?"*
- *"Search the web for today's top tech news."*

## Configuration

All tunables live in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | your Claude API key |
| `CLAUDE_MODEL` | which Claude model to use |
| `WAKE_WORD` | `hey_jarvis`, `alexa`, or `hey_mycroft` (openwakeword's bundled models) |
| `WHISPER_MODEL_SIZE` | `tiny`/`base`/`small`/`medium`/`large-v3` — bigger = more accurate, slower |
| `TTS_VOICE` | any edge-tts voice; run `edge-tts --list-voices` to see all options |

## Running tests

The pure-logic pieces (sentence chunking, reminder storage) are covered
by unit tests that don't need a mic or an API key:
```bash
python -m unittest tests/test_functions.py -v
```

## Extending it

- **Swap TTS for higher quality:** replace `src/tts.py`'s `_synthesize`
  with a call to ElevenLabs' streaming API for more natural voices —
  the queue-based architecture doesn't need to change.
- **Add a new tool:** add a schema to `TOOLS` in `src/functions.py`, add
  the implementation, and wire it into `dispatch_tool` — Claude will
  start calling it automatically.
- **Swap STT/LLM providers:** `stt.py` and `llm.py` are the only files
  that talk to Whisper/Claude directly, so either can be swapped out
  independently.

## License

MIT
