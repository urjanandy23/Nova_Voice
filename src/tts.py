"""Text-to-speech with a producer/consumer queue so audio for the FIRST
sentence starts playing while later sentences are still being synthesized
and while the LLM is still generating the rest of the answer.

This is what makes the assistant feel fast: instead of
  wait for full LLM answer -> synthesize whole thing -> play
we do
  LLM streams sentence 1 -> synth+play sentence 1  \
  LLM streams sentence 2 -> synth+play sentence 2   } happening concurrently
  LLM streams sentence 3 -> synth+play sentence 3  /
"""
import asyncio
import io
import queue
import threading

import edge_tts
from pydub import AudioSegment
from pydub.playback import play

from .config import Config


class StreamingTTS:
    def __init__(self, voice: str = None):
        self.voice = voice or Config.TTS_VOICE
        self._sentence_queue: "queue.Queue[str | None]" = queue.Queue()
        self._audio_queue: "queue.Queue[AudioSegment | None]" = queue.Queue()
        self._synth_thread = None
        self._play_thread = None

    # -- public API -----------------------------------------------------
    def start(self):
        """Starts the background synth + playback workers for one response."""
        self._sentence_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def say(self, sentence: str):
        """Call this as each sentence becomes available from the LLM stream."""
        sentence = sentence.strip()
        if sentence:
            self._sentence_queue.put(sentence)

    def finish(self):
        """Signals no more sentences are coming, then blocks until playback
        of everything queued so far has finished."""
        self._sentence_queue.put(None)
        self._synth_thread.join()
        self._play_thread.join()

    # -- internals --------------------------------------------------------
    def _synth_worker(self):
        """Pulls sentences off the queue, synthesizes them in order, and
        pushes the resulting audio onto the playback queue."""
        while True:
            sentence = self._sentence_queue.get()
            if sentence is None:
                self._audio_queue.put(None)
                return
            audio = asyncio.run(self._synthesize(sentence))
            self._audio_queue.put(audio)

    def _play_worker(self):
        while True:
            audio = self._audio_queue.get()
            if audio is None:
                return
            play(audio)

    async def _synthesize(self, text: str) -> AudioSegment:
        communicate = edge_tts.Communicate(text, self.voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return AudioSegment.from_file(buf, format="mp3")

    def speak_now(self, text: str):
        """Simple one-shot synth+play, used for out-of-band messages like
        reminder alerts, where there's no LLM stream to overlap with."""
        audio = asyncio.run(self._synthesize(text))
        play(audio)
