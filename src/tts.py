"""Text-to-speech with a producer/consumer pipeline, plus interruption.

  LLM chunk 1 -> synth -> play      \
  LLM chunk 2 -> synth -> play       } overlapped in time
  LLM chunk 3 -> synth -> play      /

Playback goes through a sounddevice OutputStream written in small blocks, so
stop() cuts the voice off within ~50 ms (the old blocking player could not be
interrupted). That is what makes barge-in possible.
"""
from __future__ import annotations
import asyncio
import io
import logging
import queue
import threading
from typing import Any

import edge_tts
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from .config import Config
from .text_utils import clean_for_speech

logger = logging.getLogger(__name__)
PLAY_BLOCK_FRAMES = 1024


def _to_pcm(audio: AudioSegment) -> tuple[np.ndarray, int]:
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return samples.reshape(-1, audio.channels), audio.frame_rate


class StreamingTTS:
    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or Config.TTS_VOICE
        self.timeline: Any = None
        self._stop = threading.Event()
        self._sentence_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._oneshot_lock = threading.Lock()

    # -- public API -----------------------------------------------------
    def start(self, timeline: Any = None) -> None:
        """Starts the synth + playback workers for one response."""
        self.timeline = timeline
        self._stop = threading.Event()
        self._sentence_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def say(self, sentence: str) -> None:
        """Queue a chunk for speech (markdown and URLs are stripped)."""
        sentence = clean_for_speech(sentence)
        if sentence and not self._stop.is_set():
            self._sentence_queue.put(sentence)

    def finish(self) -> None:
        """No more text is coming; block until everything queued has been spoken."""
        self._sentence_queue.put(None)
        if self._synth_thread:
            self._synth_thread.join()
        if self._play_thread:
            self._play_thread.join()

    def stop(self) -> None:
        """Interrupt now: discard queued speech and silence the speaker."""
        self._stop.set()
        self._sentence_queue.put(None)  # wake the synth worker

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    # -- internals --------------------------------------------------------
    def _synth_worker(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            while not self._stop.is_set():
                sentence = self._sentence_queue.get()
                if sentence is None:
                    break
                try:
                    audio = loop.run_until_complete(self._synthesize(sentence))
                    self._audio_queue.put(audio)
                except Exception:
                    # One failed sentence must not stall or kill the response.
                    logger.exception("TTS synthesis failed for one sentence")
        finally:
            loop.close()
            self._audio_queue.put(None)

    def _play_worker(self) -> None:
        stream = None
        params = None
        try:
            while True:
                audio = self._audio_queue.get()
                if audio is None:
                    return
                if self._stop.is_set():
                    continue  # drain without playing
                pcm, rate = _to_pcm(audio)
                if stream is None or params != (rate, pcm.shape[1]):
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    params = (rate, pcm.shape[1])
                    stream = sd.OutputStream(samplerate=rate, channels=pcm.shape[1], dtype="int16")
                    stream.start()
                if self.timeline:
                    self.timeline.mark("first_audio")
                for i in range(0, len(pcm), PLAY_BLOCK_FRAMES):
                    if self._stop.is_set():
                        stream.abort()
                        stream.close()
                        stream = None
                        break
                    stream.write(pcm[i:i + PLAY_BLOCK_FRAMES])
        except Exception:
            logger.exception("TTS playback failed")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    async def _synthesize(self, text: str) -> AudioSegment:
        communicate = edge_tts.Communicate(text, self.voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return AudioSegment.from_file(buf, format="mp3")

    def speak_now(self, text: str) -> None:
        """One-shot synth + play for out-of-band messages (reminders)."""
        with self._oneshot_lock:
            try:
                audio = asyncio.run(self._synthesize(clean_for_speech(text)))
                pcm, rate = _to_pcm(audio)
                with sd.OutputStream(samplerate=rate, channels=pcm.shape[1], dtype="int16") as s:
                    for i in range(0, len(pcm), PLAY_BLOCK_FRAMES):
                        s.write(pcm[i:i + PLAY_BLOCK_FRAMES])
            except Exception:
                logger.exception("could not speak reminder")
