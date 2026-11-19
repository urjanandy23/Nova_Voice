"""Per-turn timing, so latency claims come from measurements."""
from __future__ import annotations

import math
import time
from typing import Iterable


class Timeline:
    """Records the first time each named event happens in one turn.

    Marks used by the assistant: wake_detected, listening, speech_last,
    speech_end, transcribed, llm_start, first_token, first_sentence,
    first_audio, done.
    """

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self.tool_calls = 0
        self.wake_stats: dict | None = None  # filled from WakeWordDetector.stats()

    def mark(self, name: str, at: float | None = None) -> None:
        """Record `name` once. `at` is a time.perf_counter() value captured
        earlier (e.g. the instant of the last loud audio block)."""
        self.marks.setdefault(name, time.perf_counter() if at is None else at)

    def between(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return self.marks[b] - self.marks[a]
        return None

    def stages(self) -> dict[str, float | None]:
        return {
            # wake word returned -> mic stream open and first block read.
            # Speech in this gap is not captured.
            "wake_to_listen_s": self.between("wake_detected", "listening"),
            # last loud block -> endpointer decided you were done: the silence
            # wait (SILENCE_DURATION) that every spoken turn pays.
            "endpoint_wait_s": self.between("speech_last", "speech_end"),
            "stt_s": self.between("speech_end", "transcribed"),
            "llm_first_token_s": self.between("llm_start", "first_token"),
            "llm_first_sentence_s": self.between("llm_start", "first_sentence"),
            "tts_first_audio_s": self.between("first_sentence", "first_audio"),
            "heard_to_first_audio_s": self.between("speech_end", "first_audio"),
            # What the user perceives: they stop talking -> they hear a reply.
            "perceived_s": self.between("speech_last", "first_audio"),
            "total_s": self.between("llm_start", "done"),
        }

    def format(self) -> str:
        parts = []
        for key, val in self.stages().items():
            if val is not None:
                parts.append(f"{key[:-2]}={val:.2f}s")
        if self.tool_calls:
            parts.append(f"tool_calls={self.tool_calls}")
        if self.wake_stats and self.wake_stats.get("frames"):
            s = self.wake_stats
            parts.append(f"wake_infer_mean={s['mean_ms']:.1f}ms "
                         f"wake_infer_max={s['max_ms']:.1f}ms "
                         f"(budget {s['budget_ms']:.0f}ms/frame)")
        return "[timing] " + "  ".join(parts)


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile; returns None for an empty list."""
    values = list(values)
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]
