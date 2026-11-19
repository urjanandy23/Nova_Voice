"""The streaming + tool loop shared by every LLM backend.

A backend only says how to run ONE streamed round against its API and how to
write that round (and the tool results) into its own memory format. Chunking,
cancellation, the tool-round limit and tool dispatch live here, once.

Kept free of provider imports so it is unit-testable anywhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from .functions import dispatch_tool
from .memory import ConversationMemory
from .text_utils import SentenceChunker

MAX_TOOL_ROUNDS = 5
GIVE_UP = "Sorry, I couldn't finish that request."


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class Round:
    """Outcome of one streamed model call."""
    cancelled: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    result: Any = None  # what stream_response returns if this round is the last
    raw: Any = None     # provider-native payload to store in memory


class BaseVoiceLLM(ABC):
    memory_class: type[ConversationMemory] = ConversationMemory

    def new_memory(self, max_turns: int | None = None) -> ConversationMemory:
        return self.memory_class(max_turns)

    @abstractmethod
    def _stream_round(self, memory: ConversationMemory, feed: Callable[[str], None],
                      cancel: Any, timeline: Any) -> Round:
        """Run one model call. Call feed(text) for every streamed text piece
        (after marking first_token); stop and return Round(cancelled=True) as
        soon as `cancel` is set."""

    @abstractmethod
    def _append_assistant(self, memory: ConversationMemory, rnd: Round) -> None:
        """Store the assistant message of this round in memory."""

    @abstractmethod
    def _append_tool_results(self, memory: ConversationMemory,
                             results: list[tuple[ToolCall, str]]) -> None:
        """Store the tool results for this round's tool calls in memory."""

    def stream_response(self, memory: ConversationMemory, on_sentence: Callable[[str], None],
                        cancel: Any = None, timeline: Any = None) -> Any:
        """Streams one assistant turn, including tool round trips.

        on_sentence(text) is called per speakable chunk. If `cancel`
        (threading.Event) is set, generation stops and the text spoken so far
        is kept in memory. Returns the backend's final result, or None if
        cancelled or the tool-round limit was hit.
        """
        from .config import Config

        chunker = SentenceChunker(Config.FIRST_CHUNK_MIN_CHARS)
        spoken: list[str] = []

        def emit(chunk: str) -> None:
            spoken.append(chunk)
            if timeline:
                timeline.mark("first_sentence")
            on_sentence(chunk)

        def feed(text: str) -> None:
            for chunk in chunker.feed(text):
                emit(chunk)

        for _ in range(MAX_TOOL_ROUNDS):
            rnd = self._stream_round(memory, feed, cancel, timeline)
            if rnd.cancelled:
                if spoken:
                    memory.add_assistant_text(" ".join(spoken))
                return None

            for chunk in chunker.flush():
                emit(chunk)
            self._append_assistant(memory, rnd)
            if not rnd.tool_calls:
                return rnd.result

            results = []
            for call in rnd.tool_calls:
                if timeline:
                    timeline.tool_calls += 1
                results.append((call, dispatch_tool(call.name, call.args)))
            self._append_tool_results(memory, results)

        emit(GIVE_UP)
        return None
