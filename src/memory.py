"""Rolling conversation window.

Trimming only ever cuts at the start of a user turn, so a tool_use message
is never separated from its tool_result (the Anthropic API rejects that).

``ConversationMemory`` holds only what every backend shares. Each backend uses
the subclass that matches its own wire format, so neither format leaks into
the other:

* ``AnthropicMemory``: tool_use / tool_result content blocks
* ``OllamaMemory``: ``tool_calls`` on the assistant message + ``role: tool``
"""
from __future__ import annotations

from typing import Any

from .config import Config


def _is_turn_start(message: dict) -> bool:
    """A real user utterance (plain string), as opposed to a tool result."""
    return message.get("role") == "user" and isinstance(message.get("content"), str)


class ConversationMemory:
    def __init__(self, max_turns: int | None = None) -> None:
        self.max_turns = max_turns if max_turns is not None else Config.MAX_HISTORY_TURNS
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def pop_incomplete_turn(self) -> None:
        """Drop the most recent user turn and everything after it. Used when a
        turn fails midway so history stays valid."""
        for i in range(len(self.messages) - 1, -1, -1):
            if _is_turn_start(self.messages[i]):
                self.messages = self.messages[:i]
                return

    def _trim(self) -> None:
        starts = [i for i, m in enumerate(self.messages) if _is_turn_start(m)]
        if len(starts) > self.max_turns:
            self.messages = self.messages[starts[-self.max_turns]:]

    def as_list(self) -> list[dict]:
        return self.messages

    def reset(self) -> None:
        self.messages = []


class AnthropicMemory(ConversationMemory):
    def add_assistant(self, content: Any) -> None:
        """content: the list of content blocks from an Anthropic message."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        """results: list of (tool_use_id, text). All results for one assistant
        message go back in a single user message."""
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": text}
                for tid, text in results
            ],
        })


class OllamaMemory(ConversationMemory):
    def add_assistant_tool_calls(self, tool_calls: Any) -> None:
        self.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

    def add_tool_message(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content,
        })
