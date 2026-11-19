"""Ollama backend: no API key and no cost. The shared loop is in llm_base."""
from __future__ import annotations

from typing import Any, Callable

import ollama

from .config import Config
from .functions import TOOLS
from .llm_base import MAX_TOOL_ROUNDS, BaseVoiceLLM, Round, ToolCall  # noqa: F401
from .memory import OllamaMemory
from .prompts import SYSTEM_PROMPT


def _to_ollama_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OllamaVoiceLLM(BaseVoiceLLM):
    memory_class = OllamaMemory

    def __init__(self) -> None:
        self.model = Config.OLLAMA_MODEL
        self.tools = _to_ollama_tools(TOOLS)
        self.client = ollama.Client(host=Config.OLLAMA_HOST)

    def _stream_round(self, memory: OllamaMemory, feed: Callable[[str], None],
                      cancel: Any, timeline: Any) -> Round:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.as_list()
        full_text = ""
        tool_calls_seen = None

        stream = self.client.chat(
            model=self.model, messages=messages, tools=self.tools, stream=True
        )
        for chunk in stream:
            msg = _get(chunk, "message")
            piece = _get(msg, "content", "") or ""
            if piece:
                if timeline:
                    timeline.mark("first_token")
                if cancel is not None and cancel.is_set():
                    return Round(cancelled=True)
                full_text += piece
                feed(piece)
            calls = _get(msg, "tool_calls")
            if calls:
                tool_calls_seen = calls

        if not tool_calls_seen:
            return Round(result=full_text, raw=full_text)

        parsed = []
        for call in tool_calls_seen:
            fn = _get(call, "function")
            name = _get(fn, "name")
            parsed.append(ToolCall(_get(call, "id", name) or name, name,
                                   _get(fn, "arguments") or {}))
        return Round(tool_calls=parsed, result=full_text, raw=tool_calls_seen)

    def _append_assistant(self, memory: OllamaMemory, rnd: Round) -> None:
        if rnd.tool_calls:
            memory.add_assistant_tool_calls(rnd.raw)
        else:
            memory.add_assistant_text(rnd.raw)

    def _append_tool_results(self, memory: OllamaMemory,
                             results: list[tuple[ToolCall, str]]) -> None:
        for call, text in results:
            memory.add_tool_message(call.id, call.name, text)
