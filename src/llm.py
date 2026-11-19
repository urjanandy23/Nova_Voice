"""Claude backend: streaming + tool use. The shared loop is in llm_base."""
from __future__ import annotations

from typing import Any, Callable

from anthropic import Anthropic

from .config import Config
from .functions import TOOLS
from .llm_base import MAX_TOOL_ROUNDS, BaseVoiceLLM, Round, ToolCall  # noqa: F401
from .memory import AnthropicMemory
from .prompts import SYSTEM_PROMPT


class ClaudeVoiceLLM(BaseVoiceLLM):
    memory_class = AnthropicMemory

    def __init__(self) -> None:
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY, max_retries=2, timeout=30.0)

    def _stream_round(self, memory: AnthropicMemory, feed: Callable[[str], None],
                      cancel: Any, timeline: Any) -> Round:
        with self.client.messages.stream(
            model=Config.CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=memory.as_list(),
            tools=TOOLS,
        ) as stream:
            for text in stream.text_stream:
                if timeline:
                    timeline.mark("first_token")
                if cancel is not None and cancel.is_set():
                    return Round(cancelled=True)
                feed(text)
            final = stream.get_final_message()

        calls: list[ToolCall] = []
        if final.stop_reason == "tool_use":
            calls = [ToolCall(b.id, b.name, b.input)
                     for b in final.content if b.type == "tool_use"]
        return Round(tool_calls=calls, result=final, raw=final.content)

    def _append_assistant(self, memory: AnthropicMemory, rnd: Round) -> None:
        memory.add_assistant(rnd.raw)

    def _append_tool_results(self, memory: AnthropicMemory,
                             results: list[tuple[ToolCall, str]]) -> None:
        memory.add_tool_results([(call.id, text) for call, text in results])
