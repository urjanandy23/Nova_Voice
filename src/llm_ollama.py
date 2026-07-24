"""Talks to a locally-running Ollama model instead of Claude — same idea as
llm.py (streaming + sentence chunking + tool use), just pointed at your own
computer instead of Anthropic's API. No API key, no cost, ever.
"""
import ollama

from .config import Config
from .functions import TOOLS, dispatch_tool
from .llm import SYSTEM_PROMPT, split_complete_sentences


def _to_ollama_tools(tools):
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return converted


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OllamaVoiceLLM:
    def __init__(self):
        self.model = Config.OLLAMA_MODEL
        self.tools = _to_ollama_tools(TOOLS)
        self.client = ollama.Client(host=Config.OLLAMA_HOST)

    def stream_response(self, memory, on_sentence):
        while True:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.as_list()

            buffer = ""
            full_text = ""
            tool_calls_seen = None

            stream = self.client.chat(
                model=self.model,
                messages=messages,
                tools=self.tools,
                stream=True,
            )
            for chunk in stream:
                msg = _get(chunk, "message")
                piece = _get(msg, "content", "") or ""
                if piece:
                    buffer += piece
                    full_text += piece
                    sentences, buffer = split_complete_sentences(buffer)
                    for s in sentences:
                        on_sentence(s)

                calls = _get(msg, "tool_calls")
                if calls:
                    tool_calls_seen = calls

            if buffer.strip():
                on_sentence(buffer.strip())

            if not tool_calls_seen:
                memory.add_assistant_text(full_text)
                return full_text

            memory.add_assistant_tool_calls(tool_calls_seen)
            for call in tool_calls_seen:
                fn = _get(call, "function")
                name = _get(fn, "name")
                args = _get(fn, "arguments") or {}
                result = dispatch_tool(name, args)
                call_id = _get(call, "id", name)
                memory.add_tool_message(call_id, name, result)