"""Talks to Claude with streaming + tool use, and breaks the streamed text
into sentences as they complete so the caller (assistant.py) can start
speaking sentence 1 before sentence 2 has even been generated.
"""
import re

from anthropic import Anthropic

from .config import Config
from .functions import TOOLS, dispatch_tool

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Your replies are spoken out "
    "loud, so keep them short and conversational — avoid bullet points, "
    "markdown, or long lists. Get to the point in one or two sentences unless "
    "the user asks for more detail. Use the available tools (set_reminder, "
    "list_reminders, search_web) whenever they help answer the request."
)

# Matches end-of-sentence punctuation followed by whitespace (or end of string
# handled separately). Good enough for spoken-style text; not meant to be a
# perfect sentence tokenizer.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def split_complete_sentences(buffer: str):
    """Given a running text buffer, returns (list_of_complete_sentences,
    remaining_incomplete_tail)."""
    parts = _SENTENCE_END_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    # Everything except the last part is a complete sentence; the last part
    # might still be mid-sentence, so keep it in the buffer.
    complete, tail = parts[:-1], parts[-1]
    return complete, tail


class ClaudeVoiceLLM:
    def __init__(self):
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def stream_response(self, memory, on_sentence):
        """Streams a full assistant turn (including any tool-use round trips)
        against the given ConversationMemory, calling on_sentence(text) for
        each sentence as soon as it's complete. Mutates `memory` in place
        with the assistant's turns and tool results.
        """
        while True:
            buffer = ""
            with self.client.messages.stream(
                model=Config.CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=memory.as_list(),
                tools=TOOLS,
            ) as stream:
                for text in stream.text_stream:
                    buffer += text
                    sentences, buffer = split_complete_sentences(buffer)
                    for s in sentences:
                        on_sentence(s)
                final_message = stream.get_final_message()

            # Speak whatever text was left in the buffer for this turn.
            if buffer.strip():
                on_sentence(buffer.strip())

            memory.add_assistant(final_message.content)

            if final_message.stop_reason != "tool_use":
                return final_message

            # Run every requested tool call, feed results back, and loop
            # again so Claude can produce its final spoken answer.
            for block in final_message.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    memory.add_tool_result(block.id, result)
