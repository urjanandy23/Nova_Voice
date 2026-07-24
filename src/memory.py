"""Keeps a rolling window of conversation turns so the assistant has context
across multiple exchanges, without letting the prompt grow unbounded."""
from .config import Config


class ConversationMemory:
    def __init__(self, max_turns: int = Config.MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.messages = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, content):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_result(self, tool_use_id: str, result: str):
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ],
        })
        self._trim()

    def add_assistant_text(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def add_assistant_tool_calls(self, tool_calls):
        self.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        self._trim()

    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })
        self._trim()

    def _trim(self):
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def as_list(self):
        return self.messages

    def reset(self):
        self.messages = []
        