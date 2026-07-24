"""Real-world actions the assistant can take via Claude's tool use.

Each tool has:
  - a JSON schema (TOOLS) describing it to Claude
  - a Python implementation (dispatch_tool) that actually runs it
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta

from .config import Config

_reminder_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic "tools" format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "set_reminder",
        "description": (
            "Set a reminder that will alert the user after a given number of "
            "minutes. Use this whenever the user asks to be reminded of something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to remind the user about.",
                },
                "minutes_from_now": {
                    "type": "number",
                    "description": "How many minutes from now the reminder should fire.",
                },
            },
            "required": ["text", "minutes_from_now"],
        },
    },
    {
        "name": "list_reminders",
        "description": "List all currently pending (not-yet-fired) reminders.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for current information, e.g. news, facts, prices, "
            "or anything that may have changed after the assistant's training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Reminders: persisted to a small JSON file, checked by a background thread
# ---------------------------------------------------------------------------
def _load_reminders():
    if not os.path.exists(Config.REMINDERS_FILE):
        return []
    with open(Config.REMINDERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_reminders(reminders):
    with open(Config.REMINDERS_FILE, "w") as f:
        json.dump(reminders, f, indent=2)


def set_reminder(text: str, minutes_from_now: float) -> str:
    fire_at = (datetime.now() + timedelta(minutes=minutes_from_now)).isoformat()
    with _reminder_lock:
        reminders = _load_reminders()
        reminders.append({"text": text, "fire_at": fire_at, "fired": False})
        _save_reminders(reminders)
    return f"Reminder set for {minutes_from_now:g} minutes from now: '{text}'."


def list_reminders() -> str:
    with _reminder_lock:
        reminders = [r for r in _load_reminders() if not r["fired"]]
    if not reminders:
        return "No pending reminders."
    lines = [f"- {r['text']} at {r['fire_at']}" for r in reminders]
    return "Pending reminders:\n" + "\n".join(lines)


def start_reminder_watcher(on_fire):
    """Background thread that checks every few seconds for due reminders
    and calls on_fire(text) for each one — e.g. to speak it out loud."""

    def _loop():
        while True:
            time.sleep(5)
            with _reminder_lock:
                reminders = _load_reminders()
                changed = False
                for r in reminders:
                    if not r["fired"] and datetime.fromisoformat(r["fire_at"]) <= datetime.now():
                        r["fired"] = True
                        changed = True
                        on_fire(r["text"])
                if changed:
                    _save_reminders(reminders)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def search_web(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "Web search is unavailable: duckduckgo_search is not installed."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"- {title}: {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher used by the LLM layer
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, tool_input: dict) -> str:
    if name == "set_reminder":
        return set_reminder(tool_input["text"], float(tool_input["minutes_from_now"]))
    if name == "list_reminders":
        return list_reminders()
    if name == "search_web":
        return search_web(tool_input["query"])
    return f"Unknown tool: {name}"
