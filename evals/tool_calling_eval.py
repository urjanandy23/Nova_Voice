"""Measures how often the LLM picks the CORRECT tool for a fixed set of
test prompts — without actually executing any tool (no real reminders
created, no real web searches run).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.functions import TOOLS
from src.llm import SYSTEM_PROMPT

TEST_CASES = [
    {"prompt": "Remind me to call mom in 20 minutes", "expected_tool": "set_reminder"},
    {"prompt": "Set a reminder to take the laundry out in 10 minutes", "expected_tool": "set_reminder"},
    {"prompt": "Don't let me forget to submit my report in an hour", "expected_tool": "set_reminder"},
    {"prompt": "What are my pending reminders?", "expected_tool": "list_reminders"},
    {"prompt": "Do I have anything I asked you to remind me about?", "expected_tool": "list_reminders"},
    {"prompt": "Search the web for the latest iPhone release", "expected_tool": "search_web"},
    {"prompt": "What's the current weather in Tokyo right now?", "expected_tool": "search_web"},
    {"prompt": "Look up who won the game last night", "expected_tool": "search_web"},
    {"prompt": "What's the capital of France?", "expected_tool": None},
    {"prompt": "What's 12 times 8?", "expected_tool": None},
    {"prompt": "Tell me a fun fact about octopuses", "expected_tool": None},
    {"prompt": "How do you say 'hello' in Spanish?", "expected_tool": None},
    {"prompt": "What's the boiling point of water?", "expected_tool": None},
]


def predict_tool_anthropic(prompt: str):
    from anthropic import Anthropic

    client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=Config.CLAUDE_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.name
    return None


def predict_tool_ollama(prompt: str):
    import ollama

    from src.llm_ollama import _get, _to_ollama_tools

    client = ollama.Client(host=Config.OLLAMA_HOST)
    resp = client.chat(
        model=Config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=_to_ollama_tools(TOOLS),
    )
    msg = _get(resp, "message")
    calls = _get(msg, "tool_calls")
    if calls:
        fn = _get(calls[0], "function")
        return _get(fn, "name")
    return None


def predict_tool(prompt: str):
    if Config.LLM_PROVIDER == "ollama":
        return predict_tool_ollama(prompt)
    return predict_tool_anthropic(prompt)


def main():
    print(f"Evaluating tool-calling accuracy (provider: {Config.LLM_PROVIDER})\n")

    results = []
    correct = 0
    for case in TEST_CASES:
        predicted = predict_tool(case["prompt"])
        is_correct = predicted == case["expected_tool"]
        correct += int(is_correct)
        results.append({**case, "predicted_tool": predicted, "correct": is_correct})
        mark = "PASS" if is_correct else "FAIL"
        print(
            f"[{mark}] {case['prompt']!r}\n"
            f"       expected={case['expected_tool']}  got={predicted}"
        )

    total = len(TEST_CASES)
    accuracy = correct / total * 100
    print("\n" + "=" * 60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 60)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tool-calling accuracy eval\n\n")
        f.write(f"Provider: `{Config.LLM_PROVIDER}`\n\n")
        f.write(f"**Accuracy: {correct}/{total} ({accuracy:.1f}%)**\n\n")
        f.write("| Prompt | Expected | Predicted | Result |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            mark = "✅" if r["correct"] else "❌"
            f.write(
                f"| {r['prompt']} | {r['expected_tool']} | "
                f"{r['predicted_tool']} | {mark} |\n"
            )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()