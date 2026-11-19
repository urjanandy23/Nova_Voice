"""Tool-calling accuracy eval: does the model pick the right tool (or none)?

Two suites (cases live in evals/cases.py):
  * single: one user message, no history
  * multi:  scripted history (including earlier tool calls/results) + a
            follow-up message ("cancel that", "and in Paris?", "thanks")

Tools are never executed. Reports overall and per-suite accuracy, per-tool
precision/recall, a confusion matrix and every failure, and records provider,
model and the system-prompt hash so a before/after comparison is reproducible.
Results are written to evals/results_<provider>.md so runs of different
providers never overwrite each other.

Usage:
    python evals/tool_calling_eval.py                  # both suites, 1 pass
    python evals/tool_calling_eval.py --repeats 3      # models are non-deterministic
    python evals/tool_calling_eval.py --suite multi
"""
import argparse
import hashlib
import os
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import (MULTI_TURN_CASES, NONE, SINGLE_TURN_CASES, build_messages,  # noqa: E402
                   single_turn_messages)
from src.config import Config  # noqa: E402
from src.functions import TOOLS  # noqa: E402
from src.prompts import SYSTEM_PROMPT  # noqa: E402

def predict_anthropic(messages):
    from anthropic import Anthropic
    client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=Config.CLAUDE_MODEL, max_tokens=512, system=SYSTEM_PROMPT,
        messages=messages, tools=TOOLS,
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.name
    return NONE


def predict_ollama(messages):
    import ollama
    from src.llm_ollama import _get, _to_ollama_tools
    client = ollama.Client(host=Config.OLLAMA_HOST)
    resp = client.chat(
        model=Config.OLLAMA_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=_to_ollama_tools(TOOLS),
    )
    calls = _get(_get(resp, "message"), "tool_calls")
    if calls:
        return _get(_get(calls[0], "function"), "name")
    return NONE


def build_jobs(suite, provider):
    """Returns [(suite_name, case_name, messages, expected)] for the chosen suite(s)."""
    jobs = []
    if suite in ("single", "all"):
        for prompt, expected in SINGLE_TURN_CASES:
            jobs.append(("single", prompt, single_turn_messages(prompt), expected))
    if suite in ("multi", "all"):
        for case in MULTI_TURN_CASES:
            label = f"{case.name}: ...{case.user}"
            jobs.append(("multi", label, build_messages(case, provider), case.expected))
    return jobs


def accuracy_line(rows):
    n = len(rows)
    ok = sum(1 for r in rows if r["expected"] == r["predicted"])
    return f"{ok}/{n} ({ok / n * 100:.1f}%)" if n else "n/a"


def render_report(rows, meta):
    """rows: dicts with suite, name, expected, predicted. Returns markdown."""
    total = len(rows)
    labels = sorted({r["expected"] for r in rows} | {r["predicted"] for r in rows})
    confusion = Counter((r["expected"], r["predicted"]) for r in rows)
    suites = sorted({r["suite"] for r in rows})

    lines = [
        "# Tool-calling accuracy eval",
        "",
        f"- Date: {meta['date']}",
        f"- Provider / model: `{meta['provider']}` / `{meta['model']}`",
        f"- Prompt + tool-description hash: `{meta['prompt_hash']}`",
        f"- Suite(s): {', '.join(suites)}; {total} predictions over {meta['repeats']} pass(es)",
        "",
        f"**Overall accuracy: {accuracy_line(rows)}**",
        "",
        "| Suite | Accuracy |",
        "|---|---|",
    ]
    for su in suites:
        lines.append(f"| {su} | {accuracy_line([r for r in rows if r['suite'] == su])} |")

    lines += ["", "## Per-label precision / recall (all suites)", "",
              "| Label | Precision | Recall | Support |", "|---|---|---|---|"]
    for lab in labels:
        tp = confusion[(lab, lab)]
        pred_n = sum(v for (e, p), v in confusion.items() if p == lab)
        true_n = sum(v for (e, p), v in confusion.items() if e == lab)
        prec = f"{tp / pred_n:.2f}" if pred_n else "n/a"
        rec = f"{tp / true_n:.2f}" if true_n else "n/a"
        lines.append(f"| {lab} | {prec} | {rec} | {true_n} |")

    lines += ["", "## Confusion matrix (rows = expected, columns = predicted)", "",
              "| expected \\ predicted | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for e in labels:
        lines.append(f"| {e} | " + " | ".join(str(confusion[(e, p)]) for p in labels) + " |")

    failures = [r for r in rows if r["expected"] != r["predicted"]]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if failures:
        lines += ["| Suite | Case | Expected | Predicted |", "|---|---|---|---|"]
        lines += [f"| {r['suite']} | {r['name']} | {r['expected']} | {r['predicted']} |"
                  for r in failures]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--suite", choices=["single", "multi", "all"], default="all")
    args = parser.parse_args()

    provider = Config.LLM_PROVIDER
    predict = predict_ollama if provider == "ollama" else predict_anthropic
    model = Config.OLLAMA_MODEL if provider == "ollama" else Config.CLAUDE_MODEL
    prompt_hash = hashlib.sha256(
        (SYSTEM_PROMPT + str([t["description"] for t in TOOLS])).encode()).hexdigest()[:10]
    print(f"Provider: {provider}  model: {model}  prompt-hash: {prompt_hash}\n")

    jobs = build_jobs(args.suite, provider)
    rows = []
    for _ in range(args.repeats):
        for suite, name, messages, expected in jobs:
            predicted = predict(messages)
            rows.append({"suite": suite, "name": name,
                         "expected": expected, "predicted": predicted})
            print(f"[{'PASS' if predicted == expected else 'FAIL'}] ({suite}) {name!r}  "
                  f"expected={expected} got={predicted}")

    meta = {"date": date.today().isoformat(), "provider": provider, "model": model,
            "prompt_hash": prompt_hash, "repeats": args.repeats}
    report = render_report(rows, meta)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{provider}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nOverall: {accuracy_line(rows)}\nSaved: {out}")


if __name__ == "__main__":
    main()
