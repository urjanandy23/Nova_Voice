"""Latency benchmark for the streaming design.

Compares, per prompt, with repeated runs and p50/p95:
  * time to first LLM token
  * time to first speakable chunk, whole-sentence mode
  * time to first speakable chunk, clause mode (first chunk cut at a comma)
  * naive design: wait for the full reply
Optionally (--with-tts) adds the real edge-tts synthesis time for the first
chunk, giving an estimate of time-to-first-audio.

Two prompt sets: "short" (one-sentence answers, where streaming cannot help)
and "long" (multi-sentence answers, where it should). Runs in which the model
called a tool are counted and reported, since a tool round trip dominates.

Usage:
    python benchmarks/latency_benchmark.py                 # 3 repeats, both sets
    python benchmarks/latency_benchmark.py --repeats 5 --with-tts
    python benchmarks/latency_benchmark.py --sets long
"""
import argparse
import asyncio
import csv
import os
import platform
import statistics
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.metrics import Timeline, percentile

PROMPT_SETS = {
    "short": [
        "What's the capital of Japan?",
        "What's the boiling point of water in Fahrenheit?",
        "What year did the first iPhone come out?",
        "Name three primary colors.",
        "What's the tallest mountain in the world?",
    ],
    "long": [
        "Explain in four or five sentences why the sky is blue.",
        "Give me a short, step by step explanation of how to make a cup of tea.",
        "Describe how a bill becomes a law in about five sentences.",
        "Tell me about the water cycle in a few sentences.",
        "Explain how vaccines work, in four sentences.",
    ],
}
CLAUSE_CHARS = 35
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_once(llm, prompt, first_chunk_chars):
    Config.FIRST_CHUNK_MIN_CHARS = first_chunk_chars
    memory = llm.new_memory()
    memory.add_user(prompt)
    tl = Timeline()
    first_chunk = {}

    def on_sentence(text):
        first_chunk.setdefault("text", text)

    tl.mark("llm_start")
    llm.stream_response(memory, on_sentence, timeline=tl)
    tl.mark("done")
    return {
        "ttft": tl.between("llm_start", "first_token"),
        "first_chunk": tl.between("llm_start", "first_sentence"),
        "full": tl.between("llm_start", "done"),
        "tools": tl.tool_calls,
        "text": first_chunk.get("text", ""),
    }


def synth_seconds(text):
    from src.tts import StreamingTTS
    tts = StreamingTTS()
    start = time.perf_counter()
    asyncio.run(tts._synthesize(text))
    return time.perf_counter() - start


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return f"{statistics.median(values):.2f} / {percentile(values, 95):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sets", nargs="+", default=["short", "long"], choices=list(PROMPT_SETS))
    parser.add_argument("--with-tts", action="store_true",
                        help="also time edge-tts synthesis of the first chunk (needs network + ffmpeg)")
    args = parser.parse_args()

    from src.assistant import _build_llm
    model = Config.OLLAMA_MODEL if Config.LLM_PROVIDER == "ollama" else Config.CLAUDE_MODEL
    llm = _build_llm()
    print(f"Provider: {Config.LLM_PROVIDER}  model: {model}")

    rows = []
    tts_warned = False
    for set_name in args.sets:
        for prompt in PROMPT_SETS[set_name]:
            for rep in range(1, args.repeats + 1):
                for mode, chars in (("sentence", 0), ("clause", CLAUSE_CHARS)):
                    r = run_once(llm, prompt, chars)
                    r.update(set=set_name, prompt=prompt, rep=rep, mode=mode)
                    r["synth"] = None
                    if args.with_tts and r["text"]:
                        try:
                            r["synth"] = synth_seconds(r["text"])
                        except Exception as e:  # edge-tts is a network service; do not lose the run
                            if not tts_warned:
                                print(f"  (TTS synthesis failed: {type(e).__name__}; "
                                      "TTS columns will be n/a)")
                                tts_warned = True
                    rows.append(r)
                    print(f"[{set_name}/{mode} #{rep}] ttft={r['ttft']} first={r['first_chunk']} "
                          f"full={r['full']:.2f} tools={r['tools']}  {prompt!r}")

    # raw data
    with open(os.path.join(OUT_DIR, f"results_{Config.LLM_PROVIDER}.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["set", "prompt", "rep", "mode", "ttft", "first_chunk", "full", "tools", "synth"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary
    lines = [
        "# Latency benchmark",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Provider / model: `{Config.LLM_PROVIDER}` / `{model}`",
        f"- Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'}), "
        f"Python {platform.python_version()}",
        f"- Repeats per prompt per mode: {args.repeats}; clause mode cuts at >= {CLAUSE_CHARS} chars",
        "- Cells are **median / p95** in seconds. LLM only unless a TTS column is shown.",
        "",
    ]
    for set_name in args.sets:
        sub = [r for r in rows if r["set"] == set_name]
        sent = [r for r in sub if r["mode"] == "sentence"]
        clause = [r for r in sub if r["mode"] == "clause"]
        tool_runs = sum(1 for r in sub if r["tools"])
        lines += [
            f"## Prompt set: {set_name} ({len(sub)} runs, {tool_runs} involved a tool call)",
            "",
            "| Design | Time to first speakable chunk (median / p95) |",
            "|---|---|",
            f"| Naive: wait for full reply | {stat([r['full'] for r in sent])} |",
            f"| Streaming, whole sentences | {stat([r['first_chunk'] for r in sent])} |",
            f"| Streaming, clause-cut first chunk | {stat([r['first_chunk'] for r in clause])} |",
            f"| (reference) first LLM token | {stat([r['ttft'] for r in sent])} |",
            "",
        ]
        if args.with_tts:
            est_sent = [r["first_chunk"] + r["synth"] for r in sent if r["synth"] and r["first_chunk"]]
            est_clause = [r["first_chunk"] + r["synth"] for r in clause if r["synth"] and r["first_chunk"]]
            lines += [
                "| Estimated time to first audio (chunk + TTS synth) | median / p95 |",
                "|---|---|",
                f"| Whole sentences | {stat(est_sent)} |",
                f"| Clause-cut first chunk | {stat(est_clause)} |",
                "",
            ]
    lines += [
        "Notes: runs that called a tool include the tool round trip in every column. "
        f"Raw per-run data is in `results_{Config.LLM_PROVIDER}.csv`.",
    ]
    with open(os.path.join(OUT_DIR, f"results_{Config.LLM_PROVIDER}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()