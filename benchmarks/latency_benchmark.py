"""Measures the actual benefit of sentence-by-sentence streaming TTS versus
a naive design that waits for the full LLM response before speaking
anything at all.
"""
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant import _build_llm
from src.memory import ConversationMemory

PROMPTS = [
    "What's the capital of Japan?",
    "Give me a two sentence summary of why the sky is blue.",
    "Suggest a good name for a pet goldfish.",
    "What's the boiling point of water in Fahrenheit?",
    "Recommend one book for someone who likes mystery novels.",
    "What year did the first iPhone come out?",
    "Explain what photosynthesis is in one sentence.",
    "What's a healthy breakfast idea?",
    "Name three primary colors.",
    "What's the tallest mountain in the world?",
]


def measure_one(llm, prompt: str):
    memory = ConversationMemory()
    memory.add_user(prompt)

    start = time.perf_counter()
    first_sentence_time = None

    def on_sentence(_sentence):
        nonlocal first_sentence_time
        if first_sentence_time is None:
            first_sentence_time = time.perf_counter()

    llm.stream_response(memory, on_sentence)
    end = time.perf_counter()

    streaming_ttfa = (first_sentence_time or end) - start
    naive_ttfa = end - start
    return streaming_ttfa, naive_ttfa


def main():
    print("Loading LLM...")
    llm = _build_llm()

    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {prompt!r}")
        streaming_ttfa, naive_ttfa = measure_one(llm, prompt)
        improvement = (naive_ttfa - streaming_ttfa) / naive_ttfa * 100 if naive_ttfa else 0
        rows.append({
            "prompt": prompt,
            "streaming_ttfa_s": round(streaming_ttfa, 3),
            "naive_ttfa_s": round(naive_ttfa, 3),
            "improvement_pct": round(improvement, 1),
        })
        print(
            f"    streaming: {streaming_ttfa:.2f}s to first audio | "
            f"naive: {naive_ttfa:.2f}s to first audio | "
            f"{improvement:.0f}% faster to start speaking"
        )

    streaming_avg = statistics.mean(r["streaming_ttfa_s"] for r in rows)
    naive_avg = statistics.mean(r["naive_ttfa_s"] for r in rows)
    overall_improvement = (naive_avg - streaming_avg) / naive_avg * 100 if naive_avg else 0

    print("\n" + "=" * 60)
    print(f"Average time to first audio — streaming: {streaming_avg:.2f}s")
    print(f"Average time to first audio — naive (wait for full reply): {naive_avg:.2f}s")
    print(f"Overall: streaming starts speaking {overall_improvement:.0f}% sooner")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(out_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Streaming vs. naive latency benchmark\n\n")
        f.write(
            f"Average time-to-first-audio across {len(rows)} prompts:\n\n"
            f"| Design | Avg. time to first audio |\n"
            f"|---|---|\n"
            f"| Streaming (this project) | {streaming_avg:.2f}s |\n"
            f"| Naive (wait for full reply) | {naive_avg:.2f}s |\n\n"
            f"**Streaming starts speaking {overall_improvement:.0f}% sooner "
            f"on average.**\n\n"
            f"## Per-prompt results\n\n"
            f"| Prompt | Streaming (s) | Naive (s) | Improvement |\n"
            f"|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['prompt']} | {r['streaming_ttfa_s']} | "
                f"{r['naive_ttfa_s']} | {r['improvement_pct']}% |\n"
            )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()