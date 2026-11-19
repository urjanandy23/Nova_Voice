# Latency benchmark

- Date: 2026-11-20
- Provider / model: `ollama` / `llama3.2`
- Machine: Windows-11-10.0.26200-SP0 (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel), Python 3.14.4
- Repeats per prompt per mode: 1; clause mode cuts at >= 35 chars
- Cells are **median / p95** in seconds. LLM only unless a TTS column is shown.

## Prompt set: long (10 runs, 10 involved a tool call)

| Design | Time to first speakable chunk (median / p95) |
|---|---|
| Naive: wait for full reply | 20.72 / 28.79 |
| Streaming, whole sentences | 16.02 / 21.22 |
| Streaming, clause-cut first chunk | 3.34 / 18.21 |
| (reference) first LLM token | 13.42 / 17.44 |

Notes: runs that called a tool include the tool round trip in every column. Raw per-run data is in `results_ollama.csv`.
