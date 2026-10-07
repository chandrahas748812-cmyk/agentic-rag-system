# LLM Eval Harness

An **offline evaluation framework** for LLM applications. Define golden datasets, run candidate models or prompts against them, and score outputs with LLM-as-judge groundedness checks, hallucination detection, ROUGE-L, and latency/cost tracking. Compare runs side-by-side to drive model and prompt decisions with data instead of vibes.

## Why this exists

In production LLM systems, you can ship faster than you can evaluate. This harness gives you:

- **Golden datasets** — versioned question/expected-answer pairs with source documents.
- **LLM-as-judge scoring** — groundedness (is every claim supported by the retrieved context?) and hallucination rate.
- **Classical metrics** — ROUGE-L for lexical overlap, plus latency and token-cost per case.
- **Run comparison** — diff two runs (e.g. old prompt vs new prompt) on every metric.

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python example.py
```

## Project layout

```
src/
  evaluator.py   # Core runner: executes cases, collects scores, writes reports
  judges.py      # LLM-as-judge prompts: groundedness, hallucination, relevance
  metrics.py     # ROUGE-L, latency, cost, aggregation helpers
  dataset.py     # Golden dataset loading & validation (JSONL)
example.py       # Demo: evaluate two prompts, print comparison table
data/
  golden.jsonl   # Sample golden dataset (10 cases)
```

## Golden dataset format (JSONL)

```json
{"id": "g1", "question": "What is the refund window?",
 "context": "Enterprise contracts include a 30-day cancellation window...",
 "expected": "30 days from signature"}
```

## Example output

```
Run A (prompt v1) vs Run B (prompt v2) — 10 cases
─────────────────────────────────────────────────
groundedness      0.71 → 0.86  (+0.15) ✅
hallucination     0.22 → 0.09  (-0.13) ✅
rougeL            0.54 → 0.61  (+0.07)
p50 latency        1.8s → 1.9s (+0.1s)
cost / 1k cases  $4.20 → $4.35
─────────────────────────────────────────────────
Verdict: ship prompt v2 (groundedness +15pts, no latency regression)
```

## Design notes

- **Judges are calibrated, not clever** — binary/0-1 scoring prompts with strict output formats beat elaborate rubrics for consistency.
- **Separate what you measure from how you judge** — lexical metrics (ROUGE) catch regressions cheaply; LLM judges catch semantic failures.
- **Cost is a metric** — every run logs tokens so you can trade quality against spend explicitly.

## Built with

Python · OpenAI API · rouge-score · Pydantic · tabulate
