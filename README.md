# Agentic RAG System

A production-style **agentic retrieval-augmented generation (RAG)** system built with LangGraph. Instead of a single retrieve-then-generate step, a graph of specialized agents plans, retrieves, verifies, and answers — with human-in-the-loop review for low-confidence outputs.

## Architecture

```
User Query
    │
    ▼
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌────────┐
│ Planner │───▶│Retriever │───▶│ Verifier  │───▶│ Answer │
│  Agent  │    │  Agent   │    │  Agent    │    │ Agent  │
└─────────┘    └──────────┘    └───────────┘    └────────┘
                                        │ low confidence
                                        ▼
                                  ┌───────────┐
                                  │ Human     │
                                  │ Review    │
                                  └───────────┘
```

- **Planner Agent** — decomposes the query into sub-questions and picks retrieval strategies.
- **Retriever Agent** — hybrid retrieval (BM25 + dense embeddings) with configurable top-k.
- **Verifier Agent** — scores each retrieved chunk for relevance; drops weak evidence.
- **Answer Agent** — generates a grounded answer with citations, or escalates to human review when confidence is low.

State is checkpointed at every node, so any run can be resumed, replayed, or audited.

## Quickstart

```bash
pip install -r requirements.txt
python example.py
```

Set your LLM provider key first:

```bash
export OPENAI_API_KEY="sk-..."        # or
export AZURE_OPENAI_API_KEY="..."     # for Azure OpenAI
```

## Project layout

```
src/
  graph.py        # LangGraph state machine: nodes, edges, checkpointing
  agents.py       # Planner / Retriever / Verifier / Answer agent prompts & logic
  retrieval.py    # Hybrid BM25 + dense retriever over a local document store
  state.py        # Shared graph state schema (Pydantic)
  human_loop.py   # Human-in-the-loop escalation policy
example.py        # End-to-end runnable demo
```

## Key design decisions

- **Hybrid retrieval** beats dense-only on enterprise corpora with jargon and IDs — BM25 catches exact terms, dense catches semantics.
- **Verification before generation** cuts hallucinations: the answer agent only sees evidence the verifier kept.
- **Checkpointed state** (LangGraph `MemorySaver`) makes every run resumable and auditable — critical for regulated domains.
- **Confidence-gated human review** keeps automation high while routing genuinely ambiguous cases to a person.

## Example output

```
Query: "What is the refund policy for enterprise contracts?"

Planner: sub-questions = ["refund terms", "enterprise contract cancellation window"]
Retriever: 12 chunks → Verifier kept 5 (avg relevance 0.81)
Answer: "Enterprise contracts include a 30-day cancellation window with
pro-rated refunds [doc-3, doc-7]..."
Confidence: 0.87 → auto-approved (threshold 0.75)
```

## Built with

Python · LangGraph · LangChain · Pydantic · scikit-learn (BM25) · sentence-transformers
