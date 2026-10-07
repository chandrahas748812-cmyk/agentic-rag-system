"""The four agents: planner, retriever, verifier, answerer.

Each agent is a thin wrapper around an LLM call with a focused system
prompt. Swapping the LLM provider only requires changing `get_llm()`.
"""

import os
from .state import AgentState, RetrievedChunk


def get_llm():
    """Lazily build the chat model so imports never require API keys."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version="2024-02-01",
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


PLANNER_PROMPT = """You are a query planner for a RAG system. Decompose the user
question into 1-3 focused sub-questions for retrieval, and choose a retrieval
strategy: "hybrid" (default), "bm25" (for exact IDs/terms), or "dense"
(for conceptual questions).

Respond in exactly this format:
SUBQUESTIONS:
- <sub-question 1>
- <sub-question 2>
STRATEGY: <hybrid|bm25|dense>"""

VERIFIER_PROMPT = """You are an evidence verifier. Given a question and a text
chunk, rate how relevant the chunk is for answering the question.
Respond with a single number between 0 and 1, nothing else."""

ANSWER_PROMPT = """You are a grounded answering agent. Answer the question using
ONLY the evidence below. Cite sources as [doc-id]. If the evidence does not
contain the answer, say "I don't have enough information" — never invent facts.

EVIDENCE:
{evidence}

QUESTION: {question}

Also end your response with a line: CONFIDENCE: <0-1>"""


def planner_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    resp = llm.invoke([
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": state.query},
    ]).content
    subs, strategy = [], "hybrid"
    for line in resp.splitlines():
        line = line.strip()
        if line.startswith("- "):
            subs.append(line[2:])
        elif line.startswith("STRATEGY:"):
            strategy = line.split(":", 1)[1].strip().lower()
    state.sub_questions = subs or [state.query]
    state.retrieval_strategy = strategy if strategy in ("hybrid", "bm25", "dense") else "hybrid"
    return state


def verifier_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    kept = []
    for chunk in state.chunks:
        resp = llm.invoke([
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": f"Question: {state.query}\nChunk [{chunk.doc_id}]: {chunk.text[:800]}"},
        ]).content.strip()
        try:
            score = max(0.0, min(1.0, float(resp)))
        except ValueError:
            score = 0.5
        chunk.relevance = score
        chunk.kept = score >= 0.4
        if chunk.kept:
            kept.append(chunk)
    state.kept_chunks = kept
    return state


def answer_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    if state.human_feedback:
        evidence = "\n".join(f"[{c.doc_id}] {c.text}" for c in state.kept_chunks)
        evidence += f"\n\nHUMAN REVIEWER NOTE: {state.human_feedback}"
    else:
        evidence = "\n".join(f"[{c.doc_id}] {c.text}" for c in state.kept_chunks)
    resp = llm.invoke([
        {"role": "system", "content": ANSWER_PROMPT.format(evidence=evidence, question=state.query)},
        {"role": "user", "content": state.query},
    ]).content

    # Parse trailing CONFIDENCE line
    confidence, answer_lines = 0.5, []
    for line in resp.splitlines():
        if line.strip().upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        else:
            answer_lines.append(line)
    state.draft_answer = "\n".join(answer_lines).strip()
    state.confidence = confidence
    state.citations = [c.doc_id for c in state.kept_chunks]
    return state
