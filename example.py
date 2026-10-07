"""End-to-end demo: build a tiny corpus, run the agent graph, print the answer."""

from src.state import AgentState
from src.retrieval import HybridRetriever
from src.graph import build_graph

DOCUMENTS = [
    {"id": "doc-1", "text": "Enterprise contracts include a 30-day cancellation window from the signature date."},
    {"id": "doc-2", "text": "Refunds for enterprise plans are pro-rated based on unused service days."},
    {"id": "doc-3", "text": "Standard (non-enterprise) plans are billed monthly and are non-refundable after 7 days."},
    {"id": "doc-4", "text": "To request a refund, contact billing@acme.com with your contract ID."},
    {"id": "doc-5", "text": "Enterprise onboarding includes a dedicated solutions engineer for 90 days."},
    {"id": "doc-6", "text": "SLA credits are issued automatically when uptime falls below 99.9% in a month."},
]

# Optional dense embeddings — works BM25-only without sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    dense = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    dense = None
    print("(dense model unavailable — running BM25-only)")

retriever = HybridRetriever(DOCUMENTS, dense_model=dense)
graph = build_graph(retriever)

query = "What is the refund policy for enterprise contracts?"
config = {"configurable": {"thread_id": "demo-1"}}

print(f"Query: {query}\n")
result = None
for event in graph.stream(AgentState(query=query), config=config):
    for node, state in event.items():
        if node == "answer" and not isinstance(state, dict):
            result = state

if result is None:
    # Interrupted for human review — supply feedback and resume
    print("→ Escalated to human review (low confidence).")
    graph.update_state(config, {"human_feedback": "Use doc-1 and doc-2; be concise."})
    for event in graph.stream(None, config=config):
        for node, state in event.items():
            if node == "answer" and not isinstance(state, dict):
                result = state

print("Sub-questions:", result.sub_questions)
print("Strategy:", result.retrieval_strategy)
print(f"Kept {len(result.kept_chunks)}/{len(result.chunks)} chunks")
print(f"Confidence: {result.confidence:.2f}")
print("\nAnswer:\n", result.final_answer or result.draft_answer)
