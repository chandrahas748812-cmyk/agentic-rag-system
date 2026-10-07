"""LangGraph state machine wiring the four agents together."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .agents import planner_agent, verifier_agent, answer_agent
from .retrieval import HybridRetriever
from .human_loop import maybe_escalate

CONFIDENCE_THRESHOLD = 0.75


def build_graph(retriever: HybridRetriever):
    """Assemble the agent graph. Returns a compiled, checkpointed runnable."""

    def retrieve_node(state: AgentState) -> AgentState:
        # Retrieve per sub-question, then de-duplicate by doc_id
        seen, merged = set(), []
        for sq in state.sub_questions:
            alpha = {"hybrid": 0.5, "bm25": 0.0, "dense": 1.0}[state.retrieval_strategy]
            for chunk in retriever.retrieve(sq, top_k=8, alpha=alpha):
                if chunk.doc_id not in seen:
                    seen.add(chunk.doc_id)
                    merged.append(chunk)
        merged.sort(key=lambda c: c.fused_score, reverse=True)
        state.chunks = merged[:10]
        return state

    def route_after_answer(state: AgentState) -> str:
        return "human_review" if maybe_escalate(state, CONFIDENCE_THRESHOLD) else END

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_agent)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("answer", answer_agent)
    builder.add_node("human_review", lambda s: s)  # interrupt point; resumed with feedback

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retrieve")
    builder.add_edge("retrieve", "verifier")
    builder.add_edge("verifier", "answer")
    builder.add_conditional_edges("answer", route_after_answer, {"human_review": "human_review", END: END})
    builder.add_edge("human_review", "answer")  # re-answer after human feedback

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"],
    )
