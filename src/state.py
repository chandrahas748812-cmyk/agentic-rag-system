"""Shared graph state for the agentic RAG pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    doc_id: str
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    relevance: float = 0.0  # verifier-assigned, 0..1
    kept: bool = True


class AgentState(BaseModel):
    query: str
    sub_questions: List[str] = Field(default_factory=list)
    retrieval_strategy: str = "hybrid"
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    kept_chunks: List[RetrievedChunk] = Field(default_factory=list)
    draft_answer: str = ""
    citations: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    needs_human_review: bool = False
    human_feedback: Optional[str] = None
    final_answer: str = ""
