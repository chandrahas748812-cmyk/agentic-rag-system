"""Hybrid BM25 + dense retriever over an in-memory document store.

In production this would sit on Pinecone / pgvector. The fusion logic
(reciprocal rank fusion) is identical — only the vector backend changes.
"""

from typing import List
import numpy as np
from rank_bm25 import BM25Okapi

from .state import RetrievedChunk


class HybridRetriever:
    def __init__(self, documents: List[dict], dense_model=None):
        """
        documents: [{"id": str, "text": str}, ...]
        dense_model: a sentence-transformers model (optional; falls back to BM25-only)
        """
        self.documents = documents
        self.corpus = [d["text"] for d in documents]
        self.tokenized = [t.lower().split() for t in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.dense_model = dense_model
        self.doc_embeddings = None
        if dense_model is not None:
            self.doc_embeddings = dense_model.encode(self.corpus, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 8, alpha: float = 0.5) -> List[RetrievedChunk]:
        """alpha blends BM25 (1-alpha) and dense (alpha). Returns fused top-k."""
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_norm = self._minmax(bm25_scores)

        if self.doc_embeddings is not None:
            q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
            dense_scores = self.doc_embeddings @ q_emb
            dense_norm = self._minmax(dense_scores)
        else:
            dense_norm = np.zeros_like(bm25_norm)

        fused = (1 - alpha) * bm25_norm + alpha * dense_norm
        top_idx = np.argsort(fused)[::-1][:top_k]

        return [
            RetrievedChunk(
                doc_id=self.documents[i]["id"],
                text=self.documents[i]["text"],
                bm25_score=float(bm25_norm[i]),
                dense_score=float(dense_norm[i]),
                fused_score=float(fused[i]),
            )
            for i in top_idx
        ]

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)
