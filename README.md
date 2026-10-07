# Hybrid Search Lab

A hands-on lab comparing **BM25 vs dense vs hybrid retrieval** — with and without reranking. Run it, see the numbers, and understand *why* hybrid search wins on enterprise corpora. This is the exact tuning workflow behind production RAG pipelines.

## What you'll learn

1. BM25 wins on exact terms, IDs, and jargon; dense wins on paraphrases and concepts.
2. Hybrid (weighted fusion) beats either alone on mixed query workloads.
3. Reranking the top-k with a cross-encoder is the cheapest relevance boost available.
4. Chunk size and top-k are the two knobs that matter most.

## Quickstart

```bash
pip install -r requirements.txt
python example.py
```

## What it runs

```
Corpus: 12 enterprise-ish documents (policies, specs, FAQs)
Queries: 8 test queries spanning exact-match, paraphrase, and hybrid

For each query, compares:
  bm25           — classic keyword scoring
  dense          — sentence-transformer embeddings, cosine similarity
  hybrid(0.5)    — 50/50 score fusion (min-max normalized)
  hybrid+rerank  — hybrid top-10 → cross-encoder rerank → top-3

Metric: precision@3 against labeled relevant docs
```

## Example output

```
Query: "refund policy enterprise"          (exact terms)
  bm25:           P@3 = 1.00  ✅
  dense:          P@3 = 0.67
  hybrid:         P@3 = 1.00
  hybrid+rerank:  P@3 = 1.00

Query: "how do I get my money back"       (paraphrase)
  bm25:           P@3 = 0.33
  dense:          P@3 = 1.00  ✅
  hybrid:         P@3 = 1.00
  hybrid+rerank:  P@3 = 1.00

Query: "SSO setup enterprise contract 30-day"   (mixed)
  bm25:           P@3 = 0.67
  dense:          P@3 = 0.67
  hybrid:         P@3 = 1.00  ✅
  hybrid+rerank:  P@3 = 1.00

Aggregate P@3:
  bm25: 0.58 | dense: 0.71 | hybrid: 0.92 | hybrid+rerank: 0.96
```

## Try it yourself

Edit `WEIGHTS` in `example.py` to sweep the BM25/dense blend, or change `TOP_K` to see how retrieval depth affects precision. The `tune.py` script grid-searches both automatically.

## Project layout

```
src/
  corpus.py      # Demo corpus + labeled relevance judgments
  search.py      # BM25, dense, hybrid retrievers + fusion
  rerank.py      # Cross-encoder reranker (optional, graceful fallback)
  metrics.py     # precision@k, MRR
example.py       # Head-to-head comparison demo
tune.py          # Grid search over alpha (blend) and top_k
```

## Built with

Python · rank-bm25 · sentence-transformers · scikit-learn · numpy
