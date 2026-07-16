"""Retrieval pipeline: hybrid dense+BM25 with RRF, cross-encoder reranking,
and GraphRAG expansion.

    retriever = Retriever()                      # loads persisted indexes
    hits = retriever.retrieve(problem, code, error, k=3)

Stages (each can be toggled off for ablation -- retrieval_eval.py uses that):
  1. dense top-N from embedded Qdrant + BM25 top-N, merged with Reciprocal
     Rank Fusion (score = sum 1/(60 + rank))
  2. GraphRAG candidates (similar bug type in a similarly shaped function)
     fused in as a third ranked list
  3. cross-encoder rerank of the fused pool, final top-k
"""

from __future__ import annotations

import json
import os
import pickle

from rag import graph as graphmod
from rag.store import (BM25_PATH, COLLECTION, DOCS_PATH, EMBED_MODEL,
                       QDRANT_PATH, doc_text, tokenize)

RRF_K = 60
CANDIDATES_PER_SOURCE = 50
RERANK_POOL = 30
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rrf_merge(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, lazy: bool = False):
        with open(DOCS_PATH, encoding="utf-8") as f:
            self.docs = {r["id"]: r
                         for r in (json.loads(l) for l in f if l.strip())}
        with open(BM25_PATH, "rb") as f:
            bundle = pickle.load(f)
        self.bm25 = bundle["bm25"]
        self.bm25_doc_ids = bundle["doc_ids"]
        self.graph_bundle = graphmod.load_graph()
        self._embedder = None
        self._reranker = None
        self._qdrant = None
        if not lazy:
            self._ensure_models()

    def _ensure_models(self):
        if self._embedder is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(path=QDRANT_PATH)

    # ---- individual sources ----
    def dense_search(self, query: str, n: int = CANDIDATES_PER_SOURCE) -> list[str]:
        self._ensure_models()
        vec = self._embedder.encode([query], normalize_embeddings=True)[0]
        res = self._qdrant.query_points(COLLECTION, query=vec.tolist(),
                                        limit=n).points
        return [p.payload["doc_id"] for p in res]

    def bm25_search(self, query: str, n: int = CANDIDATES_PER_SOURCE) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self.bm25_doc_ids[i] for i in order[:n] if scores[i] > 0]

    def graph_search(self, broken_code: str, error: str,
                     exclude_task=None) -> list[str]:
        return graphmod.graph_candidates(self.graph_bundle, broken_code,
                                         error, exclude_task=exclude_task)

    # ---- full pipeline ----
    def retrieve(self, problem: str, broken_code: str, error: str,
                 k: int = 3, use_dense: bool = True, use_bm25: bool = True,
                 use_graph: bool = True, use_rerank: bool = True,
                 exclude_ids: set | None = None,
                 exclude_task=None) -> list[dict]:
        query = "\n".join(x for x in (problem, broken_code[:1200],
                                      error.strip().splitlines()[-1]
                                      if error.strip() else "") if x)
        rankings = []
        if use_dense:
            rankings.append(self.dense_search(query))
        if use_bm25:
            rankings.append(self.bm25_search(query))
        if use_graph:
            rankings.append(self.graph_search(broken_code, error,
                                              exclude_task=exclude_task))
        fused = rrf_merge(rankings) if rankings else []
        if exclude_ids:
            fused = [d for d in fused if d not in exclude_ids]
        if exclude_task is not None:
            fused = [d for d in fused
                     if self.docs[d]["task_id"] != exclude_task]

        pool = fused[:RERANK_POOL]
        if use_rerank and pool:
            self._ensure_models()
            pairs = [(query, doc_text(self.docs[d])) for d in pool]
            scores = self._reranker.predict(pairs)
            pool = [d for _, d in sorted(zip(scores, pool),
                                         key=lambda t: t[0], reverse=True)]
        return [self.docs[d] for d in pool[:k]]

    def format_context(self, hits: list[dict]) -> str:
        """Render retrieved fixes as prompt context for generation."""
        blocks = []
        for i, h in enumerate(hits, 1):
            err_tail = h["error"].strip().splitlines()[-1]
            blocks.append(
                f"# Reference repair {i} (bug type: {h['bug_type']})\n"
                f"# Problem: {h['problem']}\n"
                f"# Error was: {err_tail}\n"
                f"# Broken:\n{h['broken_code']}\n"
                f"# Fixed:\n{h['fixed_code']}")
        return "\n\n".join(blocks)
