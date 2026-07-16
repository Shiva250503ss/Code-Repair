"""Retrieval evaluation with a real labeled query set.

Labels come from the dataset itself, not from human or LLM judgment: for a
query built from one bug variant, the relevant documents are the OTHER bug
variants of the same underlying MBPP problem (same task_id, different broken
code and different captured error). The query document itself is excluded
from every result list.

Two query modes:
  full       problem text + broken code + error (what the UI sends)
  code_only  broken code + error, no problem text (user pasted only code --
             harder, and where dense/graph signals must carry the weight)

Systems compared, cumulative and ablated:
  dense | bm25 | graph | hybrid (dense+bm25 RRF) | hybrid+rerank |
  hybrid+graph+rerank (full pipeline)

Metrics: recall@5, recall@10, nDCG@10, MRR. All numbers are measured by
running the real pipeline over the real index.

Usage:  python rag/retrieval_eval.py [--queries 120] [--seed 7]
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from rag.retriever import Retriever

SYSTEMS = {
    "dense only":          dict(use_dense=True, use_bm25=False, use_graph=False, use_rerank=False),
    "bm25 only":           dict(use_dense=False, use_bm25=True, use_graph=False, use_rerank=False),
    "graph only":          dict(use_dense=False, use_bm25=False, use_graph=True, use_rerank=False),
    "hybrid (RRF)":        dict(use_dense=True, use_bm25=True, use_graph=False, use_rerank=False),
    "hybrid+rerank":       dict(use_dense=True, use_bm25=True, use_graph=False, use_rerank=True),
    "hybrid+graph+rerank": dict(use_dense=True, use_bm25=True, use_graph=True, use_rerank=True),
}

K_EVAL = 10


def ndcg_at_k(hit_ranks: list[int], n_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 2) for r in hit_ranks if r < k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal else 0.0


def evaluate(retriever: Retriever, queries: list[dict], mode: str,
             flags: dict) -> dict:
    recall5 = recall10 = ndcg = mrr = 0.0
    for q in queries:
        problem = q["problem"] if mode == "full" else ""
        hits = retriever.retrieve(problem, q["broken_code"], q["error"],
                                  k=K_EVAL, exclude_ids={q["id"]}, **flags)
        got = [h["id"] for h in hits]
        relevant = q["relevant_ids"]
        hit_ranks = [i for i, d in enumerate(got) if d in relevant]
        n_rel = len(relevant)
        recall5 += len([r for r in hit_ranks if r < 5]) / min(n_rel, 5)
        recall10 += len(hit_ranks) / min(n_rel, K_EVAL)
        ndcg += ndcg_at_k(hit_ranks, n_rel, K_EVAL)
        mrr += 1.0 / (hit_ranks[0] + 1) if hit_ranks else 0.0
    n = len(queries)
    return {"recall@5": recall5 / n, "recall@10": recall10 / n,
            "nDCG@10": ndcg / n, "MRR": mrr / n}


def build_query_set(retriever: Retriever, n_queries: int, seed: int):
    by_task = collections.defaultdict(list)
    for doc in retriever.docs.values():
        by_task[doc["task_id"]].append(doc["id"])
    eligible = [doc for doc in retriever.docs.values()
                if len(by_task[doc["task_id"]]) >= 2]
    rng = random.Random(seed)
    sample = rng.sample(eligible, min(n_queries, len(eligible)))
    queries = []
    for doc in sample:
        relevant = set(by_task[doc["task_id"]]) - {doc["id"]}
        queries.append({**doc, "relevant_ids": relevant})
    return queries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("loading retriever (indexes + models, CPU) ...")
    retriever = Retriever()
    queries = build_query_set(retriever, args.queries, args.seed)
    print(f"query set: {len(queries)} labeled queries "
          f"(relevant = other bug variants of the same real problem)")

    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"{len(queries)} labeled queries (seed {args.seed}) over "
        f"{len(retriever.docs)} indexed documents. Relevant documents for a "
        "query are the other bug variants of the same MBPP problem; the "
        "query document itself is always excluded. All numbers measured.",
        "",
    ]
    for mode in ("full", "code_only"):
        print(f"\n=== query mode: {mode} ===")
        lines += [f"## Query mode: {mode}", "",
                  "| System | recall@5 | recall@10 | nDCG@10 | MRR |",
                  "|---|---|---|---|---|"]
        for name, flags in SYSTEMS.items():
            t0 = time.monotonic()
            m = evaluate(retriever, queries, mode, flags)
            dt = time.monotonic() - t0
            row = (f"| {name} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                   f"| {m['nDCG@10']:.3f} | {m['MRR']:.3f} |")
            lines.append(row)
            print(f"{name:22s} recall@5={m['recall@5']:.3f} "
                  f"recall@10={m['recall@10']:.3f} nDCG@10={m['nDCG@10']:.3f} "
                  f"MRR={m['MRR']:.3f}  ({dt:.0f}s)")
        lines.append("")

    out = os.path.join(_REPO_ROOT, "rag", "eval_results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
