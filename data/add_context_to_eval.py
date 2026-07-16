"""Precompute retrieval context for the held-out HumanEval eval set.

Runs the full Part C pipeline (hybrid + graph + rerank) on CPU for every
eval item and stores the retrieved context string alongside it. The Colab
notebook (Part E) then measures fix rate with vs. without this context
without needing the retrieval stack on the GPU machine -- the retrieval
itself is still real, computed here against the real MBPP index.

No leakage: the index contains only MBPP training pairs; queries are the
held-out HumanEval items.

Usage:  python data/add_context_to_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from rag.retriever import Retriever

IN_PATH = os.path.join(_REPO_ROOT, "data", "out", "humaneval_broken.jsonl")
OUT_PATH = os.path.join(_REPO_ROOT, "data", "out",
                        "humaneval_broken_with_context.jsonl")
K = 3


def main() -> int:
    with open(IN_PATH, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    print(f"loaded {len(items)} eval items; retrieving context for each ...")

    retriever = Retriever()
    t0 = time.monotonic()
    for i, item in enumerate(items, 1):
        hits = retriever.retrieve(item["problem"], item["broken_code"],
                                  item["error"], k=K)
        item["retrieved_context"] = retriever.format_context(hits)
        item["retrieved_ids"] = [h["id"] for h in hits]
        if i % 50 == 0:
            print(f"  {i}/{len(items)} ({time.monotonic() - t0:.0f}s)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH} ({len(items)} items, k={K}, "
          f"{time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
