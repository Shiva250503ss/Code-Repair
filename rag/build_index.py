"""Build all retrieval indexes (dense, BM25, knowledge graph) from the
Part B dataset.

Usage:  python rag/build_index.py
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from rag import graph as graphmod
from rag.store import build_indexes, load_dataset


def main() -> int:
    t0 = time.monotonic()
    records = load_dataset()
    print(f"loaded {len(records)} training pairs")

    build_indexes(records)

    print("building knowledge graph ...")
    bundle = graphmod.build_graph(records)
    graphmod.save_graph(bundle)
    g = bundle["graph"]
    kinds = {}
    for _n, d in g.nodes(data=True):
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
    print(f"graph: {g.number_of_nodes()} nodes "
          f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}), "
          f"{g.number_of_edges()} edges")
    print(f"done in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
