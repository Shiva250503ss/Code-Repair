"""Document store and hybrid index for the code-repair corpus.

One document per training pair from data/out/dataset.jsonl. Two indexes over
the same documents:

  - dense:  sentence-transformers all-MiniLM-L6-v2 vectors in an embedded
            (file-based, serverless) Qdrant collection
  - sparse: BM25 (rank_bm25) over code-aware tokens

Hybrid queries merge both rankings with Reciprocal Rank Fusion in
retriever.py. Build everything with:  python rag/build_index.py
"""

from __future__ import annotations

import json
import os
import pickle
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(_REPO_ROOT, "data", "out", "dataset.jsonl")
INDEX_DIR = os.path.join(_REPO_ROOT, "rag", "index")
QDRANT_PATH = os.path.join(INDEX_DIR, "qdrant")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
DOCS_PATH = os.path.join(INDEX_DIR, "docs.jsonl")
COLLECTION = "repair_pairs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+")


def tokenize(text: str) -> list[str]:
    """Code-aware tokens: identifiers split on underscores, lowercased."""
    out = []
    for tok in _TOKEN_RE.findall(text):
        tok = tok.lower()
        parts = [p for p in tok.split("_") if p]
        out.extend(parts if len(parts) > 1 else [tok])
    return out


def doc_text(rec: dict, include_problem: bool = True) -> str:
    """The searchable text for one training pair."""
    error_tail = rec["error"].strip().splitlines()[-1] if rec["error"] else ""
    parts = []
    if include_problem:
        parts.append(rec["problem"])
    parts.append(rec["broken_code"][:1200])
    parts.append(error_tail)
    return "\n".join(parts)


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run data/build_dataset.py first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_indexes(records: list[dict], batch_size: int = 256) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    texts = [doc_text(r) for r in records]

    print(f"embedding {len(texts)} documents with {EMBED_MODEL} (CPU) ...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True)

    print("writing embedded Qdrant collection ...")
    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1],
                                    distance=Distance.COSINE))
    for start in range(0, len(records), batch_size):
        chunk = range(start, min(start + batch_size, len(records)))
        client.upsert(COLLECTION, points=[
            PointStruct(id=i, vector=vectors[i].tolist(),
                        payload={"doc_id": records[i]["id"]})
            for i in chunk])
    client.close()

    print("building BM25 index ...")
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": [r["id"] for r in records]}, f)

    print(f"indexed {len(records)} documents into {INDEX_DIR}")
