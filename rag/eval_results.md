# Retrieval Evaluation Results

120 labeled queries (seed 7) over 3760 indexed documents. Relevant documents for a query are the other bug variants of the same MBPP problem; the query document itself is always excluded. All numbers measured.

## Query mode: full

| System | recall@5 | recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| dense only | 0.990 | 1.000 | 0.996 | 0.994 |
| bm25 only | 0.988 | 1.000 | 0.985 | 0.977 |
| graph only | 0.533 | 0.650 | 0.587 | 0.547 |
| hybrid (RRF) | 0.992 | 1.000 | 0.993 | 0.990 |
| hybrid+rerank | 0.981 | 0.990 | 0.984 | 0.981 |
| hybrid+graph+rerank | 0.983 | 0.996 | 0.987 | 0.981 |

## Query mode: code_only

| System | recall@5 | recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| dense only | 0.973 | 0.998 | 0.982 | 0.977 |
| bm25 only | 0.932 | 0.973 | 0.938 | 0.924 |
| graph only | 0.533 | 0.650 | 0.587 | 0.547 |
| hybrid (RRF) | 0.975 | 0.998 | 0.973 | 0.963 |
| hybrid+rerank | 0.981 | 1.000 | 0.988 | 0.981 |
| hybrid+graph+rerank | 0.981 | 1.000 | 0.988 | 0.981 |

