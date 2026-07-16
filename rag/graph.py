"""GraphRAG: a knowledge graph over the real code-repair dataset.

Nodes (all extracted from the data, none invented):
  task:<task_id>      one per MBPP problem in the corpus
  bug:<bug_type>      the five injected bug families actually present
  topic:<token>       function-name tokens that recur across problems
                      (frequency-filtered so only real shared vocabulary
                      becomes a node)
  sig:<shape>         structural signature features parsed from the code:
                      argument count, loops, recursion

Edges connect a task to every bug/topic/sig node observed in its data.

Multi-hop retrieval: a query's error text maps to likely bug types via a
conditional table P(bug_type | error_type) counted from the dataset itself,
its code parses to topic/sig nodes, and candidate tasks are those reachable
from BOTH a bug node and a shape node -- "a similar bug in a similarly
shaped function", not nearest-neighbor lookup.
"""

from __future__ import annotations

import ast
import collections
import os
import pickle
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(_REPO_ROOT, "rag", "index", "graph.pkl")

# Topic tokens must appear in at least MIN_DF tasks and at most MAX_DF_FRAC
# of all tasks (ubiquitous verbs like "find" carry no signal).
MIN_DF = 3
MAX_DF_FRAC = 0.25

_NAME_RE = re.compile(r"assert\s+(\w+)\s*\(")


def entry_function_name(rec: dict) -> str | None:
    """The function under test: taken from the real test call if possible,
    else the last function defined in the code."""
    for t in rec.get("tests") or []:
        m = _NAME_RE.search(t)
        if m:
            return m.group(1)
    try:
        tree = ast.parse(rec["fixed_code" if "fixed_code" in rec else "broken_code"])
        fns = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return fns[-1] if fns else None
    except SyntaxError:
        return None


def name_tokens(fn_name: str | None) -> list[str]:
    if not fn_name:
        return []
    return [p for p in fn_name.lower().split("_") if len(p) > 2]


def signature_shape(code: str, fn_name: str | None) -> list[str]:
    """Structural features of the entry function, parsed from real code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fn_name is None or node.name == fn_name:
                target = node
    if target is None:
        return []
    shapes = [f"nargs={len(target.args.args)}"]
    body_nodes = list(ast.walk(target))
    if any(isinstance(n, (ast.For, ast.While)) for n in body_nodes):
        shapes.append("loop")
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
           and n.func.id == target.name for n in body_nodes):
        shapes.append("recursion")
    if any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                          ast.GeneratorExp)) for n in body_nodes):
        shapes.append("comprehension")
    return shapes


def build_graph(records: list[dict]):
    import networkx as nx

    g = nx.Graph()
    task_docs: dict[str, list[str]] = collections.defaultdict(list)
    task_meta: dict[str, dict] = {}
    df = collections.Counter()

    per_task_tokens: dict[str, set] = {}
    for rec in records:
        tid = f"task:{rec['task_id']}"
        task_docs[tid].append(rec["id"])
        if tid not in task_meta:
            fn = entry_function_name(rec)
            toks = set(name_tokens(fn))
            per_task_tokens[tid] = toks
            task_meta[tid] = {
                "fn_name": fn,
                "shapes": signature_shape(rec["fixed_code"], fn),
            }
            for tok in toks:
                df[tok] += 1

    n_tasks = len(task_meta)
    topic_vocab = {tok for tok, n in df.items()
                   if n >= MIN_DF and n <= MAX_DF_FRAC * n_tasks}

    for rec in records:
        tid = f"task:{rec['task_id']}"
        g.add_node(tid, kind="task")
        bug = f"bug:{rec['bug_type']}"
        g.add_node(bug, kind="bug")
        g.add_edge(tid, bug)
    for tid, meta in task_meta.items():
        for tok in per_task_tokens[tid] & topic_vocab:
            node = f"topic:{tok}"
            g.add_node(node, kind="topic")
            g.add_edge(tid, node)
        for shape in meta["shapes"]:
            node = f"sig:{shape}"
            g.add_node(node, kind="sig")
            g.add_edge(tid, node)

    # P(bug_type | error_type), counted from the data. Used at query time to
    # infer likely bug types from an error message (the true bug type is the
    # training label and is never assumed known for a query).
    err_bug = collections.defaultdict(collections.Counter)
    for rec in records:
        err_bug[rec["error_type"]][rec["bug_type"]] += 1
    error_to_bug = {}
    for et, counter in err_bug.items():
        total = sum(counter.values())
        error_to_bug[et] = [(bt, n / total) for bt, n in counter.most_common()]

    return {
        "graph": g,
        "task_docs": dict(task_docs),
        "task_meta": task_meta,
        "topic_vocab": topic_vocab,
        "error_to_bug": error_to_bug,
    }


def save_graph(bundle, path: str = GRAPH_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_graph(path: str = GRAPH_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_bug_types(bundle, error: str, top_n: int = 2) -> list[tuple[str, float]]:
    last = error.strip().splitlines()[-1] if error.strip() else ""
    et = last.split(":")[0].strip()
    return bundle["error_to_bug"].get(et, [])[:top_n]


def graph_candidates(bundle, broken_code: str, error: str,
                     exclude_task: str | None = None,
                     top_tasks: int = 12) -> list[str]:
    """Multi-hop lookup: tasks connected to an inferred bug node AND at
    least one topic/sig node of the query. Returns ranked doc ids."""
    g = bundle["graph"]
    fn = None
    try:
        tree = ast.parse(broken_code)
        fns = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        fn = fns[-1] if fns else None
    except SyntaxError:
        pass

    bug_types = infer_bug_types(bundle, error)
    topic_nodes = [f"topic:{t}" for t in name_tokens(fn)
                   if t in bundle["topic_vocab"]]
    sig_nodes = [f"sig:{s}" for s in signature_shape(broken_code, fn)]

    scores: collections.Counter = collections.Counter()
    bug_hits: set = set()
    shape_hits: set = set()
    for bt, prob in bug_types:
        node = f"bug:{bt}"
        if node in g:
            for task in g.neighbors(node):
                scores[task] += 2.0 * prob
                bug_hits.add(task)
    for node in topic_nodes:
        if node in g:
            for task in g.neighbors(node):
                scores[task] += 1.5
                shape_hits.add(task)
    for node in sig_nodes:
        if node in g:
            for task in g.neighbors(node):
                scores[task] += 0.5
                shape_hits.add(task)

    ranked = []
    for task, _score in scores.most_common():
        if exclude_task and task == f"task:{exclude_task}":
            continue
        # multi-hop requirement: shared bug node AND shared topic/sig node
        if task not in bug_hits or task not in shape_hits:
            continue
        ranked.append(task)
        if len(ranked) >= top_tasks:
            break

    doc_ids = []
    for task in ranked:
        doc_ids.extend(bundle["task_docs"].get(task, []))
    return doc_ids
