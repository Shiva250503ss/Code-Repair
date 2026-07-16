"""AST-level bug injection for real reference solutions.

This is the single deliberately-synthetic step in the project: taking a real,
passing reference solution and introducing one realistic bug. Everything
downstream stays real -- every mutated variant is executed in the sandbox and
only kept if it genuinely fails its problem's real tests, and the error text
stored with it is the actual captured traceback, never a template.

Five bug families:
    off_by_one        an integer constant nudged by +/-1 (range bounds,
                      slice indices, comparisons)
    wrong_operator    one arithmetic operator swapped (+ <-> -, * <-> //, ...)
    wrong_comparison  one comparison swapped (< <-> <=, == <-> !=, < <-> >)
    wrong_variable    one variable read replaced with another in-scope name
    missing_edge_case an if-guard (typically an early return for a special
                      case) deleted entirely

Mutations are enumerated as (bug_type, site_index) candidates over the parsed
AST, then applied one at a time to a fresh copy of the tree. Both the broken
variant and the reference target are produced with ast.unparse so the pair
differs only by the injected bug, not by formatting.
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass

BUG_TYPES = ("off_by_one", "wrong_operator", "wrong_comparison",
             "wrong_variable", "missing_edge_case")

_BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Div: ast.FloorDiv, ast.Mod: ast.FloorDiv,
    ast.Pow: ast.Mult,
    ast.LShift: ast.RShift, ast.RShift: ast.LShift,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
}

_CMP_SWAPS = {
    ast.Lt: (ast.LtE, ast.Gt),
    ast.LtE: (ast.Lt, ast.GtE),
    ast.Gt: (ast.GtE, ast.Lt),
    ast.GtE: (ast.Gt, ast.LtE),
    ast.Eq: (ast.NotEq,),
    ast.NotEq: (ast.Eq,),
}


@dataclass(frozen=True)
class MutationCandidate:
    bug_type: str
    site: int       # index into the enumeration for that bug type
    choice: int     # which replacement at that site (some sites offer several)


def normalize(source: str) -> str:
    """Canonical formatting for a solution (what both sides of a pair use)."""
    return ast.unparse(ast.parse(source))


def _walk_with_parents(tree: ast.AST):
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for i, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        yield parent, field, i, child
            elif isinstance(value, ast.AST):
                yield parent, field, None, value


def _int_constant_sites(tree: ast.AST) -> list[ast.Constant]:
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and type(node.value) is int
                and not isinstance(node.value, bool)):
            sites.append(node)
    return sites


def _binop_sites(tree: ast.AST) -> list[ast.AST]:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAPS:
            sites.append(node)
        elif isinstance(node, ast.AugAssign) and type(node.op) in _BINOP_SWAPS:
            sites.append(node)
    return sites


def _cmp_sites(tree: ast.AST) -> list[tuple[ast.Compare, int]]:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _CMP_SWAPS:
                    sites.append((node, i))
    return sites


def _name_load_sites(
        tree: ast.AST) -> list[tuple[ast.AST, ast.Name, list[str]]]:
    """(function, name-load, alternative-names) triples where at least one
    other in-scope name of the same function exists to swap in."""
    sites = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scope_names = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                scope_names.add(node.id)
        if len(scope_names) < 2:
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in scope_names):
                sites.append((fn, node, sorted(scope_names - {node.id})))
    return sites


def _guard_if_sites(tree: ast.AST) -> list[tuple[ast.AST, str, int, ast.If]]:
    """If statements safe to delete: no else branch, short body, and the
    containing block keeps at least one other statement."""
    sites = []
    for parent, field, idx, child in _walk_with_parents(tree):
        if (isinstance(child, ast.If) and idx is not None
                and not child.orelse and len(child.body) <= 3):
            block = getattr(parent, field)
            if len(block) > 1:
                sites.append((parent, field, idx, child))
    return sites


def enumerate_candidates(source: str) -> list[MutationCandidate]:
    tree = ast.parse(source)
    out: list[MutationCandidate] = []
    for i in range(len(_int_constant_sites(tree))):
        out.append(MutationCandidate("off_by_one", i, 0))   # +1
        out.append(MutationCandidate("off_by_one", i, 1))   # -1
    for i in range(len(_binop_sites(tree))):
        out.append(MutationCandidate("wrong_operator", i, 0))
    for i, (_node, _j) in enumerate(_cmp_sites(tree)):
        n_choices = len(_CMP_SWAPS[type(_node.ops[_j])])
        for c in range(n_choices):
            out.append(MutationCandidate("wrong_comparison", i, c))
    for i, (_fn, _node, alts) in enumerate(_name_load_sites(tree)):
        for c in range(min(len(alts), 3)):
            out.append(MutationCandidate("wrong_variable", i, c))
    for i in range(len(_guard_if_sites(tree))):
        out.append(MutationCandidate("missing_edge_case", i, 0))
    return out


def apply_mutation(source: str, cand: MutationCandidate) -> str | None:
    """Return mutated source, or None if the candidate is inapplicable /
    produces code identical to the original."""
    tree = ast.parse(source)

    if cand.bug_type == "off_by_one":
        sites = _int_constant_sites(tree)
        if cand.site >= len(sites):
            return None
        node = sites[cand.site]
        node.value = node.value + (1 if cand.choice == 0 else -1)

    elif cand.bug_type == "wrong_operator":
        sites = _binop_sites(tree)
        if cand.site >= len(sites):
            return None
        node = sites[cand.site]
        node.op = _BINOP_SWAPS[type(node.op)]()

    elif cand.bug_type == "wrong_comparison":
        sites = _cmp_sites(tree)
        if cand.site >= len(sites):
            return None
        node, j = sites[cand.site]
        choices = _CMP_SWAPS[type(node.ops[j])]
        if cand.choice >= len(choices):
            return None
        node.ops[j] = choices[cand.choice]()

    elif cand.bug_type == "wrong_variable":
        sites = _name_load_sites(tree)
        if cand.site >= len(sites):
            return None
        _fn, node, alts = sites[cand.site]
        if cand.choice >= len(alts):
            return None
        node.id = alts[cand.choice]

    elif cand.bug_type == "missing_edge_case":
        sites = _guard_if_sites(tree)
        if cand.site >= len(sites):
            return None
        parent, field, idx, _child = sites[cand.site]
        getattr(parent, field).pop(idx)

    else:
        raise ValueError(f"unknown bug type: {cand.bug_type}")

    try:
        mutated = ast.unparse(ast.fix_missing_locations(tree))
    except Exception:
        return None
    if mutated.strip() == normalize(source).strip():
        return None
    return mutated


def candidates_grouped_and_shuffled(
        source: str, seed: int) -> dict[str, list[MutationCandidate]]:
    """Candidates grouped by bug type, each group shuffled reproducibly."""
    rng = random.Random(seed)
    groups: dict[str, list[MutationCandidate]] = {b: [] for b in BUG_TYPES}
    for cand in enumerate_candidates(source):
        groups[cand.bug_type].append(cand)
    for group in groups.values():
        rng.shuffle(group)
    return groups
