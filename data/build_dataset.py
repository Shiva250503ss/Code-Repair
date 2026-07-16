"""Build the code-repair training dataset from MBPP.

Pipeline (everything below actually executes -- no step is simulated):
  1. Load all splits of MBPP from Hugging Face (real problems, real
     reference solutions, real test lists).
  2. Quality filter: run every reference solution against its own tests in
     the Part A sandbox; drop any problem whose reference does not pass.
  3. For each surviving problem, inject 3-5 single-site bugs (different bug
     types where possible) into the reference solution, run each variant in
     the sandbox, and keep it only if it genuinely fails -- storing the
     actual captured traceback as the error.
  4. Dedupe identical broken variants.
  5. Emit dataset.jsonl (training pairs), dpo_pairs.jsonl (chosen = real
     reference fix, rejected = a different verified-failing variant of the
     same problem), and report.md with real before/after counts.

HumanEval is never touched here -- it is reserved exclusively for the final
Part E evaluation (see build_eval_set.py).

Usage:  python data/build_dataset.py [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.mutations import (BUG_TYPES, apply_mutation,
                            candidates_grouped_and_shuffled, normalize)
from sandbox.executor import run_tests

OUT_DIR = os.path.join(_REPO_ROOT, "data", "out")
TARGET_VARIANTS = 5
MIN_VARIANTS = 1
MAX_SANDBOX_ATTEMPTS = 14   # per problem, bounds total runtime
SANDBOX_TIMEOUT_S = 6.0

_KEEP_FRAME = re.compile(r'  File "(solution\.py|test_case_\d+\.py|test_setup\.py)"')
_ANY_FRAME = re.compile(r'  File "')


def clean_traceback(tb: str, max_chars: int = 1800) -> str:
    """Drop sandbox-runner frames (with their absolute temp paths) from a
    real traceback, keeping only frames in the user's code and tests plus
    the exception itself. Content is otherwise verbatim."""
    if not tb:
        return tb
    lines = tb.splitlines()
    out, skipping = [], False
    for line in lines:
        if _ANY_FRAME.match(line):
            skipping = not _KEEP_FRAME.match(line)
            if not skipping:
                out.append(line)
        elif line.startswith("    ") and skipping:
            continue  # code/caret line of a skipped frame
        else:
            skipping = False
            out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned[:max_chars]


def error_type(error: str) -> str:
    last = error.strip().splitlines()[-1] if error.strip() else ""
    return last.split(":")[0].strip() or "UnknownError"


def load_mbpp():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full")
    problems = []
    for split in ds:
        for row in ds[split]:
            problems.append({
                "task_id": row["task_id"],
                "problem": row["text"],
                "code": row["code"],
                "tests": list(row["test_list"]),
                "test_setup": row.get("test_setup_code") or "",
                "mbpp_split": split,
            })
    return problems


def validate_reference(p: dict) -> dict:
    try:
        p["norm_code"] = normalize(p["code"])
    except SyntaxError:
        p["ref_ok"] = False
        p["ref_fail_reason"] = "reference does not parse"
        return p
    r = run_tests(p["norm_code"], p["tests"], setup_code=p["test_setup"],
                  timeout_s=SANDBOX_TIMEOUT_S)
    p["ref_ok"] = r.ok
    if not r.ok:
        p["ref_fail_reason"] = (
            "timeout" if r.timed_out
            else error_type(r.first_error or "unknown"))
    return p


# wrong_variable mutations nearly always break code, so without a cap they
# crowd out the other bug types (measured ~50% of pairs in a smoke run).
PER_TYPE_CAPS = {"wrong_variable": 2}


def make_variants(p: dict) -> dict:
    """Generate up to TARGET_VARIANTS verified-failing bug variants,
    round-robin across bug types so one problem covers several types."""
    groups = candidates_grouped_and_shuffled(p["norm_code"], seed=p["task_id"])
    order = [b for b in BUG_TYPES if groups[b]]
    variants, seen_code, attempts = [], {p["norm_code"].strip()}, 0
    type_counts: collections.Counter = collections.Counter()
    gi = 0
    while (len(variants) < TARGET_VARIANTS
           and attempts < MAX_SANDBOX_ATTEMPTS
           and any(groups[b] for b in order)):
        bug_type = order[gi % len(order)]
        gi += 1
        if not groups[bug_type]:
            continue
        cand = groups[bug_type].pop()
        mutated = apply_mutation(p["norm_code"], cand)
        if mutated is None or mutated.strip() in seen_code:
            continue
        seen_code.add(mutated.strip())
        attempts += 1
        r = run_tests(mutated, p["tests"], setup_code=p["test_setup"],
                      timeout_s=SANDBOX_TIMEOUT_S)
        if r.ok:
            continue  # semantically neutral mutation -- not a real bug
        error = clean_traceback(r.first_error or "")
        if not error:
            continue
        failing_test = next((t.source for t in r.tests if t.error), None)
        variants.append({
            "bug_type": bug_type,
            "broken_code": mutated,
            "error": error,
            "error_type": error_type(error),
            "failing_test": failing_test,
            "timed_out": r.timed_out,
            "tests_failed": (r.tests_total - r.tests_passed) or None,
            "tests_total": r.tests_total or None,
        })
        type_counts[bug_type] += 1
        if type_counts[bug_type] >= PER_TYPE_CAPS.get(bug_type, TARGET_VARIANTS):
            groups[bug_type].clear()
    p["variants"] = variants
    p["mutation_attempts"] = attempts
    return p


def build_records(problems: list[dict]) -> tuple[list[dict], list[dict]]:
    records, dpo_pairs = [], []
    for p in problems:
        for i, v in enumerate(p["variants"], 1):
            rec = {
                "id": f"mbpp_{p['task_id']}_v{i}",
                "source": "mbpp",
                "task_id": p["task_id"],
                "mbpp_split": p["mbpp_split"],
                "problem": p["problem"],
                "broken_code": v["broken_code"],
                "error": v["error"],
                "error_type": v["error_type"],
                "failing_test": v["failing_test"],
                "bug_type": v["bug_type"],
                "fixed_code": p["norm_code"],
                "tests": p["tests"],
                "test_setup": p["test_setup"],
            }
            records.append(rec)
        # DPO: rejected = a different variant of the same problem -- a
        # plausible fix attempt already verified to fail the real tests.
        vs = p["variants"]
        for i, v in enumerate(vs):
            if len(vs) < 2:
                break
            rej = vs[(i + 1) % len(vs)]
            dpo_pairs.append({
                "id": f"mbpp_{p['task_id']}_dpo{i + 1}",
                "task_id": p["task_id"],
                "problem": p["problem"],
                "broken_code": v["broken_code"],
                "error": v["error"],
                "chosen": p["norm_code"],
                "rejected": rej["broken_code"],
                "rejected_verified_failing": True,
                "rejected_bug_type": rej["bug_type"],
            })
    return records, dpo_pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N problems (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.monotonic()

    print("[1/5] loading MBPP from Hugging Face ...")
    problems = load_mbpp()
    total_loaded = len(problems)
    if args.limit:
        problems = problems[:args.limit]
    print(f"      loaded {total_loaded} problems "
          f"({'processing ' + str(len(problems)) if args.limit else 'all'})")

    print("[2/5] validating reference solutions in the sandbox ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        problems = list(pool.map(validate_reference, problems))
    valid = [p for p in problems if p["ref_ok"]]
    failed_refs = [p for p in problems if not p["ref_ok"]]
    print(f"      {len(valid)} references pass their own tests, "
          f"{len(failed_refs)} dropped")

    # Dedupe identical (code, tests) problems, if any.
    seen, deduped = set(), []
    for p in valid:
        key = (p["norm_code"].strip(), tuple(p["tests"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    dupes_dropped = len(valid) - len(deduped)
    print(f"      {dupes_dropped} duplicate problems dropped")

    print(f"[3/5] generating bug variants "
          f"({len(deduped)} problems x up to {TARGET_VARIANTS}, "
          f"every variant executed in the sandbox) ...")
    done = 0
    def _work(p):
        nonlocal done
        p = make_variants(p)
        done += 1
        if done % 100 == 0:
            print(f"      {done}/{len(deduped)} problems processed "
                  f"({time.monotonic() - t0:.0f}s elapsed)")
        return p
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        deduped = list(pool.map(_work, deduped))

    with_variants = [p for p in deduped if len(p["variants"]) >= MIN_VARIANTS]
    print(f"      {len(with_variants)} problems produced at least "
          f"{MIN_VARIANTS} verified-failing variant(s)")

    print("[4/5] building records and DPO pairs ...")
    records, dpo_pairs = build_records(with_variants)

    dataset_path = os.path.join(OUT_DIR, "dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dpo_path = os.path.join(OUT_DIR, "dpo_pairs.jsonl")
    with open(dpo_path, "w", encoding="utf-8") as f:
        for rec in dpo_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[5/5] writing report ...")
    bug_counts = collections.Counter(r["bug_type"] for r in records)
    err_counts = collections.Counter(r["error_type"] for r in records)
    per_problem = collections.Counter(len(p["variants"]) for p in with_variants)
    elapsed = time.monotonic() - t0

    lines = [
        "# MBPP Code-Repair Dataset Report",
        "",
        f"Generated by data/build_dataset.py in {elapsed:.0f}s "
        f"({args.workers} sandbox workers). All counts below are measured, "
        "not estimated.",
        "",
        "## Before / after",
        "",
        "| Stage | Count |",
        "|---|---|",
        f"| MBPP problems loaded (all splits) | {total_loaded} |",
        f"| Reference solutions that pass their own tests | {len(valid)} |",
        f"| Dropped: reference fails own tests | {len(failed_refs)} |",
        f"| Dropped: duplicate (code, tests) | {dupes_dropped} |",
        f"| Problems yielding >= 1 verified-failing variant | {len(with_variants)} |",
        f"| **Final training pairs (broken -> fixed)** | **{len(records)}** |",
        f"| DPO preference pairs | {len(dpo_pairs)} |",
        "",
        "## Variants per problem",
        "",
        "| Variants | Problems |",
        "|---|---|",
    ]
    for k in sorted(per_problem):
        lines.append(f"| {k} | {per_problem[k]} |")
    lines += [
        "",
        "## Breakdown by bug type",
        "",
        "| Bug type | Pairs |",
        "|---|---|",
    ]
    for bt in BUG_TYPES:
        lines.append(f"| {bt} | {bug_counts.get(bt, 0)} |")
    lines += [
        "",
        "## Captured error types (top 12)",
        "",
        "| Error | Pairs |",
        "|---|---|",
    ]
    for et, n in err_counts.most_common(12):
        lines.append(f"| {et} | {n} |")
    lines += [
        "",
        "## Dropped references (reason histogram)",
        "",
        "| Reason | Problems |",
        "|---|---|",
    ]
    for reason, n in collections.Counter(
            p["ref_fail_reason"] for p in failed_refs).most_common():
        lines.append(f"| {reason} | {n} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Both broken and fixed code are AST-normalized (ast.unparse) so a "
        "pair differs only by the injected bug, never by formatting.",
        "- Every error string is the actual traceback captured from the "
        "sandbox run of that exact variant (sandbox-runner frames stripped, "
        "user-code frames verbatim).",
        "- Mutations that did not change behavior (still passed all tests) "
        "were discarded.",
        "- HumanEval is fully held out; see build_eval_set.py.",
    ]
    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("-" * 60)
    print(f"training pairs : {len(records)}")
    print(f"dpo pairs      : {len(dpo_pairs)}")
    print("bug types      : " + ", ".join(
        f"{bt}={bug_counts.get(bt, 0)}" for bt in BUG_TYPES))
    print(f"wrote {dataset_path}")
    print(f"wrote {dpo_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
