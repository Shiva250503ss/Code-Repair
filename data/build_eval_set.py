"""Build the held-out evaluation set from HumanEval.

HumanEval is NEVER used for training. This script produces the broken-code
evaluation set for Part E only, using the same real machinery as Part B:

  1. Load HumanEval (164 real problems, canonical solutions, check() tests).
  2. Validate every canonical solution against its own real tests in the
     Part A sandbox; drop any that do not pass.
  3. Inject up to 3 bug variants per problem (same mutation engine as MBPP),
     keep only variants that genuinely fail, store the actual captured
     traceback.
  4. Emit humaneval_broken.jsonl plus a small report.

Usage:  python data/build_eval_set.py [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.build_dataset import clean_traceback, error_type
from data.mutations import (BUG_TYPES, apply_mutation,
                            candidates_grouped_and_shuffled, normalize)
from sandbox.executor import run_tests

OUT_DIR = os.path.join(_REPO_ROOT, "data", "out")
TARGET_VARIANTS = 3
MAX_SANDBOX_ATTEMPTS = 10
SANDBOX_TIMEOUT_S = 8.0


def load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval")["test"]
    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],              # e.g. "HumanEval/0"
            "problem": row["prompt"],               # signature + docstring
            "code": row["prompt"] + row["canonical_solution"],
            "test_code": row["test"],               # defines check(candidate)
            "entry_point": row["entry_point"],
        })
    return problems


def validate_reference(p: dict) -> dict:
    try:
        p["norm_code"] = normalize(p["code"])
    except SyntaxError:
        p["ref_ok"] = False
        return p
    r = run_tests(p["norm_code"], p["test_code"],
                  entry_point=p["entry_point"], timeout_s=SANDBOX_TIMEOUT_S)
    p["ref_ok"] = r.ok
    return p


def make_variants(p: dict) -> dict:
    seed = int(p["task_id"].split("/")[-1])
    groups = candidates_grouped_and_shuffled(p["norm_code"], seed=seed)
    order = [b for b in BUG_TYPES if groups[b]]
    variants, seen, attempts, gi = [], {p["norm_code"].strip()}, 0, 0
    while (len(variants) < TARGET_VARIANTS
           and attempts < MAX_SANDBOX_ATTEMPTS
           and any(groups[b] for b in order)):
        bug_type = order[gi % len(order)]
        gi += 1
        if not groups[bug_type]:
            continue
        mutated = apply_mutation(p["norm_code"], groups[bug_type].pop())
        if mutated is None or mutated.strip() in seen:
            continue
        seen.add(mutated.strip())
        attempts += 1
        r = run_tests(mutated, p["test_code"], entry_point=p["entry_point"],
                      timeout_s=SANDBOX_TIMEOUT_S)
        if r.ok:
            continue
        error = clean_traceback(r.first_error or "")
        if not error:
            continue
        variants.append({
            "bug_type": bug_type,
            "broken_code": mutated,
            "error": error,
            "error_type": error_type(error),
        })
    p["variants"] = variants
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.monotonic()

    print("[1/3] loading HumanEval (held out -- evaluation only) ...")
    problems = load_humaneval()
    print(f"      loaded {len(problems)} problems")

    print("[2/3] validating canonical solutions in the sandbox ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        problems = list(pool.map(validate_reference, problems))
    valid = [p for p in problems if p["ref_ok"]]
    print(f"      {len(valid)} canonical solutions pass their own tests, "
          f"{len(problems) - len(valid)} dropped")

    print("[3/3] generating verified-failing bug variants ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        valid = list(pool.map(make_variants, valid))

    records = []
    for p in valid:
        for i, v in enumerate(p["variants"], 1):
            records.append({
                "id": f"humaneval_{p['task_id'].split('/')[-1]}_v{i}",
                "source": "humaneval",
                "task_id": p["task_id"],
                "problem": p["problem"],
                "broken_code": v["broken_code"],
                "error": v["error"],
                "error_type": v["error_type"],
                "bug_type": v["bug_type"],
                "fixed_code": p["norm_code"],
                "test_code": p["test_code"],
                "entry_point": p["entry_point"],
            })

    out_path = os.path.join(OUT_DIR, "humaneval_broken.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    bug_counts = collections.Counter(r["bug_type"] for r in records)
    elapsed = time.monotonic() - t0
    report = [
        "# HumanEval Held-Out Evaluation Set Report",
        "",
        f"Generated by data/build_eval_set.py in {elapsed:.0f}s. "
        "This set is for Part E evaluation ONLY and never enters training.",
        "",
        "| Stage | Count |",
        "|---|---|",
        f"| HumanEval problems loaded | {len(problems)} |",
        f"| Canonical solutions passing own tests | {len(valid)} |",
        f"| Problems yielding >= 1 failing variant | "
        f"{sum(1 for p in valid if p['variants'])} |",
        f"| **Evaluation items (broken -> fixed)** | **{len(records)}** |",
        "",
        "| Bug type | Items |",
        "|---|---|",
    ] + [f"| {bt} | {bug_counts.get(bt, 0)} |" for bt in BUG_TYPES]
    report_path = os.path.join(OUT_DIR, "eval_set_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("-" * 60)
    print(f"evaluation items: {len(records)}")
    print("bug types       : " + ", ".join(
        f"{bt}={bug_counts.get(bt, 0)}" for bt in BUG_TYPES))
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
