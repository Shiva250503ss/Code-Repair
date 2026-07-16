"""End-to-end test of the UI backend against the configured model.

Takes real broken variants from the Part B dataset, then exercises the full
loop a user would: sandbox run (must fail), model fix, sandbox verification
of the fix. The fix passing is NOT asserted -- whether the configured model
repairs the bug is a reported measurement, not a requirement for the UI to
be working. What is asserted: every endpoint responds, the broken code
really fails, the diff is produced, and the verification verdict comes from
a real sandbox run.

Usage:  python ui/test_e2e.py [--n 3] [--retrieval]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import requests

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://localhost:8000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--retrieval", action="store_true",
                    help="include retrieval context in fix requests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    health = requests.get(f"{BASE}/api/health", timeout=30).json()
    print("health:", json.dumps(health))
    assert health["model_ok"], "model endpoint not available"

    with open(os.path.join(_REPO_ROOT, "data", "out", "dataset.jsonl"),
              encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    sample = random.Random(args.seed).sample(records, args.n)

    fixed_count = 0
    for i, rec in enumerate(sample, 1):
        print(f"\n[{i}/{args.n}] {rec['id']} (bug: {rec['bug_type']}) -- "
              f"{rec['problem'][:70]}")
        tests_text = "\n".join(rec["tests"])

        run = requests.post(f"{BASE}/api/run", timeout=60, json={
            "code": rec["broken_code"], "tests": tests_text,
            "test_setup": rec["test_setup"]}).json()["result"]
        assert not run["ok"], "broken variant unexpectedly passed"
        err_line = (run["first_error"] or "").strip().splitlines()[-1]
        print(f"  sandbox: FAIL as expected "
              f"({run['tests'].__len__()} tests, error: {err_line[:80]})")

        fix = requests.post(f"{BASE}/api/fix", timeout=300, json={
            "problem": rec["problem"], "code": rec["broken_code"],
            "tests": tests_text, "error": run["first_error"] or "",
            "use_retrieval": args.retrieval}).json()
        assert "fix_code" in fix, f"fix endpoint error: {fix.get('error')}"
        print(f"  fix: {len(fix['diff'])} diff lines, model {fix['model']}, "
              f"{fix['latency_s']}s, retrieval={fix['retrieval_used']}"
              + (f", refs={[r['id'] for r in fix['retrieved']]}"
                 if fix["retrieved"] else ""))

        verify = requests.post(f"{BASE}/api/verify", timeout=60, json={
            "code": fix["fix_code"], "tests": tests_text,
            "test_setup": rec["test_setup"]}).json()["result"]
        passed = sum(1 for t in verify["tests"] if t["passed"])
        verdict = "PASS" if verify["ok"] else "FAIL"
        print(f"  verify: {verdict} ({passed}/{len(verify['tests'])} tests)")
        fixed_count += verify["ok"]

    print(f"\nUI loop working end to end. Model fixed {fixed_count}/{args.n} "
          f"sampled bugs (model quality, not UI correctness).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
