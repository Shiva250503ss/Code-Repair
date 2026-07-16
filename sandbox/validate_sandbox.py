"""Validation suite for the sandbox itself.

Runs 11 known cases where the correct outcome is known in advance and
verifies the sandbox reports exactly that outcome. Run this before trusting
the sandbox with generated code:

    python sandbox/validate_sandbox.py
"""

from __future__ import annotations

import sys
import time

from executor import run_tests

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("correct solution passes MBPP-style asserts")
def _(r=None):
    r = run_tests(
        "def add(a, b):\n    return a + b\n",
        ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"])
    assert r.ok, f"expected ok, got {r}"
    assert r.tests_passed == 2 and r.tests_total == 2
    return f"ok=True, {r.tests_passed}/{r.tests_total} tests passed"


@check("wrong solution fails with a real AssertionError traceback")
def _():
    r = run_tests(
        "def add(a, b):\n    return a - b\n",
        ["assert add(1, 2) == 3"])
    assert not r.ok
    assert r.tests_passed == 0
    assert "AssertionError" in (r.first_error or "")
    return f"ok=False, captured error ends with: {r.first_error.strip().splitlines()[-1]}"


@check("runtime exception (TypeError) is captured verbatim")
def _():
    r = run_tests(
        "def f(x):\n    return x + '1'\n",
        ["assert f(1) == 2"])
    assert not r.ok
    assert "TypeError" in (r.first_error or "")
    return f"captured: {r.first_error.strip().splitlines()[-1]}"


@check("syntax error is captured as a fatal error")
def _():
    r = run_tests("def f(:\n    pass\n", ["assert True"])
    assert not r.ok
    assert r.fatal_error and "SyntaxError" in r.fatal_error
    return f"fatal: {r.fatal_error.strip().splitlines()[-1]}"


@check("infinite loop hits the timeout, machine stays healthy")
def _():
    t0 = time.monotonic()
    r = run_tests("while True:\n    pass\n", ["assert True"], timeout_s=3.0)
    wall = time.monotonic() - t0
    assert r.timed_out, f"expected timeout, got {r}"
    assert wall < 10, f"kill took too long: {wall:.1f}s"
    return f"timed_out=True after {r.duration_s:.2f}s (limit 3.0s)"


@check("memory bomb is killed by the hard memory limit")
def _():
    r = run_tests(
        "data = bytearray(2 * 1024 * 1024 * 1024)  # 2 GB\n",
        ["assert True"], memory_mb=256, timeout_s=15.0)
    assert not r.ok, f"2 GB allocation succeeded under a 256 MB limit: {r}"
    err = (r.first_error or "") + r.stderr
    assert "MemoryError" in err or r.exit_code not in (0,), (
        f"expected memory kill, got {r}")
    tail = (r.first_error or "process killed").strip().splitlines()[-1]
    return f"blocked: {tail}"


@check("network access is blocked")
def _():
    code = ("import socket\n"
            "def ping():\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.connect(('example.com', 80))\n"
            "    return True\n")
    r = run_tests(code, ["assert ping()"])
    assert not r.ok
    assert "network access is disabled" in (r.first_error or "")
    return f"blocked: {r.first_error.strip().splitlines()[-1]}"


@check("file write outside the scratch dir is blocked")
def _():
    # cwd is the scratch dir; its parent is the shared temp dir -- outside.
    code = ("import os\n"
            "def touch():\n"
            "    target = os.path.join(os.path.dirname(os.getcwd()),\n"
            "                          'sandbox_escape.txt')\n"
            "    with open(target, 'w') as f:\n"
            "        f.write('escaped')\n"
            "    return True\n")
    r = run_tests(code, ["assert touch()"])
    assert not r.ok, f"escape write was allowed: {r}"
    assert "outside the scratch directory" in (r.first_error or ""), (
        f"unexpected error: {r.first_error}")
    return f"blocked: {r.first_error.strip().splitlines()[-1]}"


@check("file write inside the scratch dir is allowed")
def _():
    code = ("def roundtrip():\n"
            "    with open('local.txt', 'w') as f:\n"
            "        f.write('hello')\n"
            "    with open('local.txt') as f:\n"
            "        return f.read()\n")
    r = run_tests(code, ["assert roundtrip() == 'hello'"])
    assert r.ok, f"expected ok, got first_error={r.first_error}"
    return "scratch-local write and read-back succeeded"


@check("stdout is captured, exit code 0 on success")
def _():
    r = run_tests(
        "print('progress message')\ndef f():\n    return 42\n",
        ["assert f() == 42"])
    assert r.ok and r.exit_code == 0
    assert "progress message" in r.stdout
    return f"stdout={r.stdout.strip()!r}, exit_code={r.exit_code}"


@check("HumanEval-style check(candidate) tests work")
def _():
    solution = "def add(a, b):\n    return a + b\n"
    test_code = ("def check(candidate):\n"
                 "    assert candidate(2, 3) == 5\n"
                 "    assert candidate(-1, -1) == -2\n")
    r = run_tests(solution, test_code, entry_point="add")
    assert r.ok, f"expected ok, got first_error={r.first_error}"
    bad = run_tests("def add(a, b):\n    return a * b\n", test_code,
                    entry_point="add")
    assert not bad.ok and "AssertionError" in (bad.first_error or "")
    return "correct candidate passes, broken candidate fails with AssertionError"


def main() -> int:
    print(f"Sandbox validation -- {len(CHECKS)} known cases "
          f"(python {sys.version.split()[0]}, platform {sys.platform})")
    print("-" * 76)
    failures = 0
    for i, (name, fn) in enumerate(CHECKS, 1):
        t0 = time.monotonic()
        try:
            detail = fn()
            status = "PASS"
        except AssertionError as exc:
            detail = str(exc)
            status = "FAIL"
            failures += 1
        dt = time.monotonic() - t0
        print(f"[{i:2d}/{len(CHECKS)}] {status}  {name}  ({dt:.2f}s)")
        print(f"        {detail}")
    print("-" * 76)
    print(f"result: {len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
