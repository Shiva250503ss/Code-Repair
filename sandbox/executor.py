"""Safe execution engine for untrusted / model-generated Python code.

Every run happens in a fresh subprocess with:
  - a hard wall-clock timeout (default 8s), enforced by the parent
  - a hard memory limit (default 512 MB): Windows Job Object on win32,
    RLIMIT_AS on POSIX (so this same file works unchanged on Colab/Linux)
  - network access disabled (socket layer is stubbed out before user code runs)
  - file writes restricted to a per-run scratch temp directory
  - imports of subprocess/ctypes blocked after guard setup
  - a minimal environment and `python -I` (isolated mode)

The child writes structured test results to a JSON file; the parent captures
stdout/stderr/exit code separately. Tracebacks are real Python tracebacks --
user code is compiled from a file named solution.py so errors point there.

Threat model: buggy or careless generated code, not a determined attacker.
OS-level containerization is out of scope; guards are defense in depth on
top of subprocess + timeout + memory-kill isolation.

Stdlib-only on purpose (psutil is used opportunistically if present) so the
exact same file can be uploaded to Colab and reused for Part E evaluation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field

DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MEMORY_MB = 512
RESULT_FILENAME = "__sandbox_results__.json"
SOLUTION_FILENAME = "solution.py"

# Runs inside the child before any user code. SCRATCH_DIR / MEM_BYTES are
# substituted in by the parent. Must be stdlib-only and must not print.
_GUARD_TEMPLATE = r'''
import builtins as _b, os as _os, sys as _sys

_SCRATCH = {scratch!r}
_MEM_BYTES = {mem_bytes}

# ---- memory limit (hard, kernel-enforced) ----
if _os.name == "nt":
    import ctypes as _ct
    from ctypes import wintypes as _wt

    class _BASIC(_ct.Structure):
        _fields_ = [("PerProcessUserTimeLimit", _ct.c_int64),
                    ("PerJobUserTimeLimit", _ct.c_int64),
                    ("LimitFlags", _wt.DWORD),
                    ("MinimumWorkingSetSize", _ct.c_size_t),
                    ("MaximumWorkingSetSize", _ct.c_size_t),
                    ("ActiveProcessLimit", _wt.DWORD),
                    ("Affinity", _ct.c_size_t),
                    ("PriorityClass", _wt.DWORD),
                    ("SchedulingClass", _wt.DWORD)]

    class _IO(_ct.Structure):
        _fields_ = [(n, _ct.c_uint64) for n in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]

    class _EXT(_ct.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC),
                    ("IoInfo", _IO),
                    ("ProcessMemoryLimit", _ct.c_size_t),
                    ("JobMemoryLimit", _ct.c_size_t),
                    ("PeakProcessMemoryUsed", _ct.c_size_t),
                    ("PeakJobMemoryUsed", _ct.c_size_t)]

    _k32 = _ct.windll.kernel32
    # 64-bit HANDLEs get truncated without explicit signatures.
    _k32.CreateJobObjectW.restype = _ct.c_void_p
    _k32.GetCurrentProcess.restype = _ct.c_void_p
    _k32.SetInformationJobObject.argtypes = [
        _ct.c_void_p, _ct.c_int, _ct.c_void_p, _ct.c_uint32]
    _k32.AssignProcessToJobObject.argtypes = [_ct.c_void_p, _ct.c_void_p]

    _job = _k32.CreateJobObjectW(None, None)
    _info = _EXT()
    _info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    _info.ProcessMemoryLimit = _MEM_BYTES
    if not _k32.SetInformationJobObject(_job, 9, _ct.byref(_info),
                                        _ct.sizeof(_info)):
        raise OSError("sandbox: SetInformationJobObject failed")
    if not _k32.AssignProcessToJobObject(_job, _k32.GetCurrentProcess()):
        raise OSError("sandbox: AssignProcessToJobObject failed")
else:
    import resource as _res
    _res.setrlimit(_res.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))

# ---- no network ----
import socket as _socket

def _no_net(*a, **k):
    raise RuntimeError("sandbox: network access is disabled")

for _name in ("socket", "create_connection", "getaddrinfo", "gethostbyname",
              "create_server", "socketpair"):
    if hasattr(_socket, _name):
        setattr(_socket, _name, _no_net)

# ---- file writes restricted to scratch dir ----
_real_open = _b.open
_scratch_real = _os.path.realpath(_SCRATCH)

def _guarded_open(file, mode="r", *a, **k):
    if any(c in str(mode) for c in "wax+"):
        try:
            target = _os.path.realpath(_os.path.abspath(_os.fspath(file)))
        except TypeError:  # file descriptors etc.
            return _real_open(file, mode, *a, **k)
        if not target.startswith(_scratch_real):
            raise PermissionError(
                "sandbox: writing outside the scratch directory is disabled")
    return _real_open(file, mode, *a, **k)

_b.open = _guarded_open

def _no_fs(*a, **k):
    raise PermissionError("sandbox: this filesystem operation is disabled")

for _name in ("remove", "unlink", "rmdir", "rename", "replace", "system",
              "popen", "startfile", "execv", "execve", "spawnv", "spawnve"):
    if hasattr(_os, _name):
        setattr(_os, _name, _no_fs)

# ---- block dangerous imports from here on ----
class _ImportBlocker:
    _blocked = frozenset(("subprocess", "ctypes", "_ctypes", "resource"))

    def find_module(self, name, path=None):
        return self if name.split(".")[0] in self._blocked else None

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self._blocked:
            raise ImportError(f"sandbox: import of {{name!r}} is disabled")
        return None

_sys.meta_path.insert(0, _ImportBlocker())
for _mod in ("ctypes", "ctypes.wintypes", "_ctypes", "resource"):
    _sys.modules.pop(_mod, None)

# NOTE: guard helper names (_real_open etc.) must stay defined -- the guarded
# functions read them from this module's globals at call time. User code
# cannot reach them: it executes inside its own globals dict.
'''

_RUNNER_TEMPLATE = r'''
import json, traceback

_result = {{"fatal": None, "tests": []}}

def _finish():
    with open({result_path!r}, "w", encoding="utf-8") as f:
        json.dump(_result, f)

_g = {{"__name__": "__main__"}}
try:
    with open({solution_path!r}, "r", encoding="utf-8") as f:
        _src = f.read()
    exec(compile(_src, "solution.py", "exec"), _g)
except BaseException:
    _result["fatal"] = traceback.format_exc()
    _finish()
    raise SystemExit(1)

_setup = {setup_code!r}
if _setup.strip():
    try:
        exec(compile(_setup, "test_setup.py", "exec"), _g)
    except BaseException:
        _result["fatal"] = traceback.format_exc()
        _finish()
        raise SystemExit(1)

for _i, _t in enumerate({tests!r}):
    _entry = {{"source": _t, "passed": False, "error": None}}
    try:
        exec(compile(_t, f"test_case_{{_i}}.py", "exec"), _g)
        _entry["passed"] = True
    except BaseException:
        _entry["error"] = traceback.format_exc()
    _result["tests"].append(_entry)

_finish()
raise SystemExit(0 if all(t["passed"] for t in _result["tests"]) else 1)
'''


@dataclass
class TestResult:
    source: str
    passed: bool
    error: str | None = None


@dataclass
class ExecutionResult:
    ok: bool = False
    timed_out: bool = False
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    fatal_error: str | None = None
    tests: list[TestResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def tests_passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def tests_total(self) -> int:
        return len(self.tests)

    @property
    def first_error(self) -> str | None:
        """The real captured error: fatal traceback, first failing test
        traceback, or a timeout marker. None if everything passed."""
        if self.timed_out:
            return "TimeoutError: sandbox: execution exceeded the time limit"
        if self.fatal_error:
            return self.fatal_error
        for t in self.tests:
            if t.error:
                return t.error
        return None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        proc.kill()


def _humaneval_tests(test_code: str, entry_point: str) -> tuple[str, list[str]]:
    """HumanEval ships a check(candidate) function; run it as one test."""
    setup = test_code
    call = f"check({entry_point})"
    return setup, [call]


def run_tests(
    code: str,
    tests: list[str] | str,
    setup_code: str = "",
    entry_point: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    scratch_root: str | None = None,
) -> ExecutionResult:
    """Execute `code` against test cases inside the sandbox.

    tests: list of assert-statement strings (MBPP style), or a single string
    containing a check(candidate) definition (HumanEval style; requires
    entry_point).
    """
    if isinstance(tests, str):
        if not entry_point:
            raise ValueError("string test code requires entry_point")
        extra_setup, tests = _humaneval_tests(tests, entry_point)
        setup_code = (setup_code + "\n" + extra_setup) if setup_code else extra_setup

    scratch = tempfile.mkdtemp(prefix=f"sbx_{uuid.uuid4().hex[:8]}_",
                               dir=scratch_root)
    solution_path = os.path.join(scratch, SOLUTION_FILENAME)
    result_path = os.path.join(scratch, RESULT_FILENAME)
    runner_path = os.path.join(scratch, "__runner__.py")

    with open(solution_path, "w", encoding="utf-8") as f:
        f.write(code)

    guard = _GUARD_TEMPLATE.format(scratch=scratch,
                                   mem_bytes=memory_mb * 1024 * 1024)
    runner = _RUNNER_TEMPLATE.format(result_path=result_path,
                                     solution_path=solution_path,
                                     setup_code=setup_code,
                                     tests=list(tests))
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(guard + runner)

    env = {"TEMP": scratch, "TMP": scratch, "TMPDIR": scratch,
           "PYTHONIOENCODING": "utf-8"}
    for keep in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "HOME"):
        if keep in os.environ:
            env[keep] = os.environ[keep]

    started = time.monotonic()
    result = ExecutionResult()
    proc = subprocess.Popen(
        [sys.executable, "-I", runner_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=scratch, env=env, text=True, encoding="utf-8", errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        result.timed_out = True
        result.exit_code = None
    result.duration_s = time.monotonic() - started
    result.stdout = stdout or ""
    result.stderr = stderr or ""

    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            result.fatal_error = payload.get("fatal")
            result.tests = [TestResult(t["source"], t["passed"], t["error"])
                            for t in payload.get("tests", [])]
        except (json.JSONDecodeError, KeyError):
            result.fatal_error = result.fatal_error or (
                "sandbox: result file was corrupt\n" + result.stderr)
    elif not result.timed_out:
        # Child died before writing results (e.g. hard memory kill).
        result.fatal_error = result.stderr.strip() or (
            f"sandbox: process exited with code {result.exit_code} "
            "before producing results")

    result.ok = (not result.timed_out
                 and result.fatal_error is None
                 and result.tests_total > 0
                 and result.tests_passed == result.tests_total)

    _cleanup(scratch)
    return result


def _cleanup(scratch: str) -> None:
    import shutil
    try:
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception:
        pass
