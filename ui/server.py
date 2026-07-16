"""Code Repair Assistant -- web UI backend.

Endpoints:
  GET  /              the single-page interface
  GET  /api/health    sandbox / model endpoint / retrieval index status
  POST /api/run       execute code against tests in the Part A sandbox
  POST /api/fix       retrieve context, ask the configured model for a fix
  POST /api/verify    re-run a proposed fix through the sandbox

Every pass/fail shown in the UI comes from a real sandbox execution.

Run:  python -m uvicorn ui.server:app --port 8000   (from the repo root)
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import threading
import time

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from sandbox.executor import run_tests
from ui import config

app = FastAPI(title="Code Repair Assistant")

_retriever = None
_retriever_error: str | None = None
_retriever_lock = threading.Lock()


def get_retriever():
    """Lazy-load the retrieval stack; degrade cleanly if the index is not
    available (not built yet, or opened by another process) rather than
    failing the whole request. Failures are retried on the next call."""
    global _retriever, _retriever_error
    with _retriever_lock:
        if _retriever is None:
            try:
                from rag.retriever import Retriever
                _retriever = Retriever()
                _retriever_error = None
            except Exception as exc:
                _retriever_error = f"{type(exc).__name__}: {exc}"
        return _retriever


class RunRequest(BaseModel):
    code: str
    tests: str            # one assert statement per line
    test_setup: str = ""


class FixRequest(BaseModel):
    problem: str = ""
    code: str
    tests: str = ""
    test_setup: str = ""
    error: str = ""
    use_retrieval: bool = True


def parse_tests(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def sandbox_payload(result) -> dict:
    return {
        "ok": result.ok,
        "timed_out": result.timed_out,
        "exit_code": result.exit_code,
        "stdout": result.stdout[-4000:],
        "duration_s": round(result.duration_s, 3),
        "fatal_error": result.fatal_error,
        "first_error": result.first_error,
        "tests": [{"source": t.source, "passed": t.passed, "error": t.error}
                  for t in result.tests],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "static", "index.html"))


@app.get("/api/health")
def health():
    model_ok, model_detail = False, ""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        model_ok = any(n == config.OLLAMA_MODEL
                       or n.split(":")[0] == config.OLLAMA_MODEL
                       for n in names)
        model_detail = ("available" if model_ok else
                        f"endpoint up, model not pulled (have: "
                        f"{', '.join(names) or 'none'})")
    except requests.RequestException as exc:
        model_detail = f"endpoint unreachable: {type(exc).__name__}"

    # Never load the retrieval stack here -- a health check must stay fast.
    # It loads lazily on the first /api/fix request instead.
    from rag.store import DOCS_PATH, QDRANT_PATH
    if _retriever is not None:
        retrieval_ok, retrieval_status = True, "index loaded"
    elif _retriever_error:
        retrieval_ok, retrieval_status = False, _retriever_error
    elif os.path.exists(DOCS_PATH) and os.path.exists(QDRANT_PATH):
        retrieval_ok = True
        retrieval_status = "index present (loads on first fix request)"
    else:
        retrieval_ok = False
        retrieval_status = "index not built -- run rag/build_index.py"
    return {
        "sandbox": "ready",
        "model": config.OLLAMA_MODEL,
        "model_status": model_detail if not model_ok else "available",
        "model_ok": model_ok,
        "retrieval_ok": retrieval_ok,
        "retrieval_status": retrieval_status,
    }


@app.post("/api/run")
def run(req: RunRequest):
    tests = parse_tests(req.tests)
    if not tests:
        return {"error": "no test cases provided"}
    result = run_tests(req.code, tests, setup_code=req.test_setup,
                       timeout_s=config.SANDBOX_TIMEOUT_S,
                       memory_mb=config.SANDBOX_MEMORY_MB)
    return {"result": sandbox_payload(result)}


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def build_prompt(req: FixRequest, context: str) -> str:
    parts = [
        "You are a precise Python code repair assistant. A solution is "
        "failing its tests. Produce a corrected, complete version of the "
        "code. Change only what is necessary to fix the bug. Reply with "
        "exactly one fenced Python code block and nothing else.",
    ]
    if context:
        parts.append("Similar past repairs for reference:\n\n" + context)
    if req.problem.strip():
        parts.append("Problem description:\n" + req.problem.strip())
    parts.append("Broken code:\n```python\n" + req.code.strip() + "\n```")
    if req.error.strip():
        parts.append("Error produced when running the tests:\n```\n"
                     + req.error.strip() + "\n```")
    if req.tests.strip():
        parts.append("The fix must pass these tests:\n```\n"
                     + req.tests.strip() + "\n```")
    return "\n\n".join(parts)


@app.post("/api/fix")
def fix(req: FixRequest):
    retrieved, context = [], ""
    if req.use_retrieval:
        retriever = get_retriever()
        if retriever is not None:
            hits = retriever.retrieve(req.problem, req.code, req.error,
                                      k=config.RETRIEVAL_K)
            context = retriever.format_context(hits)
            retrieved = [{"id": h["id"], "bug_type": h["bug_type"],
                          "problem": h["problem"]} for h in hits]

    prompt = build_prompt(req, context)
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": prompt,
                  "stream": False, "options": config.GENERATION_OPTIONS},
            timeout=config.GENERATION_TIMEOUT_S)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except requests.RequestException as exc:
        return {"error": f"model endpoint error: {type(exc).__name__}: {exc}",
                "model": config.OLLAMA_MODEL}
    latency = time.monotonic() - t0

    fix_code = extract_code(raw)
    diff = list(difflib.unified_diff(
        req.code.strip().splitlines(), fix_code.splitlines(),
        fromfile="broken.py", tofile="fixed.py", lineterm=""))
    return {
        "fix_code": fix_code,
        "diff": diff,
        "model": config.OLLAMA_MODEL,
        "latency_s": round(latency, 2),
        "retrieval_used": bool(context),
        "retrieved": retrieved,
    }


@app.post("/api/verify")
def verify(req: RunRequest):
    return run(req)


app.mount("/static", StaticFiles(
    directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static")
