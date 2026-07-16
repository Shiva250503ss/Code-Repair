"use strict";

const $ = (id) => document.getElementById(id);
let lastRunError = "";
let lastFixCode = "";

function setBadge(el, state, text) {
  el.className = "badge " + state;
  el.textContent = text;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} returned HTTP ${res.status}`);
  return res.json();
}

function renderTests(listEl, tests) {
  listEl.innerHTML = "";
  for (const t of tests) {
    const li = document.createElement("li");
    li.className = t.passed ? "pass" : "fail";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = t.passed ? "PASS" : "FAIL";
    li.appendChild(tag);
    li.appendChild(document.createTextNode(t.source));
    listEl.appendChild(li);
  }
}

function renderSandboxResult(prefix, result) {
  const status = $(prefix + "-status");
  if (result.timed_out) setBadge(status, "fail", "TIMEOUT");
  else if (result.ok) setBadge(status, "pass", "ALL TESTS PASS");
  else setBadge(status, "fail", "FAILING");

  const passed = result.tests.filter((t) => t.passed).length;
  $(prefix + "-meta").textContent =
    `${passed}/${result.tests.length} tests passed in ` +
    `${result.duration_s.toFixed(2)}s (exit code ` +
    `${result.exit_code === null ? "killed" : result.exit_code})`;

  renderTests($(prefix + "-tests"), result.tests);

  const errEl = $(prefix + "-error");
  if (result.first_error && !result.ok) {
    errEl.textContent = result.first_error;
    errEl.hidden = false;
  } else {
    errEl.hidden = true;
  }
  if (prefix === "run") {
    const outEl = $("run-stdout");
    if (result.stdout && result.stdout.trim()) {
      outEl.textContent = "stdout:\n" + result.stdout;
      outEl.hidden = false;
    } else outEl.hidden = true;
  }
}

function renderDiff(lines) {
  const el = $("fix-diff");
  el.innerHTML = "";
  if (!lines.length) {
    el.textContent = "(model returned code identical to the input)";
    return;
  }
  for (const line of lines) {
    const span = document.createElement("span");
    if (line.startsWith("+")) span.className = "add";
    else if (line.startsWith("-")) span.className = "del";
    else if (line.startsWith("@@")) span.className = "hunk";
    else span.className = "ctx";
    span.textContent = line;
    el.appendChild(span);
  }
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    $("h-sandbox").textContent = h.sandbox;
    $("h-sandbox").className = "ok";
    $("h-model").textContent = `${h.model} (${h.model_status})`;
    $("h-model").className = h.model_ok ? "ok" : "bad";
    $("h-retrieval").textContent = h.retrieval_status;
    $("h-retrieval").className = h.retrieval_ok ? "ok" : "bad";
  } catch (e) {
    $("h-sandbox").textContent = "server unreachable";
    $("h-sandbox").className = "bad";
  }
}

$("btn-run").addEventListener("click", async () => {
  const btn = $("btn-run");
  btn.disabled = true;
  btn.textContent = "Running...";
  try {
    const data = await api("/api/run", {
      code: $("code").value,
      tests: $("tests").value,
    });
    $("panel-sandbox").hidden = false;
    if (data.error) {
      setBadge($("run-status"), "fail", "ERROR");
      $("run-meta").textContent = data.error;
      $("btn-fix").disabled = true;
      return;
    }
    renderSandboxResult("run", data.result);
    lastRunError = data.result.first_error || "";
    $("btn-fix").disabled = data.result.ok;
    $("panel-fix").hidden = true;
    $("panel-verify").hidden = true;
  } catch (e) {
    $("panel-sandbox").hidden = false;
    setBadge($("run-status"), "fail", "ERROR");
    $("run-meta").textContent = String(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run in sandbox";
  }
});

$("btn-fix").addEventListener("click", async () => {
  const btn = $("btn-fix");
  btn.disabled = true;
  btn.textContent = "Generating...";
  try {
    const data = await api("/api/fix", {
      problem: $("problem").value,
      code: $("code").value,
      tests: $("tests").value,
      error: lastRunError,
      use_retrieval: $("use-retrieval").checked,
    });
    $("panel-fix").hidden = false;
    if (data.error) {
      $("fix-meta").textContent = data.error;
      $("fix-diff").textContent = "";
      $("fix-code").textContent = "";
      return;
    }
    lastFixCode = data.fix_code;
    $("fix-meta").textContent =
      `model ${data.model}, ${data.latency_s}s` +
      (data.retrieval_used ? ", with retrieval context" : ", no retrieval");
    const ret = $("fix-retrieved");
    if (data.retrieved && data.retrieved.length) {
      ret.innerHTML = "<b>Retrieved reference repairs:</b> " +
        data.retrieved.map((r) =>
          `${r.id} [${r.bug_type}]`).join(", ");
      ret.hidden = false;
    } else ret.hidden = true;
    renderDiff(data.diff);
    $("fix-code").textContent = data.fix_code;
    $("panel-verify").hidden = true;
  } catch (e) {
    $("panel-fix").hidden = false;
    $("fix-meta").textContent = String(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate fix";
  }
});

$("btn-verify").addEventListener("click", async () => {
  const btn = $("btn-verify");
  btn.disabled = true;
  btn.textContent = "Verifying...";
  try {
    const data = await api("/api/verify", {
      code: lastFixCode,
      tests: $("tests").value,
    });
    $("panel-verify").hidden = false;
    if (data.error) {
      setBadge($("verify-status"), "fail", "ERROR");
      $("verify-meta").textContent = data.error;
      return;
    }
    renderSandboxResult("verify", data.result);
  } catch (e) {
    $("panel-verify").hidden = false;
    setBadge($("verify-status"), "fail", "ERROR");
    $("verify-meta").textContent = String(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Verify fix in sandbox";
  }
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.remove("active"));
    tab.classList.add("active");
    const diffView = tab.dataset.view === "diff";
    $("fix-diff").hidden = !diffView;
    $("fix-code").hidden = diffView;
  });
});

refreshHealth();
setInterval(refreshHealth, 15000);
