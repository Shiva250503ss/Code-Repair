# Project Explained -- Plain English Walkthrough

This guide explains every real file in Parts A, B, C, and F, plus the
Colab notebook that carries Parts D and E. For each file: what it does, why
it is built that way, what to say out loud in an interview, and likely
follow-up questions with short answers. The language is kept simple on
purpose.

Not covered below, on purpose: `data/__init__.py`, `rag/__init__.py`,
`sandbox/__init__.py` (empty or thin re-export files, nothing to explain),
`requirements.txt` / `.gitignore` / `.gitattributes` (plain config, no
logic), and `data/out/report.md` / `data/out/eval_set_report.md` /
`rag/eval_results.md` (generated data reports -- their real numbers are
already quoted throughout this document, not the files themselves).

One sentence for the whole project:

> "I built a code-repair assistant. It takes broken Python code and the real
> error it produces, generates a fix with a fine-tuned model, and proves the
> fix works by running it against real test cases in a sandbox -- no LLM
> judging anywhere."

---

## Part A -- The Sandbox

### sandbox/executor.py

**What it does.** Runs any Python code safely and reports what happened.
It starts a fresh Python subprocess for every run. The subprocess has a time
limit (default 8 seconds), a memory limit (default 512 MB), no network, and
it can only write files inside one temporary folder. It runs the code
against test cases and returns: pass or fail for each test, the printed
output, and the real error traceback if something failed.

**Why it is built this way.**
- A separate process means bad code cannot crash or freeze my main program.
- The memory limit is enforced by the operating system kernel (a "Job
  Object" on Windows, `rlimit` on Linux), not by trusting the code.
- The same file works on Windows and Linux, so the Colab notebook imports
  this exact file instead of a copy. One sandbox, used everywhere.
- The code under test is saved as `solution.py` and compiled from that
  file, so error tracebacks are real Python tracebacks that point at
  `solution.py` -- perfect for training data.

**Say this out loud.**
> "Before anything else, I built the execution sandbox, because every other
> part depends on trusting execution results. Each run is an isolated
> subprocess with a kernel-enforced memory cap, a hard timeout, no network,
> and file writes only inside a scratch folder. It returns structured
> results: per-test pass/fail plus the genuine traceback."

**Likely follow-ups.**
- *Q: Is this really secure?* A: "It is defense in depth for buggy generated
  code, not a container for hostile malware. For production against
  untrusted users I would add Docker or gVisor around it. I documented that
  threat model in the file."
- *Q: How do you know the sandbox itself works?* A: "I wrote 11 validation
  cases where the correct outcome is known -- infinite loop, memory bomb,
  network attempt, file escape, and so on -- and all 11 pass. I found two
  real bugs this way, including a 64-bit Windows handle bug that silently
  disabled the memory limit."
- *Q: Why capture the error inside the child, not parse stderr?* A: "The
  child writes structured JSON results to a file. Parsing stdout is fragile
  because user code can print anything."

### sandbox/validate_sandbox.py

**What it does.** Runs 11 known cases through the sandbox and checks the
sandbox gives exactly the expected outcome for each. Examples: correct code
must pass, an infinite loop must hit the timeout, a 2 GB allocation must be
killed by the 256 MB limit, a network call must be blocked.

**Why.** If the sandbox lies, the whole dataset and benchmark lie. So the
sandbox is tested first, before it is trusted with anything.

---

## Part B -- The Dataset

### data/mutations.py

**What it does.** Takes a correct solution and injects exactly one realistic
bug into it. It parses the code into a syntax tree (AST) and makes one small
change: an integer nudged by one (off-by-one), one arithmetic operator
swapped, one comparison flipped, one variable name replaced by another
in-scope variable, or one `if` guard deleted (missing edge case).

**Why AST and not text edits.** Editing the syntax tree guarantees the
result is still valid Python and the change is precise. Both the broken and
fixed versions are printed from the AST the same way, so the pair differs
only by the bug, never by formatting noise.

**Say this out loud.**
> "The only synthetic step in the whole project is bug injection, and it is
> synthetic on purpose -- that is how you get supervised repair pairs. Five
> realistic bug families, one single-site change each, applied on the AST.
> Everything else -- problems, solutions, tests, error messages -- is real."

**Likely follow-ups.**
- *Q: How do you know the injected bug is a real bug?* A: "Every variant is
  executed in the sandbox against the problem's real tests. If it still
  passes, the mutation changed nothing important and it is thrown away."
- *Q: Why cap wrong_variable at two per problem?* A: "I measured a smoke
  run and that bug type was crowding out the others -- about half the data
  -- because renaming a variable almost always breaks code. The cap keeps
  the distribution balanced."

### data/build_dataset.py

**What it does.** The full pipeline. Loads all 974 MBPP problems from
Hugging Face. Runs every reference solution against its own tests in the
sandbox -- 972 pass, 2 are dropped, 1 duplicate removed. Then it makes up to
5 bug variants per problem, runs each one in the sandbox, and keeps only
variants that really fail, storing the actual captured traceback. Result:
**3,760 training pairs** and **3,752 DPO preference pairs**, plus a report
with all real counts (`data/out/report.md`).

**Why the DPO pairs look like they do.** DPO needs a "chosen" answer and a
"rejected" answer. Chosen is the real reference fix. Rejected is a
*different* broken variant of the same problem -- plausible-looking code
that was already proven to fail in the sandbox. Not trivially bad garbage.

**Likely follow-ups.**
- *Q: Why do errors matter so much?* A: "The error message is a huge hint
  for the model. A NameError points at a wrong variable; an AssertionError
  with a failing test tells you the expected value. Templated or invented
  errors would teach the model wrong associations, so every error is
  captured from a real run of that exact variant."
- *Q: What is the class balance?* A: "Off-by-one 973, wrong operator 829,
  wrong comparison 496, wrong variable 1,342, missing edge case 120. The
  last one is rare because most MBPP solutions have no removable guard."

### data/build_eval_set.py

**What it does.** The same pipeline, but on HumanEval (164 problems), to
create the held-out evaluation set: **467 items**. HumanEval never enters
training in any form.

**Why hold out a whole dataset instead of a split.** Different distribution
(docstring prompts, `check()` test functions) and zero risk of leakage. The
fine-tuned model is graded on problems it has truly never seen.

### data/add_context_to_eval.py

**What it does.** For each of the 467 eval items, runs the full retrieval
pipeline (on this CPU machine, for real) and saves the retrieved repair
examples next to the item. The Colab notebook then measures "fix rate with
context vs without" without needing the retrieval stack on the GPU.

---

## Part C -- Retrieval (RAG + GraphRAG)

### rag/store.py and rag/build_index.py

**What they do.** Turn every training pair into a searchable document
(problem text + broken code + last error line) and build two indexes over
the same 3,760 documents: dense vectors (MiniLM embeddings in an embedded
Qdrant database -- no server needed) and BM25 keyword search.

**Why two indexes.** They fail differently. BM25 finds exact identifier and
keyword matches; dense vectors find "means the same thing" matches even
with different words. Merging both is more robust than either alone.

### rag/retriever.py

**What it does.** The query pipeline: dense top-50 and BM25 top-50 are
merged with Reciprocal Rank Fusion (a simple, robust formula: score =
sum of 1/(60+rank)); graph candidates join as a third list; a cross-encoder
then re-reads the top 30 and reorders them; top-k become prompt context.

**Why RRF and not score mixing.** Dense scores and BM25 scores live on
different scales. RRF only uses ranks, so nothing needs calibration.

**Why a cross-encoder on top.** The first stage encodes query and document
separately (fast, but shallow). The cross-encoder reads them *together* and
catches things the first stage misses. It is too slow to run on everything,
so it only re-scores the short list.

### rag/graph.py

**What it does.** Builds a knowledge graph from the real dataset: 930 task
nodes, 5 bug-type nodes, 166 topic nodes (recurring words from real function
names), 9 signature-shape nodes (argument count, has-loop, recursion,
comprehension). 5,605 edges connect each task to what was observed in it.

At query time it does a real multi-hop walk: the error message maps to
likely bug types using a probability table counted from the data (for
example, NameError almost always means wrong_variable); the code parses
into topic and shape nodes; candidate tasks must be connected to BOTH a
matching bug node AND a matching shape/topic node. That is "find a past fix
for a similar bug in a similarly shaped function".

**Say this out loud.**
> "GraphRAG here is not decoration. The bug type of a query is unknown at
> inference time -- it is the label -- so I learned a conditional table,
> probability of bug type given error type, from the data itself, and use
> the graph to demand candidates that share both the suspected bug type and
> the function shape. That is a genuine multi-hop constraint a vector
> search cannot express."

**Likely follow-ups.**
- *Q: Where do graph categories come from?* A: "Extracted from the data:
  function-name tokens filtered by document frequency, and structural
  features parsed from the AST. Nothing hand-invented."
- *Q: Does the graph actually help?* A: "See rag/eval_results.md -- I
  report where it helps and where it does not. The labeled query set is
  real: for a query built from one bug variant, the relevant documents are
  the other variants of the same problem."

### rag/retrieval_eval.py

**What it does.** Builds 120 labeled queries from the dataset itself and
measures recall@5, recall@10, nDCG@10 and MRR for six system variants
(dense only, BM25 only, graph only, hybrid, hybrid+rerank, full pipeline),
in two modes: with the problem text, and code-plus-error only (harder).
Results are written to `rag/eval_results.md` -- real numbers, including any
configuration where a component does not help.

---

## Part F -- The Web UI

### ui/server.py

**What it does.** A FastAPI backend with four endpoints: health check, run
code in the sandbox, generate a fix (retrieval context + Ollama model), and
verify a fix in the sandbox. The diff between broken and fixed code is
computed server-side with Python's difflib.

**Why Ollama.** The model name is one config value (`ui/config.py`,
`OLLAMA_MODEL`). Today it points at the base `qwen2.5-coder:1.5b` to prove
the UI works; after the Colab run, the fine-tuned GGUF is registered with
`ollama create` and the config changes by one line. No rebuild.

**Likely follow-ups.**
- *Q: What happens if the model returns garbage?* A: "The verify step
  catches it -- the fix is executed against the real tests and the UI
  shows FAIL honestly. In my end-to-end test the small base model fixed 1
  of 3 sampled bugs; that is reported as model quality, not hidden."
- *Q: Why does the health endpoint not load the retrieval index?* A: "It
  did at first -- and it made health checks hang while models loaded. I
  moved heavy loading to the first fix request; health only reports cheap
  facts. That was a real bug I found by testing."

### ui/config.py

**What it does.** Every setting the UI needs in one small file: the Ollama
model name, generation options (temperature, max tokens), how many
retrieved examples to include (`RETRIEVAL_K`), and the sandbox timeout and
memory limit to use for run/verify requests.

**Why it is its own file.** Swapping in the fine-tuned model later means
changing one line -- `OLLAMA_MODEL` -- and restarting the server. Nothing
else in the codebase needs to change, and there is nothing to rebuild.

**Say this out loud.**
> "The config file is small on purpose. The one line that matters is
> `OLLAMA_MODEL` -- today it points at the base model to prove the UI
> works end to end; the moment the fine-tuned GGUF is registered with
> Ollama, this is a one-line, one-restart swap."

### ui/static/ (index.html, style.css, app.js)

**What it does.** A single-page internal tool: paste problem, code, tests;
run; see per-test PASS/FAIL and the real traceback; generate a fix; see a
color diff or the full code; verify the fix live in the sandbox. No emoji,
no spinner theater -- status text and measured numbers only.

### ui/test_e2e.py

**What it does.** Scripted proof the loop works: samples real broken
variants from the dataset, checks they fail in the sandbox, requests a fix
from the configured model, verifies the fix in the sandbox. Asserts the
machinery, reports the model quality.

---

## Parts D/E -- Fine-Tuning and Benchmark (notebook/code_repair_colab.ipynb)

This is one notebook, written and validated (every code cell parses and
runs without a syntax error), but **not executed here** -- it needs a GPU
this machine doesn't have. It is meant to run on Colab Pro with an L4. Below
is what each section does; treat the whole notebook as one file with
several jobs.

### Environment gate (first cell)

**What it does.** Checks for a GPU, checks it reports enough VRAM, checks
it is an L4 specifically (with an override flag if you accept a different
GPU), checks for High-RAM. Fails loudly with an exact fix-it instruction
if any check fails, before installing anything or touching data.

**Why.** A silent wrong-runtime failure (running on a CPU-only or T4
Colab instance) wastes an hour before you notice. Failing on cell 1 costs
nothing.

### Data loading (Drive mount + manual-upload fallback)

**What it does.** Looks for four files in `MyDrive/code-repair/`: the
3,760-pair dataset, the DPO pairs, the HumanEval eval set with
precomputed retrieval context, and the sandbox `executor.py` itself. If
Drive isn't mounted or the files aren't there, a fallback cell lets you
upload them by hand.

**Why reuse `executor.py` instead of rewriting it for Colab.** One sandbox,
tested once, trusted everywhere -- the same file that built the training
data also grades the benchmark. Two implementations could quietly drift
apart and grade differently.

### Base model choice: Qwen2.5-Coder-3B-Instruct

**What it does.** A markdown cell states the model size and why.

**Say this out loud.**
> "I originally sized this for the 7B model -- it's the largest Qwen2.5-Coder
> that lets LoRA, QLoRA, and DoRA all run on one L4 in bf16, roughly 15 GB
> of weights. I switched to 3B for this run specifically because I had an
> interview deadline and 3B trains and generates in roughly a third of the
> wall-clock time on the same GPU -- it's a time trade-off I made
> deliberately, not a hardware limit. The pipeline -- data, sandbox
> verification, RAG, DPO, benchmarking -- is identical either way; only the
> base model size changed, and switching back to 7B is a one-line edit."

### LoRA / QLoRA / DoRA training cells

**What it does.** One shared `run_sft()` function, called three times with
different flags: plain LoRA (bf16 base), QLoRA (4-bit NF4 base, same
adapter shape), DoRA (weight-decomposed LoRA, bf16 base). Each run trains,
evaluates, saves the adapter, and immediately copies it to Google Drive
before the next method starts -- so a Colab disconnect never loses a
finished stage. GPU memory is explicitly freed between runs.

**Why compare three methods instead of picking one.** They make different
trade-offs: LoRA is simplest; QLoRA trades a little quality for a much
smaller memory footprint (useful on smaller GPUs); DoRA changes how the
adapter decomposes the weight update and often trains a bit better than
plain LoRA at the same rank. Measuring eval_loss for all three, on the same
data, same hyperparameters, is what makes the comparison honest instead of
a guess.

**Likely follow-ups.**
- *Q: Why back up to Drive after every run instead of at the end?* A:
  "Colab can disconnect mid-notebook. Backing up immediately means a
  disconnect during, say, the DoRA run only costs that one run, not LoRA
  and QLoRA too."
- *Q: What decides "best" adapter?* A: "Measured eval_loss on a held-out
  2% slice of the training data -- lowest wins, whichever method that
  turns out to be. It's picked in code, not by assumption."

### DPO cell

**What it does.** Takes the measured-best SFT adapter and runs DPO
(Direct Preference Optimization) on top of it, using the DPO pairs from
Part B. Before training starts, it re-verifies a random sample of 20
"rejected" answers in the sandbox, right there on the GPU machine, and
asserts all 20 still fail -- so the DPO signal is checked twice, not just
trusted from when the dataset was built.

**Say this out loud.**
> "DPO needs a chosen and a rejected answer for each example. Chosen is the
> real reference fix. Rejected is a different bug variant of the same
> problem that the sandbox already proved fails -- a plausible wrong
> answer, not a strawman. I re-verify a sample of those rejections again
> right before training, on the actual training machine, so that claim
> isn't taken on faith from an earlier run."

### Part E: benchmark harness

**What it does.** Generates a fix for each held-out HumanEval item with
every model variant (base, LoRA, QLoRA, DoRA, DPO), then with the best
variant plus retrieval context on. For each generated fix it extracts the
code from the model's reply and executes it through the same sandbox
(`run_tests`, imported, not reimplemented) against that problem's real
`check()` tests. pass@1 is one greedy generation graded pass/fail; pass@3
is three sampled generations, counted as a pass if any one succeeds.
Nothing here is judged by an LLM -- every number comes from a real process
exit code.

**Why pass@1 and pass@3, and why execute instead of comparing text.**
Comparing generated code to the reference text as a string would penalize
correct-but-differently-written fixes. Executing against real tests is the
only check that measures what actually matters: does it work.

**Likely follow-ups.**
- *Q: How is the RAG effect isolated?* A: "Same held-out items, same best
  model, run twice -- once with retrieved context in the prompt, once
  without. Only that one variable changes, so the difference in pass@1 is
  attributable to retrieval."
- *Q: Why reduce N_EVAL or DPO_SUBSET for a faster run -- does that weaken
  the result?* A: "It shrinks the sample size, so the numbers get noisier,
  but they're still real measured values on real held-out data, just with
  a wider margin of error. Nothing about the method changes."

### Export cells (merge, GGUF quantize, Modelfile)

**What it does.** Merges the best adapter's weights into the base model,
converts to GGUF with llama.cpp (building it from source, installing cmake
if needed), quantizes to q4_k_m (4-bit, a good size/quality balance for
local use), and writes an Ollama `Modelfile` so the result can be
registered locally with one command.

**Why q4_k_m specifically.** It is a widely-used quantization level that
keeps most of the model's quality while cutting its file size roughly to a
quarter of full precision -- practical for running locally in Ollama
without a GPU.

---

## The two ideas to repeat if you are nervous

1. **Execution is the source of truth.** Every claim in this project --
   dataset quality, DPO rejections, benchmark scores, UI verdicts -- comes
   from actually running code against real tests in the sandbox.
2. **Real data everywhere except the bug itself.** Problems, solutions,
   tests: MBPP and HumanEval. Errors: captured from real runs. The only
   synthetic thing is the injected bug, which is the point of the task.
