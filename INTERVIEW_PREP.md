# Interview Prep -- Presentation Script

Order to follow live: **(1) Explain the project -> (2) Show the architecture
diagram -> (3) Live demo -> (4) Walk through the code -> (5) Answer
questions.** This document gives you exact words for each step. Read it out
loud a few times before the call -- you do not need to memorize it word for
word, just know the shape of each part.

If this is the Neon AI "LLM Fine-Tuning Engineer" screen: their posting asks
about **RAG optimizations, training dataset optimization, model benchmark
design, and fine-tuning experiments** almost word for word. This project has
all four. Say those exact words early -- it is not a coincidence, point it
out.

---

## Part 1 -- Explain the project

### 30-second version (say this first, before anyone asks a question)

> "I built a code-repair assistant. You give it a broken Python function, the
> problem it is supposed to solve, and the real error it produces. A
> fine-tuned model suggests a fix. And here is the important part: I never
> trust the model's own judgment about whether the fix is correct. I run the
> fix in a sandbox against real test cases and only a real pass counts as a
> fix. Everything is built on real data -- real problems from MBPP, real
> held-out problems from HumanEval -- the only thing I generate myself is the
> bug, because that is the whole point of the task."

### 2-minute version (if they want more)

> "The project has five parts. First, a sandbox: it runs untrusted Python
> code safely, with a timeout, a memory limit, no network access, and no
> file access outside one folder. Second, a dataset: I took 974 real MBPP
> problems, and for each one I injected a small realistic bug -- an
> off-by-one, a wrong operator, a wrong variable, that kind of thing -- using
> the code's syntax tree, not text editing. I ran every single broken
> version through the sandbox and only kept it if it genuinely failed, and I
> saved the real error message it produced. That gave me 3,760 training
> pairs: broken code plus its real error, mapped to the real correct fix.
> Third, a retrieval system -- when the model is about to fix a new bug, it
> first looks up similar past fixes, using both keyword search and vector
> search combined, plus a knowledge graph that connects bugs by type and by
> function shape. Fourth, fine-tuning -- I compare three parameter-efficient
> methods, LoRA, QLoRA, and DoRA, then run a preference-tuning step called
> DPO on top of the best one. Fifth, a web interface where you paste in
> broken code and watch the whole loop happen live: sandbox run, model fix,
> sandbox verification of the fix."

### The one sentence to repeat if you get nervous

> "Every claim in this project is backed by actually running the code, not
> by an AI model saying 'looks right to me.'"

---

## Part 2 -- Show the architecture diagram

The diagram is already made -- it's `docs/architecture.png`, shown in
`README.md` under "Architecture." Have that image open (or the README
rendered on GitHub) to share your screen. Five groups, left to right:

1. **Execution Sandbox** (top left). "`executor.py` -- subprocess
   isolation, a timeout, a memory cap, no network, and it can only write
   files inside one scratch folder. Below it, the validation suite: 11
   known cases I tested it against before trusting it with anything."
2. **Dataset Pipeline** (top middle). "974 real MBPP problems go through
   AST bug injection -- five bug families, one change at a time. Every
   variant runs through that same sandbox box on the left, and only
   variants that genuinely fail become training data: `dataset.jsonl`,
   3,760 pairs with real captured tracebacks, plus `dpo_pairs.jsonl` for
   preference tuning. HumanEval sits to the side, untouched by training --
   467 held-out evaluation items."
3. **Retrieval** (bottom left). "Dense vectors, BM25 keyword search, and a
   knowledge graph of bug types and function shapes all feed into
   Reciprocal Rank Fusion, then a cross-encoder reranks the result. This
   is what finds a similar past repair before the model even generates a
   fix."
4. **Fine-tuning and Benchmark** (bottom middle, marked pending). "Qwen2.5-
   Coder-1.5B, compared across LoRA, QLoRA, and DoRA, then DPO on top of
   whichever wins. pass@1 and pass@3 are measured by executing every
   generated fix through that same sandbox again -- not judged by the
   model itself. This box needs a GPU I run separately on Colab."
5. **Web UI** (right). "FastAPI and a static frontend -- paste code, run
   it, generate a fix, see the diff, verify the fix. It talks to an Ollama
   endpoint that's one config value, so swapping in the fine-tuned model
   later is a one-line change."

Then point at the arrows and tie it together:

> "Notice the sandbox box has arrows going to almost everywhere else --
> dataset building, retrieval evaluation, benchmarking, and the live UI
> all call the exact same execution engine. That's deliberate: one trusted
> way to check 'does this code actually work,' reused everywhere instead
> of reimplemented four times. The dashed arrow on the right is the one
> piece that's still pending -- once the Colab run finishes, the exported
> model plugs into the same UI you're about to see, no rebuild."

If they ask for more detail, that is when you switch to the **full
technical diagram** in `README.md` and go subgraph by subgraph (A through
F) -- it matches the five parts from your 2-minute explanation exactly.

---

## Part 3 -- Live demo

**Before the call:** start Ollama, then start the UI server from the repo
root:
```
.\.venv\Scripts\python.exe -m uvicorn ui.server:app --port 8000
```
Open `http://localhost:8000` in a browser tab and leave it ready. Confirm
the health line at the top says sandbox: ready, model: available, retrieval:
index loaded (index loaded only appears after your first fix request in
this session, so run one demo once beforehand to warm it up).

### Three examples that are proven to work (tested live before this doc was written)

Use these -- they are real dataset entries, not cherry-picked toy examples,
and all three were verified end-to-end with retrieval turned on:

**Example 1**
- Problem: *Write a function to multiply two integers without using the `*`
  operator.*
- What is broken: a wrong comparison operator was injected.
- What you will see: sandbox FAILS with AssertionError, the model proposes a
  fix, verify PASSES 3/3.

**Example 2**
- Problem: *Write a function to convert the given tuple to a floating-point
  number.*
- What is broken: a wrong variable was injected.
- What you will see: sandbox FAILS with a ValueError, model fix, verify
  PASSES 3/3.

**Example 3**
- Problem: *Write a function to find the volume of a cube.*
- What is broken: a wrong operator was injected.
- What you will see: sandbox FAILS with AssertionError, model fix, verify
  PASSES 3/3.

To pull the exact broken code and tests for any of these to paste into the
UI, run:
```
.\.venv\Scripts\python.exe -c "import json; [print(json.dumps(json.loads(l), indent=2)) for l in open('data/out/dataset.jsonl', encoding='utf-8') if json.loads(l)['id'] in ('mbpp_127_v2','mbpp_553_v1','mbpp_234_v3')]"
```

### What to say while the demo runs

> "I'll paste in this broken function and its tests. First I run it in the
> sandbox -- you can see it fails, and this is the real traceback, not a
> made-up message. Now I click Generate Fix -- retrieval is on, so it first
> looked up similar past repairs from the same bug family before asking the
> model. Here's the diff. Now Verify -- and this re-runs the exact same
> sandbox against the exact same tests. Pass means pass, for real."

### Be honest if a fix fails during the live demo

It might, especially without retrieval or on an example you didn't
pre-test. Say this, calmly:

> "That's a genuine result, not a bug in my code -- the base model is small
> and doesn't always get it right on the first try. That's exactly why the
> verify step exists: a wrong fix is caught here, not silently shipped. Once
> my fine-tuned adapter is in place this rate should go up, and I have a
> benchmark set up to measure that precisely -- pass@1 and pass@3, executed,
> not guessed."

This answer is a strength, not a weakness -- it shows the system does what
it claims even when the model is imperfect.

---

## Part 4 -- Walk through the code

Suggested order to open files, with the one sentence to say for each. Full
depth (why-built-this-way, follow-up Q&A) is in `PROJECT_EXPLAINED.md` --
open that in a second tab if you get a question you want backup for.

1. **`sandbox/executor.py`** -- "This is the trust boundary. Subprocess
   isolation, kernel-enforced memory limit, no network, scratch-dir-only
   writes. Everything else in the project calls into this file; it's never
   duplicated."
2. **`sandbox/validate_sandbox.py`** -- "Before I trusted this with
   generated code, I proved it against 11 known cases -- infinite loop,
   memory bomb, network escape, file escape. All 11 pass."
3. **`data/mutations.py`** -- "The one deliberately synthetic step: bug
   injection on the syntax tree, five bug families, one change at a time."
4. **`data/build_dataset.py`** -- "The full pipeline: load MBPP, validate
   references in the sandbox, inject bugs, keep only variants that really
   fail, save the real captured error. 3,760 pairs, real counts in
   `data/out/report.md`."
5. **`rag/graph.py`** -- "This is the GraphRAG piece. Nodes are bug types
   and function shapes pulled from the real data. At query time I infer
   likely bug types from the error message using a probability table
   learned from the dataset, then require a candidate to share both a bug
   node and a shape node -- a genuine multi-hop constraint, not just nearest
   neighbor."
6. **`rag/retriever.py`** -- "Dense search and BM25 merged with Reciprocal
   Rank Fusion, a cross-encoder reranks the short list, graph candidates
   join as a third signal."
7. **`ui/server.py`** -- "Four endpoints: run, fix, verify, health. The
   model is one config value in `ui/config.py` -- swapping in the
   fine-tuned model later is a one-line change, not a rebuild."
8. **`notebook/code_repair_colab.ipynb`** -- "Written and validated, meant
   for a GPU I run separately on Colab: LoRA vs QLoRA vs DoRA, then DPO,
   then a benchmark that executes every generated fix in this same
   sandbox. Currently using Qwen2.5-Coder-1.5B -- the same size as the base
   model my UI demo already runs, so the benchmark's baseline and the demo
   are the same weights, and the whole pipeline trains in under two hours.
   7B was my original target for the largest model all three PEFT methods
   fit on one L4; the pipeline itself doesn't change, only `MODEL_ID`."

---

## Part 5 -- If asked about fine-tuning results specifically

Use whichever is true when you're asked:

**If the Colab run is not finished / did not finish in time:**
> "The sandbox, the dataset, and the retrieval system are fully built and
> executed with real, measured numbers -- I can show you those right now.
> The fine-tuning notebook is written and validated end to end, including
> the benchmark methodology, but it needs a GPU I run separately on Colab,
> and that run is still in progress. I'd rather tell you that honestly than
> show you a number I didn't actually measure."

**If it finished:**
> "Here's the real benchmark table -- pass@1 and pass@3 for the base model,
> LoRA, QLoRA, DoRA, and DPO, each one measured by actually executing the
> generated fix against HumanEval's real tests, plus the same comparison
> with and without retrieval context." (Open `benchmark_results.csv` /
> the notebook's final table cell.)

Either answer is a good answer. The dishonest answer -- making up a number --
is the only bad one.

---

## Part 6 -- Likely questions and short answers

**"Why not just use GPT-4 / Claude to fix the code directly, no fine-tuning?"**
> "That works fine for a demo, but the task here was specifically to build
> and evaluate a fine-tuning pipeline -- dataset construction, LoRA/QLoRA/DoRA
> comparison, DPO, quantized local deployment. A frontier API call skips all
> of that. Also, a small local model plus retrieval plus verification can be
> a lot cheaper and more controllable for a narrow, repeated task like this."

**"How is this different from GitHub Copilot?"**
> "Copilot suggests code as you type and trusts you to check it. This system
> makes verification the product, not an afterthought -- every fix is
> executed against real tests before it's ever called 'fixed,' and that
> verification loop is also how the training data itself was built."

**"Why MBPP and not real GitHub bug-fix commits?"**
> "MBPP gives me a clean, small, fully-testable problem with a ground-truth
> solution and real tests -- I can prove a bug variant truly fails and prove
> a fix truly passes. Real-world commits are noisier: tests are often
> missing or flaky, and 'the fix' is entangled with unrelated changes. For a
> first version, that trade-off was worth it."

**"How do you know there's no leakage between training and evaluation?"**
> "HumanEval is a completely separate dataset from MBPP, loaded by a
> separate script, and it is never touched by the training data builder. I
> assert that explicitly in code -- every training record's source field is
> 'mbpp' and every eval record's is 'humaneval.'"

**"What would you improve with more time?"**
> "Three things: real GitHub bug-fix data as a second training source, not
> just injected bugs; an LLM-based mutation generator validated against the
> sandbox, to get more naturalistic bugs than pure AST edits can produce;
> and multi-file/multi-function repair instead of single-function."

**"What was the hardest bug you hit building this?"**
> "The sandbox's memory limit silently didn't work at first, on Windows --
> a 2 GB allocation succeeded when it should have been killed at 256 MB. The
> cause was a 64-bit process handle getting truncated by a ctypes call that
> defaulted to a 32-bit return type. I only caught it because I'd already
> written a validation suite with a known-bad case, not because the demo
> looked wrong."

---

## Closing line

> "The theme across every part of this is the same: don't trust a model's
> opinion about correctness, trust execution. That's true for how I built
> the training data, how I built retrieval, and how the UI reports results."
