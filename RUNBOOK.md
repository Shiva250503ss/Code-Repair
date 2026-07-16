# RUNBOOK

Exact steps to run everything. Times are from the real runs on this machine
(Windows 11, Python 3.11, 12 sandbox workers) except where marked as
estimates for the Colab side.

## A. Run the completed local parts (A/B/C/F)

All commands from the repo root. The virtualenv already exists (`.venv`);
if starting fresh: `python -m venv .venv` then step 1.

1. **Install dependencies** (one time, ~5-10 min):
   ```
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. **Validate the sandbox** (~10 s). Expect `11/11 checks passed`:
   ```
   cd sandbox
   ..\.venv\Scripts\python.exe validate_sandbox.py
   cd ..
   ```

3. **Build the MBPP dataset** (~10 min; already built -> `data/out/`):
   ```
   .\.venv\Scripts\python.exe data\build_dataset.py --workers 12
   ```
   Outputs: `dataset.jsonl` (3,760 pairs), `dpo_pairs.jsonl` (3,752),
   `report.md`.

4. **Build the HumanEval eval set** (~4 min; already built):
   ```
   .\.venv\Scripts\python.exe data\build_eval_set.py --workers 12
   ```
   Outputs: `humaneval_broken.jsonl` (467 items), `eval_set_report.md`.

5. **Build retrieval indexes** (~5 min; already built -> `rag/index/`):
   ```
   .\.venv\Scripts\python.exe rag\build_index.py
   ```

6. **Precompute eval retrieval context** (~15 min; already built):
   ```
   .\.venv\Scripts\python.exe data\add_context_to_eval.py
   ```
   Output: `humaneval_broken_with_context.jsonl` -- one of the four files
   the Colab notebook needs.

7. **Retrieval evaluation** (~30-45 min, CPU cross-encoder is the slow part):
   ```
   .\.venv\Scripts\python.exe rag\retrieval_eval.py --queries 120
   ```
   Output: `rag/eval_results.md`.

   Note: the embedded Qdrant index allows ONE process at a time. Do not run
   this while the UI server is up (stop the server first, or vice versa).

8. **Run the UI** (Ollama must be running; model `qwen2.5-coder:1.5b` is
   already pulled):
   ```
   .\.venv\Scripts\python.exe -m uvicorn ui.server:app --port 8000
   ```
   Open http://localhost:8000. Scripted end-to-end check in a second
   terminal (~1-2 min):
   ```
   .\.venv\Scripts\python.exe ui\test_e2e.py --n 3 --retrieval
   ```

## B. Run the Colab notebook (Parts D/E) yourself

1. **Copy four files to Google Drive**, folder `MyDrive/code-repair/`
   (~2 min upload):
   - `data/out/dataset.jsonl`
   - `data/out/dpo_pairs.jsonl`
   - `data/out/humaneval_broken_with_context.jsonl`
   - `sandbox/executor.py`

2. **Open the notebook**: upload `notebook/code_repair_colab.ipynb` to
   colab.research.google.com. Runtime > Change runtime type >
   **GPU: L4** and **High-RAM**. (Colab Pro required.)

3. **Run cells top to bottom, in order.** Realistic wall-clock at default
   settings (estimates -- the notebook prints the real measured times):
   - Environment gate + installs: ~5 min
   - Data loading: ~2 min
   - LoRA SFT: ~45-90 min
   - QLoRA SFT: ~60-100 min
   - DoRA SFT: ~60-110 min
   - DPO (on best adapter): ~30-60 min
   - Benchmark, 7 arms x 150 items x 4 generations: ~2.5-4 h
   - GGUF convert + q4_k_m quantize: ~20-30 min
   Total: roughly a full day of L4 time. Each adapter is backed up to
   Drive the moment its training finishes, so a disconnect never loses a
   completed stage. For a fast first pass set `NUM_EPOCHS = 1` and
   `N_EVAL = 50`.

4. **Bring the model home** (~10 min): download
   `code-repair-qwen-q4_k_m.gguf` and `Modelfile` from Drive into one
   folder, then:
   ```
   ollama create code-repair-qwen -f Modelfile
   ```

5. **Swap the UI to the fine-tuned model** (one line): in `ui/config.py`
   set `OLLAMA_MODEL = "code-repair-qwen"`, restart the server from A.8.
