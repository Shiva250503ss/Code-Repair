"""UI server configuration.

To swap in the fine-tuned model later, change OLLAMA_MODEL below (after
registering the GGUF export with `ollama create`, see RUNBOOK.md). Nothing
else needs to change.
"""

OLLAMA_URL = "http://localhost:11434"

# The one-line model swap. Currently the base model used to verify the UI
# end to end; replace with e.g. "code-repair-qwen" once the fine-tuned
# adapter from the Colab notebook is exported and registered.
OLLAMA_MODEL = "qwen2.5-coder:1.5b"

GENERATION_TIMEOUT_S = 180
GENERATION_OPTIONS = {"temperature": 0.2, "num_predict": 1024}

RETRIEVAL_K = 3          # reference repairs added to the prompt
SANDBOX_TIMEOUT_S = 8.0
SANDBOX_MEMORY_MB = 512
