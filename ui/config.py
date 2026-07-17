"""UI server configuration.

To swap in the fine-tuned model later, change OLLAMA_MODEL below (after
registering the GGUF export with `ollama create -f Modelfile`). Nothing
else needs to change.
"""

OLLAMA_URL = "http://localhost:11434"

# The one-line model swap. Points at the fine-tuned adapter (LoRA,
# quantized to GGUF) registered locally with Ollama; set back to
# "qwen2.5-coder:1.5b" to compare against the base model.
OLLAMA_MODEL = "code-repair-qwen"

GENERATION_TIMEOUT_S = 180
GENERATION_OPTIONS = {"temperature": 0.2, "num_predict": 1024}

RETRIEVAL_K = 3          # reference repairs added to the prompt
SANDBOX_TIMEOUT_S = 8.0
SANDBOX_MEMORY_MB = 512
