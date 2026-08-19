#!/usr/bin/env bash
#
# check_prompt_tokens.sh — verify Task 2.1 system prompts are under the
# 500-token budget, using the REAL Phi-3-mini tokenizer via llama-tokenize
# (not a word-count approximation).
#
# Usage: bash scripts/check_prompt_tokens.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/models/phi3_mini_4k_instruct.gguf"
TOKENIZE_BIN="${REPO_ROOT}/llama.cpp/build/bin/llama-tokenize"

if [[ ! -x "$TOKENIZE_BIN" ]]; then
  echo "ERROR: llama-tokenize not found at $TOKENIZE_BIN"
  echo "It should have been built alongside llama-cli/llama-bench in Task 1.2."
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model not found at $MODEL_PATH"
  exit 1
fi

echo "======================================================================"
echo "System Prompt Token Counts (real Phi-3 tokenizer via llama-tokenize)"
echo "======================================================================"

python3 - "$REPO_ROOT" << 'PYEOF'
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from prompt_engineer import PROMPTS

for name, prompt in PROMPTS.items():
    with open(f"/tmp/rk_agridig_prompt_{name}.txt", "w") as f:
        f.write(prompt)
PYEOF

for name in conservative balanced aggressive; do
  file="/tmp/rk_agridig_prompt_${name}.txt"
  count=$("$TOKENIZE_BIN" -m "$MODEL_PATH" -f "$file" 2>/dev/null | wc -l)
  status="OK"
  if [[ "$count" -gt 500 ]]; then
    status="EXCEEDS 500-TOKEN BUDGET"
  fi
  printf "%-14s %4d tokens   [%s]\n" "$name" "$count" "$status"
done

rm -f /tmp/rk_agridig_prompt_*.txt
echo "======================================================================"