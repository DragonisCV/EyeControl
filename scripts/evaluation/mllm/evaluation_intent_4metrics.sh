#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_ROOT="${REPO_ROOT}/results/ControlArt_Bench"
TARGET_DIR="${BENCH_ROOT}/eyecontrol/checkpoint-10000/focus/output"
INPUT_DIR="${BENCH_ROOT}/inputs"
PROMPT_DIR="${BENCH_ROOT}/prompts"
SAVE_DIR="${BENCH_ROOT}/eyecontrol/checkpoint-10000/focus/mllm-fa-pq-o"

for required_dir in "$TARGET_DIR" "$INPUT_DIR" "$PROMPT_DIR"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "[ERROR] Required directory not found: $required_dir" >&2
        exit 1
    fi
done

mkdir -p "$SAVE_DIR"
cd "$REPO_ROOT"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"

python scripts/evaluation/mllm/test_global_metrics.py \
    --input_dir "$INPUT_DIR" \
    --target_dir "$TARGET_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --save_dir "$SAVE_DIR" \
    --backbone qwen25vl \
    --suffix mask

echo "MLLM evaluation complete: $SAVE_DIR"
