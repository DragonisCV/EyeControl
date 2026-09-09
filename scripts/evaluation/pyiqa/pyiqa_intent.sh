#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_ROOT="${REPO_ROOT}/results/ControlArt_Bench"
TARGET_DIR="${BENCH_ROOT}/eyecontrol/checkpoint-10000/focus/output"
REF_DIR="${BENCH_ROOT}/targets"
SAVE_DIR="${BENCH_ROOT}/eyecontrol/checkpoint-10000/focus/pyiqa"

for required_dir in "$TARGET_DIR" "$REF_DIR"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "[ERROR] Required directory not found: $required_dir" >&2
        exit 1
    fi
done

mkdir -p "$SAVE_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for metric in psnr ssim; do
    python scripts/evaluation/pyiqa/inference_iqa.py \
        --target "$TARGET_DIR" \
        --ref "$REF_DIR" \
        --metric_mode FR \
        --metric_name "$metric" \
        --save_file "$SAVE_DIR/${metric}.csv"
done

for metric in BasedSaliency_KL BasedSaliency_SIM BasedSaliency_CC; do
    python scripts/evaluation/pyiqa/inference_iqa_intent.py \
        --target "$TARGET_DIR" \
        --ref "$REF_DIR" \
        --metric_mode based_saliency \
        --metric_name "$metric" \
        --save_file "$SAVE_DIR/${metric}.csv"
done

echo "PyIQA evaluation complete: $SAVE_DIR"
