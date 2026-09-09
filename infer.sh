#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT_DIRS=(
  "checkpoints/eyecontrol/checkpoint-10000"
)

DATASET_NAMES=(
  "ControlArt_Bench"
)

MODES=(
  "focus"
  # "global"
  # "local"
)

DATA_ROOT="results"
SAVE_ROOT="results"
NUM_GPUS=1

# Add optional inference flags here, for example: --vis_attn.
EXTRA_ARGS=(
)

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  for dataset_name in "${DATASET_NAMES[@]}"; do
    for mode in "${MODES[@]}"; do
      input_dir="$DATA_ROOT/$dataset_name/inputs"
      mask_dir="$DATA_ROOT/$dataset_name/masks"
      prompt_dir="$DATA_ROOT/$dataset_name/prompts"
      checkpoint_name="${checkpoint_dir#checkpoints/}"
      save_dir="$SAVE_ROOT/$dataset_name/$checkpoint_name/$mode"

      echo "Running $dataset_name with $checkpoint_name ($mode)"
      mkdir -p "$save_dir"

      torchrun --nproc_per_node "$NUM_GPUS" src/inference/infer.py \
        --input_dir "$input_dir" \
        --style_dir "$mask_dir" \
        --prompt_dir "$prompt_dir" \
        --save_dir "$save_dir" \
        --checkpoint_dir "$checkpoint_dir" \
        --suffix "$mode" \
        "${EXTRA_ARGS[@]}"
    done
  done
done
