#!/usr/bin/env bash
set -e

# Run from the repository root so all relative paths resolve consistently.
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="configs/train.yaml"
MODEL="pretrained/FLUX.1-Kontext-dev"
OUTPUT="checkpoints/eyecontrol"

EYECONTROL_ENV="${EYECONTROL_ENV:-/root/miniconda3/envs/eyecontrol}"
if [[ -x "$EYECONTROL_ENV/bin/accelerate" ]]; then
  ACCELERATE_BIN="$EYECONTROL_ENV/bin/accelerate"
else
  ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
fi

"$ACCELERATE_BIN" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  src/training/train.py \
  --config_path "$CONFIG" \
  --pretrained_model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT" \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --ranks 64 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate 2e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 10000 \
  --checkpointing_step 500 \
  --validation_step 100 \
  --use_8bit_adam \
  --seed 0 \
  --mask_loss \
  --consistency_method \
  --consistency_cpu_offload \
  --prob_consistency_update_local 0.5 \
  "$@"
