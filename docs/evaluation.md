# Evaluation

[← Back to README](../README.md#evaluation)

Run all commands below from the repository root.

Run inference first, then use the evaluation utilities appropriate for your task:

| Evaluation | Location |
| --- | --- |
| Image quality metrics | [scripts/evaluation/pyiqa/](../scripts/evaluation/pyiqa/) |
| MLLM assessment | [scripts/evaluation/mllm/](../scripts/evaluation/mllm/) |

Configure the input and output paths in the evaluation scripts for your dataset.

## PyIQA

Set the generated-image, reference-image, and evaluation-output paths in the
script, activate the project environment, and run:

```bash
conda activate eyecontrol
bash scripts/evaluation/pyiqa/pyiqa_intent.sh
```

The launcher computes PSNR and SSIM against the reference images, together with the saliency-distribution metrics KL, SIM, and CC. Per-image CSV files and logs are written to the configured evaluation output directory.

If evaluation was started while inference was still producing images, remove the incomplete `pyiqa` result directory and rerun it after inference has finished.

## MLLM Evaluation (FA, PQ, and O)

The MLLM evaluator uses an OpenAI-compatible vLLM server. The following command loads the local Qwen2.5-VL-72B checkpoint on a single 80 GB GPU. 

```bash
conda activate eyecontrol

vllm serve pretrained/Qwen2.5-VL-72B-Instruct \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 80 \
  --max-model-len 4096 \
  --enforce-eager
```

The served name preserves compatibility with the current evaluator; the model weights are loaded from the 72B path in the first argument. After the server is ready, open another terminal and run:

```bash
cd /path/to/eyecontrol
conda activate eyecontrol
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
bash scripts/evaluation/mllm/evaluation_intent_4metrics.sh
```

Configure the source images, prompts, generated images, and output paths in the evaluation script. The evaluator writes per-image FA, PQ, and O scores plus their averages to the configured output directory.

