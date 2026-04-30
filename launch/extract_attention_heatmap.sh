#!/bin/bash
# =============================================================================
# Standalone Image-to-Mask Attention Heatmap Extraction
# =============================================================================
# Usage:
#   bash launch/extract_attention_heatmap.sh \
#     --sample_ids LIDC-IDRI-0251_0001,LIDC-IDRI-0657_0001 \
#     --layer_index 20
#
# This launcher is for qualitative diagnostics only. It does not run metrics.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-Shitao/OmniGen-v1}"
LORA_PATH="${LORA_PATH:-results/joint_mask_cogen/checkpoints/0010000/}"
MASK_MODULES_PATH="${MASK_MODULES_PATH:-results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin}"
JSONL_PATH="${JSONL_PATH:-/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis_attention_heatmaps}"

INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
IMG_GUIDANCE_SCALE="${IMG_GUIDANCE_SCALE:-2.0}"
SEED="${SEED:-42}"
MASK_LATENT_CHANNELS="${MASK_LATENT_CHANNELS:-4}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.0}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1024}"
DEVICE="${DEVICE:-cuda:0}"
HEATMAP_ALPHA="${HEATMAP_ALPHA:-0.45}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${OUTPUT_DIR}"

PY_ARGS=(
    --model_path "${MODEL_PATH}"
    --lora_path "${LORA_PATH}"
    --mask_modules_path "${MASK_MODULES_PATH}"
    --mask_latent_channels "${MASK_LATENT_CHANNELS}"
    --jsonl_path "${JSONL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --inference_steps "${INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --img_guidance_scale "${IMG_GUIDANCE_SCALE}"
    --seed "${SEED}"
    --mask_threshold "${MASK_THRESHOLD}"
    --max_image_size "${MAX_IMAGE_SIZE}"
    --device "${DEVICE}"
    --heatmap_alpha "${HEATMAP_ALPHA}"
)

"${PYTHON_BIN}" extract_attention_heatmap.py "${PY_ARGS[@]}" "$@"
