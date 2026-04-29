#!/bin/bash
# =============================================================================
# Joint Image-Mask Co-Generation Attention Extraction Launch Script
# =============================================================================
# Usage:
#   bash launch/test_joint_mask_with_attn.sh --sample_ids LIDC-IDRI-0251_0001
#   bash launch/test_joint_mask_with_attn.sh --max_samples 8
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Paths ---
MODEL_PATH="${MODEL_PATH:-Shitao/OmniGen-v1}"
LORA_PATH="${LORA_PATH:-results/joint_mask_cogen/checkpoints/0010000/}"
MASK_MODULES_PATH="${MASK_MODULES_PATH:-results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin}"
JSONL_PATH="${JSONL_PATH:-/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/joint_mask_attention}"

# --- Attention extraction is memory-heavy; use smaller batches by default. ---
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
IMG_GUIDANCE_SCALE="${IMG_GUIDANCE_SCALE:-2.0}"
SEED="${SEED:-42}"

# --- Joint mask settings ---
MASK_LATENT_CHANNELS="${MASK_LATENT_CHANNELS:-4}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.0}"
HEATMAP_ALPHA="${HEATMAP_ALPHA:-0.45}"

# --- Resolution settings (match training preprocessing) ---
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1024}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${OUTPUT_DIR}"

PY_ARGS=(
    --model_path "${MODEL_PATH}"
    --lora_path "${LORA_PATH}"
    --mask_modules_path "${MASK_MODULES_PATH}"
    --mask_latent_channels "${MASK_LATENT_CHANNELS}"
    --jsonl_path "${JSONL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --num_gpus "${NUM_GPUS}"
    --inference_steps "${INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --img_guidance_scale "${IMG_GUIDANCE_SCALE}"
    --seed "${SEED}"
    --save_masks
    --mask_threshold "${MASK_THRESHOLD}"
    --keep_raw_resolution
    --max_image_size "${MAX_IMAGE_SIZE}"
    --heatmap_alpha "${HEATMAP_ALPHA}"
)

"${PYTHON_BIN}" test_joint_mask_with_attn.py "${PY_ARGS[@]}" "$@"
