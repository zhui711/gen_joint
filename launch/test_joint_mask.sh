#!/bin/bash
# =============================================================================
# Joint Image-Mask Co-Generation Inference Launch Script
# =============================================================================
# Usage:
#   bash launch/test_joint_mask.sh
#   bash launch/test_joint_mask.sh --skip_generation
#   bash launch/test_joint_mask.sh --skip_eval
#   bash launch/test_joint_mask.sh --max_samples 100
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Paths ---
MODEL_PATH="Shitao/OmniGen-v1"
LORA_PATH="results/joint_mask_cogen/checkpoints/0010000/"
MASK_MODULES_PATH="results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin"
JSONL_PATH="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl"
OUTPUT_DIR="results/joint_mask_inference"

# --- Multi-GPU throughput settings ---
NUM_GPUS=8
BATCH_SIZE=32
INFERENCE_STEPS=50
GUIDANCE_SCALE=2.5
IMG_GUIDANCE_SCALE=2.0
SEED=42

# --- Joint mask settings ---
MASK_LATENT_CHANNELS=4
MASK_THRESHOLD=0.0
MASK_SCALE_FACTOR=10.0

# --- Resolution settings (match training preprocessing) ---
MAX_IMAGE_SIZE=1024

# Override CUDA_VISIBLE_DEVICES externally if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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
    --mask_scale_factor "${MASK_SCALE_FACTOR}"
    --keep_raw_resolution
    --max_image_size "${MAX_IMAGE_SIZE}"
)

"${PYTHON_BIN}" test_joint_mask.py "${PY_ARGS[@]}" "$@"
