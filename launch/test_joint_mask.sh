#!/bin/bash
# =============================================================================
# Joint Image-Mask Co-Generation Inference Launch Script
# =============================================================================
# Generates both edited images and 10-channel anatomy masks.
# =============================================================================

set -e

# --- Paths ---
MODEL_PATH="Shitao/OmniGen-v1"
LORA_PATH="results/joint_mask_cogen/checkpoints/0010000/"
MASK_MODULES_PATH="results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin"
JSONL_PATH="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl"
OUTPUT_DIR="results/joint_mask_inference"

# --- Inference settings ---
BATCH_SIZE=1
NUM_GPUS=1
INFERENCE_STEPS=50
GUIDANCE_SCALE=2.5
IMG_GUIDANCE_SCALE=2.0
SEED=42
MASK_THRESHOLD=0.0

python test_joint_mask.py \
    --model_path ${MODEL_PATH} \
    --lora_path ${LORA_PATH} \
    --mask_modules_path ${MASK_MODULES_PATH} \
    --mask_latent_channels 4 \
    --jsonl_path ${JSONL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_gpus ${NUM_GPUS} \
    --inference_steps ${INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --img_guidance_scale ${IMG_GUIDANCE_SCALE} \
    --seed ${SEED} \
    --save_masks \
    --mask_threshold ${MASK_THRESHOLD}

echo "Done. Images saved to ${OUTPUT_DIR}, masks saved to ${OUTPUT_DIR}/masks/"
