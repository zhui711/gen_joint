#!/bin/bash
# =============================================================================
# test.sh — Launch CXR OmniGen Inference & Evaluation
# =============================================================================
# Usage:
#   bash lanuch/test.sh                        # Run the default config below
#   bash lanuch/test.sh --max_samples 100      # Quick mini-test override
# =============================================================================

set -euo pipefail

# ---- Environment ----
export HF_HOME="~/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Paths (edit these) ----
MODEL_PATH="Shitao/OmniGen-v1"
LORA_PATH="./results/cxr_finetune_lora_30ksteps_maskmse_timefilter/checkpoints/0004000"
# LORA_PATH="./results/cxr_finetune_lora/checkpoints/0030000"
JSONL_PATH="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl"
# OUTPUT_DIR="/raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/cxr_omnigen_results"
OUTPUT_DIR="/home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30ksteps_maskmse_timefilter"
# OUTPUT_DIR="/home/wenting/zr/gen_code/outputs/cxr_finetune_lora"

# ---- Inference params ----
BATCH_SIZE=16
NUM_GPUS=4
INFERENCE_STEPS=50
GUIDANCE_SCALE=2.5
IMG_GUIDANCE_SCALE=2.0
SEED=42

# =============================================================================
# Example 1: Full evaluation (all ~24k samples, 2 GPUs)
# =============================================================================
python ./test_omnigen_cxr.py \
    --model_path "${MODEL_PATH}" \
    --lora_path "${LORA_PATH}" \
    --jsonl_path "${JSONL_PATH}" \
    --output_dir "${OUTPUT_DIR}_4000" \
    --batch_size ${BATCH_SIZE} \
    --num_gpus ${NUM_GPUS} \
    --inference_steps ${INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --img_guidance_scale ${IMG_GUIDANCE_SCALE} \
    --seed ${SEED} \
    "$@"  # pass through any extra args (e.g. --max_samples 100)

# =============================================================================
# Example 2: Quick mini-test (uncomment to use)
# =============================================================================
# sudo ./python3 ./test_omnigen_cxr.py \
#     --model_path "${MODEL_PATH}" \
#     --lora_path "${LORA_PATH}" \
#     --jsonl_path "${JSONL_PATH}" \
#     --output_dir "${OUTPUT_DIR}_8000mini" \
#     --batch_size 4 \
#     --num_gpus 2 \
#     --max_samples 10000 \
#     --seed 42

# =============================================================================
# Example 3: Evaluate-only (skip generation, reuse existing images)
# =============================================================================
# python3 test_omnigen_cxr.py \
#     --jsonl_path "${JSONL_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --skip_generation

# =============================================================================
# Example 4: Compare different LoRA checkpoints
# =============================================================================
# for STEP in 0000500 0001000 0001500 0002000; do
#     echo "===== Testing checkpoint ${STEP} ====="
#     python3 test_omnigen_cxr.py \
#         --model_path "${MODEL_PATH}" \
#         --lora_path "./results/cxr_finetune_lora/checkpoints/${STEP}" \
#         --jsonl_path "${JSONL_PATH}" \
#         --output_dir "${OUTPUT_DIR}_ckpt${STEP}" \
#         --batch_size 4 \
#         --num_gpus 2 \
#         --max_samples 200
# done
