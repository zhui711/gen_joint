#!/bin/bash
# =============================================================================
# Joint Image-Mask Co-Generation Training Launch Script
# =============================================================================
# This script launches the joint training with:
#   - Expanded LoRA (attention + MLP) at rank 32
#   - Mask latent co-generation with profiled lambda_mask=0.25
#   - Proper checkpointing of both LoRA and mask modules
# =============================================================================

set -e

# --- Paths ---
MODEL_PATH="Shitao/OmniGen-v1"
JSON_FILE="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"
RESULTS_DIR="results/joint_mask_cogen"

# --- Optional: Resume from previous checkpoint ---
# Uncomment and set these to resume training:
# LORA_RESUME_PATH="results/joint_mask_cogen/checkpoints/0010000/"
# MASK_MODULES_RESUME="results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin"

# --- Optional: Pretrained mask autoencoder ---
# MASK_AE_CKPT="results/mask_ae_pretrain/mask_autoencoder.pt"

# --- Optional: Resume from base LoRA (Plan 1 checkpoint) ---
# LORA_RESUME_PATH="/home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/"

# --- Hyperparameters ---
BATCH_SIZE=4
GRAD_ACCUM=4
LR=2e-5
LORA_RANK=32
LORA_ALPHA=32
LORA_TARGETS="qkv_proj o_proj gate_up_proj down_proj"
LAMBDA_MASK=0.25
MASK_LATENT_CH=4
MAX_TRAIN_STEPS=100000
LOG_EVERY=50
CKPT_EVERY=5000

# --- Build command ---
CMD="accelerate launch \
    --mixed_precision bf16 \
    train_joint_mask.py \
    --model_name_or_path ${MODEL_PATH} \
    --json_file ${JSON_FILE} \
    --results_dir ${RESULTS_DIR} \
    --batch_size_per_device ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_target_modules ${LORA_TARGETS} \
    --lambda_mask ${LAMBDA_MASK} \
    --mask_latent_channels ${MASK_LATENT_CH} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --keep_raw_resolution \
    --max_image_size 256 \
    --condition_dropout_prob 0.1 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --log_every ${LOG_EVERY} \
    --ckpt_every ${CKPT_EVERY} \
    --max_grad_norm 1.0 \
    --report_to tensorboard \
    --mixed_precision bf16"

# --- Add optional resume flags ---
if [ -n "${LORA_RESUME_PATH:-}" ]; then
    CMD="${CMD} --lora_resume_path ${LORA_RESUME_PATH}"
fi
if [ -n "${MASK_MODULES_RESUME:-}" ]; then
    CMD="${CMD} --mask_modules_resume_path ${MASK_MODULES_RESUME}"
fi
if [ -n "${MASK_AE_CKPT:-}" ]; then
    CMD="${CMD} --mask_ae_ckpt ${MASK_AE_CKPT}"
fi

echo "=== Joint Mask Co-Generation Training ==="
echo "Command: ${CMD}"
echo "========================================="

eval ${CMD}
