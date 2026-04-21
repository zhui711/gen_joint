#!/bin/bash
# =============================================================================
# Joint Image-Mask Co-Generation Training Launch Script
# =============================================================================
# This script launches production joint training with:
#   - Expanded LoRA (attention + MLP) at rank 32
#   - Mask latent co-generation with profiled lambda_mask=0.25
#   - Proper checkpointing of both LoRA and mask modules
#   - Explicit 8-GPU Accelerate/DDP launch
#   - Training from the base OmniGen model, not an old LoRA checkpoint
# =============================================================================

set -e

# --- Paths ---
MODEL_PATH="Shitao/OmniGen-v1"
JSON_FILE="/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl"
RESULTS_DIR="results/joint_mask_cogen"

# --- Optional: Pretrained mask autoencoder ---
# MASK_AE_CKPT="results/mask_ae_pretrain/mask_autoencoder.pt"

# --- Optional: Resume from a joint-mask checkpoint only ---
# Keep unset for from-scratch training from the base OmniGen model.
# LORA_RESUME_PATH="results/joint_mask_cogen/checkpoints/0010000/"
# MASK_MODULES_RESUME="results/joint_mask_cogen/checkpoints/0010000/mask_modules.bin"

# --- Hyperparameters ---
BATCH_SIZE=32
GRAD_ACCUM=1
LR=3e-4
NUM_WORKERS=8 

LORA_RANK=32
LORA_ALPHA=32
LORA_TARGETS="qkv_proj o_proj gate_up_proj down_proj"
LAMBDA_MASK=0.25
MASK_LATENT_CH=4
MAX_TRAIN_STEPS=100000
LOG_EVERY=1
CKPT_EVERY=500

# --- Distributed launch ---
NUM_GPUS=8

# --- Build command ---
CMD="accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_GPUS} \
    --main_process_port 29500 \
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
    --max_image_size 1024 \
    --max_input_length_limit 18000 \
    --condition_dropout_prob 0.01 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --log_every ${LOG_EVERY} \
    --ckpt_every ${CKPT_EVERY} \
    --num_workers ${NUM_WORKERS} \
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
