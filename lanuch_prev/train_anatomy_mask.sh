#!/bin/bash
# Mask-based anatomy-aware training launch script for OmniGen LoRA fine-tuning.
#
# This script launches the clean Plan 1 pipeline:
#   diffusion loss + sigmoid(mask logits) MSE against 10-channel GT masks.
# The older feature-matching and v3-specific flags are intentionally removed.

export HF_HOME="~/.cache/huggingface"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    train_anatomy_mask.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 16 \
    --condition_dropout_prob 0.01 \
    --lr 1e-4 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 500 \
    --use_lora \
    --lora_rank 8 \
    --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
    --image_path ./ \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 4 \
    --ckpt_every 500 \
    --epochs 10 \
    --log_every 1 \
    --num_workers 4 \
    --results_dir ./results/cxr_finetune_lora_30ksteps_maskmse_timefilter \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 1.0 \
    --anatomy_alpha 4.0 \
    --anatomy_subbatch_size 16 \
    --lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/
