#!/bin/bash
# Anatomy-Aware Training Launch Script v3 (Conservative Recovery)
#
# KEY CHANGES from v2:
#   1. --lambda_anatomy 0.02 (reduced from 0.1 for gentler recovery)
#   2. --use_gen_mask False (use GT mask as stable spatial anchor)
#   3. --t_threshold 0.5 (only compute anatomy loss when t > 0.5)
#
# This script continues from the 30k baseline with conservative anatomy loss
# to recover generative quality while improving anatomical structure.

export HF_HOME="~/.cache/huggingface"

# === Conservative Recovery from 30k Baseline ===
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    train_anatomy.py \
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
    --results_dir ./results/cxr_finetune_lora_30ksteps_feaLay2_v3_modifyLossBoundary \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 0.05 \
    --anatomy_subbatch_size 16 \
    --feature_layer_idx 2 \
    --use_gen_mask False \
    --t_threshold 0.0 \
    --loss_version v3 \
    --lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/

# === Alternative: Even More Conservative (for first test) ===
# Uncomment below for an even gentler start:
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     train_anatomy.py \
#     --model_name_or_path Shitao/OmniGen-v1 \
#     --batch_size_per_device 16 \
#     --condition_dropout_prob 0.01 \
#     --lr 5e-5 \
#     --lr_scheduler constant_with_warmup \
#     --lr_warmup_steps 500 \
#     --use_lora \
#     --lora_rank 8 \
#     --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
#     --image_path ./ \
#     --max_input_length_limit 18000 \
#     --keep_raw_resolution \
#     --max_image_size 1024 \
#     --gradient_accumulation_steps 4 \
#     --ckpt_every 500 \
#     --epochs 10 \
#     --log_every 1 \
#     --num_workers 4 \
#     --results_dir ./results/cxr_finetune_lora_30ksteps_v3_ultraconservative \
#     --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
#     --lambda_anatomy 0.01 \
#     --anatomy_subbatch_size 16 \
#     --feature_layer_idx 2 \
#     --use_gen_mask False \
#     --t_threshold 0.7 \
#     --lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/
