#!/bin/bash
# Anatomy-Aware Training Launch Script for OmniGen LoRA Fine-tuning
# Scenario A: Transition from pure-diffusion LoRA (step 8000) to anatomy-aware training.
# Uses --lora_resume_path to load weights only; optimizer starts fresh.

export HF_HOME="~/.cache/huggingface"

# 叠加
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     train_anatomy.py \
#     --model_name_or_path Shitao/OmniGen-v1 \
#     --batch_size_per_device 64 \
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
#     --gradient_accumulation_steps 1 \
#     --ckpt_every 500 \
#     --epochs 10 \
#     --log_every 1 \
#     --num_workers 4 \
#     --results_dir ./results/cxr_finetune_lora_SegMSE_lamda0.1_subbatch4 \
#     --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
#     --lambda_anatomy 0.1\
#     --anatomy_subbatch_size 4 \
#     --lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0008000/
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
    --results_dir ./results/cxr_finetune_lora_30ksteps_feature_lamda0.1_subbatch16 \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 0.1 \
    --anatomy_subbatch_size 16 \
    --feature_layer_idx 2 \
    --use_gen_mask True \
    --loss_version v2 \
    --lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/

# 从头开始
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     train_anatomy.py \
#     --model_name_or_path Shitao/OmniGen-v1 \
#     --batch_size_per_device 64 \
#     --condition_dropout_prob 0.01 \
#     --lr 3e-4 \
#     --use_lora \
#     --lora_rank 8 \
#     --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
#     --image_path ./ \
#     --max_input_length_limit 18000 \
#     --keep_raw_resolution \
#     --max_image_size 1024 \
#     --gradient_accumulation_steps 1 \
#     --ckpt_every 500 \
#     --epochs 100 \
#     --log_every 1 \
#     --num_workers 4 \
#     --results_dir ./results/cxr_finetune_lora_anatomy \
#     --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
#     --lambda_anatomy 0.1 \
#     --anatomy_subbatch_size 4
