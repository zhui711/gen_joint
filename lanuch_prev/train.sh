
export HF_HOME="/raid/home/CAMCA/hj880/wt/ckpts/huggingface"
# export PATH="~/bin:$PATH"

# sudo CUDA_VISIBLE_DEVICES=2,3 /raid/home/CAMCA/hj880/miniconda3/bin/conda run -n omnigen accelerate launch \
# 
# sudo CUDA_VISIBLE_DEVICES=2,3 accelerate launch --python ./python3 \
# sudo env CUDA_VISIBLE_DEVICES=2,3 ./python3 -m accelerate.commands.launch \
# sudo env CUDA_VISIBLE_DEVICES=2,3 PYTHONEXECUTABLE="$(pwd)/python3" \
#   ./python3 -m accelerate.commands.launch \
# source ~/.bashrc

sudo env CUDA_VISIBLE_DEVICES=2,3 ./python3 -m accelerate.commands.launch \
    --num_processes=2 \
    train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 128 \
    --condition_dropout_prob 0.01 \
    --lr 3e-4 \
    --use_lora \
    --lora_rank 8 \
    --json_file /raid/home/CAMCA/hj880/wt/dataset/cxr_sythn/anno/cxr_synth_anno_train.jsonl \
    --image_path ./ \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 500 \
    --epochs 100 \
    --log_every 1 \
    --num_workers 2 \
    --results_dir ./results/cxr_finetune_lora