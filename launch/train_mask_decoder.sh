CKPT_DIR="results/joint_mask_cogen/checkpoints/0030000"

CUDA_VISIBLE_DEVICES=2 /home/wenting/miniconda3/envs/omnigen/bin/python train_mask_decoder_only.py \
  --mask_modules_path "${CKPT_DIR}/mask_modules.bin" \
  --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --mixed_precision no