# Anatomy-Aware Loss for OmniGen LoRA Fine-tuning — Implementation Summary

## 1. Files Created/Modified

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `/home/wenting/zr/gen_code/gen_data/generate_pseudo_masks.py` | **Created** | Offline pseudo-mask generator using `torch.multiprocessing.spawn` across 4 GPUs |
| 2 | `/home/wenting/zr/gen_code/gen_data/gen_mask_jsonl.py` | **Created** | JSONL generator that appends `output_mask` field pointing to `.npz` files |
| 3 | `/home/wenting/zr/gen_code/OmniGen/train_helper/data.py` | **Modified** | Added `load_output_mask()`, optional mask loading in `get_example()`, and `output_anatomy_masks` collation |
| 4 | `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py` | **Created** | `training_losses_with_anatomy()` — combined rectified-flow MSE + anatomy segmentation loss |
| 5 | `/home/wenting/zr/gen_code/train_anatomy.py` | **Created** | Anatomy-aware training script (duplicated from `train.py`) |
| 6 | `/home/wenting/zr/gen_code/lanuch/train_anatomy.sh` | **Created** | Launch script with anatomy arguments |

**Original files (`train.py`, `loss.py`) were NOT modified.**

---

## 2. Data Flow of the Anatomy Loss

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                │
│                                                                     │
│  1. DataLoader loads batch:                                         │
│     output_images (PIL→tensor→[-1,1]) + output_anatomy_masks (.npz)│
│                                                                     │
│  2. VAE Encode (no_grad):                                           │
│     x1 = vae.encode(output_images) * scaling - shift                │
│                                                                     │
│  3. Rectified Flow Sampling:                                        │
│     x0 ~ N(0,1)                                                    │
│     t ~ sigmoid(N(0,1))                                             │
│     x_t = t * x1 + (1-t) * x0                                      │
│     u_t = x1 - x0                                                   │
│                                                                     │
│  4. Model Forward:                                                  │
│     u_hat = model(x_t, t, **model_kwargs)                           │
│                                                                     │
│  5. Diffusion Loss:                                                 │
│     L_diff = MSE(u_hat, u_t)                                       │
│                                                                     │
│  6. Reconstruct Clean Latent:                                       │
│     x1_hat = x_t + (1-t) * u_hat      ← has autograd graph!       │
│                                                                     │
│  7. Sub-batch (VRAM safety):                                        │
│     idx = randperm(B)[:anatomy_subbatch_size]                       │
│     x1_hat_sub = x1_hat[idx]                                       │
│     mask_sub = output_anatomy_masks[idx]                             │
│                                                                     │
│  8. Inverse VAE Scale + Decode (NO no_grad!):                       │
│     x1_hat_sub_scaled = x1_hat_sub / scale + shift                 │
│     decoded = vae.decode(x1_hat_sub_scaled).sample                  │
│     decoded = clamp(decoded, -1, 1)     ← stays in [-1,1]         │
│                                                                     │
│  9. Frozen Seg Model Forward:                                       │
│     logits = seg_model(decoded)          ← (n, 10, 256, 256)      │
│                                                                     │
│  10. Anatomy Loss:                                                  │
│     L_anat = 0.5 * BCEWithLogits + 0.5 * DiceLoss                 │
│                                                                     │
│  11. Total Loss:                                                    │
│     L_total = L_diff + λ_anatomy * L_anat                          │
│                                                                     │
│  12. accelerator.backward(L_total)                                  │
│      → grads flow: L_anat → seg_model ops → decoded → vae.decode  │
│        → x1_hat_sub → u_hat → LoRA parameters                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Gradient Flow Path
```
L_anatomy
  ↓ (backprop through frozen seg_model operations — no weight updates)
seg_model(decoded)
  ↓
decoded = vae.decode(x1_hat_sub_scaled)  — (frozen VAE, fp32, no no_grad)
  ↓
x1_hat_sub = x_t + (1 - t) * u_hat
  ↓
u_hat = model(x_t, t, ...)  — LoRA parameters receive gradients
```

---

## 3. VRAM Safety (Sub-batching) and Checkpoint Cleanliness

### VRAM Safety
- **Problem:** VAE decoding from latent to pixel space is memory-intensive. With batch_size=128 and a full decode, GPU OOM is guaranteed.
- **Solution:** Only decode a random sub-batch of `anatomy_subbatch_size=4` samples per step.
  ```python
  n = min(anatomy_subbatch_size, B)
  idx = torch.randperm(B)[:n]
  x1_hat_sub = x1_hat[idx]
  ```
- This limits the decode to 4 images at a time regardless of batch size.
- The sub-batch is randomly selected each step, providing stochastic coverage of the full batch.

### Checkpoint Cleanliness
- **Problem:** If the seg_model's ~93MB weights are saved into OmniGen checkpoints, it bloats storage and creates confusion.
- **Solution:** The seg_model is:
  1. Loaded as a **standalone local variable** in `main()` — never attached to the OmniGen model.
  2. **NOT passed to `accelerator.prepare()`** — accelerator never touches it.
  3. Set to `eval()` and `requires_grad_(False)` — no optimizer state, no gradient buffers.
  4. Moved to `accelerator.device` manually.
- Result: `model.save_pretrained()` (LoRA) saves only the LoRA adapter weights (~3MB), completely excluding the seg_model and VAE.

---

## 4. Pipeline Execution Instructions

### Prerequisites
- Frozen segmentation checkpoint: `/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth`
- `segmentation_models_pytorch` library: `/home/wenting/zr/Segmentation/segmentation_models_pytorch/`
- SV-DRR dataset: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/`

### Step 1: Generate Pseudo-Masks (Offline, One-Time)

```bash
cd /home/wenting/zr/gen_code
python gen_data/generate_pseudo_masks.py \
    --image_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256 \
    --mask_root  /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch \
    --seg_ckpt   /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --num_gpus 4 \
    --batch_size 64
```

- Produces `<mask_root>/<patient>/<view>.npz` files with boolean masks of shape `(10, 256, 256)`.
- Supports resume: existing valid `.npz` files are skipped automatically.
- Uses atomic saves (`.tmp` → `os.replace`) to prevent corruption.

### Step 2: Generate JSONL with Mask Paths

```bash
cd /home/wenting/zr/gen_code

# Train split
python gen_data/gen_mask_jsonl.py \
    --data_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256 \
    --mask_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch \
    --output_dir /home/wenting/zr/wt_dataset/LIDC_IDRI/anno \
    --orientation PA --split train

# Test split
python gen_data/gen_mask_jsonl.py \
    --data_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256 \
    --mask_root /home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch \
    --output_dir /home/wenting/zr/wt_dataset/LIDC_IDRI/anno \
    --orientation PA --split test
```

- Produces `cxr_synth_anno_mask_train.jsonl` and `cxr_synth_anno_mask_test.jsonl`.
- **Hard error** if any `.npz` mask file is missing — ensures Step 1 completed fully.

### Step 3: Launch Anatomy-Aware Training

```bash
cd /home/wenting/zr/gen_code
bash lanuch/train_anatomy.sh
```

Or manually:
```bash
sudo env CUDA_VISIBLE_DEVICES=2,3 ./python3 -m accelerate.commands.launch \
    --num_processes=2 \
    train_anatomy.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 128 \
    --condition_dropout_prob 0.01 \
    --lr 3e-4 \
    --use_lora --lora_rank 8 \
    --json_file /path/to/cxr_synth_anno_mask_train.jsonl \
    --image_path ./ \
    --keep_raw_resolution --max_image_size 1024 \
    --results_dir ./results/cxr_finetune_lora_anatomy \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 0.1 \
    --anatomy_subbatch_size 4
```

### Key Hyperparameters to Tune
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--lambda_anatomy` | 0.1 | Weight of anatomy loss. Start with 0.1, increase if anatomy detail is poor |
| `--anatomy_subbatch_size` | 4 | Increase if VRAM allows. Decrease to 2 if OOM occurs |
| `--lr` | 3e-4 | Standard LoRA learning rate |
| `--batch_size_per_device` | 128 | Original pipeline batch size |
