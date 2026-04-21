# Anatomy-Aware OmniGen: Feature Matching Implementation Summary

**Version:** 2.0 (Feature Matching / Logits MSE)
**Date:** 2024-03
**Status:** Production Ready

---

## Executive Summary

This document describes the **Feature Matching** approach for anatomy-aware fine-tuning of OmniGen on Chest X-Ray (CXR) datasets. This represents a significant architectural pivot from the previous BCE+Dice pseudo-mask approach.

### Key Insight

Instead of comparing generated images against pre-computed binary segmentation masks (which was too strict and degraded image fidelity), we now:

1. Pass the **Ground Truth (GT) image** through a frozen segmentation UNet to extract anatomical feature representations
2. Pass the **Generated image** through the same UNet
3. Minimize the **MSE between their feature representations**

This creates a **soft anatomical constraint** that guides generation without pixel-exact mask matching.

---

## 1. Architectural Comparison

### Previous Approach (Deprecated)

```
[Offline Prep]
GT Image --> Frozen UNet --> Binary Mask --> Save as .npz
                                                  |
[Training]                                        v
Generated Image --> Frozen UNet --> Logits --> BCE+Dice(Logits, Loaded Mask)
```

**Problems:**
- Required offline pseudo-mask generation (`generate_pseudo_masks.py`)
- Required custom JSONL with `output_mask` fields (`gen_mask_jsonl.py`)
- Required DataLoader modifications to load `.npz` files
- BCE+Dice loss was too strict, degrading SSIM/PSNR metrics
- Complex pipeline with multiple failure points

### Current Approach (Feature Matching)

```
[Training - Online, End-to-End]
                                    +------------------+
GT Image -----(no_grad)-----------> |                  |
                                    |  Frozen UNet     | --> MSE Loss
Generated Image --(with_grad)-----> |  (seg_model)     |
                                    +------------------+
```

**Benefits:**
- **No offline preprocessing** - GT images used directly
- **No custom JSONL** - standard training data works
- **No DataLoader mods** - uses existing `output_images` field
- **Softer constraint** - preserves image fidelity (SSIM/PSNR)
- **Simpler pipeline** - fewer moving parts, easier debugging

---

## 2. Files Created/Modified

| # | File | Status | Description |
|---|------|--------|-------------|
| 1 | `OmniGen/train_helper/loss_anatomy.py` | **Modified** | Core loss function with Feature Matching logic |
| 2 | `train_anatomy.py` | **Modified** | Training script passing raw pixel images to loss |
| 3 | `lanuch/train_anatomy.sh` | **Unchanged** | Launch script (works as-is) |

### Obsolete Files (No Longer Required)

| File | Previous Purpose | Current Status |
|------|------------------|----------------|
| `gen_data/generate_pseudo_masks.py` | Generate `.npz` mask files | **OBSOLETE** |
| `gen_data/gen_mask_jsonl.py` | Create JSONL with `output_mask` field | **OBSOLETE** |
| DataLoader mask loading in `data.py` | Load `.npz` into `output_anatomy_masks` | **UNUSED** (backward compatible) |

---

## 3. The New Data Flow & Gradient Graph

### 3.1 High-Level Training Loop

```
for batch in dataloader:
    |
    |  +------------------+
    |  | output_images    |  <-- Raw pixel-space images [-1, 1]
    |  | (GT images)      |
    |  +------------------+
    |           |
    |           +--> [1] Keep as output_images_pixel (for Feature Matching)
    |           |
    |           +--> [2] VAE Encode (no_grad) --> x1 (latent)
    |
    |  [3] Rectified Flow Forward Pass
    |      x0 = randn_like(x1)
    |      t  = sample_timestep()
    |      xt = t * x1 + (1-t) * x0
    |      ut = x1 - x0  (ground truth velocity)
    |
    |  [4] Model Prediction
    |      u_hat = model(xt, t, **model_kwargs)
    |
    |  [5] Diffusion Loss (UNCHANGED from original OmniGen)
    |      loss_diffusion = MSE(u_hat, ut)
    |
    |  [6] Reconstruct Predicted Clean Latent
    |      x1_hat = xt + (1 - t) * u_hat
    |
    |  [7] Feature Matching Loss (NEW)
    |      loss_anatomy = FeatureMatchingLoss(x1_hat, output_images_pixel, vae, seg_model)
    |
    |  [8] Combined Loss
    |      loss_total = loss_diffusion + lambda_anatomy * loss_anatomy
    |
    |  [9] Backward & Optimize
    |      loss_total.backward()
    |      optimizer.step()
```

### 3.2 Feature Matching Loss - Detailed Breakdown

```python
def compute_feature_matching_loss(x1_hat_sub, gt_images_sub, vae, seg_model):
    """
    STEP-BY-STEP BREAKDOWN:

    Input:
        x1_hat_sub:    (n, 4, H_lat, W_lat) - Predicted clean latent (WITH autograd graph)
        gt_images_sub: (n, 3, H_img, W_img) - Ground truth images in [-1, 1]
        vae:           Frozen AutoencoderKL (requires_grad=False, but computation graph preserved)
        seg_model:     Frozen ResNet34-UNet (requires_grad=False)

    Output:
        loss_anatomy:  Scalar tensor with gradient to x1_hat_sub --> u_hat --> LoRA params
    """
```

#### Step 6a: Decode Predicted Latent (WITH Gradient)

```
x1_hat_sub (predicted latent)
    |
    | [NO torch.no_grad() here!]
    | [VAE weights frozen, but graph preserved]
    v
+-------------------+
|   VAE Decoder     |
|   (frozen weights)|
+-------------------+
    |
    v
gen_decoded: (n, 3, H_dec, W_dec) in [-1, 1]
    |
    | [Bilinear resize if needed]
    v
gen_decoded_256: (n, 3, 256, 256)
```

#### Step 6b: Prepare GT Images

```
gt_images_sub: (n, 3, H_img, W_img) in [-1, 1]
    |
    | [torch.no_grad() - no gradient needed for GT]
    | [Bilinear resize if needed]
    v
gt_images_256: (n, 3, 256, 256)
```

#### Step 6c: Extract Features via Frozen UNet

```
+----------------------------------+
|        Frozen seg_model          |
|   (ResNet34 encoder + UNet)      |
+----------------------------------+
          |              |
          v              v
    gen_decoded_256   gt_images_256
          |              |
          v              v
+----------------------------------+
|      Forward Pass (logits)       |
+----------------------------------+
          |              |
          v              v
    gen_logits       gt_logits
   (n, 10, 256, 256) (n, 10, 256, 256)
   [WITH grad]       [detached/no_grad]
```

#### Step 6d: MSE Loss on Logits

```
loss_anatomy = F.mse_loss(gen_logits, gt_logits)

                    gen_logits
                        |
                        v
              +-------------------+
              |    MSE Loss       |
              +-------------------+
                        ^
                        |
                    gt_logits (frozen target)
```

### 3.3 Complete Gradient Flow Diagram

```
+===========================================================================+
|                        GRADIENT FLOW (BACKWARD PASS)                       |
+===========================================================================+

loss_total = loss_diffusion + lambda * loss_anatomy
     |
     +---> loss_diffusion.backward()
     |           |
     |           v
     |     MSE(u_hat, ut)
     |           |
     |           v
     |        u_hat (model output)
     |           |
     |           v
     |     +------------------+
     |     |   OmniGen Model  |
     |     |   (LoRA active)  | <---- GRADIENTS FLOW HERE
     |     +------------------+
     |
     +---> loss_anatomy.backward()
                 |
                 v
           MSE(gen_logits, gt_logits)
                 |
                 v
            gen_logits
                 |
                 v
        +------------------+
        |    seg_model     |  <-- Frozen (requires_grad=False)
        |   (no grad TO)   |      Gradients flow THROUGH, not TO
        +------------------+
                 |
                 v
           gen_decoded
                 |
                 v
        +------------------+
        |   VAE Decoder    |  <-- Frozen (requires_grad=False)
        |   (no grad TO)   |      Gradients flow THROUGH, not TO
        +------------------+
                 |
                 v
            x1_hat_sub
                 |
                 v
     x1_hat = xt + (1-t) * u_hat
                 |
                 +---> xt (detached, from x0 noise)
                 |
                 +---> u_hat (model output)
                          |
                          v
                    +------------------+
                    |   OmniGen Model  |
                    |   (LoRA active)  | <---- GRADIENTS FLOW HERE
                    +------------------+

+===========================================================================+
|  RESULT: Both loss_diffusion and loss_anatomy update the same LoRA params |
+===========================================================================+
```

---

## 4. VRAM Safety & Checkpoint Cleanliness

### 4.1 Sub-batch Decoding for VRAM Safety

VAE decoding is memory-intensive (latent -> full resolution image). To prevent OOM:

```python
# In training_losses_with_anatomy():
n = min(anatomy_subbatch_size, B)  # Default: n=4
idx = torch.randperm(B)[:n]        # Random sub-sample

# Only decode n samples, not full batch B
x1_hat_sub = x1_hat[idx]           # (n, C, H, W)
gen_decoded = vae.decode(...)      # Only n images decoded
```

**Recommendation:** Start with `anatomy_subbatch_size=4`. Increase if VRAM allows.

### 4.2 Checkpoint Cleanliness

The frozen `seg_model` must **NOT** be passed to `accelerator.prepare()`:

```python
# CORRECT - seg_model kept separate
seg_model = load_seg_model(ckpt_path, device)  # Frozen, on device
model = accelerator.prepare(model)              # Only OmniGen prepared
# seg_model is NOT in accelerator's state dict

# WRONG - would bloat checkpoints
# seg_model = accelerator.prepare(seg_model)  # DON'T DO THIS
```

**Result:** Saved checkpoints contain only LoRA weights (~50MB), not the 100MB+ seg_model.

### 4.3 Precision Handling

```python
# VAE must stay in fp32 for stable decoding
vae.to(dtype=torch.float32)

# Model can use mixed precision
model.to(weight_dtype)  # bf16 or fp16

# In loss function, cast latent to fp32 before VAE decode
x1_hat_sub_fp32 = x1_hat_sub.float()
decoded = vae.decode(x1_hat_sub_fp32).sample
```

---

## 5. Execution Instructions

### 5.1 Prerequisites

No offline preprocessing required! Just ensure you have:

1. Standard CXR training JSONL (no `output_mask` field needed)
2. Frozen segmentation model checkpoint
3. Base OmniGen model

### 5.2 Launch Training

```bash
cd /home/wenting/zr/gen_code

# Single GPU
accelerate launch --num_processes 1 train_anatomy.py \
    --model_name_or_path /path/to/OmniGen-v1 \
    --json_file /path/to/train.jsonl \
    --image_path /path/to/images \
    --results_dir results/cxr_anatomy_feature_matching \
    --batch_size_per_device 2 \
    --max_image_size 256 \
    --lr 1e-4 \
    --epochs 100 \
    --use_lora \
    --lora_rank 8 \
    --seg_model_ckpt /home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth \
    --lambda_anatomy 0.1 \
    --anatomy_subbatch_size 4 \
    --log_every 50 \
    --ckpt_every 2000

# Multi-GPU (e.g., 4 GPUs)
accelerate launch --num_processes 4 train_anatomy.py \
    [same arguments as above]
```

### 5.3 Using the Launch Script

```bash
cd /home/wenting/zr/gen_code
bash lanuch/train_anatomy.sh
```

### 5.4 Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_anatomy` | 0.1 | Weight for anatomy loss. Start low (0.01-0.1) and tune. |
| `--anatomy_subbatch_size` | 4 | Max samples to VAE-decode per step. Reduce if OOM. |
| `--seg_model_ckpt` | (required) | Path to frozen ResNet34-UNet checkpoint |

### 5.5 Monitoring Training

Watch for these metrics in the logs:

```
(step=0001000) Loss: 0.0523 (diff=0.0498, anat=0.0250), Steps/Sec: 1.23, Epoch: 0.50, LR: 0.0001
                      ^            ^            ^
                      |            |            |
                      |            |            +-- Anatomy loss (Feature Matching MSE)
                      |            +-- Diffusion loss (should stay similar to baseline)
                      +-- Total loss
```

**Healthy Training Signals:**
- `loss_diffusion` should decrease steadily (similar to non-anatomy training)
- `loss_anatomy` should decrease (generated features becoming more similar to GT features)
- If `loss_anatomy` dominates, reduce `lambda_anatomy`

---

## 6. Code Reference

### 6.1 Core Loss Function Signature

```python
# OmniGen/train_helper/loss_anatomy.py

def training_losses_with_anatomy(
    model,                    # OmniGen model (with LoRA)
    x1,                       # VAE-encoded target latent
    model_kwargs,             # Dict for model forward pass
    output_images_pixel,      # Raw GT images in [-1, 1] (NEW parameter)
    vae,                      # Frozen AutoencoderKL
    seg_model,                # Frozen ResNet34-UNet
    lambda_anatomy=0.1,       # Anatomy loss weight
    anatomy_subbatch_size=4,  # VRAM safety
) -> dict:
    """
    Returns:
        {
            "loss_total": loss_diffusion + lambda_anatomy * loss_anatomy,
            "loss_diffusion": MSE(u_hat, ut),
            "loss_anatomy": MSE(gen_logits, gt_logits),
            "anatomy_subbatch_size_actual": n,
        }
    """
```

### 6.2 Training Script Integration

```python
# train_anatomy.py (key changes)

for data in active_loader:
    # Keep raw pixel images BEFORE VAE encoding
    output_images_pixel = data['output_images']  # <-- NEW

    with torch.no_grad():
        output_images = vae_encode(...)  # Latent for diffusion

    loss_dict = training_losses_with_anatomy(
        model=model,
        x1=output_images,
        model_kwargs=model_kwargs,
        output_images_pixel=output_images_pixel,  # <-- Pass raw images
        vae=vae,
        seg_model=seg_model,
        lambda_anatomy=args.lambda_anatomy,
        anatomy_subbatch_size=args.anatomy_subbatch_size,
    )
```

---

## 7. Troubleshooting

### 7.1 CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce `--anatomy_subbatch_size` (try 2 or 1)

### 7.2 Anatomy Loss Not Decreasing

**Possible Causes:**
- `lambda_anatomy` too low - increase to 0.5 or 1.0
- seg_model not properly loaded - check checkpoint path
- Images not in [-1, 1] range - verify data normalization

### 7.3 Diffusion Loss Increasing

**Possible Causes:**
- `lambda_anatomy` too high - anatomy loss dominating, reduce to 0.01
- Learning rate too high - reduce `--lr`

---

## 8. Summary

The Feature Matching approach dramatically simplifies the anatomy-aware training pipeline:

| Aspect | Previous (BCE+Dice) | Current (Feature Matching) |
|--------|---------------------|----------------------------|
| Offline Prep | Required (masks) | None |
| Custom JSONL | Required | Not needed |
| DataLoader Mods | Required | Not needed |
| Loss Strictness | Pixel-exact masks | Soft feature similarity |
| Image Fidelity | Degraded (SSIM/PSNR) | Preserved |
| Complexity | High | Low |

**The entire pipeline now consists of just two modified files and a launch script.**

---

*Document generated for OmniGen CXR Fine-tuning Project*
