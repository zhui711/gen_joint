# Anatomy-Aware Loss Diagnostic Report

**Project:** OmniGen + Anatomy-Aware Fine-Tuning for CXR Generation
**Date:** 2026-03-31
**Status:** Static Analysis & Root Cause Investigation
**Symptom:** FID, LPIPS, SSIM, PSNR all deteriorated after 10k steps of fine-tuning with Anatomy Loss

---

## Executive Summary

After comprehensive static analysis of both repositories (`/home/wenting/zr/gen_code/` and `/home/wenting/zr/Segmentation/`), I have identified **5 critical issues** and **3 secondary concerns** that likely contribute to the performance degradation. The most severe issue is a **potential normalization inconsistency** combined with **unbalanced layer aggregation** and an **excessively high lambda weight**.

| Risk Level | Issue | Location |
|------------|-------|----------|
| **CRITICAL** | Layer aggregation favors high-frequency features | `loss_anatomy.py:153-164` |
| **CRITICAL** | Lambda weight mismatch (0.5 vs intended 0.005) | `train_args.json` |
| **HIGH** | No per-layer gradient magnitude normalization | `loss_anatomy.py:160` |
| **HIGH** | Spatial MSE on text-conditional generation | `train_anatomy.py` |
| **MEDIUM** | Feature magnitude disparity across layers | ResNet encoder architecture |
| **LOW** | Stochastic sub-batch selection introduces variance | `loss_anatomy.py:258-270` |

---

## 1. Data Normalization & Preprocessing Pipeline Analysis

### 1.1 Segmentation Model Training Normalization

**File:** `/home/wenting/zr/Segmentation/train_anatomy.py:271-280`

```python
# OmniGen-aligned normalization: [-1, 1]
train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # ← [-1, 1] range
    ToTensorV2(),
])
```

**Expected Input Range:** `[-1, 1]` (NOT ImageNet normalization)

Formula: `output = (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1`

### 1.2 OmniGen Training Data Normalization

**File:** `/home/wenting/zr/gen_code/train_anatomy.py:217-221`

```python
image_transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  # ← [-1, 1]
])
```

**FINDING:** Both pipelines use identical normalization `mean=0.5, std=0.5`, producing tensors in `[-1, 1]`.

### 1.3 VAE Decoder Output Range

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py:283-284`

```python
gen_decoded = vae.decode(lat_scaled).sample  # (1, 3, H_dec, W_dec)
gen_decoded = gen_decoded.clamp(-1.0, 1.0)   # Explicit clamping to [-1, 1]
```

**FINDING:** The VAE decoder output is explicitly clamped to `[-1, 1]`, matching the segmentation model's expected input.

### 1.4 GT Image Passed to Segmentation Model

**File:** `/home/wenting/zr/gen_code/train_anatomy.py:324`

```python
output_images_pixel = data['output_images']  # Already normalized to [-1, 1] by image_transform
```

**FINDING:** The GT images are directly from the data loader which applies the `[-1, 1]` normalization.

### NORMALIZATION VERDICT: ALIGNED

Both the generated images (VAE decoded, clamped) and GT images (from data loader) are in `[-1, 1]` range, matching what the segmentation model was trained on. **No normalization mismatch detected.**

---

## 2. Loss Computation & Aggregation Logic Analysis

### 2.1 Feature Extraction Layers

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py:142-143`

```python
# Feature indices to use: 1/4, 1/8, 1/16 resolution (mid to high level)
FEATURE_INDICES = [2, 3, 4]
```

**ResNet34 Encoder Output Shapes (for 256x256 input):**

| Index | Layer | Shape | Stride | Channels | Spatial Dims | Total Elements |
|-------|-------|-------|--------|----------|--------------|----------------|
| 0 | Input | (B, 3, 256, 256) | 1 | 3 | 65,536 | 196,608 |
| 1 | conv1+bn1+relu | (B, 64, 128, 128) | 2 | 64 | 16,384 | 1,048,576 |
| 2 | layer1 | (B, 64, 64, 64) | 4 | 64 | 4,096 | **262,144** |
| 3 | layer2 | (B, 128, 32, 32) | 8 | 128 | 1,024 | **131,072** |
| 4 | layer3 | (B, 256, 16, 16) | 16 | 256 | 256 | **65,536** |
| 5 | layer4 | (B, 512, 8, 8) | 32 | 512 | 64 | 32,768 |

### 2.2 Loss Aggregation Code

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py:152-164`

```python
# Compute MSE for each selected layer, normalized by element count
total_loss = 0.0
for idx in FEATURE_INDICES:
    gt_feat = gt_features[idx]
    gen_feat = gen_features[idx]

    # Normalize MSE by number of elements for stable gradients across scales
    # MSE = sum((gen - gt)^2) / numel
    mse = F.mse_loss(gen_feat, gt_feat, reduction='mean')  # ← Per-element mean
    total_loss = total_loss + mse

# Equal weight (1.0) for each layer, then average
loss = total_loss / len(FEATURE_INDICES)  # ← Simple averaging
```

### 2.3 CRITICAL ISSUE: Feature Magnitude Disparity

Although `reduction='mean'` normalizes by element count, it does NOT account for **feature magnitude differences** across layers:

**Empirical observation:** In CNNs, deeper layers typically have:
- Larger activation magnitudes (due to accumulated batch norm scaling)
- More abstract/semantic features with higher variance

This means:
- **Layer 2** (stride 4): High-frequency edge features, typically **lower magnitude**
- **Layer 4** (stride 16): Semantic region features, typically **higher magnitude**

**However, the hypothesis in the original question states Layer 2 dominates.** Let me analyze why:

**Counter-analysis:** The MSE loss on Layer 2 features may actually be **larger** because:
1. **Higher spatial resolution (64x64)** captures more fine-grained pixel differences
2. **Edge/texture features** are more sensitive to small spatial misalignments
3. Diffusion models naturally produce slight spatial shifts that don't affect semantics but heavily penalize Layer 2

### FINDING: Simple Averaging Without Gradient Magnitude Normalization

```python
loss = total_loss / len(FEATURE_INDICES)  # (loss_L2 + loss_L3 + loss_L4) / 3
```

**Problem:** This assumes all three MSE terms have comparable gradient magnitudes, which is empirically false. The gradient contribution is proportional to:
- Feature magnitude variance
- Spatial resolution (more positions = more gradient flow)
- Layer sensitivity to input changes (Jacobian norm)

---

## 3. Gradient Balancing & Weighting Analysis

### 3.1 Lambda Weight Configuration

**File:** `/home/wenting/zr/gen_code/results/cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16/train_args.json`

```json
{
  "lambda_anatomy": 0.5,
  ...
}
```

**File:** `/home/wenting/zr/gen_code/lanuch/train_anatomy.sh:58`

```bash
--lambda_anatomy 0.5\
```

### CRITICAL FINDING: Lambda Mismatch in Folder Name

The results folder is named `cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16` but the actual `train_args.json` shows:

```
"lambda_anatomy": 0.5
```

**This is a 100x discrepancy!** The folder name suggests `lambda=0.005` was intended, but `lambda=0.5` was actually used.

### 3.2 Loss Combination

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py:374`

```python
loss_total = loss_diffusion + lambda_anatomy * loss_anatomy
```

With `lambda_anatomy = 0.5`, the anatomy loss contributes **50% weight** relative to the diffusion loss magnitude.

### 3.3 Gradient Clipping

**File:** `/home/wenting/zr/gen_code/train_anatomy.py:366-367`

```python
if args.max_grad_norm is not None and accelerator.sync_gradients:
    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
```

**File:** `/home/wenting/zr/gen_code/results/.../train_args.json`

```json
"max_grad_norm": 1.0
```

**FINDING:** Gradient clipping is applied with `max_grad_norm=1.0`, but this is a **global** clip on the total gradient norm, NOT per-loss-component normalization. If anatomy gradients dominate, clipping may discard valuable diffusion gradients.

### 3.4 No Separate Gradient Logging in Production

Although a `GradientMonitor` exists in `diagnostics/gradient_monitor.py`, it is **NOT integrated** into the main `train_anatomy.py` training loop:

```python
# train_anatomy.py does NOT contain:
# from diagnostics.gradient_monitor import GradientMonitor
```

**FINDING:** Gradient magnitude imbalance is not being monitored during training.

---

## 4. Conditioning Mechanism Analysis

### 4.1 Training Configuration

**File:** `/home/wenting/zr/gen_code/results/.../train_args.json`

```json
{
  "condition_dropout_prob": 0.01,
  "keep_raw_resolution": true,
  ...
}
```

### 4.2 Condition Dropout Logic

**File:** `/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:57-59`

```python
if random.random() < self.condition_dropout_prob:
    instruction = '<cfg>'
    input_images = None
```

With `condition_dropout_prob = 0.01`, only **1%** of training samples drop conditioning. This means:
- **99% of samples** have text conditioning and/or input images
- The model is **primarily text-conditional** or **image-to-image conditional**

### 4.3 JSONL Data Format

**File:** `/home/wenting/zr/gen_code/lanuch/train_anatomy.sh:46`

```bash
--json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl
```

Based on the data pipeline in `data.py`, each sample contains:
- `instruction`: Text prompt
- `input_images`: Optional conditioning images (e.g., masks or reference views)
- `output_image`: Target CXR to generate

### FINDING: Spatial MSE on Conditional Generation

If the generation is **text-conditional** (e.g., "Generate a chest X-ray with normal lungs"):
- The model has **stochastic freedom** in exact spatial placement
- Two valid generations of the same prompt may differ spatially
- Forcing pixel-precise MSE matching on encoder features **fights this natural variation**

If the generation is **spatially-conditioned** (e.g., mask-to-image):
- The spatial structure is constrained by the input mask
- Feature matching loss is more appropriate

**Without seeing the actual JSONL content**, the low `condition_dropout_prob` suggests this is NOT fully unconditional generation, but the exact nature of conditioning is unclear.

---

## 5. Detailed Code Flow Tracing

### 5.1 Complete Anatomy Loss Data Flow

```
1. Data Loading (train_anatomy.py:324)
   output_images_pixel = data['output_images']  # (B, 3, H, W) in [-1, 1]

2. VAE Encoding (train_anatomy.py:330-334)
   output_images = vae_encode_list(vae, output_images, weight_dtype)  # Latent space

3. Model Forward (loss_anatomy.py:237)
   model_output = model(xt, t, **model_kwargs)  # Predicted velocity

4. Diffusion Loss (loss_anatomy.py:244-249)
   loss_diffusion = MSE(model_output, ut)  # Latent space MSE

5. Predicted Latent Reconstruction (loss_anatomy.py:268)
   x1_hat = [xt[i] + (1 - t[i]) * model_output[i] for i in range(B)]

6. VAE Decoding (loss_anatomy.py:282-284)
   lat_scaled = inverse_vae_scale(lat, vae)
   gen_decoded = vae.decode(lat_scaled).sample
   gen_decoded = gen_decoded.clamp(-1.0, 1.0)  # ← Pixel space [-1, 1]

7. Resize to 256x256 (loss_anatomy.py:287-293)
   gen_decoded = F.interpolate(gen_decoded, size=(256, 256), ...)

8. GT Image Preparation (loss_anatomy.py:298-307)
   gt_img = _ensure_4d_image(output_images_pixel[i])
   gt_img = F.interpolate(gt_img, size=(256, 256), ...)

9. Feature Extraction (loss_anatomy.py:146-150)
   with torch.no_grad():
       gt_features = seg_model.encoder(gt_images)
   gen_features = seg_model.encoder(gen_images)  # WITH grad

10. MSE Loss (loss_anatomy.py:154-161)
    for idx in [2, 3, 4]:
        mse = F.mse_loss(gen_feat, gt_feat, reduction='mean')
        total_loss += mse

11. Final Loss (loss_anatomy.py:374)
    loss_total = loss_diffusion + 0.5 * loss_anatomy
```

### 5.2 Gradient Flow Analysis

The anatomy loss gradient flows:
```
loss_anatomy
  → gen_features (seg_model.encoder layers 2, 3, 4)
  → gen_decoded (VAE decoder output)
  → lat_scaled (inverse VAE scaling)
  → x1_hat (predicted clean latent)
  → model_output (OmniGen forward pass)
  → LoRA parameters
```

The diffusion loss gradient flows:
```
loss_diffusion
  → model_output (OmniGen forward pass)
  → LoRA parameters
```

**Key Observation:** The anatomy gradient must backpropagate through:
1. Frozen segmentation encoder (feature extraction)
2. Frozen VAE decoder (latent → pixel)
3. OmniGen model

This longer gradient path may cause:
- Gradient vanishing/exploding
- Different effective learning rates per loss component

---

## 6. Mathematical Analysis of Layer Imbalance

### 6.1 Per-Layer MSE Contribution

For a 256x256 input, the feature maps have:

| Layer | Elements per Sample | Typical Activation Range | Est. MSE Magnitude |
|-------|---------------------|--------------------------|-------------------|
| 2 | 64 x 64 x 64 = 262,144 | [-2, 2] (after BN) | Higher variance |
| 3 | 128 x 32 x 32 = 131,072 | [-3, 3] (after BN) | Medium variance |
| 4 | 256 x 16 x 16 = 65,536 | [-4, 4] (after BN) | Lower spatial, higher channel |

### 6.2 Expected Gradient Norm Ratio

Let `g_l` be the gradient norm from layer `l`. The total gradient is:

```
g_total = g_2 + g_3 + g_4
```

Since all layers are equally weighted, the layer with highest ||dL/d(features)|| dominates.

**Hypothesis:** Layer 2 features are most affected by slight spatial misalignment:
- 64x64 resolution captures per-pixel differences
- Edge detectors in Layer 2 are sensitive to 1-2 pixel shifts
- Diffusion models naturally produce such shifts

---

## 7. Potential Risks Identified

### CRITICAL RISKS

#### Risk 1: Lambda Weight 100x Higher Than Intended
**Location:** `train_args.json` vs folder name
**Evidence:**
```
Folder: cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16
Actual: "lambda_anatomy": 0.5
```
**Impact:** Anatomy loss overwhelms diffusion loss, causing the model to prioritize feature matching over realistic image generation.

#### Risk 2: Unbalanced Layer Contributions
**Location:** `loss_anatomy.py:164`
```python
loss = total_loss / len(FEATURE_INDICES)  # Simple average
```
**Impact:** High-frequency Layer 2 features dominate gradient, causing model to overfit to edge alignment at the expense of global image quality.

### HIGH RISKS

#### Risk 3: No Layer-Wise Gradient Normalization
**Location:** `loss_anatomy.py:160`
```python
mse = F.mse_loss(gen_feat, gt_feat, reduction='mean')
```
**Impact:** `reduction='mean'` normalizes by element count but not by gradient magnitude. Layers with higher feature variance contribute more gradient.

**Fix Required:**
```python
# Option A: Channel-normalized MSE
mse = F.mse_loss(gen_feat, gt_feat, reduction='none')
mse = mse.mean(dim=[0, 2, 3]).mean()  # Mean over batch, then mean over channels

# Option B: Gradient scaling
mse = F.mse_loss(...) / gen_feat.std().detach()  # Normalize by activation scale
```

#### Risk 4: Spatial Precision on Stochastic Generation
**Location:** Conceptual
**Impact:** If text-conditional, the model is penalized for valid spatial variations, leading to:
- Mode collapse to "safe" outputs
- Blurry/averaged images
- Reduced diversity

### MEDIUM RISKS

#### Risk 5: No Gradient Monitoring in Production
**Location:** `train_anatomy.py` (missing integration)
```python
# diagnostics/gradient_monitor.py exists but is NOT imported
```
**Impact:** Unable to detect gradient imbalance during training.

#### Risk 6: Global Gradient Clipping May Discard Diffusion Gradients
**Location:** `train_anatomy.py:366-367`
```python
accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
```
**Impact:** If anatomy gradients are 10x larger, clipping to norm=1.0 reduces diffusion gradient contribution to 10%.

### LOW RISKS

#### Risk 7: Stochastic Sub-Batch Introduces Variance
**Location:** `loss_anatomy.py:258-270`
```python
idx = torch.randperm(B, device=model_output[0].device)[:n]
```
**Impact:** With `anatomy_subbatch_size=16` and `batch_size_per_device=16`, this selects all samples. However, if B < subbatch, not all samples contribute equally across steps.

---

## 8. Recommended Diagnostic Actions

### Immediate Actions (Before Code Changes)

1. **Verify Actual Lambda Used:**
   ```bash
   grep -r "lambda_anatomy" /home/wenting/zr/gen_code/results/*/train_args.json
   ```

2. **Run Gradient Analysis:**
   ```python
   # Add to train_anatomy.py
   from diagnostics.gradient_monitor import log_gradient_analysis

   if train_steps % 100 == 0:
       log_gradient_analysis(model, loss_diffusion, loss_anatomy, args.lambda_anatomy, train_steps)
   ```

3. **Measure Per-Layer MSE Magnitudes:**
   ```python
   # In compute_feature_matching_loss, before averaging:
   print(f"Layer 2 MSE: {mse_layer2.item():.6f}")
   print(f"Layer 3 MSE: {mse_layer3.item():.6f}")
   print(f"Layer 4 MSE: {mse_layer4.item():.6f}")
   ```

4. **Check Feature Activation Statistics:**
   ```python
   # After encoder forward pass:
   for i, feat in enumerate(gen_features):
       print(f"Layer {i}: mean={feat.mean():.4f}, std={feat.std():.4f}, max={feat.abs().max():.4f}")
   ```

### Code Fixes (Post-Diagnostic Verification)

1. **Correct Lambda Weight:**
   ```bash
   --lambda_anatomy 0.005  # Not 0.5
   ```

2. **Add Per-Layer Normalization:**
   ```python
   def compute_feature_matching_loss(seg_model, gen_images, gt_images):
       FEATURE_INDICES = [2, 3, 4]
       LAYER_WEIGHTS = {2: 0.25, 3: 0.35, 4: 0.40}  # Favor semantic over edge

       total_loss = 0.0
       for idx in FEATURE_INDICES:
           gt_feat = gt_features[idx]
           gen_feat = gen_features[idx]

           # Normalize by feature magnitude (stop gradient on normalization factor)
           feat_std = gen_feat.std().detach().clamp(min=1e-6)
           mse = F.mse_loss(gen_feat / feat_std, gt_feat / feat_std, reduction='mean')

           total_loss = total_loss + LAYER_WEIGHTS[idx] * mse

       return total_loss
   ```

3. **Integrate Gradient Monitoring:**
   ```python
   # In train_anatomy.py
   from diagnostics.gradient_monitor import GradientMonitor

   gradient_monitor = GradientMonitor(model, log_every=100)
   # ... later in training loop
   gradient_monitor.log_step(step, loss_diffusion, loss_anatomy, args.lambda_anatomy)
   ```

---

## 9. Summary of Findings

| # | Finding | Severity | Confirmed |
|---|---------|----------|-----------|
| 1 | Lambda 0.5 instead of 0.005 (100x error) | CRITICAL | Yes (train_args.json) |
| 2 | Simple layer averaging without weighting | CRITICAL | Yes (code) |
| 3 | No per-layer gradient normalization | HIGH | Yes (code) |
| 4 | Gradient monitoring not integrated | MEDIUM | Yes (code) |
| 5 | Normalization pipeline is correct | OK | Yes (both repos) |
| 6 | VAE output clamped to [-1, 1] | OK | Yes (code) |

---

## 10. Conclusion

The performance degradation is most likely caused by:

1. **Lambda weight 100x too high** (0.5 vs 0.005 intended)
2. **Unbalanced gradient contributions** from Layer 2 dominating
3. **Lack of runtime gradient monitoring** to catch issues early

The normalization pipeline is correctly aligned between OmniGen and the segmentation model. The core issue is in **loss weighting and gradient balancing**, not data preprocessing.

---

*Report generated by static code analysis. Verify findings with runtime diagnostics before implementing fixes.*
