# OmniGen Anatomy-Loss Refactor Investigation

## Scope

Inspected files:

- `/home/wenting/zr/gen_code/train_anatomy.py`
- `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py`
- `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py`
- `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v3.py`
- `/home/wenting/zr/gen_code/OmniGen/train_helper/data.py`
- `/home/wenting/zr/gen_code/OmniGen/utils.py`
- `/home/wenting/zr/gen_code/lanuch/train_anatomy.sh`
- `/home/wenting/zr/gen_code/lanuch/train_anatomy_v3.sh`

I also verified one real JSONL sample and one real `.npz` mask file from:

- `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`
- `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz`

---

## 1. Data Flow & Gradient Path

### 1.1 Current training path in `train_anatomy.py`

Relevant references:

- `train_anatomy.py:322-345`
- `train_anatomy.py:348-387`
- `loss_anatomy.py:215-338`
- `loss_anatomy_v2.py:294-376`
- `loss_anatomy_v3.py:379-465`

Current flow:

1. `data["output_images"]` comes from the dataloader as GT pixel images in `[-1, 1]`.
2. Inside `with torch.no_grad():`, GT images are VAE-encoded into `output_images` latents via `vae_encode` / `vae_encode_list`.
3. Those GT latents become `x1`, the rectified-flow target.
4. The model samples `x0`, `t`, builds `xt`, and predicts `model_output = model(xt, t, **model_kwargs)`.
5. The predicted clean latent is reconstructed as:
   - variable-res: `x1_hat[i] = xt[i] + (1 - t[i]) * model_output[i]`
   - fixed-res: `x1_hat = xt + (1 - t_) * model_output`
6. `x1_hat` is decoded by the frozen VAE outside `torch.no_grad()`:
   - `lat_scaled = inverse_vae_scale(...)`
   - `gen_decoded = vae.decode(lat_scaled).sample`
7. `gen_decoded` is resized to `(256, 256)` if needed.
8. The frozen segmentation model consumes `gen_decoded`.
9. `loss_anatomy` is computed and added to diffusion loss.

### 1.2 Does gradient flow back from the segmentation branch?

Yes.

Why:

- The GT VAE encode is inside `torch.no_grad()` in `train_anatomy.py:322-335`, but that only affects the GT target latents `x1`.
- The generated branch is created from `model_output`, which is produced by the trainable OmniGen model and stays on the autograd graph.
- VAE decode for generated latents is performed outside `torch.no_grad()` in:
  - `loss_anatomy.py:243-251`, `loss_anatomy.py:310-312`
  - `loss_anatomy_v2.py:303-307`, `loss_anatomy_v2.py:344-346`
  - `loss_anatomy_v3.py:393-396`, `loss_anatomy_v3.py:432-435`
- The segmentation model is frozen with `model.requires_grad_(False)` in `train_anatomy.py:63-76`, but frozen weights still allow gradients to flow with respect to the input image.
- In v2/v3, only the GT feature branch is wrapped in `with torch.no_grad()`:
  - `loss_anatomy_v2.py:163-165`
  - `loss_anatomy_v3.py:191-193`
- The generated feature branch is not wrapped in `no_grad()`:
  - `loss_anatomy_v2.py:167-169`
  - `loss_anatomy_v3.py:195-197`

Conclusion:

- The generated image tensor does not need to be a leaf tensor with an explicit manual `requires_grad=True`.
- It already participates in autograd because it is computed from `model_output`.
- There is no `.detach()` on the generated branch before the segmentation model.
- Therefore gradients can flow:

`loss_anatomy -> seg_model(gen_decoded) -> VAE decode -> x1_hat -> model_output -> OmniGen/LoRA params`

### 1.3 Important nuance about clamping

- `loss_anatomy.py` and `loss_anatomy_v2.py` clamp decoded images to `[-1, 1]`.
- `loss_anatomy_v3.py` explicitly removed that clamp.

Impact:

- Clamp is differentiable only inside range and zero-gradient outside range.
- So gradient is preserved, but can be partially suppressed for OOD decoded pixels.

This is not a detachment bug, but it is a gradient-strength difference between old and v3 behavior.

### 1.4 Old vs current anatomy loss branch

- `loss_anatomy.py`: segmentation logits -> BCEWithLogits + Dice.
- `loss_anatomy_v2.py`: feature MSE with mask-weighted encoder features.
- `loss_anatomy_v3.py`: feature MSE with timestep gating + area normalization logic.

For Plan 1, the desired new branch is structurally closer to `loss_anatomy.py` than to v2/v3:

`gen image -> frozen seg model -> 10-channel predicted mask -> per-channel MSE vs GT mask -> sum`

---

## 2. DataLoader & GT Mask Shape

### 2.1 How the GT mask is loaded

Relevant references:

- `data.py:46-51`
- `data.py:66-70`
- `data.py:125-128`

Actual loading path:

1. Each JSONL record may include `output_mask`.
2. `DatasetFromJson.get_example()` reads `example["output_mask"]`.
3. `load_output_mask(mask_path, key="mask")` runs:
   - `np.load(mask_path)`
   - `data["mask"].astype(np.float32)`
   - `torch.from_numpy(mask)`

Verified sample JSONL row:

- `output_mask`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz`

Verified real file contents:

- keys: `["mask"]`
- stored shape: `(10, 256, 256)`
- stored dtype: `bool`

After loading:

- dataset sample mask shape: `(10, 256, 256)`
- dataset sample mask dtype: `float32`

### 2.2 How batch shape is formed

`TrainDataCollator.__call__()` stacks masks with:

- `torch.stack([f[2] for f in features], dim=0)` at `data.py:125-128`

So the batch tensor is:

- `output_anatomy_masks.shape == (B, 10, 256, 256)`

This is true regardless of `keep_raw_resolution`.

### 2.3 Exact shape at the loss boundary

In `train_anatomy.py`, the loss receives:

- `output_anatomy_masks = data.get("output_anatomy_masks", None)` at `train_anatomy.py:348-355`

Then the loss helpers slice it as:

- variable-res single-sample path:
  - `output_anatomy_masks[i:i+1]` -> `(1, 10, 256, 256)`
- fixed-res sub-batch path:
  - `output_anatomy_masks[idx]` -> `(n, 10, 256, 256)`

So the exact tensor shape reaching anatomy-loss calculation is:

- batch-level: `(B, 10, 256, 256)`
- sub-batch-level inside loss: `(n, 10, 256, 256)`

It is not channel-last and not missing the batch dimension.

### 2.4 Resolution alignment note

The GT mask is always 256x256 from disk.

The GT/output images may be variable-resolution because `crop_arr` preserves aspect ratio and only constrains size/multiple-of-16. That is why all current anatomy-loss helpers resize generated images, and in v2/v3 also GT pixel images, to `(256, 256)` before segmentation/loss.

This means the current anatomy branch is already organized around:

- fixed GT mask resolution: `256 x 256`
- potentially variable decoded image resolution -> resized to `256 x 256`

That matches the new mask-MSE plan well.

---

## 3. Loss Integration

### 3.1 Where `anatomy_loss` is added

All existing anatomy loss modules combine losses the same way:

- `loss_anatomy.py:338`
- `loss_anatomy_v2.py:376`
- `loss_anatomy_v3.py:465`

Formula:

`loss_total = loss_diffusion + lambda_anatomy * loss_anatomy`

### 3.2 What controls the weight

The controlling argument is:

- `--lambda_anatomy` in `train_anatomy.py:549-552`

It is passed into the selected loss function at:

- `train_anatomy.py:367`
- `train_anatomy.py:382`

So the anatomy branch weight is entirely controlled by `args.lambda_anatomy`.

### 3.3 Current launch-script values

From `lanuch/train_anatomy.sh`:

- active run uses `--loss_version v2`
- `--lambda_anatomy 0.1`
- `--anatomy_subbatch_size 16`
- `--use_gen_mask True`

From `lanuch/train_anatomy_v3.sh`:

- active run uses `--loss_version v3`
- `--lambda_anatomy 0.05`
- `--anatomy_subbatch_size 16`
- `--use_gen_mask False`
- `--t_threshold 0.0`

Important mismatch:

- The header comment in `train_anatomy_v3.sh` says `0.02` and `0.5`.
- The actual active command passes `0.05` and `0.0`.

So the script comment is stale; the command itself is authoritative.

### 3.4 Other anatomy-related controls currently in the training entrypoint

Current `train_anatomy.py` also exposes:

- `--feature_layer_idx`
- `--use_gen_mask`
- `--t_threshold`
- `--loss_version`

These exist because v2/v3 are feature-matching variants.

For Plan 1, these are either unnecessary or should be removed from the new clean entrypoint to avoid confusion.

---

## 4. Refactoring Strategy (Blueprint)

### 4.1 Goal

Create a clean mask-based anatomy-loss pipeline without bloating the existing v2/v3 training script.

Recommended new files:

- `/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_mask.py`
- `/home/wenting/zr/gen_code/train_anatomy_mask.py`
- optionally `/home/wenting/zr/gen_code/lanuch/train_anatomy_mask.sh`

Keep these existing files intact for reproducibility:

- `loss_anatomy.py`
- `loss_anatomy_v2.py`
- `loss_anatomy_v3.py`
- `train_anatomy.py`

### 4.2 What to reuse vs drop

Reuse:

- dataset and collator from `OmniGen/train_helper/data.py`
- segmentation-model loader pattern from `train_anatomy.py`
- rectified-flow core from current loss functions
- sub-batch decode logic
- 256x256 resize before seg inference

Drop from the new mask pipeline:

- feature extraction on GT image
- `output_images_pixel` dependency
- `feature_layer_idx`
- `use_gen_mask` flag
- `t_threshold`
- v3 area normalization
- v3 timestep gating

### 4.3 Exact insertion points for `train_anatomy_mask.py`

Recommended structure:

1. Copy `train_anatomy.py` into `train_anatomy_mask.py`.
2. Replace imports:
   - remove `training_losses_with_anatomy_v2`
   - remove `training_losses_with_anatomy_v3`
   - import `training_losses_with_anatomy_mask`
3. Keep `load_seg_model()` exactly as-is.
4. Keep dataset/collator setup exactly as-is.
5. In the training loop:
   - keep GT VAE encode block for `output_images -> x1`
   - remove `output_images_pixel` handling, because mask-MSE does not need GT pixel images
   - keep `output_anatomy_masks = data.get(...)`
   - replace the current v2/v3 branch with a single call to `training_losses_with_anatomy_mask(...)`
6. Simplify CLI args:
   - keep `seg_model_ckpt`
   - keep `lambda_anatomy`
   - keep `anatomy_subbatch_size`
   - remove `feature_layer_idx`
   - remove `use_gen_mask`
   - remove `t_threshold`
   - remove `loss_version`

### 4.4 Exact insertion points for `loss_anatomy_mask.py`

Recommended contents:

1. Reuse or copy these utility helpers from current loss files:
   - `mean_flat`
   - `sample_x0`
   - `sample_timestep`
   - `inverse_vae_scale`
   - `_ensure_4d_latent`
2. Implement a new core loss:
   - `compute_mask_mse_loss(seg_model, gen_images, mask_gt)`
3. Implement:
   - `training_losses_with_anatomy_mask(...)`

The new high-level loss flow should be:

1. Compute standard diffusion loss exactly as current scripts do.
2. Reconstruct `x1_hat` from `xt` and `model_output`.
3. Select a random sub-batch of size `n = min(anatomy_subbatch_size, B)`.
4. Decode `x1_hat` with the frozen VAE outside `no_grad()`.
5. Resize decoded image to `(256, 256)` if needed.
6. Run `seg_model(gen_decoded)` to get `(n, 10, 256, 256)`.
7. Convert segmentation output to predicted masks.
8. Compute per-channel MSE against GT masks and sum the 10 channel losses.
9. Return:
   - `loss_total`
   - `loss_diffusion`
   - `loss_anatomy`
   - `anatomy_subbatch_size_actual`

### 4.5 Predicted-mask representation: logits or sigmoid?

Source-based observation:

- The segmentation model is created with `activation=None` in `train_anatomy.py:49-57`.
- That means the model outputs raw logits, not probabilities.
- In v2/v3, when the code needs a mask from the segmentation model, it explicitly uses `torch.sigmoid(logits_gen)`.

Inference:

- If the new objective is defined as MSE between "Gen Mask" and GT mask, the safer interpretation is:
  - `mask_gen = torch.sigmoid(seg_model(gen_decoded))`
  - then compare `mask_gen` to GT mask in `[0, 1]`

This matches the semantics of "mask" better than using raw logits directly.

### 4.6 Per-channel MSE logic

Required behavior:

- compute MSE independently for each of 10 channels
- sum the 10 channel losses

Two valid PyTorch-native implementations:

#### Explicit loop version

This is the clearest and matches the requirement literally:

```python
loss_total = 0.0
for c in range(10):
    mse_c = F.mse_loss(mask_gen[:, c:c+1], mask_gt[:, c:c+1], reduction="mean")
    loss_total = loss_total + mse_c
```

Semantics:

- each `mse_c` averages over batch and spatial dimensions for channel `c`
- the final loss is the sum of the 10 channel-wise MSE terms

#### Vectorized equivalent

This is mathematically equivalent to the loop above:

```python
per_channel = ((mask_gen - mask_gt) ** 2).mean(dim=(0, 2, 3))  # (10,)
loss_total = per_channel.sum()
```

Recommendation:

- Use the explicit loop in the first clean implementation.
- It is easier to audit and makes the per-channel design intent obvious.

### 4.7 Suggested training-loss signature

Proposed signature:

```python
training_losses_with_anatomy_mask(
    model,
    x1,
    model_kwargs,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy=0.1,
    anatomy_subbatch_size=4,
)
```

Notably absent:

- `output_images_pixel`
- `feature_layer_idx`
- `use_gen_mask`
- `t_threshold`

That keeps the new file aligned with Plan 1 and avoids carrying forward feature-MSE-specific complexity.

### 4.8 Recommended decode behavior for the new mask loss

There are two possible choices:

1. Follow old v1/v2 behavior and clamp decoded image to `[-1, 1]`.
2. Follow v3 behavior and avoid clamp to preserve stronger gradients.

Source-based recommendation:

- Since Plan 1 is reverting the loss definition, not necessarily reverting every gradient-suppression choice, the new file can still keep the cleaner v3 decode behavior if desired.
- This choice is orthogonal to the new mask-MSE objective.

If the goal is a minimal, easiest-to-reason-about replacement, start with:

- no timestep gating
- no area normalization
- no GT feature branch
- no predicted-feature branch
- no detach
- no `torch.no_grad()` on generated decode path

Then decide clamp behavior explicitly as a separate experiment variable.

---

## 5. Recommended Minimal New Pipeline

### 5.1 Clean conceptual pipeline

`GT image -> VAE encode -> x1`

`x1 + sampled noise/timestep -> OmniGen -> model_output`

`model_output -> x1_hat -> VAE decode -> gen image`

`gen image -> frozen ResUNet34 seg model -> logits_gen -> sigmoid -> mask_gen`

`mask_gen vs GT mask -> per-channel MSE sum -> anatomy_loss`

`loss_total = loss_diffusion + lambda_anatomy * anatomy_loss`

### 5.2 Why this is a better fit than extending v3

- v3 contains logic that Plan 1 explicitly wants to discard.
- Extending `train_anatomy.py` further will keep accumulating incompatible flags and branches.
- A dedicated `train_anatomy_mask.py` will make experiments cleaner and reduce accidental carry-over of v2/v3 behavior.

---

## 6. Key Findings Summary

1. The current generated-image branch does preserve autograd correctly. There is no detach before the segmentation model.
2. The GT mask reaches the loss code as a batch tensor of shape `(B, 10, 256, 256)`, and sub-batch slices become `(n, 10, 256, 256)`.
3. The anatomy-loss weight is controlled by `--lambda_anatomy`, and total loss is always `loss_diffusion + lambda_anatomy * loss_anatomy`.
4. The new Plan 1 mask-MSE loss should be implemented as a fresh loss module and fresh train entrypoint, not as another branch inside v2/v3 code.
5. Because the segmentation model uses `activation=None`, the new "Gen Mask" should most likely be `torch.sigmoid(seg_model(gen_decoded))` before MSE against GT mask.
6. The requested per-channel loss is straightforward in PyTorch and should be implemented as 10 independent channel MSE terms summed together.

---

## 7. One Non-Blocking Observation

`DatasetFromJson.__getitem__()` currently returns immediately at `data.py:75-76`, so the retry and `max_input_length_limit` logic below it is dead code.

This does not block the anatomy-loss refactor, but it is worth knowing during future cleanup.
