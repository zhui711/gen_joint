# Anatomy-Aware Loss v2 Diagnostic Report

## Scope

This report analyzes the regression observed after switching from the 30k-step baseline OmniGen LoRA checkpoint to the anatomy-aware training run using `training_losses_with_anatomy_v2`.

Key files:

- [`loss_anatomy_v2.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py)
- [`train_anatomy.py`](/home/wenting/zr/gen_code/train_anatomy.py)
- [`utils.py`](/home/wenting/zr/gen_code/OmniGen/utils.py)
- [`pipeline.py`](/home/wenting/zr/gen_code/OmniGen/pipeline.py)

## Metric Regression Summary

Baseline metrics from [`metrics_report.json`](/home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30000/metrics_report.json):

- SSIM: `0.6754`
- PSNR: `20.0464`
- LPIPS: `0.1844`
- FID: `21.9875`

Anatomy-loss run metrics from [`metrics_report.json`](/home/wenting/zr/gen_code/outputs/cxr_finetune_lora_30ksteps_feaLayer2_lamda0.1_subbatch16_10500/metrics_report.json):

- SSIM: `0.6667`
- PSNR: `19.8336`
- LPIPS: `0.1936`
- FID: `26.9913`

Delta:

- SSIM: `-0.0087`
- PSNR: `-0.2129`
- LPIPS: `+0.0092`
- FID: `+5.0037`

This is a real quality regression, not noise.

## Executive Diagnosis

The main issue is not that the anatomy idea is wrong. The main issue is that the auxiliary loss is being applied at the wrong points of the rectified-flow trajectory and with an unstable gradient path.

The highest-probability failure chain is:

1. `x1_hat` is decoded for all sampled timesteps, including very noisy ones.
2. For low `t`, reconstruction error is amplified by `(1 - t)`, so the decoded image can be badly off-manifold.
3. The segmentation network then sees these off-manifold images and produces unstable features and, when `use_gen_mask=True`, unstable masks.
4. The anatomy loss backpropagates through both the segmentation encoder and the predicted mask branch into the generator.
5. That auxiliary gradient competes with the rectified-flow vector field objective and drags the model away from the pretrained generative prior.

In short: the auxiliary loss is mathematically most aggressive exactly where the RF state is least image-like.

## First-Principles Analysis

### 1. Rectified-Flow Reconstruction Error Is Worst At Low `t`

In your code, the RF interpolation is:

- `x_t = t * x1 + (1 - t) * x0` at [`loss_anatomy_v2.py:268`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L268)
- `u_hat = model(x_t, t)` at [`loss_anatomy_v2.py:274`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L274)
- `x1_hat = x_t + (1 - t) * u_hat` at [`loss_anatomy_v2.py:340`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L340)

Let the model error be `e = u_hat - (x1 - x0)`.

Then:

- `x1_hat = x1 + (1 - t) * e`

This is the key physics.

Even if the velocity error `e` is moderate, the decoded clean estimate error is scaled by `(1 - t)`.

- If `t -> 1`, then `x1_hat ~= x1`, good.
- If `t -> 0`, then `x1_hat ~= x1 + e`, worst case.

So the auxiliary loss is strongest on the least trustworthy decoded samples unless you gate it.

### 2. The Sampled `t` Distribution Makes This Worse

`sample_timestep()` uses a logistic-normal transform at [`loss_anatomy_v2.py:79`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L79). That distribution is centered near `0.5`, but it still has plenty of mass below `0.3` and above `0.7`.

That means a non-trivial fraction of training sees decoded samples where:

- the latent is still highly mixed with Gaussian noise,
- the VAE decoder is asked to decode latents that are not on the data manifold,
- the segmentation network is then used as if those outputs were anatomical images.

This is a textbook setup for destructive auxiliary supervision.

## A. Hypothesis: Time-Step Distribution Trap

### Verdict

Strongly correct. This is likely the single biggest mathematical flaw.

### Why it hurts

The anatomy loss assumes that the decoded image is meaningful enough for structure-sensitive supervision. That assumption only holds when `t` is sufficiently large.

At low `t`:

- `x_t` is still noise-dominated.
- `x1_hat` depends heavily on model extrapolation.
- the VAE decoder receives off-manifold latents.
- feature matching then punishes the model for not producing anatomy in a state where anatomy is not yet well-formed.

That distorts the RF velocity field. Instead of learning a smooth denoising flow, the model is forced to make low-`t` states look prematurely anatomical.

This can absolutely worsen FID and LPIPS while only modestly changing diffusion MSE.

### Recommendation

Add timestep gating. Do not compute anatomy loss for all samples.

Recommended first safe version:

- compute anatomy loss only when `t >= 0.6`
- return zero anatomy loss if no sample passes the gate

Better than a hard binary gate:

- use a continuous weight such as `w_t = clamp((t - 0.5) / 0.5, 0, 1)`
- multiply each sample's anatomy loss by `w_t`

Best practical starting point:

- hard gate first, because it is easier to verify
- once stable, switch to soft weighting

## B. Hypothesis: `use_gen_mask=True` Instability

### Verdict

Also correct, and severe.

### Why it hurts

When `use_gen_mask=True`, the loss path is:

- `gen_images -> seg encoder -> F_gen`
- `F_gen -> seg decoder/head -> mask_gen`
- final loss uses `mask_gen * F_gen`

See [`loss_anatomy_v2.py:174`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L174) through [`loss_anatomy_v2.py:205`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L205).

So the generator receives gradients through two coupled branches:

1. direct feature branch via `F_gen`
2. self-gating mask branch via `mask_gen(F_gen)`

This is unstable because the same poor generated image both creates the feature error and defines where that error should be applied.

If the decoded image is blurry or anatomically wrong, `mask_gen` can be wrong. Then the loss says:

- emphasize the wrong region,
- suppress the right region,
- and backpropagate that spatial error into the generator.

That is a noisy bootstrap loop.

### Recommendation

Set `use_gen_mask=False` by default for training recovery.

Use GT masks for both generated and GT features during the stabilization phase.

This changes the objective from:

- "match features inside regions predicted from a bad generated image"

to:

- "match generated features to GT features inside the known anatomical support"

That is much more stable.

If you later want to reintroduce generated masks, do it only after:

- timestep gating is in place,
- the model is already stable,
- and the mask branch is detached, or mixed conservatively with GT masks.

For example:

- `mask_weight = mask_gt_down`
- or `mask_weight = 0.8 * mask_gt_down + 0.2 * mask_gen_down.detach()`

Do not backpropagate through `mask_gen` initially.

## C. Hypothesis: VAE Scaling and Normalization Check

### Verdict

The inverse VAE scaling itself looks correct. This is not the main bug.

### Evidence

Encoding uses:

- `(latent - shift_factor) * scaling_factor` in [`utils.py:96`](/home/wenting/zr/gen_code/OmniGen/utils.py#L96)

Inference decoding uses:

- `latent / scaling_factor + shift_factor` in [`pipeline.py:296`](/home/wenting/zr/gen_code/OmniGen/pipeline.py#L296)

Your anatomy loss uses the same inverse mapping:

- [`loss_anatomy_v2.py:345`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L345)

So the scaling math is aligned with OmniGen's own pipeline.

### What is still risky

Two subtler issues remain:

1. `vae.decode(...).sample.clamp(-1.0, 1.0)` at [`loss_anatomy_v2.py:346`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L346)

This creates zero gradients outside `[-1, 1]`. On bad low-`t` decodes, that can produce dead zones exactly where the image is already unstable.

2. The segmentation model is being fed generated images that can be off-manifold even if scaling is correct.

So the problem is not "wrong scale factor"; the problem is "correct decoder applied to bad latent states".

### Recommendation

Keep the inverse scaling code.

Treat `clamp(-1, 1)` as secondary. It is not the root cause, but you can make it less harsh after the major fixes:

- either keep it for now and fix `t` gating first,
- or replace it with a softer bound later if needed

I would not start by changing the VAE scaling logic.

## D. Hypothesis: Feature Magnitude and MSE Scaling

### Verdict

Correct in part. This is likely a secondary but real issue.

### Why it hurts

`F_gen` and `F_gt` are raw ResNet34 intermediate activations from the segmentation encoder at [`loss_anatomy_v2.py:168`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L168) and [`loss_anatomy_v2.py:165`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L165).

These activations are not normalized. Therefore:

- high-activation channels dominate MSE,
- large organs dominate because masked tensors are compared with `reduction='mean'` over all spatial positions,
- the final loss sums all 10 organs, further amplifying scale drift

See [`loss_anatomy_v2.py:193`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L193) through [`loss_anatomy_v2.py:206`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_v2.py#L206).

So the current loss does not purely say "match structure". It also says:

- match absolute activation magnitudes,
- with strong bias toward channels and organs that happen to have larger energy.

That is not ideal for a perceptual anatomical constraint.

### Recommendation

Normalize features before comparison.

Safest option:

- L2-normalize along the channel dimension before masking or before MSE

Example:

```python
F_gt = F.normalize(F_gt, dim=1, eps=1e-6)
F_gen = F.normalize(F_gen, dim=1, eps=1e-6)
```

Then compute masked loss.

Even better:

- normalize by mask area per organ, so tiny organs are not diluted by `mean` over the full feature map

Example concept:

```python
diff2 = (OrganFeature_gen_c - OrganFeature_gt_c).pow(2)
denom = M_gt_c.sum().clamp_min(1.0) * F_gt.shape[1]
mse_c = diff2.sum() / denom
```

Cosine similarity is also reasonable, but L2-normalized MSE is easier to integrate and debug.

## Additional Non-Hypothesis Confounders

These are not the primary math bugs inside `loss_anatomy_v2.py`, but they matter.

### 1. Training Setup Changed In More Than One Way

Baseline `train_args.json`:

- learning rate `3e-4`
- batch size per device `128`
- grad accumulation `1`

Anatomy run `train_args.json`:

- learning rate `1e-4`
- batch size per device `16`
- grad accumulation `4`
- `use_gen_mask=true`
- new dataset path

Also, the anatomy run loads LoRA weights only and creates a fresh optimizer at [`train_anatomy.py:164`](/home/wenting/zr/gen_code/train_anatomy.py#L164) through [`train_anatomy.py:197`](/home/wenting/zr/gen_code/train_anatomy.py#L197).

So this was not a clean ablation of one variable.

### 2. Anatomy Loss Magnitude Is Not Tiny

Training logs show:

- `loss_diffusion` around `0.44~0.46`
- `loss_anatomy` around `0.7~1.0`

With `lambda_anatomy=0.1`, the auxiliary branch contributes roughly `0.07~0.10` to total loss, about `15%~22%` of the diffusion term.

That is large enough to steer the model meaningfully, especially when applied on unstable low-`t` states.

## Recommended Fix Order

### Tier 1: Immediate Stabilization

1. Set `use_gen_mask=False`.
2. Apply anatomy loss only for `t >= 0.6`.
3. Reduce `lambda_anatomy` from `0.1` to `0.02` or `0.01`.
4. Normalize segmentation features with `F.normalize(..., dim=1)`.
5. Normalize each organ loss by mask area instead of plain global mean.

This is the highest-probability recovery path.

### Tier 2: Safer Objective Design

1. Use GT mask only for weighting.
2. Compute anatomy loss per sample, then multiply by timestep weight `w_t`.
3. Keep low-resolution `feature_layer_idx=2` for now.
4. Only after stability returns, test adding deeper layers.

### Tier 3: Experimental Add-ons

Only after Tier 1 works:

1. Mix in `mask_gen.detach()` with a small weight.
2. Replace hard timestep gate with a smooth ramp.
3. Consider Huber loss instead of MSE for feature matching.

## Concrete Code-Level Recommendations For `loss_anatomy_v2.py`

### 1. Add Timestep Filtering

Inside `training_losses_with_anatomy_v2`, after sampling `t`, define a gate:

```python
t_min_anatomy = 0.6
anat_keep = t >= t_min_anatomy
```

Then, in the fixed-resolution branch:

```python
idx_all = torch.randperm(B, device=x1_hat.device)
idx_keep = idx_all[anat_keep[idx_all]]
idx = idx_keep[:n]

if idx.numel() == 0:
    loss_anatomy = loss_diffusion.new_zeros(())
else:
    ...
```

Do the analogous thing in the variable-resolution branch.

### 2. Force GT Mask Weighting

Change the training default in [`train_anatomy.py:362`](/home/wenting/zr/gen_code/train_anatomy.py#L362) to `False`, and preferably change the parser default too.

Inside `compute_mask_weighted_feature_loss`, use:

```python
mask_gen = mask_gt
```

or remove the generated-mask branch entirely for the stabilization phase.

### 3. Normalize Features

Right after feature extraction:

```python
with torch.no_grad():
    gt_features_all = seg_model.encoder(gt_images)
    F_gt = gt_features_all[feature_layer_idx]
    F_gt = F.normalize(F_gt, dim=1, eps=1e-6)

gen_features_all = seg_model.encoder(gen_images)
F_gen = gen_features_all[feature_layer_idx]
F_gen = F.normalize(F_gen, dim=1, eps=1e-6)
```

### 4. Normalize By Mask Area

Replace:

```python
mse_c = F.mse_loss(OrganFeature_gen_c, OrganFeature_gt_c, reduction='mean')
```

with:

```python
diff2_c = (OrganFeature_gen_c - OrganFeature_gt_c).pow(2)
denom_c = (M_gt_c.sum() * F_gt.shape[1]).clamp_min(1.0)
mse_c = diff2_c.sum() / denom_c
```

If you want symmetric weighting while still using GT support only:

```python
support_c = M_gt_c
diff2_c = ((F_gen - F_gt).pow(2) * support_c).sum()
denom_c = (support_c.sum() * F_gt.shape[1]).clamp_min(1.0)
mse_c = diff2_c / denom_c
```

This version is cleaner than multiplying both features separately.

### 5. Lower Auxiliary Weight

Start with:

```python
lambda_anatomy = 0.01
```

Then re-evaluate metrics before trying `0.02` or `0.05`.

## Proposed Revised Loss Shape

The safer objective is:

```text
Loss_total = Loss_diffusion + lambda_anatomy * w(t) * Loss_anatomy_masked_feature
```

where:

- `w(t) = 0` for `t < 0.6`
- `w(t) = 1` for `t >= 0.6`
- mask support comes from `Mask_gt`
- feature tensors are channel-normalized
- organ losses are normalized by mask area

This respects the RF geometry much better.

## Bottom Line

### What is most likely wrong

The anatomy loss is being asked to supervise decoded samples that are not yet valid images, and `use_gen_mask=True` turns that into a self-referential unstable gradient.

### What is probably not wrong

The VAE inverse scaling formula appears consistent with OmniGen's own encode/decode path.

### What to change first

1. gate anatomy loss to high `t`
2. disable generated masks
3. normalize features
4. normalize by mask area
5. lower `lambda_anatomy`

If you only make one change, make it timestep gating. If you make two, combine timestep gating with `use_gen_mask=False`.
