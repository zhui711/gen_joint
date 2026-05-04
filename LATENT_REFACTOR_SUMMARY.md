# Latent Refactor Summary

## Purpose

This refactor implements a pseudo-VAE stabilization path for joint mask co-generation without changing the `MaskEncoder` / `MaskDecoder` architecture or latent channel count.

The image-generation path remains the same. The change only makes the mask latent space easier for flow matching to learn and safer for `MaskDecoder` to decode at inference.

## Parameters

All new values are configured from the launch scripts and passed through argparse.

### `mask_scale_factor`

Training:

```python
z_m0_scaled = z_m0_raw * mask_scale_factor
```

Inference:

```python
pred_z_mask_unscaled = pred_z_mask / mask_scale_factor
```

Purpose: raw mask latents have a small empirical standard deviation, while the ODE starts from unit Gaussian noise. Scaling makes the mask flow target numerically closer to the ODE trajectory scale. The decoder still receives unscaled latents, preserving compatibility with existing `MaskDecoder` weights and dimensions.

Default launcher value:

```bash
MASK_SCALE_FACTOR=10.0
```

### `mask_smooth_std`

Training reconstruction path:

```python
z_m0_noisy = z_m0_scaled + torch.randn_like(z_m0_scaled) * mask_smooth_std
z_m0_unscaled = z_m0_noisy / mask_scale_factor
```

Purpose: injects controlled Gaussian perturbations before the reconstruction loss so `MaskDecoder` learns a continuous, robust neighborhood around clean mask latents instead of only memorizing sharp discrete points.

Default launcher value:

```bash
MASK_SMOOTH_STD=0.1
```

This only affects reconstruction when `lambda_recon > 0`.

### `lambda_latent_l2`

Training:

```python
L_latent_l2 = torch.mean(z_m0_raw ** 2)
loss += lambda_latent_l2 * L_latent_l2
```

Purpose: lightly pulls raw mask latents toward a compact zero-centered support. This is a pseudo-KL regularizer that does not require turning the autoencoder into a full VAE.

Default launcher value:

```bash
LAMBDA_LATENT_L2=0.01
```

## Tuning

Edit these variables in `launch/train_joint_mask.sh`:

```bash
MASK_SCALE_FACTOR=10.0
MASK_SMOOTH_STD=0.1
LAMBDA_LATENT_L2=0.01
```

Edit this variable in `launch/test_joint_mask.sh` and keep it matched to training:

```bash
MASK_SCALE_FACTOR=10.0
```

Tuning guidance:

- Increase `MASK_SCALE_FACTOR` if mask flow predictions remain too weak relative to unit-noise initialization.
- Decrease `MASK_SCALE_FACTOR` if mask latents become unstable or mask loss dominates.
- Increase `MASK_SMOOTH_STD` if decoded masks remain patchy from small latent perturbations.
- Decrease `MASK_SMOOTH_STD` if reconstructions become overly soft.
- Increase `LAMBDA_LATENT_L2` if latent magnitudes drift or become spiky.
- Decrease `LAMBDA_LATENT_L2` if anatomical detail is suppressed.

## Why This Preserves Image Metrics

The image branch, image VAE latents, text/image conditioning, CFG logic, and image metrics are untouched.

The refactor only changes the numerical representation used by the mask branch during flow matching and the mask decode path. The joint architecture can still use anatomical mask tokens as structural priors for image generation, while the mask branch gets a smoother and better-scaled target distribution for inference.
