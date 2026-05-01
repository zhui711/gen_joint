# Latent Shift Evidence Report

## Execution

Diagnostic script:

```bash
python diagnose_latent_shift.py \
  --output_dir latent_shift_diagnostic \
  --device cuda:0
```

Control run matching the standard KV-cache inference path:

```bash
python diagnose_latent_shift.py \
  --output_dir latent_shift_diagnostic_kvcache \
  --device cuda:0 \
  --use_kv_cache
```

Both runs produced identical latent statistics for the diagnosed sample, so the result is not an artifact of disabling KV cache.

## Sample

- `sample_id`: `LIDC-IDRI-0657_0001`
- `input_image`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0000.png`
- `gt_image`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0001.png`
- `gt_mask`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz`
- checkpoint: `results/30000`
- mask modules: `results/30000/mask_modules.bin`

## Latent Statistics

| Latent | min | max | mean | std |
|---|---:|---:|---:|---:|
| `Z_m0_true = MaskEncoder(GT_mask)` | -0.039474 | 0.022287 | -0.002503 | 0.006251 |
| `Z_m0_cfg`, guidance `2.5`, image guidance `2.0` | -3.671875 | 3.437500 | 0.004616 | 1.006210 |
| `Z_m0_nocfg`, guidance `1.0`, image guidance `1.0` | -3.671875 | 3.437500 | 0.004616 | 1.006210 |

Distances:

- `||Z_m0_cfg - Z_m0_true||_2 = 64.402664`
- `||Z_m0_nocfg - Z_m0_true||_2 = 64.402664`

## Interpretation

The OOD hypothesis is strongly supported, but the failure is not CFG-specific for this sample.

The clean encoder latent lives in a tiny numerical support:

```text
max_abs(Z_m0_true) ~= 0.0395
std(Z_m0_true)     ~= 0.00625
```

The inferred ODE mask latent remains at approximately unit Gaussian scale:

```text
max_abs(Z_hat_m0) ~= 3.67
std(Z_hat_m0)     ~= 1.006
```

That is roughly a `93x` larger max-absolute range and a `161x` larger standard deviation than the clean latent distribution the decoder was trained on. This explains why the decoder produces mosaic noise from inferred latents even though it reconstructs anatomical masks from `MaskEncoder(GT_mask)` latents.

## Visual Evidence

Saved files:

- `latent_shift_diagnostic/mask_true_recon_grid.png`
- `latent_shift_diagnostic/mask_true_recon_union.png`
- `latent_shift_diagnostic/mask_cfg_grid.png`
- `latent_shift_diagnostic/mask_cfg_union.png`
- `latent_shift_diagnostic/mask_nocfg_grid.png`
- `latent_shift_diagnostic/mask_nocfg_union.png`
- `latent_shift_diagnostic/generated_cfg.png`
- `latent_shift_diagnostic/generated_nocfg.png`
- `latent_shift_diagnostic/latent_shift_metrics.json`

Observed visual result:

- `MaskDecoder(MaskEncoder(GT_mask))` is anatomical and preserves recognizable organ geometry.
- `MaskDecoder(Z_m0_cfg)` is a high-frequency mosaic.
- `MaskDecoder(Z_m0_nocfg)` is also a high-frequency mosaic and is visually identical to the CFG output for this sample.

## Answers

1. Did the latent explode under CFG?

   Yes in absolute terms: the CFG inference latent is far outside the clean decoder-training support. However, the no-CFG latent has the same statistics, so the data does not support a CFG-only explanation.

2. Does the decoded mask look fundamentally better without CFG?

   No. The no-CFG decoded mask remains mosaic-like and is numerically identical to the CFG result in this run.

3. Does the decoded mask look fundamentally better from the GT latent?

   Yes. The GT encoder latent decodes into anatomical organ-like structures, confirming that the decoder can work when fed in-distribution latents.

## Recommended Fixes

The most likely root cause is a scale/support mismatch between `MaskEncoder(GT_mask)` latents and the learned/inferred ODE mask-latent trajectory. The inferred `z_mask` is still at noise scale when decoded.

Test these fixes in order:

1. Clamp or normalize the final and per-step mask latent to the empirical clean-latent distribution.

   Compute per-channel statistics over training masks:

   ```python
   z = MaskEncoder(gt_mask)
   channel_mean, channel_std, channel_p001, channel_p999
   ```

   Then during inference:

   ```python
   z_mask = torch.clamp(z_mask, p001, p999)
   ```

   or:

   ```python
   z_mask = (z_mask - z_mask.mean(dim=(2, 3), keepdim=True)) / z_mask.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
   z_mask = z_mask * clean_std + clean_mean
   ```

2. Train or calibrate the mask denoising branch against the clean encoder-latent scale.

   The inferred `z_mask` std of `~1.0` versus clean std of `~0.006` suggests the mask branch may be learning the image-latent/noise scale rather than the mask-autoencoder latent scale.

3. Disable CFG specifically for the mask branch only after fixing scale.

   Since no-CFG was identical here, branch-specific CFG alone is unlikely to solve the mosaic issue. It is still mathematically cleaner to avoid extrapolating fragile mask latents once the scale problem is fixed.

4. Add an inference-time sanity assert before decoding:

   ```python
   if z_mask.std() > clean_std_global * threshold:
       warn_or_rescale(z_mask)
   ```

   This prevents the decoder from receiving latents far outside its training support.

