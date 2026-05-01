# Latent Shift Evidence Report

## Sample

- `sample_id`: `LIDC-IDRI-0657_0001`
- `input_image`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0000.png`
- `gt_image`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0001.png`
- `gt_mask`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz`

## Latent Statistics

| Latent | min | max | mean | std |
|---|---:|---:|---:|---:|
| Z_m0_true = MaskEncoder(GT mask) | -0.039474 | 0.022287 | -0.002503 | 0.006251 |
| Z_m0_cfg, guidance=CFG | -3.671875 | 3.437500 | 0.004616 | 1.006210 |
| Z_m0_nocfg, guidance=1.0 | -3.671875 | 3.437500 | 0.004616 | 1.006210 |

## Distances to Clean GT Latent

- `||Z_m0_cfg - Z_m0_true||_2`: `64.402664`
- `||Z_m0_nocfg - Z_m0_true||_2`: `64.402664`

## Saved Visual Evidence

- `latent_shift_diagnostic/mask_true_recon_grid.png`
- `latent_shift_diagnostic/mask_true_recon_union.png`
- `latent_shift_diagnostic/mask_cfg_grid.png`
- `latent_shift_diagnostic/mask_cfg_union.png`
- `latent_shift_diagnostic/mask_nocfg_grid.png`
- `latent_shift_diagnostic/mask_nocfg_union.png`
- `latent_shift_diagnostic/generated_cfg.png`
- `latent_shift_diagnostic/generated_nocfg.png`

## Interpretation

1. Did the latent explode under CFG? **Yes**.
   The clean latent max-absolute range is `0.039474`; CFG range is `3.671875`; no-CFG range is `3.671875`.
2. Is no-CFG numerically closer to the clean latent? **No**.
3. Visual assessment should compare `mask_true_recon_grid.png`, `mask_cfg_grid.png`, and `mask_nocfg_grid.png`. The GT-latent reconstruction is the decoder sanity check; if CFG/no-CFG outputs are mosaic-like while GT reconstruction is anatomical, the failure is upstream of the decoder.

## Mathematically Sound Fixes to Test

1. Disable CFG for the mask branch while keeping image CFG. In `forward_with_cfg`, compute the guided image prediction normally, but use the conditional mask prediction directly for `mask_out` instead of applying the CFG extrapolation formula to mask tokens.
2. Clamp or normalize `z_mask` during each ODE step to the empirical clean-latent support measured from training masks, e.g. per-channel clamp to training percentiles or global clamp to `[p0.1, p99.9]` of `MaskEncoder(GT_mask)`.
3. Add a mask-latent regularizer during training or inference calibration: penalize drift in mean/std relative to the clean encoder-latent distribution, or rescale inferred `z_mask` to match clean latent per-channel mean/std before decoding.
4. If no-CFG is much closer than CFG, prefer branch-specific guidance: image branch uses CFG, mask branch uses conditional prediction or a much smaller guidance scale.
