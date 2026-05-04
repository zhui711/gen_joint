# Latent Scaling Refactor Summary

This refactor keeps only the Occam's Razor fix: `mask_scale_factor`.

Training scales the raw `MaskEncoder` latent before mask flow matching:

```python
z_m0 = z_m0_raw * mask_scale_factor
```

Reconstruction unscales before `MaskDecoder`:

```python
z_m0_unscaled = z_m0 / mask_scale_factor
recon_mask = mask_decoder(z_m0_unscaled)
```

Inference applies the same inverse scale before decoding the ODE output:

```python
pred_z_mask_unscaled = pred_z_mask / mask_scale_factor
pred_mask = mask_decoder(pred_z_mask_unscaled)
```

No noise injection and no latent L2 regularization are used. This preserves the previous `lambda_recon` magnitude because `z_m0 / mask_scale_factor` is mathematically identical to the raw encoder latent.
