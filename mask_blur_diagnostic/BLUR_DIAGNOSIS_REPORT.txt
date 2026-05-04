# Mask Blur Diagnosis Report

## Setup

- checkpoint: `/home/wenting/zr/gen_code_plan2_1/results/10000/mask_modules.bin`
- jsonl: `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`
- samples: `16`
- test: `GT mask -> MaskEncoder -> MaskDecoder -> threshold > 0`
- no ODE / no diffusion / no image model used

## Visual Verdict

**Autoencoder upper-bound masks are structurally sharp after thresholding.**

Saved comparison grids:

- `mask_blur_diagnostic/LIDC-IDRI-0657_0001_ae_roundtrip_grid.png`
- `mask_blur_diagnostic/LIDC-IDRI-0657_0002_ae_roundtrip_grid.png`
- `mask_blur_diagnostic/LIDC-IDRI-0657_0003_ae_roundtrip_grid.png`
- `mask_blur_diagnostic/LIDC-IDRI-0657_0004_ae_roundtrip_grid.png`

## Mathematical Verdict

**The dominant blur/patchiness is not an inherent MaskEncoder/MaskDecoder bottleneck. It comes from the flow-matching trajectory not landing exactly on the clean mask-latent manifold.**

Round-trip binary metrics are used only as an autoencoder sanity check, not as test-set segmentation evaluation.

- mean autoencoder Dice: `0.953158`
- mean autoencoder IoU: `0.910511`
- mean GT edge density: `0.010767`
- mean recon edge density: `0.010179`

## Latent Distribution

| statistic | value |
|---|---:|
| `min` | -0.153792 |
| `max` | 0.161384 |
| `mean` | 0.024595 |
| `std` | 0.092974 |
| `skewness` | -0.399802 |
| `excess_kurtosis` | -1.240908 |
| `p01` | -0.134587 |
| `p50` | 0.040036 |
| `p99` | 0.145128 |

KL-divergence hypothesis note: these clean encoder latents are not constrained to a unit Gaussian. A unit-Gaussian prior would have mean near 0, std near 1, skew near 0, and excess kurtosis near 0. The reported statistics show the actual learned latent support that the flow model must hit.

## Per-Sample CSV

- `mask_blur_diagnostic/sample_metrics.csv`
