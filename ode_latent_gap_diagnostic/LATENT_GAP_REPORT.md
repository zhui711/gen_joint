# Latent Gap Report

## Setup

- sample_id: `LIDC-IDRI-0657_0001`
- training jsonl: `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`
- base model: `Shitao/OmniGen-v1`
- LoRA checkpoint: `results/10000`
- mask modules: `results/10000/mask_modules.bin`
- inference steps: `50`
- guidance_scale: `2.5`
- img_guidance_scale: `2.0`
- seed: `42`

## 1. Scale Factor Code Check

**No hardcoded `MASK_SCALE_FACTOR`, `mask_scale`, or equivalent static mask-latent scaling assignment was found in the inspected active code paths.**

The current loss path uses raw `MaskEncoder(GT_mask)` latents, and the current inference path initializes and updates `mask_latents` directly with no unscale step.

- clean latent std for this sample: `0.092997`
- implied scaled target std from current code: not applicable

Inspected lines matching scale-related terms:

- No scale-related lines matched in the inspected files.

## 2. Face-to-Face Latent Probe

| latent | min | max | mean | std | p01 | p50 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| z_clean = MaskEncoder(GT mask) | -0.151406 | 0.160774 | 0.024593 | 0.092997 | -0.135707 | 0.041184 | 0.145115 |
| z_ode_unscaled = current 50-step ODE output | -0.176758 | 0.199219 | 0.021111 | 0.100037 | -0.160156 | 0.039185 | 0.166016 |

- `||z_clean - z_ode_unscaled||_2`: `1.479087`
- relative L2 vs `||z_clean||_2`: `0.240252`
- cosine similarity: `0.974706`
- ODE/clean std ratio: `1.075698`

## 3. Saved Evidence

- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_gt_mask_grid.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_clean_recon_grid.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_ode_recon_grid.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_clean_recon_union.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_ode_recon_union.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_generated_image.png`
- `ode_latent_gap_diagnostic/LIDC-IDRI-0657_0001_latent_histogram.png`

## 4. Strategic Conclusion

**The ODE latent is numerically close to the clean latent under these probes.** If decoded masks remain patchy, the likely issue is not global scale but local latent topology or decoder sensitivity to small off-manifold perturbations.

Recommended next fix: add a lightweight KL or latent smoothness regularizer to make the encoder latent manifold more connected, while preserving the already validated autoencoder reconstruction capacity.
