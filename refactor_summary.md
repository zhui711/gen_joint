# Polynomial Timestep Weighting Refactor Summary

## Why replace hard thresholding with continuous `t^alpha` weighting?

The visual debugging confirmed a first-principles failure mode: when the sampled diffusion timestep is very low, the predicted `x1_hat` decodes to images that are visually OOD for the frozen ResUNet34 anatomy segmenter. In that regime, the segmentation masks are not semantically trustworthy, so a plain anatomy MSE term injects noisy supervision into the diffusion model.

Hard thresholding would remove that supervision abruptly, but it introduces a discontinuity in the objective: two nearly identical samples on opposite sides of a threshold receive very different anatomy gradients. A continuous polynomial weight,

`weight(t) = t^alpha`,

keeps the objective smooth while still encoding the desired prior:

- low timesteps receive strong suppression because `t^alpha -> 0` rapidly as `t -> 0`
- high timesteps retain meaningful structure-aware gradients because `t^alpha` remains non-zero and increases monotonically toward `1`
- the transition is differentiable and avoids hand-crafted phase boundaries

This gives us a softer curriculum: anatomy supervision is naturally weak when decoded predictions are noisy and naturally stronger when decoded predictions become structurally meaningful.

## Code changes made

### 1. [`OmniGen/train_helper/loss_anatomy_mask.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy_mask.py)

- Added `anatomy_alpha=4.0` and `debug_vis=False` to `training_losses_with_anatomy_mask(...)`.
- Applied per-sample weighting in both the variable-resolution and fixed-resolution anatomy sub-batch paths:
  - compute `loss_i = compute_mask_mse_loss(...)`
  - compute `weighted_loss_i = loss_i * (t_i ** anatomy_alpha)`
  - append `weighted_loss_i` to `anatomy_losses`
- Isolated the debug image dump behind `if debug_vis:` so production training does not save images or request mask tensors unnecessarily.
- Kept the debug save path detached from the training graph by only calling `.detach().cpu()` inside the visualization helper.

### 2. [`train_anatomy_mask.py`](/home/wenting/zr/gen_code/train_anatomy_mask.py)

- Added `--anatomy_alpha` with default `4.0`.
- Added `--debug_vis` as a boolean flag.
- Passed both values into `training_losses_with_anatomy_mask(...)`.

### 3. [`lanuch/train_anatomy_mask.sh`](/home/wenting/zr/gen_code/lanuch/train_anatomy_mask.sh)

- Added `--anatomy_alpha 4.0`.
- Left `--debug_vis` disabled by default for production runs.
- Changed `--gradient_accumulation_steps 4` to `--gradient_accumulation_steps 8` for the requested fair-baseline setting.

### 4. [`analyze_alpha_weights.py`](/home/wenting/zr/gen_code/analyze_alpha_weights.py)

- Added a standalone script that prints `t^alpha` weights for:
  - `t = [0.1, 0.2, ..., 1.0]`
  - `alpha = [1.0, 2.0, ..., 6.0]`
- Added a compact summary showing:
  - maximum low-t weight for `t < 0.3`
  - minimum and average high-t weight for `t > 0.7`
  - which alphas satisfy the low-t suppression criterion `< 0.05`

## Current training settings for fair comparison

The updated launch configuration now includes:

- `batch_size_per_device = 16`
- `gradient_accumulation_steps = 8`
- `anatomy_alpha = 4.0`
- `lambda_anatomy = 0.1`
- `anatomy_subbatch_size = 16`
- `num_processes = 4`
- `use_lora = true`

Requested baseline-comparison interpretation:

- per-process effective batch size = `16 * 8 = 128`

Distributed-launch note:

- with `--num_processes=4`, the aggregate cross-process effective batch size is `16 * 8 * 4 = 512`

That distinction matters when comparing against prior 30k-step runs. If the baseline's reported batch size of `128` referred to the full distributed job rather than the per-process batch, then the launch configuration is still not globally matched and would need a further adjustment in either per-device batch size, accumulation, or number of processes.
