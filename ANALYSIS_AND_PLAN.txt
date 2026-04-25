# Analysis And Plan

## Scope

This report is based on two executed probes:

- [probe_option_a.py](/home/wenting/zr/gen_code_plan2_1/probe_option_a.py:1)
- [probe_option_c.py](/home/wenting/zr/gen_code_plan2_1/probe_option_c.py:1)

Executed artifacts:

- `mask_modules.bin`: `/home/wenting/zr/gen_code_plan2_1/results/mask_modules.bin`
- Real training JSONL: `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`

Important checkpoint caveat:

- I found `results/mask_modules.bin`.
- I did **not** find a companion LoRA adapter under `/home/wenting/zr/gen_code_plan2_1/results/`.
- Therefore the executed Option C probe used:
  - base `Shitao/OmniGen-v1`
  - loaded `mask_modules.bin`
  - no LoRA adapter

That means the exact absolute `L_img` value may differ from the fully trained production run, but the relative loss-scale calculation is still useful for choosing a conservative `lambda_recon`.

## Probe C Results

Configuration:

- 1 real batch
- `keep_raw_resolution=True`
- `max_image_size=1024`
- batch size `1`
- image size `256 x 256`
- raw losses computed in the actual joint training path
- reconstruction MSE computed in the decoder's native `[-1, 1]` space against GT masks mapped from `{0,1}` to `[-1,1]`

Measured raw losses:

- `L_img = 0.302734375`
- `L_flow_mask = 1.046875`
- `L_recon = 0.97265625`

Derived recommendation:

- Target rule: weighted reconstruction term should equal exactly `0.1 * L_img`
- Formula: `lambda_recon = 0.1 * L_img / L_recon`
- Computation: `0.1 * 0.302734375 / 0.97265625 = 0.03112449799196787`
- Recommended exact value: `lambda_recon = 0.0311245`

Interpretation:

- `L_recon` is about `3.21x` larger than `L_img`
- `L_flow_mask` is about `3.46x` larger than `L_img`
- So even a fairly small reconstruction weight already has a visible effect
- `lambda_recon ~= 0.03` is already a **soft** regularizer by design

SOTA-first operational note:

- If the rule is strictly "image metrics first", then `0.0311` should be treated as the mathematically matched upper bound for the **first** ablation, not as a license to increase further immediately.
- A safer first sweep would be:
  - `0.01`
  - `0.02`
  - `0.0311`

## Probe A Results

Configuration:

- Frozen `MaskEncoder` loaded from `results/mask_modules.bin`
- Fresh randomly initialized `MaskDecoder`
- `64` real GT masks
- batch size `16`
- `300` optimization steps
- viability rule: final full-subset MSE must drop below `0.05`

Measured results:

- Initial full-subset MSE: `1.2807408273220062`
- Final full-subset MSE after 300 steps: `0.25717997550964355`
- Best batch MSE observed: `0.20189562439918518`
- Decision: `option_a_viable = false`

Trajectory:

- step `0`: `1.2807`
- step `1`: `0.8049`
- step `50`: `0.4013`
- step `100`: `0.3273`
- step `150`: `0.2807`
- step `200`: `0.2471`
- step `250`: `0.2234`
- step `300`: `0.2572`

Interpretation:

- The decoder does learn **something** from the frozen latent.
- So this is **not** total collapse to pure noise.
- But it clearly does **not** retain enough information to let a fresh decoder overfit cleanly.
- On a tiny training subset, a viable latent should have allowed the fresh decoder to drive MSE much lower than `0.257`.
- The late-stage flattening and rebound also suggest the latent is missing fine anatomical detail rather than merely needing more optimization.

Bottom line:

- Option A, as a pure post-hoc decoder training strategy, looks **weak**.
- The encoder latent appears to preserve coarse structure but not enough detail for reliable mask reconstruction.

## Strategic Analysis

### Option A: Post-hoc Decoder Training

Pros:

- Zero risk to the already improved image-generation training objective
- Simple to implement and cheap to test
- Useful as a diagnostic for whether the latent still carries anatomy

Cons:

- The executed probe failed the viability threshold by a wide margin
- Final full-subset MSE `0.257` is far from the target `< 0.05`
- This suggests the problem is not "decoder untrained only"; the latent itself is already too compressed or partially collapsed
- Even if trained longer, the likely ceiling is coarse masks, not high-fidelity anatomy

Conclusion:

- Option A is not the best primary path if the goal is meaningful predicted masks

### Option C: Auxiliary Reconstruction Loss

Pros:

- Attacks the failure mode at the correct point: it regularizes the encoder-decoder pair during joint training
- Can be made quantitatively very soft
- The measured scale says `lambda_recon ~= 0.03` already limits recon to only `10%` of image flow loss on the probe batch
- This is the best available path for improving masks **without** heavily disturbing image training

Cons:

- Any extra constraint can still hurt image metrics if weighted too aggressively
- The exact balance should be validated with short ablations, not assumed
- Because the probe used base OmniGen plus saved mask modules and no LoRA adapter, the absolute optimal value may shift slightly in the full trained setup

Conclusion:

- **Option C is the recommended next move**, but only as a deliberately weak regularizer

## Recommendation

The strict rule should remain:

- **Image Metrics SOTA is the #1 priority.**

Given that rule, the most defensible path is:

1. Do **not** prioritize Option A as the main recovery strategy.
2. Implement Option C as an opt-in auxiliary term with a very small coefficient.
3. Start from `lambda_recon = 0.0311` as the mathematically matched reference point.
4. For safer first ablations, sweep `0.01`, `0.02`, and `0.0311`.
5. Stop immediately if FID or LPIPS regress meaningfully, even if masks improve.

## Proposed Next Implementation Plan

When you are ready to touch the main training code, the implementation plan should be:

1. Add `L_recon = MSE(mask_decoder(mask_encoder(gt_mask)), gt_mask_cont)` to the joint training loop.
2. Keep it behind a new flag, for example `--lambda_recon`, default `0.0`.
3. Preserve the existing image and mask flow losses exactly.
4. Start with `lambda_recon=0.01` for the first safety run.
5. Compare against the current best run on:
   - FID
   - LPIPS
   - PSNR
   - SSIM
6. Only increase toward `0.02` or `0.0311` if image metrics remain effectively unchanged.

## Final Decision

Based on executed evidence:

- Option A: **not sufficiently viable**
- Option C: **recommended**, but only with a **very soft** coefficient

Best next step:

- implement Option C conservatively
- keep `lambda_recon` small
- treat any image-metric regression as a hard failure
