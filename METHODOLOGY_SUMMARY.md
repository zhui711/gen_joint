# Methodology Summary

## 1. Why the Old System Was OOD-Broken

The old pipeline used a frozen RGB segmentation network to supervise anatomy on decoded intermediate images along the Rectified Flow denoising path.

That created a fundamental out-of-distribution problem:

- Rectified Flow trains the model to predict the velocity field between noise and data in latent space.
- Early and mid-trajectory latent states decode to noisy, unrealistic images.
- A frozen ResUnet was never trained on those noisy decoded intermediates.
- Its gradients therefore did not reflect true anatomy preservation on the data manifold.
- Instead, those gradients pushed the model to make off-manifold intermediate decodes artificially segmentable.

This is the core failure mode:

- the auxiliary anatomy loss no longer aligned with the true velocity target
- so it distorted the intended flow field
- which is exactly why the old setup could hurt generation quality

## 2. The Joint Token Flow Architecture

The new system makes anatomy a native generative variable instead of a post-hoc constraint.

### 2.1 Mask representation

Ground-truth anatomy masks start as:

- shape: `(10, 256, 256)`
- value domain: `{0, 1}`

They are first mapped to continuous space:

- `m_cont = 2 * m_bin - 1`
- new value domain: `[-1, 1]`

This matches the `tanh` output range of the `MaskDecoder`.

### 2.2 Mask latent space

The continuous 10-channel mask is encoded by a lightweight convolutional `MaskEncoder`:

- input: `(10, 256, 256)`
- output: `(4, 32, 32)`

This mirrors the image-VAE latent geometry:

- image latent: `(4, 32, 32)`
- mask latent: `(4, 32, 32)`

That symmetry is the key engineering choice that makes joint co-generation clean.

### 2.3 Joint Transformer sequence

During training:

1. The GT image is encoded into an image latent.
2. The GT mask is encoded into a mask latent.
3. Shared noise and shared timestep `t` are sampled.
4. Noisy image and noisy mask states are formed.
5. Both are patchified into token sequences.
6. The transformer input sequence is assembled as:

`[condition tokens, time token, image tokens, mask tokens]`

The image and mask token blocks are distinguished by separate modality embeddings.

The shared transformer then predicts:

- image velocity
- mask velocity

using:

- one shared trunk
- one image output head
- one mask output head

This is a true joint-distribution model, not an auxiliary classifier bolted onto noisy decodes.

## 3. Option A: Decoupled Lossless Path

Option A is now the default production path because the top priority is preserving the best observed image metrics.

### 3.1 Training-time rule

Set:

- `lambda_recon = 0.0`

Then the training loss is exactly:

`L_total = L_img + lambda_mask * L_flow_mask`

with no reconstruction term.

### 3.2 Why this is zero-risk for image metrics

In the current implementation, when `lambda_recon == 0.0`:

- the `MaskDecoder` forward pass is completely bypassed
- `L_recon` is not added to the total loss
- no reconstruction gradients reach the `MaskEncoder`
- the main joint-training objective is mathematically identical to the successful SOTA run

This is the critical guarantee.

The image branch sees exactly the same optimization problem as before.

### 3.3 Post-hoc decoder training

After joint training finishes, `train_mask_decoder_only.py` can be used to train only the `MaskDecoder` on top of the frozen learned `MaskEncoder`.

That gives the project a safe two-phase workflow:

1. train the joint image-mask model with `lambda_recon = 0.0`
2. extract whatever anatomy information remains in the learned mask latent by post-hoc decoder fitting

This is why Option A is the **decoupled lossless path**:

- image metrics are fully protected during the main run
- decoder recovery happens afterward
- there is no backward interference from reconstruction into the image-training objective

## 4. Option C: Auxiliary Reconstruction Path

Option C adds:

`L_recon = MSE(MaskDecoder(MaskEncoder(m_cont)), m_cont)`

and optimizes:

`L_total = L_img + lambda_mask * L_flow_mask + lambda_recon * L_recon`

### 4.1 What it helps

This encourages the mask latent to remain more explicitly invertible.

In principle, that can:

- reduce information collapse in the mask latent
- improve post-training mask fidelity
- keep the decoder from being completely unsupported

### 4.2 What it risks

It also changes the optimization target of the mask encoder during the main joint run.

That matters because:

- the mask encoder output is the latent distribution fed into the shared transformer
- constraining that latent for clean autoencoding can alter its geometry
- that indirectly changes the coupled optimization seen by the joint model

So Option C is not free.

Even a small reconstruction term can:

- improve masks
- but still perturb the training dynamics that produced the best image metrics

That is why Option C remains experimental, not the default.

## 5. Empirical Findings From the Full Model

I re-ran the probe with the full stack:

- base model: `Shitao/OmniGen-v1`
- LoRA: `/home/wenting/zr/gen_code_plan2_1/results/0030000`
- mask modules: `/home/wenting/zr/gen_code_plan2_1/results/mask_modules.bin`
- data: 1 real batch from `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`
- preprocessing: `keep_raw_resolution=True`, `max_image_size=1024`

Measured raw losses:

- `L_img = 0.43359375`
- `L_flow_mask = 1.046875`
- `L_recon = 0.97265625`

Exact matched reconstruction coefficient:

- formula: `lambda_recon = 0.1 * L_img / L_recon`
- computation: `0.1 * 0.43359375 / 0.97265625`
- result: `lambda_recon = 0.04457831325301206`

Recommended experimental value:

- `lambda_recon = 0.04458`

Interpretation:

- this coefficient makes the weighted reconstruction term exactly one tenth of the image flow loss on the measured batch
- it is therefore a mathematically calibrated "soft" regularizer
- but it is still **not** the default because the project priority is to preserve image SOTA first

## 6. Practical Decision

The production recommendation is:

- **Default:** Option A with `lambda_recon = 0.0`
- **Experimental only:** Option C with `lambda_recon = 0.04458`

This keeps the codebase unified while preserving a strict separation between:

- the safe SOTA path
- the optional reconstruction-regularized path
