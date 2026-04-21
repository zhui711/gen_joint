# DEVELOPMENT DESIGN

## Objective

Refactor the current anatomy-auxiliary OmniGen training setup into a true joint image-mask co-generation model:

- Training input: `input image + text instruction`
- Training targets: `edited image + 10-channel anatomy mask`
- Inference input: `input image + text instruction`
- Inference latent state: `image noise + mask noise`
- Inference output kept: `edited image`
- Inference output discarded: `predicted mask`

This design removes the current failure mode where decoded intermediate images are passed through a frozen RGB segmenter and produce unstable OOD gradients at early timesteps.

---

## 1. Current System Constraints

From code inspection:

- OmniGen is an image-latent rectified-flow model with one output latent branch.
- Current output image latent shape for `256 x 256` CXR is `(4, 32, 32)`.
- Output image tokens are produced by patchifying that latent with `patch_size=2`, yielding `16 x 16 = 256` tokens.
- The Transformer sequence today is:
  - text / condition embeddings
  - one time token
  - output image tokens
- The current `train_anatomy_mask.py` uses LoRA only on:
  - `qkv_proj`
  - `o_proj`
- Verified actual OmniGen module names in the installed `omnigen` environment:
  - `llm.layers.*.self_attn.qkv_proj`
  - `llm.layers.*.self_attn.o_proj`
  - `llm.layers.*.mlp.gate_up_proj`
  - `llm.layers.*.mlp.down_proj`

Relevant code anchors:

- `/home/wenting/zr/gen_code_plan2/OmniGen/model.py`
- `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py`
- `/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py`
- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py`

---

## 2. Design Principles

The redesign should satisfy four constraints:

1. Preserve as much of OmniGen's pretrained image-generation prior as possible.
2. Avoid decoding intermediate states to RGB during training.
3. Make mask prediction a native generative variable, not a post-hoc constraint.
4. Keep inference simple: co-denoise image and mask, decode image only.

The key idea is:

- image latent and mask latent are both flow-matched variables
- both are tokenized and passed through the same Transformer
- losses are computed directly on predicted velocities in latent space

This is the least disruptive path that still achieves genuine joint-distribution modeling.

---

## 3. Recommended Mask Representation

## 3.1 Candidate options

### Option A. Direct continuous mask latent with lightweight learned projection

Pipeline:

- GT mask `(10, 256, 256)` in `{0, 1}`
- map to `[-1, 1]`
- downsample / encode to mask latent `(C_m, 32, 32)`
- patchify into mask tokens
- Transformer predicts mask velocity in the same latent space

Pros:

- Minimal conceptual overhead
- Compatible with flow matching directly
- Does not require bit decomposition
- Easy to concatenate with existing image token stream
- Easiest to implement while preserving pretrained image branch

Cons:

- Requires introducing new mask encoder / decoder modules
- Mask latent prior is learned from scratch

### Option B. Analog bits / bitplane representation like PDM

Pipeline:

- convert per-pixel class/state encoding to multi-bit analog channels
- learn flow over continuous analog-bit space

Pros:

- Strong precedent in discrete-layout generation
- makes discrete structure more explicit

Cons:

- Your masks are already multi-label 10-channel anatomy maps, not a single categorical map
- Adds complexity without obvious benefit for this dataset
- More invasive and less aligned with minimizing disruption to OmniGen

### Option C. Reuse image VAE for masks by forcing 10-channel masks into pseudo-RGB or 4-channel latent

Pros:

- Fewer new modules

Cons:

- Conceptually wrong
- fights the pretrained RGB image VAE prior
- likely entangles mask semantics with image statistics
- high risk of unstable training

Recommendation:

- Do not use this path

## 3.2 Recommended mask-space design

Recommended approach:

- Option A with a dedicated lightweight mask encoder/decoder

Specifically:

1. Convert GT mask from `{0,1}` to `[-1,1]`:

```text
m_cont = 2 * m_bin - 1
```

2. Encode with a small trainable mask autoencoder:

```text
E_mask : R^{10 x 256 x 256} -> R^{C_m x 32 x 32}
D_mask : R^{C_m x 32 x 32} -> R^{10 x 256 x 256}
```

3. Choose `C_m = 4`.

Why `C_m = 4`:

- matches OmniGen's image latent channel count
- allows reusing the same patch geometry and output head shape logic
- minimizes code disruption
- makes sequence splitting and scheduler updates simpler

Important nuance:

- `C_m = 4` does not mean reusing the image VAE
- it only means matching the latent dimensionality for architectural compatibility

## 3.3 Why not project masks directly with a single `Conv2d(10 -> hidden)`?

That is tempting, but incomplete.

If you only project masks into tokens with no latent decoder:

- training loss either stays in token space only, which is weakly constrained
- or you need an ad hoc reconstruction head from hidden states back to mask space

A small mask autoencoder is better because it gives:

- a stable continuous latent space
- an invertible path back to mask space
- a meaningful latent velocity target for flow matching

## 3.4 Best practical mask autoencoder

Recommended minimal module family:

- `MaskEncoder`
  - strided conv stack reducing `256 -> 128 -> 64 -> 32`
  - input channels `10`
  - output channels `4`
- `MaskDecoder`
  - mirrored upsampling / conv stack
  - input channels `4`
  - output channels `10`
  - final `tanh` if reconstructing `[-1,1]`

Training of mask autoencoder:

- Phase 1: optional short standalone pretraining on mask reconstruction
- Phase 2: joint training with the main rectified-flow model

Recommendation:

- Pretrain the mask autoencoder first for stability, then jointly finetune

Reason:

- avoids starting the joint model with a meaningless mask latent space
- reduces early coupling instability

---

## 4. Mathematical Data Flow

## 4.1 Variables

Let:

- `x_img` = image latent in `R^{4 x 32 x 32}`
- `x_mask` = mask latent in `R^{4 x 32 x 32}`
- `x0_img, x0_mask` = Gaussian noise samples
- `t in [0,1]`

Forward interpolation:

```text
x_t_img  = t * x_img  + (1 - t) * x0_img
x_t_mask = t * x_mask + (1 - t) * x0_mask
```

Velocity targets:

```text
u_img  = x_img  - x0_img
u_mask = x_mask - x0_mask
```

Model outputs:

```text
v_img_theta, v_mask_theta = f_theta(x_t_img, x_t_mask, t, cond)
```

Loss:

```text
L_img  = ||v_img_theta  - u_img||^2
L_mask = ||v_mask_theta - u_mask||^2
L_total = L_img + lambda_mask * L_mask
```

Optional reconstruction regularizer through the mask decoder:

```text
m_hat = D_mask(x_mask_hat)
L_mask_recon = BCEWithLogits(m_hat, m_gt) or MSE(tanh-space)
```

Recommendation:

- Use latent-space flow loss as the primary mask loss
- Add a small decoded-mask reconstruction loss only after the core joint model is stable

## 4.2 Why this solves the current flaw

Current flawed path:

- model latent -> VAE decode -> RGB image -> frozen ResUnet -> GT mask loss

Proposed path:

- model predicts mask velocity directly in its own latent space

Benefits:

- no OOD frozen segmenter in the gradient path
- no dependence on visually meaningful early decoded RGB states
- anatomy becomes part of the generated state itself

---

## 5. Recommended Sequence Assembly

## 5.1 Current sequence

Today:

```text
[condition tokens] [time token] [image output tokens]
```

## 5.2 Proposed sequence

Recommended:

```text
[condition tokens] [time token] [image output tokens] [mask output tokens]
```

For current CXR geometry:

- image latent `(4, 32, 32)` -> `256` tokens
- mask latent `(4, 32, 32)` -> `256` tokens

Total generated-token suffix:

- `512` tokens

## 5.3 Why append mask tokens after image tokens

Pros:

- minimally invasive relative to current code
- preserves the current image token block semantics
- easy output split:
  - first generated block = image
  - second generated block = mask
- easy cache cropping logic because both are output-side generative tokens

Alternative:

- interleave image and mask tokens spatially

Cons:

- higher implementation complexity
- more invasive cache/split logic
- unnecessary for first implementation

Recommendation:

- append mask tokens after image tokens

## 5.4 Positional encoding

Both image and mask tokens use 2D positional encodings derived from their latent spatial grids.

Recommended implementation:

- reuse `cropped_pos_embed(height, width)` for both branches
- add a learned modality embedding:
  - `image_modality_embed`
  - `mask_modality_embed`

So token construction becomes:

```text
img_tokens  = PatchEmbed_img(x_t_img)  + pos_embed + image_modality_embed
mask_tokens = PatchEmbed_mask(x_t_mask) + pos_embed + mask_modality_embed
```

Why modality embeddings are important:

- image and mask tokens otherwise share spatial positions and hidden size
- modality embeddings tell the Transformer whether a token is radiograph latent or anatomy latent

---

## 6. Recommended Model Architecture Changes

## 6.1 New modules to add in `OmniGen/model.py`

Add:

1. `mask_x_embedder`
   - same shape contract as `x_embedder`
   - `PatchEmbedMR(patch_size=2, in_chans=4, embed_dim=hidden_size)`

2. `mask_final_layer`
   - same structure as current `final_layer`
   - outputs `patch_size * patch_size * 4`

3. `image_modality_embed`
   - learnable `(1, 1, hidden_size)`

4. `mask_modality_embed`
   - learnable `(1, 1, hidden_size)`

5. Optional:
   - `mask_input_x_embedder` only if you later want mask-conditioning inputs from user side
   - not needed for the current training/inference plan

## 6.2 Forward signature change

Current:

```python
forward(x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, ...)
```

Recommended:

```python
forward(
    x_img,
    x_mask,
    timestep,
    input_ids,
    input_img_latents,
    input_image_sizes,
    attention_mask,
    position_ids,
    ...
)
```

## 6.3 Forward internals

Recommended flow:

1. Patchify image latent:

```text
img_tokens, img_num_tokens, img_shapes
```

2. Patchify mask latent:

```text
mask_tokens, mask_num_tokens, mask_shapes
```

3. Add modality embeddings
4. Concatenate:

```text
generated_tokens = [img_tokens, mask_tokens]
input_emb = [condition_embeds, time_token, generated_tokens]
```

5. Transformer forward
6. Slice output suffix into:

```text
image_embedding = output[:, -(img_tokens + mask_tokens) : -mask_tokens]
mask_embedding  = output[:, -mask_tokens :]
```

7. Apply separate output heads:

```text
pred_img_latent_velocity  = image_final_layer(...)
pred_mask_latent_velocity = mask_final_layer(...)
```

8. Unpatchify each branch back to:

- image: `(B, 4, 32, 32)`
- mask: `(B, 4, 32, 32)`

## 6.4 Why separate mask output head is necessary

Do not share the current `final_layer` across image and mask.

Reason:

- image latent and mask latent are different modalities even if both use 4 channels
- separate output heads allow the pretrained image head to stay closer to its original function
- mask head can learn from scratch without distorting image decoding behavior

Recommendation:

- keep current `final_layer` for image
- add `mask_final_layer` for mask

---

## 7. Mask Tokenization Strategy Recommendation

## 7.1 Final recommendation

Use:

- binary mask in `{0,1}`
- map to `[-1,1]`
- encode with a small trainable mask autoencoder to latent `(4, 32, 32)`
- patchify with a dedicated `mask_x_embedder`

## 7.2 Why this is the optimal path

Compared with direct decoded-image supervision:

- stable latent-space objective
- no frozen segmentation network in the gradient path
- direct supervision at all timesteps

Compared with analog bits:

- simpler
- better aligned with multi-channel anatomy masks
- faster to implement in this codebase

Compared with large mask VAE:

- much lower disruption
- enough capacity for structured anatomical masks

---

## 8. Sequence Split and Loss Computation

## 8.1 New joint loss helper

Replace current `training_losses_with_anatomy_mask(...)` with something like:

```text
training_losses_with_joint_mask(...)
```

Inputs:

- image latent target `x1_img`
- mask latent target `x1_mask`
- model kwargs

Process:

1. sample `x0_img`, `x0_mask`
2. sample same timestep `t`
3. build `xt_img`, `xt_mask`
4. run model
5. receive `pred_u_img`, `pred_u_mask`
6. compute:
   - `loss_img = MSE(pred_u_img, u_img)`
   - `loss_mask = MSE(pred_u_mask, u_mask)`
7. combine:

```text
loss = loss_img + lambda_mask * loss_mask
```

## 8.2 Shared timestep vs separate timestep

Recommendation:

- use the same timestep `t` for image and mask per sample

Reason:

- image and mask should co-evolve as a coupled state
- simpler scheduler and training
- matches the joint-distribution intuition

## 8.3 Optional decoded-mask auxiliary loss

After the base joint latent loss is stable, optionally add:

```text
x1_hat_mask = xt_mask + (1 - t) * pred_u_mask
m_hat = D_mask(x1_hat_mask)
L_mask_recon = BCE or MSE
```

Recommendation:

- start without this term
- only add later if latent-only supervision under-constrains the mask branch

Reason:

- keep first redesign focused and stable
- avoid repeating the same mistake of relying heavily on decoded auxiliary supervision too early

---

## 9. LoRA Scope Expansion

## 9.1 Current state

Today `train_anatomy_mask.py` hardcodes:

```python
target_modules=["qkv_proj", "o_proj"]
```

This is in:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:155-160`

## 9.2 Required change

Yes, this requires modifying the hardcoded list in the script, unless you introduce a new CLI arg for target module names.

Recommended target list:

```python
target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
```

This is the correct Phi-3 naming in your actual OmniGen environment.

Verified by live module inspection:

- `llm.layers.*.self_attn.qkv_proj`
- `llm.layers.*.self_attn.o_proj`
- `llm.layers.*.mlp.gate_up_proj`
- `llm.layers.*.mlp.down_proj`

## 9.3 Why expand LoRA to MLP

Attention-only LoRA helps with routing information between tokens.

But your new task requires:

- representing a new modality
- translating between image-token and mask-token semantics
- learning new cross-token nonlinear interactions

Those transformations are not purely attention operations.

MLP LoRA is important for:

- modality mixing after attention aggregation
- feature remapping between image and anatomy manifolds
- allowing the network to adapt its intermediate hidden representations

## 9.4 Recommended rank and alpha

Current:

- `r=8`
- `alpha=8`

Recommended first serious setting for joint image-mask learning:

- `r=32`
- `alpha=32` or `64`

Suggested progression:

1. conservative:
   - `r=16`, `alpha=32`
2. stronger:
   - `r=32`, `alpha=32`
3. maximal within LoRA regime:
   - `r=64`, `alpha=64`

Given your stated priority of SOTA over cost:

- recommended default: `r=32`, `alpha=32`

## 9.5 PEFT implementation path

In `train_anatomy_mask.py`, replace:

```python
transformer_lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_rank,
    init_lora_weights="gaussian",
    target_modules=["qkv_proj", "o_proj"],
)
```

with a configurable version such as:

```python
transformer_lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=args.lora_target_modules,
)
```

Add parser args:

- `--lora_alpha`
- `--lora_target_modules`

Recommended default:

```text
["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
```

Why add CLI control:

- makes ablation easy
- avoids future hardcoded edits

## 9.6 PEFT constraint note

The new mask encoder, mask decoder, `mask_x_embedder`, and `mask_final_layer` will not automatically be covered by the current PEFT wrapping, because `get_peft_model(model, config)` targets matching submodules.

Recommendation:

- train the newly introduced mask-specific modules fully
- use LoRA for the pretrained Transformer

So the intended trainable set should be:

1. Always train full:
   - mask encoder
   - mask decoder
   - mask token embedder
   - mask output head
   - modality embeddings
2. Train via LoRA:
   - Transformer attention and MLP targets
3. Keep frozen:
   - pretrained image VAE
   - most base OmniGen image branch weights unless explicitly unfrozen

This gives the best stability/performance tradeoff.

---

## 10. Inference Pipeline Adjustment

## 10.1 Current state

Current pipeline only initializes:

```text
z_img ~ N(0, I)
```

Then it iteratively updates `z_img`.

## 10.2 Proposed joint latent state

Initialize:

```text
z_img  ~ N(0, I)  in R^{B x 4 x 32 x 32}
z_mask ~ N(0, I)  in R^{B x 4 x 32 x 32}
```

Both must be duplicated for CFG exactly the same way.

## 10.3 Model interface during inference

Recommended scheduler/model contract:

```python
pred_img, pred_mask, cache = func(
    z_img,
    z_mask,
    timesteps,
    past_key_values=cache,
    **model_kwargs
)
```

Update rule:

```python
z_img  = z_img  + dt * pred_img
z_mask = z_mask + dt * pred_mask
```

## 10.4 Scheduler refactor options

### Option 1. Keep separate tensors throughout scheduler

```python
scheduler(z_img, z_mask, func, model_kwargs, ...)
```

Pros:

- explicit
- easy to reason about
- easier debugging

Cons:

- requires touching scheduler signature

### Option 2. Pack into one larger tensor channel-wise

```text
z_joint = concat([z_img, z_mask], dim=1)  # (B, 8, 32, 32)
```

Pros:

- one state tensor

Cons:

- breaks current patch embed assumptions
- would require changing image patch embed from 4 to 8 channels
- disturbs pretrained image path too much

Recommendation:

- Option 1: keep separate tensors

## 10.5 Safe discard before image decode

At the end of inference:

1. keep `z_img_final`
2. ignore `z_mask_final`
3. decode only `z_img_final` through image VAE

Optionally:

- for debugging/validation, decode `z_mask_final` through `mask_decoder` and save it
- but keep this off by default in production inference

## 10.6 CFG behavior

Recommendation:

- apply CFG to both image and mask predictions jointly because they arise from one forward pass

Implementation:

- `forward_with_cfg` and `forward_with_separate_cfg` should return a pair:
  - guided image prediction
  - guided mask prediction

You should not guide only image and leave mask unguided, because that would break the coupled state dynamics.

---

## 11. File-by-File Modification Plan

## 11.1 `/home/wenting/zr/gen_code_plan2/OmniGen/model.py`

Required changes:

1. Add new modules:
   - `mask_x_embedder`
   - `mask_final_layer`
   - `image_modality_embed`
   - `mask_modality_embed`
2. Add helper to patch mask latents
3. Update `forward(...)` to accept `x_mask`
4. Concatenate image and mask generated tokens
5. Split output into image and mask suffixes
6. Return both predicted latent velocities
7. Update `forward_with_cfg(...)`
8. Update `forward_with_separate_cfg(...)`

Recommended return contract:

```python
return pred_img, pred_mask, past_key_values
```

## 11.2 `/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py`

Required changes:

1. Update scheduler signature to accept:
   - `z_img`
   - `z_mask`
2. Initialize cache length using total generated suffix tokens:
   - `num_img_tokens + num_mask_tokens`
3. At each step:
   - call joint model
   - update both states
4. Adjust cache cropping logic so it removes both image and mask generated suffix tokens after first step

Important detail:

Current crop logic assumes only:

- one time token
- one generated image suffix

New logic must crop:

- image tokens
- mask tokens
- keep the condition cache only

## 11.3 `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py`

Required changes:

1. Initialize `mask_latents`
2. Duplicate them for CFG exactly like image latents
3. Pass both into scheduler
4. Receive both outputs
5. Decode image only with VAE
6. Optionally decode mask with `mask_decoder` under a debug flag

Also:

- load `mask_encoder/mask_decoder` weights along with model checkpoint if bundled

## 11.4 `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py`

Required changes:

1. Keep current `output_anatomy_masks`
2. No major dataloader redesign required
3. Optionally normalize masks to float in `[-1,1]` here instead of in the loss

Recommendation:

- keep raw float mask loading in dataset
- do `[-1,1]` mapping inside the training loss/helper

Reason:

- avoids changing dataset semantics globally

## 11.5 `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py`

Replace with a new module, for example:

- `loss_joint_mask.py`

Required functionality:

1. Encode GT image to image latent
2. Encode GT mask to mask latent
3. Sample joint noise
4. Compute joint noisy state
5. Call joint model
6. Compute:
   - `loss_img`
   - `loss_mask`
7. Return:
   - total loss
   - component losses

Remove entirely:

- VAE decode in loss
- frozen segmentation model path
- timestep-weighted decoded-mask MSE branch

## 11.6 `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py`

This file should become the new main joint-training script, or be replaced with a new file such as:

- `train_joint_mask.py`

Required changes:

1. Remove segmentation model loading
2. Remove old anatomy-mask loss import
3. Load and train:
   - mask encoder
   - mask decoder
4. Expand LoRA targets and parser args
5. Encode GT masks before loss computation
6. Call new joint loss helper
7. Save extra state for mask modules in checkpoints

Recommended parser additions:

- `--mask_latent_channels`
- `--lambda_mask`
- `--lambda_mask_recon`
- `--lora_alpha`
- `--lora_target_modules`
- `--mask_ae_ckpt`

## 11.7 `/home/wenting/zr/gen_code_plan2/test_omnigen_cxr.py`

Required changes:

1. No change to user-facing inputs
2. Internally pipeline now creates both image and mask noise
3. Discard mask before image decode/save
4. Optionally add debug argument:
   - `--save_pred_masks`

The evaluation block can remain image-only.

## 11.8 New file recommendation

Add a dedicated file such as:

- `/home/wenting/zr/gen_code_plan2/OmniGen/mask_autoencoder.py`

Contents:

- `MaskEncoder`
- `MaskDecoder`
- optional `MaskAutoencoder` wrapper

This keeps mask-specific logic out of the main OmniGen model file.

---

## 12. Checkpointing and Backward Compatibility

## 12.1 New checkpoint contents

Joint model checkpoints should include:

- OmniGen base/LoRA weights
- mask encoder weights
- mask decoder weights
- mask token embedder
- mask output head
- modality embeddings

## 12.2 Backward compatibility strategy

Recommendation:

- create a new training entrypoint instead of mutating the current one in place

Suggested names:

- `train_joint_mask.py`
- `loss_joint_mask.py`

Reason:

- cleaner experiments
- easier regression comparison
- avoids confusion with the old segmentation-mediated approach

---

## 13. Training Strategy Recommendation

## 13.1 Phase plan

### Phase 0. Mask autoencoder warm-up

Train:

- `MaskEncoder`
- `MaskDecoder`

On:

- GT masks only

Objective:

- reconstruction in `[-1,1]` or BCE space

### Phase 1. Joint latent training with pretrained image OmniGen

Train:

- mask modules fully
- Transformer via expanded LoRA

Loss:

- `L_img + lambda_mask * L_mask`

Recommendation:

- start with `lambda_mask = 0.5`

### Phase 2. Optional decoded-mask auxiliary refinement

If needed, add small:

- `lambda_mask_recon * L_mask_recon`

Recommendation:

- keep this term small, such as `0.05` to `0.1`

## 13.2 Why not start with a large mask weight

Even in the joint-latent design, the image branch remains the main task.

A too-large mask loss may still over-regularize image quality.

Suggested starting range:

- `lambda_mask in [0.25, 1.0]`

Recommended first run:

- `lambda_mask = 0.5`

---

## 14. Risks and Mitigations

## 14.1 Risk: mask branch dominates the shared Transformer

Mitigation:

- separate mask output head
- moderate initial `lambda_mask`
- expanded LoRA rather than full unfrozen Transformer

## 14.2 Risk: mask autoencoder latent is poorly structured

Mitigation:

- pretrain mask autoencoder first
- use small decoded-mask reconstruction regularizer later

## 14.3 Risk: cache cropping breaks after adding mask tokens

Mitigation:

- explicitly compute:
  - `num_img_tokens`
  - `num_mask_tokens`
  - `num_generated_tokens = num_img_tokens + num_mask_tokens`
- crop cache using total generated suffix length

## 14.4 Risk: LoRA target names are wrong

Mitigation:

- use verified names from your environment:
  - `qkv_proj`
  - `o_proj`
  - `gate_up_proj`
  - `down_proj`

---

## 15. Final Recommended Path

The best path for this codebase is:

1. Add a small dedicated mask autoencoder producing mask latent `(4, 32, 32)`.
2. Treat image latent and mask latent as a joint rectified-flow state.
3. Sequence-concatenate:
   - image tokens first
   - mask tokens second
4. Add modality embeddings and a dedicated mask output head.
5. Compute flow-matching loss directly in latent space for both branches.
6. Expand LoRA from attention-only to:
   - `qkv_proj`
   - `o_proj`
   - `gate_up_proj`
   - `down_proj`
7. Use `r=32`, `alpha=32` as the first strong setting.
8. Modify the scheduler and pipeline to co-denoise image noise and mask noise together.
9. Decode only the image at inference and discard the mask by default.

This path is optimal because it gives you true co-generation with minimal disruption to OmniGen's pretrained image prior, avoids the frozen-segmenter OOD gradient problem entirely, and maps cleanly onto the existing code structure.

---

## 16. Verified Engineering Facts Used in This Design

The following were verified directly in your environment and codebase:

- Image latent geometry is `(4, 32, 32)` for `256 x 256` outputs.
- Output image token count is `256`.
- Current LoRA target list in code is hardcoded in `train_anatomy_mask.py`.
- Real Phi-3 / OmniGen module names include:
  - `llm.layers.*.self_attn.qkv_proj`
  - `llm.layers.*.self_attn.o_proj`
  - `llm.layers.*.mlp.gate_up_proj`
  - `llm.layers.*.mlp.down_proj`

This means the proposed LoRA expansion is directly compatible with the installed `peft` / OmniGen stack and does require modifying the hardcoded target list in the training script unless you expose it as a new CLI argument.
