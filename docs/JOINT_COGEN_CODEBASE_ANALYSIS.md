# OmniGen CXR Anatomy-Mask Pipeline Analysis

## Scope

This report is based on direct code inspection of:

- `/home/wenting/zr/gen_code_plan2/`
- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py`
- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py`
- `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh`
- `/home/wenting/zr/gen_code_plan2/test_omnigen_cxr.py`
- `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch`
- `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256`

It also includes spot verification of:

- One JSONL sample from `cxr_synth_anno_mask_train.jsonl`
- One real GT image file
- One real GT mask file

## Evidence Summary

- Verified JSONL sample:
  - `output_image`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0001.png`
  - `output_mask`: `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz`
- Verified GT image file:
  - `0001.png` is `256 x 256`, grayscale PNG
- Verified GT mask file:
  - `.npz` contains key `mask`
  - shape `(10, 256, 256)`
  - dtype `bool`

Note:

- I could not run a live PyTorch dataloader instantiation in the shell that was available to me because that interpreter did not have `torch` installed.
- The tensor shapes below are therefore derived from code plus verified on-disk data dimensions, not from a live runtime printout.

---

## 1. OmniGen Transformer and VAE Architecture

### 1.1 Current image modality path

The current training and inference stack is strictly image-latent based:

1. Images are loaded as RGB tensors.
2. They are encoded by a Diffusers `AutoencoderKL`.
3. The resulting latent is patchified and fed into OmniGen's Transformer.
4. The Transformer predicts a latent-space velocity / flow field.
5. The final latent prediction is decoded back to RGB image space by the same VAE.

Relevant code:

- RGB conversion in dataset and processor:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:40-45`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:53-56`
- VAE loading:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:119-127`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:87-95`
- VAE encode scaling:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/utils.py:94-109`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:110-117`

### 1.2 Exact latent shape of the image

For the current CXR dataset, the effective image path is:

- GT PNG is grayscale `256 x 256`
- It is converted to RGB by `.convert('RGB')`
- It becomes a tensor of shape `(3, 256, 256)`

The OmniGen latent shape is effectively:

- `(4, 32, 32)` per image

Why:

- In inference, the pipeline explicitly creates noise latents of shape `(B, 4, height // 8, width // 8)`:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:230-237`
- For `height = width = 256`, this is `(B, 4, 32, 32)`.
- Training uses the same VAE family and the same scaling utilities, so the encoded target latent is also the same spatial format:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:304-314`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/utils.py:94-109`

Under `--keep_raw_resolution`, training keeps latents as a Python list of tensors. For the current dataset, each output image is still effectively:

- image tensor: `(1, 3, 256, 256)`
- latent tensor after VAE encode: `(1, 4, 32, 32)`

### 1.3 How images are tokenized / patchified before the Transformer

OmniGen does not feed pixel-space patches into the Transformer. It feeds latent-space patches.

Patchification implementation:

- `PatchEmbedMR` is a `Conv2d` patch embedder with:
  - `patch_size = 2`
  - `in_chans = 4`
  - output dimension = Transformer hidden size
- Code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:133-149`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:156-173`

Therefore, for a latent `(4, 32, 32)`:

- patch size = `2 x 2`
- token grid = `16 x 16`
- number of output image tokens = `256`

This matches the processor/collator logic:

- output token count is computed as `H * W // 16 // 16`
- for `256 x 256`, token count is `256`
- code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:241-245`

So the output image path is:

- RGB image: `(3, 256, 256)`
- VAE latent: `(4, 32, 32)`
- latent patch tokens: `(256, hidden_size)`

### 1.4 How the Transformer sequence is assembled

The OmniGen forward pass concatenates:

1. Text token embeddings
2. Replaced input-image placeholder embeddings
3. One time token
4. Output image latent patch tokens

Key code:

- Placeholder token ranges are inserted in the processor:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:63-91`
- Input-image placeholders are replaced with patch embeddings:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:322-334`
- Final sequence assembly:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:334-338`

Important observation:

- Input images are represented as latent patch tokens embedded into the text stream.
- The output image is represented by appended latent tokens after the time token.
- There is no parallel mask stream anywhere in the current OmniGen architecture.

### 1.5 VAE channel assumptions: can it handle 10-channel masks?

In the current codebase, the VAE path is effectively 3-channel image only.

Evidence:

- All images are forced to RGB before VAE encode:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:40-45`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:53-56`
- The segmentation model is also hard-coded to `in_channels=3`:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:56-64`
- The VAE decode output is used as RGB image for segmentation and final save:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:235-243`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:294-317`

Conclusion:

- The current VAE path is not designed for 10-channel segmentation masks.
- There is no code path that encodes or decodes a 10-channel tensor with the current `AutoencoderKL`.
- For a joint image-mask co-generation redesign, you will need either:
  - a separate mask encoder/decoder or mask VAE, or
  - a dedicated learned projection path for masks into a mask-latent space

First-principles note:

- A single lightweight `Conv2d(10 -> 4)` could project masks into a 4-channel tensor, but that is not equivalent to giving masks a proper generative latent model.
- If the redesign goal is true co-evolution of image latent and mask latent, a separate mask latent branch is the more faithful architectural move.

### 1.6 Hidden size

The exact Transformer hidden size is taken from the pretrained OmniGen config at runtime:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:219-223`

The repository default collator fallback is `3072`:

- `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:155-158`

This strongly suggests the active checkpoint is using hidden size `3072`, but the downloaded model config was not directly available in the inspected workspace.

Status:

- Hidden size `3072`: likely
- Exact checkpoint confirmation: Requires further investigation

---

## 2. LoRA Configuration

### 2.1 Where LoRA is applied

LoRA is created in:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:151-163`

The exact config is:

- `r = args.lora_rank`
- `lora_alpha = args.lora_rank`
- `target_modules = ["qkv_proj", "o_proj"]`
- `init_lora_weights = "gaussian"`

This means:

- LoRA is only attached to attention projection modules whose names match `qkv_proj` and `o_proj`
- It is not attached to MLP layers
- It is not attached to separate `q_proj`, `k_proj`, `v_proj` names

So the current fine-tuning scope is:

- attention fused QKV projection
- attention output projection

Not included:

- MLP / feed-forward blocks
- `gate_proj`
- `up_proj`
- `down_proj`

### 2.2 Current rank and alpha

Defaults in the Python training script:

- `lora_rank = 8`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:473-475`
- `lora_alpha = lora_rank`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:155-158`

Therefore default effective settings are:

- `r = 8`
- `alpha = 8`

The launch script explicitly sets:

- `--use_lora`
- `--lora_rank 8`

Relevant lines:

- `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:19-20`

So the currently launched configuration is:

- LoRA enabled
- rank `8`
- alpha `8`
- targets `qkv_proj`, `o_proj`

### 2.3 Resume behavior

If `--lora_resume_path` is provided, the script loads adapter weights only and creates a fresh optimizer:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:164-192`

The current launch script resumes from:

- `/home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/`
  - `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:36`

---

## 3. Current Data Flow and Loss Hooking

### 3.1 How images and masks are paired in the dataset

Dataset class:

- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:19-91`

Pairing logic:

1. The JSONL example provides:
   - `instruction`
   - `input_images`
   - `output_image`
   - optional `output_mask`
2. `output_image` is loaded as RGB tensor
3. `output_mask` is loaded from `.npz` key `"mask"`
4. The dataset returns:
   - `(mllm_input, output_image, output_anatomy_mask)`

Relevant lines:

- output image load:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:40-45`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:64`
- output mask load:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:46-52`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:66-70`

Verified JSONL sample:

```json
{
  "task_type": "image_edit",
  "instruction": "<img><|image_1|></img> Edit the view using delta pose: d_theta=-0.0247 rad, sin(d_azimuth)=0.0269, cos(d_azimuth)=0.9996.",
  "input_images": ["/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0000.png"],
  "output_image": "/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0657/0001.png",
  "output_mask": "/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0657/0001.npz"
}
```

### 3.2 Tensor shapes yielded by the dataloader

Current launch uses:

- `--keep_raw_resolution`
  - `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:24`

Therefore the collator does not stack images into a single dense tensor. It keeps them as lists.

Relevant code:

- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:101-130`

For the current CXR dataset, the derived shapes are:

- `output_images`
  - Python list of length `B`
  - each element shape `(1, 3, 256, 256)`
- `output_anatomy_masks`
  - stacked tensor of shape `(B, 10, 256, 256)`
- `input_pixel_values`
  - Python list over all input conditioning images in the batch
  - each element shape `(1, 3, 256, 256)` for current data
- `input_ids`
  - tensor of shape `(effective_batch_cfg, seq_len)`
- `attention_mask`
  - tensor of shape `(effective_batch_cfg, seq_len_total, seq_len_total)`
- `position_ids`
  - tensor of shape `(effective_batch_cfg, seq_len_total)`

Notes:

- `effective_batch_cfg` is larger than the user batch because OmniGen duplicates conditions for CFG and image CFG in the collator:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:269-293`
- For output images, token count is based on target size:
  - `256 * 256 // 16 // 16 = 256`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:241-245`

After VAE encode in training:

- `output_images` becomes list of target latents
- each latent is effectively `(1, 4, 32, 32)` for current data
- `input_pixel_values` also becomes list of conditioning latents of shape `(1, 4, 32, 32)`

Relevant code:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:304-314`

### 3.3 Exact place where `loss_anatomy_mask.py` is called

The call site is:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:333-345`

This occurs:

1. After `output_images` has already been VAE-encoded into latent space
2. After `input_pixel_values` has also been VAE-encoded
3. After `model_kwargs` has been assembled
4. Before `accelerator.backward(loss)`

Backprop happens at:

- `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:352`

### 3.4 Is the anatomy loss computed from predicted `x1` or predicted velocity?

The anatomy loss is not applied directly to the velocity prediction.

The model's main output is trained against:

- `ut = x1 - x0`
- diffusion loss: `MSE(model_output, ut)`

Code:

- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:198-218`

So `model_output` is the predicted rectified-flow velocity / target vector field.

Then the code reconstructs a predicted clean latent:

- `x1_hat = xt + (1 - t) * model_output`

Code:

- list case:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:222-223`
- tensor case:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:276-279`

Therefore:

- diffusion loss is on predicted velocity
- anatomy loss is on predicted clean latent `x1_hat`

This matches the rectified-flow identity:

- `xt = t * x1 + (1 - t) * x0`
- `u = x1 - x0`
- `x1 = xt + (1 - t) * u`

### 3.5 Exact anatomy-loss path

The current anatomy branch is:

1. Sample `x0` and timestep `t`
2. Build noisy latent `xt`
3. Predict `model_output`
4. Reconstruct `x1_hat`
5. Decode `x1_hat` with VAE
6. Resize decoded image to `256 x 256` if needed
7. Run frozen segmentation model
8. Apply sigmoid to segmentation logits
9. Compute 10-channel MSE against GT mask
10. Weight anatomy loss by `t ** anatomy_alpha`
11. Add to diffusion loss

Relevant code:

- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:198-330`

The actual combined loss returned is:

- `loss = loss_diffusion + lambda_anatomy * loss_anatomy`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:330-337`

### 3.6 How gradients backpropagate to OmniGen

The code explicitly preserves gradient flow through the decode branch:

- it does not wrap VAE decode or segmentation inference in `torch.no_grad()`
- comments in the file explicitly state this

Relevant lines:

- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:12-17`
- `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:230-235`

Gradient path is:

- `loss_anatomy`
  -> `mask_gen = sigmoid(seg_model(gen_decoded))`
  -> `gen_decoded = vae.decode(x1_hat_scaled).sample`
  -> `x1_hat`
  -> `model_output`
  -> OmniGen / LoRA parameters

Important detail:

- `seg_model` is frozen:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:67-80`
- `vae` encoder is frozen and target latents are created under `torch.no_grad()`:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:150`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:303-314`
- But the decode path inside the loss is differentiable.

So the current auxiliary supervision is:

- post-hoc
- image-space
- segmentation-model mediated
- attached to reconstructed clean latent `x1_hat`

There is no mask latent inside OmniGen itself.

### 3.7 Current time filtering already present

The current `loss_anatomy_mask.py` is not the naive version described in the historical failure mode. It already includes timestep weighting:

- `weighted_loss_i = loss_i * (t_i ** anatomy_alpha)`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:270`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:325`

Defaults:

- Python default: `anatomy_alpha = 4.0`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:523-528`
- Launch override:
  - `--anatomy_alpha 4.0`
  - `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:33-35`

So the currently launched code is already trying to suppress early-timestep anatomy gradients, but it still remains a post-hoc decoded-image constraint rather than a co-generated mask modality.

### 3.8 Active launch hyperparameters relevant to anatomy training

Current launch script:

- `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:10-36`

Active relevant settings:

- `--use_lora`
- `--lora_rank 8`
- `--batch_size_per_device 16`
- `--gradient_accumulation_steps 4`
- `--lambda_anatomy 1.0`
- `--anatomy_alpha 4.0`
- `--anatomy_subbatch_size 16`
- `--keep_raw_resolution`
- `--lora_resume_path /home/wenting/zr/gen_code/results/cxr_finetune_lora/checkpoints/0030000/`

Important mismatch to note:

- training script default `lambda_anatomy` is `0.1`
- launch script overrides it to `1.0`

---

## 4. Inference and Test Flow

### 4.1 How `test_omnigen_cxr.py` loads the model

Each worker process:

1. Loads `OmniGenPipeline.from_pretrained(args.model_path)`
2. Optionally merges LoRA via `pipe.merge_lora(args.lora_path)`
3. Moves the pipeline to `cuda:{rank}`

Code:

- `/home/wenting/zr/gen_code_plan2/test_omnigen_cxr.py:121-143`

LoRA merge path in pipeline:

- `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:97-102`

### 4.2 How generation is run

For each batch:

1. Prompts and input image paths are assembled
2. `OmniGenPipeline.__call__` is invoked
3. Generated images are saved to deterministic output paths matching GT structure

Code:

- `/home/wenting/zr/gen_code_plan2/test_omnigen_cxr.py:152-212`

The test script uses fixed output size:

- `IMAGE_SIZE = 256`
  - `/home/wenting/zr/gen_code_plan2/test_omnigen_cxr.py:61`

And calls the pipeline with:

- `height=256`
- `width=256`
- `num_inference_steps=args.inference_steps`
- `guidance_scale=args.guidance_scale`
- `img_guidance_scale=args.img_guidance_scale`

### 4.3 Does inference use an ODE solver?

Yes.

Inference uses `OmniGenScheduler`, which performs explicit iterative latent updates:

- `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:285-287`
- `/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py:155-181`

The update rule is:

```python
z = z + (sigma_next - sigma) * pred
```

from:

- `/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py:162-167`

This is an explicit Euler-style solver over the rectified-flow ODE.

Conclusion:

- Yes, inference is ODE-based
- It is using a first-order explicit Euler-style update
- It is not using DPM-Solver
- It is not using a higher-order adaptive solver in the inspected code

### 4.4 Timestep schedule used at inference

The scheduler constructs:

```python
t = torch.linspace(0, 1, num_steps + 1)
t = t / (t + time_shifting_factor - time_shifting_factor * t)
self.sigma = t
```

Code:

- `/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py:116-124`

Default:

- `time_shifting_factor = 1`

So in the default configuration this reduces to the standard linear schedule from `0` to `1`.

### 4.5 How output latents are decoded at inference

After the scheduler finishes:

1. CFG-expanded samples are chunked and only the first chunk is kept
2. VAE scaling is inverted
3. Latents are decoded by `vae.decode`
4. Output is mapped from `[-1, 1]` to `[0, 1]`
5. PIL images are produced

Code:

- `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:285-317`

---

## 5. First-Principles Architectural Findings

### 5.1 What the current system actually is

The current system is not a joint image-mask model.

It is:

- an image-only latent rectified-flow model
- with a post-hoc auxiliary segmentation consistency loss
- applied after reconstructing `x1_hat` into RGB image space

There is no:

- mask tokenizer
- mask latent
- mask branch in the Transformer
- mask decoder
- mask co-sampling process during inference

### 5.2 Why the current anatomy loss is structurally fragile

Even with timestep weighting, the current anatomy branch still depends on:

- decoding predicted latent into an image
- passing that image through a frozen RGB segmentation model
- trusting that segmentation model as a surrogate anatomy oracle

This creates two structural risks:

1. The mask supervision is external to the generative state space.
2. The gradients are mediated by a frozen network trained on clean RGB anatomy, not on noisy or partially denoised generative states.

This is exactly why a true co-generation redesign would be qualitatively different:

- the mask would become part of the state being generated
- not a post-hoc constraint on a decoded image

### 5.3 Minimal factual implication for redesign

A true joint-distribution or dual-stream redesign will require new components that do not exist in the current code:

- mask latent representation
- mask patch/token embedding path
- mask positional/token bookkeeping
- mask output head / unpatchify path
- training loss defined directly in mask latent or mask output space
- inference scheduler state that includes both image latent and mask latent

The current codebase only has one generative latent:

- image latent `(4, H/8, W/8)`

---

## 6. Direct Answers to the Requested Questions

### Q1. How are input images tokenized or patchified? What is the exact image latent shape?

Answer:

- Images are converted to RGB, encoded by a Diffusers `AutoencoderKL`, then patchified in latent space.
- The latent patch embedder is `Conv2d(in_chans=4, kernel_size=2, stride=2)`.
- For the current `256 x 256` CXR data:
  - image tensor: `(3, 256, 256)`
  - image latent: `(4, 32, 32)`
  - latent patch tokens: `16 x 16 = 256` tokens
- Relevant code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py:230-237`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:133-149`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/model.py:276-311`

### Q2. Is the VAE strictly 3-channel RGB? Can it handle 10-channel masks?

Answer:

- In the current codebase, yes, it is functionally a 3-channel RGB image VAE path.
- All loaded images are converted to RGB before encoding.
- There is no code path for encoding or decoding 10-channel masks with the existing VAE.
- A separate mask encoder/decoder or mask latent projection path will be needed for joint co-generation.
- Relevant code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:40-45`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/processor.py:53-56`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:56-64`

### Q3. How is LoRA currently applied? Which modules are tuned? What are `r` and `alpha`?

Answer:

- LoRA is applied in `train_anatomy_mask.py`.
- Target modules are:
  - `qkv_proj`
  - `o_proj`
- No MLP layers are targeted.
- Current effective settings:
  - `r = 8`
  - `alpha = 8`
- Relevant code:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:151-163`
  - `/home/wenting/zr/gen_code_plan2/lanuch/train_anatomy_mask.sh:19-20`

### Q4. How are image and mask loaded and paired? What are their dataloader shapes?

Answer:

- They are paired through the JSONL fields `output_image` and `output_mask`.
- `output_mask` is loaded from `.npz` key `mask`.
- Under the current `--keep_raw_resolution` launch:
  - `output_images`: list of `(1, 3, 256, 256)`
  - `output_anatomy_masks`: tensor `(B, 10, 256, 256)`
- After VAE encode:
  - output latents: list of `(1, 4, 32, 32)`
- Relevant code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:53-70`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py:101-130`
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:304-314`

### Q5. Where is `loss_anatomy_mask.py` called in training?

Answer:

- It is called at:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:333-345`
- Backprop follows immediately at:
  - `/home/wenting/zr/gen_code_plan2/train_anatomy_mask.py:352`

### Q6. Is segmentation loss computed using predicted `x1` or predicted velocity `v_theta`?

Answer:

- Diffusion loss is computed on the predicted velocity / flow target.
- Anatomy loss is computed on `x1_hat`, the predicted clean latent reconstructed from `xt` and `model_output`.
- Then `x1_hat` is decoded to RGB and fed to the segmentation model.
- Relevant code:
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:210-218`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:222-235`
  - `/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_anatomy_mask.py:276-283`

### Q7. How does gradient backpropagate to OmniGen?

Answer:

- `loss_anatomy`
  -> frozen `seg_model`
  -> differentiable `vae.decode`
  -> `x1_hat`
  -> `model_output`
  -> OmniGen / LoRA parameters

The segmentation model is frozen, but its forward pass still transmits gradients to its input. The VAE decode branch is intentionally not wrapped in `torch.no_grad()`.

### Q8. How does `test_omnigen_cxr.py` run inference? Does it use an ODE solver?

Answer:

- It loads `OmniGenPipeline`, optionally merges LoRA, and runs batched generation per GPU worker.
- Sampling is performed by `OmniGenScheduler`.
- The scheduler uses explicit Euler-style updates:
  - `z = z + (sigma_next - sigma) * pred`
- Therefore it is an ODE-style rectified-flow sampler, not DPM-Solver.

---

## 7. Items Requiring Further Investigation

### 7.1 Exact pretrained checkpoint config dump

Not directly available in the inspected workspace:

- exact `Phi3Config` JSON for the active checkpoint
- exact hidden size confirmation from the downloaded model files

Impact:

- low for latent-shape analysis
- moderate for redesign planning if you want exact parameter budgeting

### 7.2 Runtime verification of one collated batch inside the exact training environment

I statically derived the dataloader shapes from source plus verified real files, but could not instantiate a live batch in the available shell interpreter because `torch` was missing there.

Impact:

- low, because the shape logic in the collator is explicit
- still useful to verify in your exact training env before implementation

### 7.3 Exact internal module paths after PEFT wrapping

The configured target names are clear:

- `qkv_proj`
- `o_proj`

But the exact full dotted module paths after model loading were not enumerated from a live model object in this session.

Impact:

- low for conceptual redesign
- useful if you later want layerwise selective LoRA for the dual-stream redesign

---

## 8. Bottom Line

The current codebase is an image-only latent rectified-flow OmniGen with a post-hoc segmentation consistency loss applied on decoded predicted clean images. The image latent is currently `(4, 32, 32)` for your `256 x 256` CXR data, and the Transformer sees `256` latent patch tokens per output image. LoRA only touches attention `qkv_proj` and `o_proj` with `r=8`, `alpha=8`.

Most importantly for your redesign goal, there is no native mask latent or mask stream anywhere in the current architecture. If you want true joint image-mask co-generation, the redesign will need to introduce a second latent modality and corresponding tokenization, Transformer integration, and output head, rather than continuing to supervise anatomy only through a decoded RGB image and a frozen segmenter.
