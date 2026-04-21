# Anatomy-Aware OmniGen Development Blueprint

## 1. Executive Summary
This document defines the implementation blueprint for adding an auxiliary Anatomy-Aware Loss to OmniGen fine-tuning for multi-view CXR generation on the SV-DRR dataset, while preserving the original OmniGen core training path.

The design follows three hard constraints:
- **Minimal intrusion**: do not modify the original OmniGen core entrypoints in a way that risks baseline regression. Duplicate training/loss scripts instead.
- **VRAM safety**: anatomy supervision must not decode and segment the full training batch. Use a decode-and-segment sub-batch.
- **Operational throughput**: pseudo-mask generation must scale across all 4 RTX A6000 GPUs.

The target result is a new training path that optimizes:

```python
loss_total = loss_diffusion + lambda_anatomy * loss_anatomy
```

where:
- `loss_diffusion` is OmniGen’s existing rectified-flow MSE loss in latent space.
- `loss_anatomy` is a BCE + Dice loss computed by a frozen ResNet34-UNet on decoded generated images versus precomputed pseudo-masks.

## 2. Grounded Environment Facts
### 2.1 OmniGen codebase
The current OmniGen training stack lives under:
- [`/home/wenting/zr/gen_code`](/home/wenting/zr/gen_code)

Relevant current files:
- [`/home/wenting/zr/gen_code/train.py`](/home/wenting/zr/gen_code/train.py)
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py)
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py)
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py)

### 2.2 SV-DRR image dataset
The target image dataset is:
- [`/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256`](/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256)

Observed layout:
- patient-folder structure such as:
  - [`/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0001`](/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0001)
- sample image:
  - [`/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0001/0000.png`](/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0001/0000.png)
- sample image metadata:
  - size `256x256`
  - PIL mode `L` (grayscale)

### 2.3 Frozen anatomy model contract
The frozen segmentation model contract is already defined in:
- [`/home/wenting/zr/Segmentation/train_anatomy.py`](/home/wenting/zr/Segmentation/train_anatomy.py)

Model definition:

```python
return smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=10,
    activation=None,
)
```

This implies:
- the segmentation model consumes **3-channel** image tensors
- the output is **10-channel raw logits**
- loss is multilabel, not single-class segmentation

### 2.4 Frozen checkpoint path
Use the current concrete checkpoint path as the default:
- [`/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth`](/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth)

The checkpoint format stores:
- `model_state_dict`
- `val_dice`
- `global_step`

## 3. Design Principles
### 3.1 Isolation strategy
Do not overwrite:
- [`/home/wenting/zr/gen_code/train.py`](/home/wenting/zr/gen_code/train.py)
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py)
- [`/home/wenting/zr/gen_code/OmniGen/model.py`](/home/wenting/zr/gen_code/OmniGen/model.py)

Instead create:
- `generate_pseudo_masks.py`
- `gen_mask_jsonl.py`
- `train_anatomy.py`
- `OmniGen/train_helper/loss_anatomy.py`

This keeps the baseline training path fully reproducible.

### 3.2 Memory strategy
Full-batch anatomy supervision is too expensive because it adds:
- predicted latent reconstruction
- differentiable VAE decode
- frozen UNet forward
- segmentation loss graph

Therefore:
- keep the existing diffusion loss on the full batch
- compute anatomy supervision on only a small random sub-batch per step

### 3.3 Data strategy
Pseudo-masks should be generated **offline** once, not on-the-fly during OmniGen training.

This reduces:
- runtime instability
- training-step latency
- repeated UNet inference
- multi-GPU training complexity

## 4. Phase A: Offline Pseudo-Mask Generation
## 4.1 Deliverables
Create a standalone script:
- `generate_pseudo_masks.py`

Create a JSONL companion script:
- `gen_mask_jsonl.py`

## 4.2 Purpose
Generate pseudo ground-truth anatomy masks for every rendered SV-DRR image:
- input: grayscale `.png` CXR images
- output: compressed `.npz` mask files with shape `(10, 256, 256)`

## 4.3 Output directory convention
Use a sibling directory tree to the current image dataset:

```text
/home/wenting/zr/wt_dataset/LIDC_IDRI/
├── img_complex_fb_256/
│   ├── LIDC-IDRI-0001/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
└── img_complex_fb_256_pseudomask_10ch/
    ├── LIDC-IDRI-0001/
    │   ├── 0000.npz
    │   ├── 0001.npz
    │   └── ...
```

Recommended default root:
- `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch`

This gives a deterministic 1:1 mapping between image path and mask path.

## 4.4 Shared path-mapping helper
Define a shared helper contract:

```python
def image_path_to_mask_path(
    image_path: str,
    image_root: str,
    mask_root: str,
    suffix: str = ".npz",
) -> str:
    """
    Map:
      /image_root/<patient>/<view>.png
    to:
      /mask_root/<patient>/<view>.npz
    """
```

Example:

```python
image_path = "/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256/LIDC-IDRI-0001/0000.png"
mask_path = "/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256_pseudomask_10ch/LIDC-IDRI-0001/0000.npz"
```

This helper must be reused by:
- `generate_pseudo_masks.py`
- `gen_mask_jsonl.py`
- future training sanity checks

## 4.5 Pseudo-mask storage format
Recommended disk format:

```python
np.savez_compressed(mask_path, mask=mask_bool)
```

Recommended on-disk dtype:
- `bool`

Recommended in-memory training dtype:
- `torch.float32`

Rationale:
- the segmentation target is multi-label and typically thresholded/binary
- boolean storage is significantly smaller than `float32`
- training losses expect floating tensors

Target tensor shape:
- `(10, 256, 256)`

## 4.6 Input preprocessing contract
The segmentation model expects 3-channel input, but the SV-DRR image files are grayscale (`L`).

Therefore `generate_pseudo_masks.py` must:
1. load grayscale PNG
2. convert it to a single tensor/image
3. replicate or convert to 3 channels
4. apply the **same normalization convention used by the anatomy segmentation project**

Important rule:
- do **not** invent a new normalization just because OmniGen uses `Normalize(mean=0.5, std=0.5)`.
- reuse the segmentation project’s preprocessing contract, or explicitly centralize it into one shared helper if it needs to be restated.

Recommended helper:

```python
def prepare_segmentation_input(pil_img) -> torch.Tensor:
    """
    Convert grayscale CXR PNG -> normalized 3-channel tensor expected by the
    frozen anatomy UNet.
    """
```

Pseudo-code:

```python
pil = Image.open(image_path).convert("L")
arr = np.array(pil)                      # (H, W)
arr3 = np.stack([arr, arr, arr], axis=-1)
tensor = seg_transform(image=arr3)["image"]   # (3, 256, 256)
```

## 4.7 Model loading contract
The standalone script should instantiate the frozen segmentation model with the exact architecture used in training:

```python
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=10,
    activation=None,
)
ckpt = torch.load(seg_model_ckpt, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
model.requires_grad_(False)
model.to(device)
```

Use:
- `encoder_weights=None` when loading for inference from checkpoint
- `torch.inference_mode()` for forward passes

## 4.8 Multi-GPU strategy
Use `torch.multiprocessing.spawn` with one process per GPU.

Recommended world size:
- `4`

High-level flow:

```python
def main():
    image_paths = sorted(all_pngs_under_root)
    mp.spawn(worker, nprocs=4, args=(image_paths, args))

def worker(rank, image_paths, args):
    torch.cuda.set_device(rank)
    shard = image_paths[rank::args.world_size]
    model = load_frozen_seg_model(...)
    loader = build_rank_local_loader(shard, batch_size=args.batch_size_per_gpu)
    for batch in loader:
        with torch.inference_mode():
            logits = model(images)
            probs = torch.sigmoid(logits)
            mask_bool = probs > args.threshold
            save_npz(...)
```

### Why `rank::world_size` is recommended
Either contiguous chunks or strided partitioning works, but `rank::world_size` is preferable here because:
- it is trivial to implement
- it remains stable if the input list is globally sorted
- it reduces the risk that one rank gets a pathological block with many corrupted files or unusually slow I/O

## 4.9 Worker responsibilities
Each rank should:
- bind to one GPU with `torch.cuda.set_device(rank)`
- load its own copy of the segmentation model
- process only its own shard
- log rank-local progress
- skip already-existing outputs

Recommended log prefix:

```python
print(f"[rank {rank}] processed={done}/{len(shard)} skipped={skipped} failed={failed}")
```

## 4.10 Dataloader inside pseudo-mask generation
Each worker should use a local DataLoader over only its assigned paths.

Recommended settings:
- `batch_size_per_gpu`: configurable, default chosen empirically
- `num_workers`: configurable per process
- `pin_memory=True`
- `shuffle=False`

Because this is offline inference:
- use `torch.inference_mode()`
- no gradient graph
- no need for Accelerate or DDP

## 4.11 Resume and idempotency behavior
The script must be resumable.

For each image:
- compute target `.npz` path
- if output exists and passes a light integrity check, skip it

Recommended integrity check:
- file exists
- `np.load(...)[key]` succeeds
- array shape is `(10, 256, 256)`

If integrity check fails:
- delete/overwrite stale file
- recompute

## 4.12 Atomic save strategy
Never write directly to the final path.

Use:
1. temp path in the same directory
2. `np.savez_compressed(temp_path, mask=mask_bool)`
3. `os.replace(temp_path, final_path)`

Pseudo-code:

```python
tmp_path = final_path + ".tmp"
np.savez_compressed(tmp_path, mask=mask_bool)
os.replace(tmp_path, final_path)
```

This avoids half-written files if the process dies mid-save.

## 4.13 Failure handling
If a single image fails:
- log the absolute path
- increment a rank-local failure counter
- continue processing

At the end:
- write a rank-local summary JSON or log line
- main process prints merged totals:
  - total images
  - generated
  - skipped
  - failed

## 4.14 `gen_mask_jsonl.py`
This script should be a clone/fork of the current JSONL generator logic in:
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py)

It must preserve:
- `task_type`
- `instruction`
- `input_images`
- `output_image`

and append:

```json
"output_mask": "/absolute/path/to/<patient>/<view>.npz"
```

Pseudo-code:

```python
jsonl_entry = {
    "task_type": "image_edit",
    "instruction": instruction,
    "input_images": [cond_img_path],
    "output_image": target_img_path,
    "output_mask": image_path_to_mask_path(
        target_img_path, image_root=args.image_root, mask_root=args.mask_root
    ),
}
```

### Integrity rule for JSONL generation
If the target `output_mask` file is missing:
- raise a hard error

Reason:
- training-set integrity matters more than silently producing partially supervised data
- missing masks should be fixed in Phase A, not tolerated downstream

## 5. Phase B: Minimal Dataloader Extension
## 5.1 Design goal
Touch only the narrow training-data boundary in:
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py)

Do not modify:
- `OmniGen/processor.py`
- multimodal prompt semantics
- model-side input formatting

## 5.2 `DatasetFromJson` changes
Current behavior:

```python
return (mllm_input, output_image)
```

New behavior:

```python
return (mllm_input, output_image, output_anatomy_mask)
```

Required additions:
- read `example["output_mask"]`
- load with `np.load(mask_path)["mask"]`
- cast to `torch.float32`

Recommended helper:

```python
def load_output_mask(mask_path: str, key: str = "mask") -> torch.Tensor:
    arr = np.load(mask_path)[key]              # (10, 256, 256)
    tensor = torch.from_numpy(arr).to(torch.float32)
    return tensor
```

Pseudo-code patch shape:

```python
def get_example(self, index):
    example = self.data[index]
    instruction = example["instruction"]
    input_images = example["input_images"]
    output_image = example["output_image"]
    output_mask = example["output_mask"]

    ...
    mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images)
    output_image = self.process_image(output_image)
    output_anatomy_mask = load_output_mask(output_mask)

    return (mllm_input, output_image, output_anatomy_mask)
```

## 5.3 `TrainDataCollator` changes
Current collator reads:
- `f[0]` -> multimodal input
- `f[1]` -> output image

New collator reads:
- `f[0]` -> multimodal input
- `f[1]` -> output image
- `f[2]` -> output anatomy mask

Add a new batch field:
- `output_anatomy_masks`

Recommended behavior:
- when `keep_raw_resolution=False`, stack into `(B, 10, H, W)`
- when `keep_raw_resolution=True`, still stack if shapes are uniform
- only fall back to a list if mixed shapes ever become possible

Given the current dataset is fixed at `256x256`, the blueprint should prefer:

```python
output_anatomy_masks = torch.stack([f[2] for f in features], dim=0)
```

Pseudo-code:

```python
def __call__(self, features):
    mllm_inputs = [f[0] for f in features]
    output_images = [f[1].unsqueeze(0) for f in features]
    output_anatomy_masks = torch.stack([f[2] for f in features], dim=0)
    ...
    data = {
        "input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "input_pixel_values": all_pixel_values,
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        "output_images": output_images,
        "output_anatomy_masks": output_anatomy_masks,
    }
    return data
```

## 5.4 Processor boundary remains unchanged
The processor contract in:
- [`/home/wenting/zr/gen_code/OmniGen/processor.py`](/home/wenting/zr/gen_code/OmniGen/processor.py)

does not need semantic changes because:
- `output_mask` is not part of prompt/token construction
- it is supervision-only data
- it belongs on the dataset/collator side, not the prompt-processing side

## 6. Phase C: Anatomy-Aware Training Path
## 6.1 New files
Create:
- [`/home/wenting/zr/gen_code/train_anatomy.py`](/home/wenting/zr/gen_code/train_anatomy.py)
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss_anatomy.py)

Do not overwrite:
- [`/home/wenting/zr/gen_code/train.py`](/home/wenting/zr/gen_code/train.py)
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py)

## 6.2 `train_anatomy.py` initialization
Start from the existing `train.py` flow and preserve:
- Accelerator setup
- model loading
- VAE loading
- LoRA setup
- optimizer
- scheduler
- dataset and collator wiring

Add new CLI arguments:

```python
parser.add_argument("--seg_model_ckpt", type=str,
    default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth")
parser.add_argument("--lambda_anatomy", type=float, default=0.1)
parser.add_argument("--anatomy_subbatch_size", type=int, default=4)
parser.add_argument("--pseudo_mask_key", type=str, default="mask")
```

### Load the frozen segmentation model
Use the exact architecture from the segmentation project:

```python
def get_frozen_anatomy_model(seg_model_ckpt):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    ckpt = torch.load(seg_model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model
```

Rules:
- keep it in `eval()`
- set `requires_grad_(False)`
- keep it in **fp32**

Important clarification:
- frozen does **not** mean wrapped in `torch.no_grad()` globally
- gradients should not update segmentation weights, but the forward graph can still participate in backpropagation through its inputs

## 6.3 Why a separate `loss_anatomy.py` is necessary
The current OmniGen loss helper only returns diffusion loss. The new path needs:
- diffusion loss
- predicted clean latent reconstruction
- random sub-batch selection
- differentiable decode
- frozen segmentation forward
- anatomy BCE + Dice
- final weighted combination

This is enough additional logic that a dedicated helper is cleaner and safer.

## 6.4 Rectified-flow interception logic
The current OmniGen training helper samples:
- `x0`
- `t`
- `x_t`
- `u_t = x1 - x0`

The anatomy-aware helper should preserve that exact baseline path, then intercept the predicted velocity field.

Pseudo-code:

```python
x0 = sample_x0(x1)
t = sample_timestep(x1)
t_ = t.view(B, 1, 1, 1)
x_t = t_ * x1 + (1 - t_) * x0
u_t = x1 - x0

u_hat = model(x_t, t, **model_kwargs)
loss_diffusion = mean_flat((u_hat - u_t) ** 2).mean()
```

## 6.5 Predicted clean latent reconstruction
Under rectified flow:

```python
x_t = t * x1 + (1 - t) * x0
u_hat ≈ x1 - x0
```

So a predicted clean latent can be reconstructed as:

```python
x1_hat = x_t + (1 - t) * u_hat
```

With broadcasting:

```python
t_ = t.view(B, 1, 1, 1)
x1_hat = x_t + (1 - t_) * u_hat
```

This `x1_hat` is the latent estimate that should be decoded for anatomy supervision.

## 6.6 Sub-batching policy
Do **not** decode the full batch.

At each training step:
1. let `B` be current batch size
2. choose `n = min(anatomy_subbatch_size, B)`
3. sample `n` unique indices
4. slice both predicted latents and target masks

Pseudo-code:

```python
n = min(anatomy_subbatch_size, x1_hat.shape[0])
idx = torch.randperm(x1_hat.shape[0], device=x1_hat.device)[:n]
x1_hat_sub = x1_hat[idx]
mask_sub = output_anatomy_masks[idx]
```

Properties:
- diffusion loss still uses full batch
- only `x1_hat_sub` is decoded
- only `mask_sub` participates in anatomy loss

## 6.7 Inverse VAE scaling
The current training path encodes images with VAE scaling. Before decode, reverse that transform using the same logic as OmniGen inference.

Recommended helper:

```python
def inverse_vae_scale(latents, vae):
    if vae.config.shift_factor is not None:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor
    return latents
```

Then:

```python
x1_hat_sub_scaled = inverse_vae_scale(x1_hat_sub.float(), vae)
decoded = vae.decode(x1_hat_sub_scaled).sample
```

## 6.8 Critical warning: do not use `torch.no_grad()` around VAE decode
This is the most important implementation trap.

### What must happen
The anatomy loss must backpropagate to OmniGen through:

```text
loss_anatomy
 -> segmentation logits
 -> decoded image
 -> decoded latent x1_hat_sub
 -> predicted velocity u_hat
 -> OmniGen LoRA parameters
```

### Why `torch.no_grad()` is wrong here
If decode is wrapped in:

```python
with torch.no_grad():
    decoded = vae.decode(x1_hat_sub_scaled).sample
```

then autograd will stop at the decoded image tensor. The computation graph between:
- `decoded`
- `x1_hat_sub_scaled`
- `x1_hat_sub`
- `u_hat`

is severed.

That would make anatomy loss numerically measurable but **non-trainable** with respect to OmniGen.

### Correct mental model
These two statements are both true:
- VAE parameters should be frozen.
- Gradients must still flow **through** the VAE operations into the latent input.

Freezing weights means:

```python
for p in vae.parameters():
    p.requires_grad = False
```

It does **not** mean:

```python
torch.no_grad()
```

for the decode branch.

### Practical rule
- Keep VAE parameters frozen.
- Keep VAE in fp32.
- Run `vae.decode(...)` with autograd enabled.

## 6.9 Why VAE should stay in fp32 for decode branch
The decode branch is numerically sensitive because:
- it converts latent predictions into image-space supervision
- those decoded pixels become inputs to a second network
- gradients must remain stable over two model stacks

Therefore:
- keep the VAE module in `float32`
- cast only the selected sub-batch latents to fp32 before decode

Pseudo-code:

```python
vae.to(dtype=torch.float32)
x1_hat_sub_scaled = x1_hat_sub_scaled.to(torch.float32)
decoded = vae.decode(x1_hat_sub_scaled).sample
```

This reduces the risk of:
- NaN gradients
- Inf activations
- unstable mixed-precision decode behavior

## 6.10 Preparing decoded images for the segmentation model
The segmentation model was trained with its own preprocessing convention. The OmniGen VAE decode output must be converted to the range expected by that model.

Recommended helper:

```python
def prepare_segmentation_input(decoded_images: torch.Tensor) -> torch.Tensor:
    """
    Convert decoded OmniGen output into normalized 3-channel tensors expected
    by the frozen anatomy UNet.
    """
```

The helper should:
1. assume decoded images are already 3-channel
2. convert them to the correct numeric range
3. apply the segmentation normalization convention

Important note:
- if segmentation training used `[0, 1]` images plus a specific normalization, reproduce exactly that same contract
- do not silently reuse OmniGen’s training normalization unless the two pipelines are confirmed to match

Pseudo-code:

```python
decoded = decoded.clamp(-1, 1)
decoded_01 = (decoded + 1.0) / 2.0
seg_input = seg_normalize(decoded_01)
```

If the segmentation project used a different convention, centralize it here.

## 6.11 Frozen segmentation forward and anatomy loss
Feed the prepared image sub-batch through the frozen anatomy model:

```python
logits = seg_model(seg_input_sub)
loss_bce = bce_loss(logits, mask_sub)
loss_dice = dice_loss(logits, mask_sub)
loss_anatomy = 0.5 * loss_bce + 0.5 * loss_dice
```

Recommended loss objects:

```python
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
```

Target mask dtype:
- `torch.float32`

Target mask shape:
- `(n, 10, 256, 256)`

## 6.12 Combined loss helper contract
`loss_anatomy.py` should export one primary helper, for example:

```python
def training_losses_with_anatomy(
    model,
    x1,
    model_kwargs,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy,
    anatomy_subbatch_size,
):
    ...
    return {
        "loss_total": loss_total,
        "loss_diffusion": loss_diffusion,
        "loss_anatomy": loss_anatomy,
        "anatomy_subbatch_size_actual": n,
    }
```

This is the cleanest integration surface for `train_anatomy.py`.

## 6.13 Training loop integration in `train_anatomy.py`
Inside the duplicated training loop:
- keep the original latent-encoding block
- then call the new anatomy-aware loss helper instead of the original `training_losses(...)`

Pseudo-code:

```python
with accelerator.accumulate(model):
    with torch.no_grad():
        output_images = data["output_images"]
        input_pixel_values = data["input_pixel_values"]
        output_images = vae_encode(...)
        input_pixel_values = vae_encode(...)

    loss_dict = training_losses_with_anatomy(
        model=model,
        x1=output_images,
        model_kwargs=model_kwargs,
        output_anatomy_masks=data["output_anatomy_masks"].to(device),
        vae=vae,
        seg_model=seg_model,
        lambda_anatomy=args.lambda_anatomy,
        anatomy_subbatch_size=args.anatomy_subbatch_size,
    )

    loss = loss_dict["loss_total"]
    accelerator.backward(loss)
```

Key subtlety:
- the initial VAE encode of ground-truth target images can remain under `torch.no_grad()`
- the later decode of `x1_hat_sub` must not be under `torch.no_grad()`

## 6.14 Logging and monitoring
Log at least:
- `loss_total`
- `loss_diffusion`
- `loss_anatomy`
- `anatomy_subbatch_size_actual`

Recommended optional debug stats:
- decoded image mean/std
- segmentation logits min/max
- fraction of positive mask pixels in `mask_sub`

Pseudo-code:

```python
accelerator.log({
    "loss_total": loss_total.item(),
    "loss_diffusion": loss_diffusion.item(),
    "loss_anatomy": loss_anatomy.item(),
    "anatomy_subbatch_size_actual": n,
}, step=train_steps)
```

## 7. Locked Public Interfaces
### 7.1 New JSONL field

```json
"output_mask": "/absolute/path/to/<patient>/<view>.npz"
```

### 7.2 New batch field

```python
data["output_anatomy_masks"]
```

### 7.3 New CLI arguments

```python
--seg_model_ckpt
--lambda_anatomy
--anatomy_subbatch_size
--pseudo_mask_key
```

### 7.4 New helper names
The implementation should define and centralize the following helpers:
- `image_path_to_mask_path(...)`
- `load_output_mask(...)`
- `inverse_vae_scale(...)`
- `prepare_segmentation_input(...)`
- `compute_anatomy_loss(...)`

## 8. Test Plan
## 8.1 Offline pseudo-mask generation
- Single-GPU dry run on one patient folder only.
- 4-GPU production run where each rank processes only its shard.
- Resume test:
  - generate a subset
  - rerun
  - verify existing files are skipped
- Validate one saved `.npz`:
  - key `mask` exists
  - shape `(10, 256, 256)`
  - dtype decodes correctly

## 8.2 JSONL generation
- `gen_mask_jsonl.py` keeps `input_images` and `output_image` unchanged.
- Each record includes `output_mask`.
- Every `output_mask` path exists on disk.
- Missing mask file raises hard error instead of silently emitting bad training records.

## 8.3 Dataloader tests
- One sample returns:

```python
(mllm_input, output_image, output_anatomy_mask)
```

- One collated batch includes:

```python
data["output_anatomy_masks"].shape == (B, 10, 256, 256)
data["output_anatomy_masks"].dtype == torch.float32
```

## 8.4 Anatomy-aware training dry run
- Run with tiny batch and `anatomy_subbatch_size=1`.
- Assert:
  - segmentation model params `requires_grad=False`
  - VAE params `requires_grad=False`
  - decoded image tensor `requires_grad=True`
- After backward:
  - OmniGen/LoRA trainable params get gradients
  - segmentation params do not
  - VAE params do not

## 8.5 Numerical stability tests
- Check for NaNs/Infs in:
  - `x1_hat_sub`
  - decoded images
  - segmentation logits
  - `loss_anatomy`
- Compare memory with:
  - anatomy branch disabled
  - anatomy branch enabled with `subbatch_size=4`
- Verify only sub-batch decode occurs, not full-batch decode.

## 8.6 Behavioral baseline test
- Set `lambda_anatomy=0.0`.
- Verify the duplicated training path numerically behaves like baseline OmniGen training, aside from extra logging and batch fields.

## 9. Implementation Notes and Recommended Defaults
### 9.1 Recommended defaults
- `seg_model_ckpt`:
  - `/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth`
- `lambda_anatomy = 0.1`
- `anatomy_subbatch_size = 4`
- pseudo-mask storage key:
  - `mask`
- on-disk mask dtype:
  - `bool`
- training mask dtype:
  - `float32`

### 9.2 Why `subbatch_size=4` is the initial safe choice
Even on 48 GB GPUs, the anatomy branch adds:
- fp32 latent decode
- decoded image activation graph
- frozen UNet forward graph through inputs

So `4` is a conservative starting point. Increase only after profiling.

### 9.3 Compatibility note on older docs
Some older segmentation notes reference `train_Seg_v2`, but the currently existing concrete checkpoint path in the environment is `train_Seg`. The OmniGen-side blueprint should follow the real current checkpoint unless the team explicitly promotes a newer artifact.

## 10. Final Build Sequence
Recommended implementation order:
1. Build and validate `generate_pseudo_masks.py`
2. Build `gen_mask_jsonl.py`
3. Extend `DatasetFromJson` and `TrainDataCollator`
4. Duplicate `train.py` -> `train_anatomy.py`
5. Implement `loss_anatomy.py`
6. Run tiny dry run with `lambda_anatomy=0.0`
7. Run tiny dry run with `lambda_anatomy>0`
8. Profile VRAM and tune `anatomy_subbatch_size`

This ordering minimizes risk and isolates bugs by stage.
