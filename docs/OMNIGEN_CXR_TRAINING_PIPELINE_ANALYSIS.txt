# OmniGen CXR Training Pipeline Analysis for Anatomy-Aware Loss Integration

## Summary
This report maps the current SV-DRR multi-view CXR training stack in `/home/wenting/zr/gen_code` with emphasis on where an anatomy-aware auxiliary loss can later be attached. The analysis is based on local source code only and focuses on dataset schema, dataloader flow, rectified-flow loss computation, VAE behavior, LoRA scope, and runtime settings that affect memory risk.

Key takeaways:
- OmniGen is vendored locally under [`/home/wenting/zr/gen_code/OmniGen`](/home/wenting/zr/gen_code/OmniGen), even though pretrained weights are loaded from Hugging Face.
- Training currently stays in latent space after VAE encoding; there is no training-time decode path in `train.py`.
- The cleanest future `output_mask` path is: JSONL field -> `DatasetFromJson.get_example()` -> `TrainDataCollator.__call__()` -> `for data in loader` inside `train.py`.
- The primary training objective is a rectified-flow MSE between the model-predicted latent velocity and `u_t = x1 - x0`.

## 1. Repository Topology
The project contains a local OmniGen implementation rather than only wrapping an external Python package.

Evidence:
- [`/home/wenting/zr/gen_code/train.py:30-32`](/home/wenting/zr/gen_code/train.py#L30) imports `OmniGen`, `OmniGenProcessor`, `DatasetFromJson`, `TrainDataCollator`, and `training_losses` from local modules.
- [`/home/wenting/zr/gen_code/OmniGen/model.py:152-204`](/home/wenting/zr/gen_code/OmniGen/model.py#L152) defines the `OmniGen` model class and its `from_pretrained(...)` loader.

Training-relevant local modules:
- [`/home/wenting/zr/gen_code/train.py`](/home/wenting/zr/gen_code/train.py): main training entrypoint.
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py): dataset and collator used by training.
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py): rectified-flow training loss.
- [`/home/wenting/zr/gen_code/OmniGen/model.py`](/home/wenting/zr/gen_code/OmniGen/model.py): latent-prediction backbone.
- [`/home/wenting/zr/gen_code/OmniGen/processor.py`](/home/wenting/zr/gen_code/OmniGen/processor.py): multimodal prompt processing and collator helpers.
- [`/home/wenting/zr/gen_code/lanuch/train.sh`](/home/wenting/zr/gen_code/lanuch/train.sh): launch configuration.

Short source excerpt:

```python
from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper import training_losses
```

Source: [`/home/wenting/zr/gen_code/train.py:30-32`](/home/wenting/zr/gen_code/train.py#L30)

## 2. Dataset and JSONL Mechanics
### 2.1 JSONL generator and current schema
The JSONL generator is [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py). It builds training or test annotations by iterating over patients, choosing one fixed condition image per patient, and pairing that condition against every valid target view in `camera_views.json`.

The generated JSONL schema is:

```json
{
  "task_type": "image_edit",
  "instruction": "...",
  "input_images": ["<condition_image_path>"],
  "output_image": "<target_image_path>"
}
```

Evidence:
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:110-115`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L110)
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:123-125`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L123)

Short source excerpt:

```python
jsonl_entry = {
    "task_type": "image_edit",
    "instruction": instruction,
    "input_images": [cond_img_path],
    "output_image": target_img_path,
}
```

### 2.2 View pairing logic
The generator reads `camera_views.json`, filters views by orientation, and constructs an `id -> coordinate` dictionary.

Evidence:
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:49-60`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L49)

Each patient uses a fixed condition view:
- `cond_id = "0000"` at [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:73-74`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L73)

The generator then loops over every valid target view in `views_dict`, skipping only the condition view itself:
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:84-91`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L84)

The delta-pose prompt is encoded from:
- `d_theta = theta_cond - theta_target`
- `d_azimuth = (azimuth_cond - azimuth_target) % (2 * math.pi)`
- `sin(d_azimuth)`
- `cos(d_azimuth)`

Evidence:
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:95-108`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L95)

Short source excerpt:

```python
instruction = (
    f"<img><|image_1|></img> Edit the view using delta pose: "
    f"d_theta={d_theta:.4f} rad, "
    f"sin(d_azimuth)={sin_d_azimuth:.4f}, "
    f"cos(d_azimuth)={cos_d_azimuth:.4f}."
)
```

### 2.3 Path assumptions and dataset-root mismatch
The generator’s default paths still point to `/raid/...`, not the user-provided dataset root `/home/wenting/zr/wt_dataset/LIDC_IDRI/img_complex_fb_256`.

Evidence:
- Function defaults at [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:6-12`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L6)
- Main block at [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py:133-148`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py#L133)

This matters for architecture planning because any later `output_mask` convention should be aligned with the dataset layout actually used for the next training run, not just the historical `/raid/...` defaults.

### 2.4 How the dataloader reads JSONL records
Training uses `DatasetFromJson` in [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py).

The dataset:
- loads the JSONL file with `load_dataset('json', data_files=json_file)['train']`
- reads `instruction`, `input_images`, and `output_image` from each example
- optionally drops conditioning for classifier-free guidance
- transforms input images and output image into normalized tensors
- returns `(mllm_input, output_image)`

Evidence:
- Dataset load at [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:37-38`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L37)
- Example parsing at [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:46-59`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L46)

Short source excerpt:

```python
instruction, input_images, output_image = example['instruction'], example['input_images'], example['output_image']
...
mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images)
output_image = self.process_image(output_image)
return (mllm_input, output_image)
```

### 2.5 Processor and collator contract
`DatasetFromJson` delegates multimodal prompt packing to `OmniGenProcessor.process_multi_modal_prompt(...)`, which returns a dict with:
- `input_ids`
- `pixel_values`
- `image_sizes`

Evidence:
- [`/home/wenting/zr/gen_code/OmniGen/processor.py:57-61`](/home/wenting/zr/gen_code/OmniGen/processor.py#L57)

`TrainDataCollator` then:
- receives tuples from the dataset
- extracts `mllm_inputs` and `output_images`
- derives `target_img_size`
- calls `process_mllm_input(...)`
- returns a training batch dict containing model inputs plus `output_images`

Evidence:
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:82-111`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L82)
- Supporting processor-side packing at [`/home/wenting/zr/gen_code/OmniGen/processor.py:241-266`](/home/wenting/zr/gen_code/OmniGen/processor.py#L241)

Short source excerpt:

```python
data = {"input_ids": all_padded_input_ids,
"attention_mask": all_attention_mask,
"position_ids": all_position_ids,
"input_pixel_values": all_pixel_values,
"input_image_sizes": all_image_sizes,
"padding_images": all_padding_images,
"output_images": output_images,
}
```

### 2.6 Most likely `output_mask` injection seam
The current codebase has one straightforward propagation chain for a new target-side supervision tensor such as `output_mask`:

- Data hook: [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:46-59`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L46)
  - Read a new JSONL field and load the `.npz` mask here alongside `output_image`.
- Batch hook: [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:88-111`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L88)
  - Add the loaded mask tensor into the collated batch dict here.
- Loop hook: [`/home/wenting/zr/gen_code/train.py:194-233`](/home/wenting/zr/gen_code/train.py#L194)
  - Consume the batch mask in the training loop where `output_images` and loss are handled.

## 3. Training and Loss Path
### 3.1 Training entrypoint and model setup
`train.py` is the main training script. It creates:
- an `Accelerator`
- a local `OmniGen` model from pretrained weights
- an `AutoencoderKL` VAE
- an `OmniGenProcessor`
- a `DatasetFromJson`
- a `TrainDataCollator`
- a `DataLoader`

Evidence:
- Accelerator setup at [`/home/wenting/zr/gen_code/train.py:43-55`](/home/wenting/zr/gen_code/train.py#L43)
- Model/VAE/processor setup at [`/home/wenting/zr/gen_code/train.py:67-99`](/home/wenting/zr/gen_code/train.py#L67)
- Data setup at [`/home/wenting/zr/gen_code/train.py:125-153`](/home/wenting/zr/gen_code/train.py#L125)

### 3.2 VAE usage during training
The VAE is explicitly frozen:

```python
requires_grad(vae, False)
```

Source: [`/home/wenting/zr/gen_code/train.py:100`](/home/wenting/zr/gen_code/train.py#L100)

During training, both the target images and optional conditioning images are encoded inside `torch.no_grad()`:

```python
with torch.no_grad():
    output_images = data['output_images']
    input_pixel_values = data['input_pixel_values']
    ...
    output_images = vae_encode(...)
    input_pixel_values = vae_encode(...)
```

Evidence:
- [`/home/wenting/zr/gen_code/train.py:195-206`](/home/wenting/zr/gen_code/train.py#L195)

The encode helpers in [`/home/wenting/zr/gen_code/OmniGen/utils.py:94-109`](/home/wenting/zr/gen_code/OmniGen/utils.py#L94) only define `vae_encode` and `vae_encode_list`; there is no corresponding training helper for decode.

Conclusion:
- The current training loop is latent-space-only after encoding.
- Introducing an anatomy-aware segmentation loss would require an additional decode path if the segmentation network expects image-space tensors.

### 3.3 Rectified-flow loss computation
The actual diffusion-style training objective is implemented in [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py).

The loss helper:
- samples `x0` as Gaussian noise
- samples a scalar timestep `t`
- forms the interpolated latent `x_t`
- defines the rectified-flow target `u_t = x1 - x0`
- calls the model on `(x_t, t, **model_kwargs)`
- computes plain MSE between `model_output` and `u_t`

Evidence:
- [`/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py:23-71`](/home/wenting/zr/gen_code/OmniGen/train_helper/loss.py#L23)

Short source excerpt:

```python
x0 = sample_x0(x1)
t = sample_timestep(x1)
xt = t_ * x1 + (1 - t_) * x0
ut = x1 - x0
model_output = model(xt, t, **model_kwargs)
terms["loss"] = mean_flat(((model_output - ut) ** 2))
```

### 3.4 How `train.py` calls the loss
After VAE encoding, `train.py` assembles `model_kwargs` from the collated batch and calls `training_losses(...)`.

Evidence:
- [`/home/wenting/zr/gen_code/train.py:230-233`](/home/wenting/zr/gen_code/train.py#L230)

Short source excerpt:

```python
model_kwargs = dict(
    input_ids=data['input_ids'],
    input_img_latents=input_pixel_values,
    input_image_sizes=data['input_image_sizes'],
    attention_mask=data['attention_mask'],
    position_ids=data['position_ids'],
    padding_latent=data['padding_images'],
    past_key_values=None,
    return_past_key_values=False,
)
loss_dict = training_losses(model, output_images, model_kwargs)
loss = loss_dict["loss"].mean()
```

### 3.5 Exact model prediction site
`OmniGen.forward(...)` in [`/home/wenting/zr/gen_code/OmniGen/model.py:314-358`](/home/wenting/zr/gen_code/OmniGen/model.py#L314) is the model-side site where the predicted latent field is produced.

What it does:
- patchifies the noised latent target `x`
- patchifies conditioning image latents if present
- inserts image-latent tokens into the text embedding sequence
- appends a timestep token and latent tokens
- runs the internal Phi-3 transformer
- applies the final layer
- unpatchifies back to latent-shaped tensors

Short source excerpt:

```python
output = self.llm(inputs_embeds=input_emb, ...)
output, past_key_values = output.last_hidden_state, output.past_key_values
...
x = self.final_layer(image_embedding, time_emb)
latents = self.unpatchify(x, shapes[0], shapes[1])
```

Interpretation for future design:
- The returned `latents` from `OmniGen.forward(...)` are the predicted flow/velocity tensors used by `training_losses(...)`.

## 4. Model and Adaptation Stack
### 4.1 OmniGen architecture dependency
The repo contains the full model wrapper and major internals locally:
- [`/home/wenting/zr/gen_code/OmniGen/model.py`](/home/wenting/zr/gen_code/OmniGen/model.py)
- [`/home/wenting/zr/gen_code/OmniGen/transformer.py`](/home/wenting/zr/gen_code/OmniGen/transformer.py)
- [`/home/wenting/zr/gen_code/OmniGen/processor.py`](/home/wenting/zr/gen_code/OmniGen/processor.py)
- [`/home/wenting/zr/gen_code/OmniGen/scheduler.py`](/home/wenting/zr/gen_code/OmniGen/scheduler.py)

Pretrained weights and tokenizer/config artifacts are still loaded from a local model directory or downloaded Hugging Face snapshot:
- [`/home/wenting/zr/gen_code/OmniGen/model.py:190-204`](/home/wenting/zr/gen_code/OmniGen/model.py#L190)
- [`/home/wenting/zr/gen_code/OmniGen/processor.py:41-50`](/home/wenting/zr/gen_code/OmniGen/processor.py#L41)

### 4.2 LoRA configuration and scope
LoRA is configured only in `train.py`.

Evidence:
- [`/home/wenting/zr/gen_code/train.py:101-115`](/home/wenting/zr/gen_code/train.py#L101)

Short source excerpt:

```python
requires_grad(model, False)
transformer_lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_rank,
    init_lora_weights="gaussian",
    target_modules=["qkv_proj", "o_proj"],
)
model = get_peft_model(model, transformer_lora_config)
```

Implications:
- The base model is frozen first when `--use_lora` is enabled.
- The current setup restricts training to inserted LoRA adapters.
- The target modules are attention projection layers named `qkv_proj` and `o_proj`.
- Based on the training code, LoRA is applied to the transformer-backed model path, not to the VAE or any auxiliary image module.

## 5. Runtime and OOM-Relevant Settings
### 5.1 Distributed launch mode
The active launch script uses Accelerate directly:

```bash
sudo env CUDA_VISIBLE_DEVICES=2,3 ./python3 -m accelerate.commands.launch \
    --num_processes=2 \
    train.py \
```

Source: [`/home/wenting/zr/gen_code/lanuch/train.sh:13-15`](/home/wenting/zr/gen_code/lanuch/train.sh#L13)

There is no active DeepSpeed configuration in the inspected training path. `train.py` imports `DistributedType` and contains an FSDP guard for LoRA, but the launch script is plain Accelerate.

Evidence:
- [`/home/wenting/zr/gen_code/train.py:22`](/home/wenting/zr/gen_code/train.py#L22)
- [`/home/wenting/zr/gen_code/train.py:102-103`](/home/wenting/zr/gen_code/train.py#L102)

### 5.2 Current CXR run configuration
The launch script and the saved training args agree on the main settings for the representative CXR run:

- `batch_size_per_device=128`
- `mixed_precision=bf16`
- `gradient_accumulation_steps=1`
- `num_workers=2`
- `keep_raw_resolution=true`
- `max_image_size=1024`
- `use_lora=true`
- `lora_rank=8`

Evidence:
- Launch script: [`/home/wenting/zr/gen_code/lanuch/train.sh:16-32`](/home/wenting/zr/gen_code/lanuch/train.sh#L16)
- Saved args: [`/home/wenting/zr/gen_code/results/cxr_finetune_lora/train_args.json:1`](/home/wenting/zr/gen_code/results/cxr_finetune_lora/train_args.json#L1)

### 5.3 Mixed precision and gradient checkpointing
`train.py` defaults `--mixed_precision` to `bf16`:
- [`/home/wenting/zr/gen_code/train.py:376-385`](/home/wenting/zr/gen_code/train.py#L376)

The model enables gradient checkpointing on the internal transformer:
- [`/home/wenting/zr/gen_code/train.py:76`](/home/wenting/zr/gen_code/train.py#L76)

Short source excerpt:

```python
model = OmniGen.from_pretrained(args.model_name_or_path)
model.llm.config.use_cache = False
model.llm.gradient_checkpointing_enable()
```

### 5.4 Memory-relevant observations
Current memory pressure is dominated by:
- large per-device batch size `128`
- raw-resolution mode
- 1024 max image size
- multimodal token packing

Current mitigations already present:
- bf16 mixed precision
- LoRA instead of full-parameter tuning
- transformer gradient checkpointing
- frozen VAE with encode inside `torch.no_grad()`

Observation relevant to the next design phase:
- Any future training-time image decode plus segmentation-network forward/backward path would add memory beyond the current latent-only training path.

## 6. VAE Decode: Training vs Inference
There is no VAE decode path in `train.py` or `OmniGen/utils.py`.

Evidence:
- Training only calls `vae_encode(...)` and `vae_encode_list(...)` in [`/home/wenting/zr/gen_code/train.py:200-206`](/home/wenting/zr/gen_code/train.py#L200)
- Utility helpers only define encode routines in [`/home/wenting/zr/gen_code/OmniGen/utils.py:94-109`](/home/wenting/zr/gen_code/OmniGen/utils.py#L94)

Inference does decode latents back to images:
- [`/home/wenting/zr/gen_code/OmniGen/pipeline.py:294-300`](/home/wenting/zr/gen_code/OmniGen/pipeline.py#L294)

Short source excerpt:

```python
self.vae.to(self.device)
samples = samples.to(torch.float32)
...
samples = self.vae.decode(samples).sample
```

Conclusion:
- Training is encode-only and latent-space-only today.
- Decode currently belongs to inference-time generation, not the training loop.

## 7. Integration Readiness Notes
For the upcoming anatomy-aware loss design, the current codebase exposes three natural hook locations:

- Data hook:
  - [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:46-59`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L46)
  - This is where a new JSONL field such as `output_mask` would first be read and loaded.

- Batch hook:
  - [`/home/wenting/zr/gen_code/OmniGen/train_helper/data.py:88-111`](/home/wenting/zr/gen_code/OmniGen/train_helper/data.py#L88)
  - This is where the new tensor would be added to the batch dictionary so it reaches the loop.

- Loss hook:
  - [`/home/wenting/zr/gen_code/train.py:195-233`](/home/wenting/zr/gen_code/train.py#L195)
  - This is where encoded targets are prepared and the current rectified-flow loss is computed, making it the most direct place to combine the primary latent loss with any future auxiliary anatomy-aware term.

## Assumptions Used In This Report
- [`/home/wenting/zr/gen_code/results/cxr_finetune_lora/train_args.json`](/home/wenting/zr/gen_code/results/cxr_finetune_lora/train_args.json) is treated as the most representative completed CXR run configuration.
- [`/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py`](/home/wenting/zr/gen_code/gen_data/gen_train_test_jsonl.py) is treated as the authoritative JSONL generator for the current schema unless another production generator is later adopted.
