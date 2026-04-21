# IMPLEMENTATION SUMMARY REPORT

## Status

The joint image-mask co-generation implementation has now been debugged successfully in both single-GPU and multi-GPU settings.

Validated end to end:

1. Single-step training forward + backward + optimizer step
2. Custom checkpoint saving:
   - PEFT adapter checkpoint
   - `mask_modules.bin`
3. Inference generation loop
4. Predicted mask decoding and saving

All of the above completed without runtime exceptions in the real `omnigen` environment.

Additionally validated:

5. Two-process distributed training via `accelerate launch --num_processes 2`
6. Gradient accumulation with `gradient_accumulation_steps=3`
7. Two optimizer steps without DDP reducer errors

---

## 1. Debugging Summary

## 1.0 Final DDP root cause

The reported multi-GPU crash

```text
RuntimeError: Expected to mark a variable ready only once.
... down_proj.lora_B.default.weight has been marked as ready twice
```

was reproduced exactly with a real two-process launch.

### First-principles diagnosis against the 4 candidate causes

1. Multiple forward passes:
   - Not the cause.
   - [loss_joint_mask.py](/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_joint_mask.py) invokes `model(...)` exactly once per training iteration.
2. CFG / unconditional second forward:
   - Not the cause.
   - There is no classifier-free guidance training branch or second unconditional forward in `train_joint_mask.py`.
3. Improper gradient accumulation:
   - Not the cause.
   - The forward, backward, clip, optimizer step, and scheduler step are already wrapped inside `with accelerator.accumulate(model):` in [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py).
4. Gradient checkpointing + PEFT + DDP interaction:
   - This was the actual root cause.
   - Specifically, `find_unused_parameters=True` in DDP interacted badly with LoRA adapter weights under gradient checkpointing, causing duplicate reducer-hook firing on LoRA parameters such as `down_proj.lora_B.default.weight`.

### Verified fix

In [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py), the DDP config was changed from:

```python
kwargs = DDPK(find_unused_parameters=True)
```

to:

```python
kwargs = DDPK(find_unused_parameters=False)
```

Why this is correct:

- the joint training graph is static
- the image branch, mask branch, and LoRA-adapted Transformer are all used every iteration
- the two-process repro emitted PyTorch’s own warning that no unused parameters were found
- removing the extra unused-parameter traversal eliminated the duplicate ready-hook error

The training graph remained stable with:

- gradient checkpointing still enabled
- PEFT LoRA still enabled
- `model.llm.enable_input_require_grads()` still enabled

## 1.1 Training failure found and fixed

### Error 1: TensorBoard tracker config rejected list-valued args

Observed traceback:

```text
ValueError: value should be one of int, float, str, bool, or torch.Tensor
```

Root cause:

- `accelerator.init_trackers(..., config=args.__dict__)` passed list-valued args such as `lora_target_modules`
- TensorBoard hparams only accepts scalar-like values

Fix:

- added `_sanitize_tracker_config(...)` in [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py)
- lists/tuples are converted to comma-separated strings before tracker initialization

### Error 2: List-based image tensors were not moved to the VAE device before encoding

Observed during profiling:

```text
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

Root cause:

- under `--keep_raw_resolution`, `output_images` and `input_pixel_values` remain Python lists
- the list elements were being passed to `vae_encode_list(...)` without first moving them onto the same device as the VAE

Fix:

- patched [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py)
- patched [profiler.py](/home/wenting/zr/gen_code_plan2/profiler.py)
- list elements are now explicitly moved to `device` in `float32` before VAE encode

### Error 3: DDP reducer crash with LoRA + gradient checkpointing

Observed traceback:

```text
RuntimeError: Expected to mark a variable ready only once.
Parameter ... base_model.model.llm.layers.31.mlp.down_proj.lora_B.default.weight has been marked as ready twice.
```

Root cause:

- DDP was configured with `find_unused_parameters=True`
- the model also used:
  - LoRA adapters
  - Transformer gradient checkpointing
  - PEFT input-grad enabling
- this combination caused duplicate reducer-hook accounting on LoRA parameters during backward

Fix:

- patched [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py)
- changed DDP to `find_unused_parameters=False`
- preserved the one-forward training path in [loss_joint_mask.py](/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_joint_mask.py)
- added an explicit runtime check that joint training returns `(pred_img, pred_mask)` from one model call

## 1.2 Inference status

Inference did not require a code fix after the training-side hardening. The following paths were validated successfully:

- plain image generation dry run
- full inference with:
  - saved PEFT adapter
  - saved `mask_modules.bin`
  - mask decoding and `.npz` saving

---

## 2. Dry Run Validation

## 2.1 Training dry run

Executed successfully:

```bash
/home/wenting/miniconda3/envs/omnigen/bin/python /home/wenting/zr/gen_code_plan2/train_joint_mask.py \
  --model_name_or_path Shitao/OmniGen-v1 \
  --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
  --image_path ./ \
  --results_dir /home/wenting/zr/gen_code_plan2/tmp_dryrun_train \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 1 \
  --epochs 1 \
  --max_train_steps 1 \
  --num_workers 0 \
  --log_every 1 \
  --ckpt_every 100 \
  --lr 1e-5 \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_target_modules qkv_proj o_proj gate_up_proj down_proj \
  --lambda_mask 0.25 \
  --mask_latent_channels 4 \
  --keep_raw_resolution \
  --max_image_size 256 \
  --condition_dropout_prob 0.0 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --report_to tensorboard \
  --mixed_precision bf16
```

Observed successful log:

```text
(step=0000001) Loss: 0.9915 (img=0.6955, mask=1.1843)
Done!
```

## 2.2 Checkpoint dry run

Executed successfully with forced checkpoint save on step 1.

Observed save results:

- accelerator state:
  - `checkpoints/checkpoint-1/`
- PEFT adapter:
  - `checkpoints/0000001/adapter_model.safetensors`
- custom mask module checkpoint:
  - `checkpoints/0000001/mask_modules.bin`

Confirmed checkpoint tree:

```text
checkpoints/
  0000001/
    adapter_config.json
    adapter_model.safetensors
    mask_modules.bin
  checkpoint-1/
    model.safetensors
    optimizer.bin
    scheduler.bin
    random_states_0.pkl
    ...
```

## 2.3 Multi-GPU DDP dry run

Executed successfully on 2 GPUs with 3 accumulation steps and 2 optimizer steps:

```bash
cd /home/wenting/zr/gen_code_plan2
CUDA_VISIBLE_DEVICES=0,1 /home/wenting/miniconda3/envs/omnigen/bin/accelerate launch \
  --num_processes 2 \
  train_joint_mask.py \
  --model_name_or_path Shitao/OmniGen-v1 \
  --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
  --image_path ./ \
  --results_dir /home/wenting/zr/gen_code_plan2/tmp_ddp_repro \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 3 \
  --epochs 1 \
  --max_train_steps 2 \
  --num_workers 0 \
  --log_every 1 \
  --ckpt_every 100 \
  --lr 1e-5 \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_target_modules qkv_proj o_proj gate_up_proj down_proj \
  --lambda_mask 0.25 \
  --mask_latent_channels 4 \
  --keep_raw_resolution \
  --max_image_size 256 \
  --condition_dropout_prob 0.0 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --report_to tensorboard \
  --mixed_precision bf16
```

Observed successful log:

```text
(step=0000001) Loss: 0.8640 (img=0.5527, mask=1.2454)
(step=0000002) Loss: 0.8428 (img=0.5410, mask=1.2075)
Done!
```

## 2.4 Inference dry run

Executed successfully:

```bash
/home/wenting/miniconda3/envs/omnigen/bin/python /home/wenting/zr/gen_code_plan2/test_joint_mask.py \
  --model_path Shitao/OmniGen-v1 \
  --lora_path /home/wenting/zr/gen_code_plan2/tmp_dryrun_ckpt/checkpoints/0000001 \
  --mask_modules_path /home/wenting/zr/gen_code_plan2/tmp_dryrun_ckpt/checkpoints/0000001/mask_modules.bin \
  --mask_latent_channels 4 \
  --jsonl_path /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl \
  --output_dir /home/wenting/zr/gen_code_plan2/tmp_dryrun_test_with_mask \
  --batch_size 1 \
  --num_gpus 1 \
  --inference_steps 2 \
  --guidance_scale 2.5 \
  --img_guidance_scale 2.0 \
  --seed 42 \
  --save_masks \
  --mask_threshold 0.0 \
  --max_samples 1
```

Observed success:

```text
[GPU 0] Done: 1/1 success
All done.
```

Confirmed outputs:

- generated image:
  - `/home/wenting/zr/gen_code_plan2/tmp_dryrun_test_with_mask/LIDC-IDRI-0251/0001.png`
- predicted mask:
  - `/home/wenting/zr/gen_code_plan2/tmp_dryrun_test_with_mask/masks/LIDC-IDRI-0251/0001.npz`

Verified predicted mask file:

- key: `mask`
- shape: `(10, 256, 256)`
- dtype: `float32`
- min/max: `0.0 / 1.0`

---

## 3. Mathematical and Data Flow Finalization

## 3.1 Image path

### Raw image

- dataset image file:
  - grayscale PNG `256 x 256`
- loader converts to RGB:
  - shape `(3, 256, 256)`

### Image latent

- frozen image VAE encodes:
  - `(3, 256, 256) -> (4, 32, 32)`

### Image tokenization

- `PatchEmbedMR` uses:
  - `patch_size = 2`
  - `in_chans = 4`
- latent `(4, 32, 32)` becomes:
  - token grid `16 x 16`
  - `256` image tokens

So the image path is:

```text
Image PNG
  -> RGB tensor (3, 256, 256)
  -> VAE latent (4, 32, 32)
  -> 256 image tokens
```

## 3.2 Mask path

### Raw mask

- GT or predicted mask tensor:
  - `(10, 256, 256)`

### Training-time mask normalization

- binary `{0,1}` is mapped to `[-1,1]`:

```text
m_cont = 2 * m_bin - 1
```

### Mask latent

- `MaskEncoder` maps:
  - `(10, 256, 256) -> (4, 32, 32)`

### Mask tokenization

- dedicated `mask_x_embedder` also uses:
  - `patch_size = 2`
  - `in_chans = 4`
- mask latent `(4, 32, 32)` becomes:
  - `256` mask tokens

So the mask path is:

```text
Mask tensor
  -> [-1, 1]
  -> MaskEncoder latent (4, 32, 32)
  -> 256 mask tokens
```

## 3.3 Transformer sequence assembly

Final sequence order:

```text
[condition tokens] [time token] [image tokens] [mask tokens]
```

For the validated one-sample dry run:

- `input_ids`: `(1, 327)`
- `time token`: `1`
- image tokens: `256`
- mask tokens: `256`
- total Transformer sequence length:
  - `327 + 1 + 256 + 256 = 840`

Validated tensor shapes:

- `position_ids`: `(1, 840)`
- `attention_mask`: `(1, 840, 840)`

## 3.4 Output split

The Transformer output suffix is split as:

```text
[image token outputs] [mask token outputs]
```

Then:

- image suffix -> `final_layer` -> unpatchify -> predicted image velocity latent
- mask suffix -> `mask_final_layer` -> unpatchify -> predicted mask velocity latent

## 3.5 Training objective

For shared timestep `t`:

```text
x_t_img  = t * x_img  + (1 - t) * x0_img
x_t_mask = t * x_mask + (1 - t) * x0_mask
```

Velocity targets:

```text
u_img  = x_img  - x0_img
u_mask = x_mask - x0_mask
```

Losses:

```text
L_img  = ||v_img_theta  - u_img||^2
L_mask = ||v_mask_theta - u_mask||^2
L_total = L_img + lambda_mask * L_mask
```

---

## 4. Hyperparameters and LoRA Settings

## 4.1 Profiled `lambda_mask`

Measured on one real batch:

- `L_img = 0.296875`
- `L_mask = 1.2421875`
- ratio `L_img / L_mask ≈ 0.239`

Recommended and adopted starting value:

- `lambda_mask = 0.25`

Reason:

- balances the weighted mask contribution to the same order as image loss

## 4.2 Mask autoencoder profiling result

Single-batch probe:

- initial recon MSE: `1.3046875`
- final recon MSE after 25 probe steps: `0.21484375`
- improvement: `83.53%`

Conclusion:

- Phase 0 pretraining is useful but not strictly required
- joint training from scratch is viable with this lightweight encoder/decoder

## 4.3 LoRA configuration

Final recommended LoRA setup:

- `rank = 32`
- `alpha = 32`
- `target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`

These module names were validated directly in the real OmniGen/Phi-3 environment.

## 4.4 DDP setting

Final distributed-training setting:

- `find_unused_parameters = False`

Reason:

- all trainable branches are exercised in every iteration
- this avoids the duplicate reducer-hook issue seen with LoRA + gradient checkpointing

---

## 5. Custom Checkpointing Mechanism

## 5.1 Why custom checkpointing is required

PEFT only saves adapter weights by default.

It does not automatically save the newly introduced joint-mask modules:

- `MaskEncoder`
- `MaskDecoder`
- `mask_x_embedder`
- `mask_final_layer`
- `image_modality_embed`
- `mask_modality_embed`

Therefore a custom save/load path is required.

## 5.2 What is saved

In [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py):

- PEFT adapter checkpoint is saved to:
  - `checkpoints/<step>/adapter_model.safetensors`
- custom joint-mask weights are saved to:
  - `checkpoints/<step>/mask_modules.bin`

`mask_modules.bin` contains:

- `mask_encoder.*`
- `mask_decoder.*`
- `mask_x_embedder.*`
- `mask_final_layer.*`
- `image_modality_embed`
- `mask_modality_embed`

## 5.3 How loading works

### During training resume

- explicit path:
  - `--mask_modules_resume_path`
- automatic companion discovery:
  - if `--lora_resume_path <dir>` is given and `<dir>/mask_modules.bin` exists, it is loaded automatically
- if resuming from `accelerate` state via `--resume_from_checkpoint`, the code also looks for the sibling step folder and loads `mask_modules.bin` when found

### During inference

In [test_joint_mask.py](/home/wenting/zr/gen_code_plan2/test_joint_mask.py):

1. load base OmniGen
2. call `init_mask_modules(...)`
3. merge LoRA from `--lora_path`
4. load `mask_modules.bin` from `--mask_modules_path`
5. attach `MaskEncoder` / `MaskDecoder`

This gives a complete joint model for:

- image generation
- mask generation
- mask decoding

---

## 6. Final Code Files

Implemented / validated files:

- [OmniGen/mask_autoencoder.py](/home/wenting/zr/gen_code_plan2/OmniGen/mask_autoencoder.py)
- [OmniGen/model.py](/home/wenting/zr/gen_code_plan2/OmniGen/model.py)
- [OmniGen/scheduler.py](/home/wenting/zr/gen_code_plan2/OmniGen/scheduler.py)
- [OmniGen/pipeline.py](/home/wenting/zr/gen_code_plan2/OmniGen/pipeline.py)
- [OmniGen/train_helper/loss_joint_mask.py](/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/loss_joint_mask.py)
- [OmniGen/train_helper/data.py](/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/data.py)
- [OmniGen/train_helper/__init__.py](/home/wenting/zr/gen_code_plan2/OmniGen/train_helper/__init__.py)
- [train_joint_mask.py](/home/wenting/zr/gen_code_plan2/train_joint_mask.py)
- [test_joint_mask.py](/home/wenting/zr/gen_code_plan2/test_joint_mask.py)
- [launch/train_joint_mask.sh](/home/wenting/zr/gen_code_plan2/launch/train_joint_mask.sh)
- [launch/test_joint_mask.sh](/home/wenting/zr/gen_code_plan2/launch/test_joint_mask.sh)
- [profiler.py](/home/wenting/zr/gen_code_plan2/profiler.py)

---

## 7. Execution Guide

## 7.1 Training

Recommended command:

```bash
cd /home/wenting/zr/gen_code_plan2
bash launch/train_joint_mask.sh
```

The launcher now uses:

- real manifest path:
  - `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl`
- default profiled mask weight:
  - `lambda_mask=0.25`

For explicit multi-GPU launch, the verified distributed command is:

```bash
cd /home/wenting/zr/gen_code_plan2
CUDA_VISIBLE_DEVICES=0,1 /home/wenting/miniconda3/envs/omnigen/bin/accelerate launch \
  --num_processes 2 \
  train_joint_mask.py \
  --model_name_or_path Shitao/OmniGen-v1 \
  --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
  --image_path ./ \
  --results_dir results/joint_mask_cogen \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 3 \
  --epochs 1 \
  --max_train_steps 2 \
  --num_workers 0 \
  --log_every 1 \
  --ckpt_every 100 \
  --lr 1e-5 \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_target_modules qkv_proj o_proj gate_up_proj down_proj \
  --lambda_mask 0.25 \
  --mask_latent_channels 4 \
  --keep_raw_resolution \
  --max_image_size 256 \
  --condition_dropout_prob 0.0 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --report_to tensorboard \
  --mixed_precision bf16
```

## 7.2 Testing / inference

Recommended command:

```bash
cd /home/wenting/zr/gen_code_plan2
bash launch/test_joint_mask.sh
```

The launcher expects:

- base model path
- LoRA adapter directory
- `mask_modules.bin`
- test JSONL:
  - `/home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_test.jsonl`

## 7.3 Where outputs are written

### Training outputs

Under:

- `results/joint_mask_cogen/`

Important subpaths:

- logs:
  - `results/joint_mask_cogen/log.txt`
- tensorboard:
  - `results/joint_mask_cogen/tensorboard_log/`
- PEFT adapters:
  - `results/joint_mask_cogen/checkpoints/<step>/adapter_model.safetensors`
- custom mask modules:
  - `results/joint_mask_cogen/checkpoints/<step>/mask_modules.bin`

### Inference outputs

Under:

- `results/joint_mask_inference/`

Generated images:

- `results/joint_mask_inference/<patient_id>/<view_id>.png`

Generated masks:

- `results/joint_mask_inference/masks/<patient_id>/<view_id>.npz`

The `.npz` contains:

- key: `mask`
- tensor shape: `(10, 256, 256)`

---

## 8. Final Recommendations

Use this as the default serious training setup:

- `lambda_mask = 0.25`
- `lora_rank = 32`
- `lora_alpha = 32`
- `lora_target_modules = qkv_proj o_proj gate_up_proj down_proj`
- `mask_latent_channels = 4`

This configuration is now:

- mathematically aligned with the joint-distribution design
- empirically profiled on real data
- dry-run validated for:
  - training
  - checkpointing
  - inference
  - mask byproduct saving

---

## 9. Final Outcome

The implementation now satisfies the project requirements:

1. joint image-mask co-generation in latent space
2. retention and saving of predicted masks during inference
3. empirically profiled `lambda_mask`
4. expanded LoRA over attention + MLP
5. robust save/load of all new modules via `mask_modules.bin`
6. successful dry-run debugging of both training and inference

The system is ready for multi-step training runs and larger-scale evaluation.
