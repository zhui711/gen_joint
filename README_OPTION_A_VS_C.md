# README: Option A vs Option C

This guide explains how to run the two supported training modes in the unified codebase.

The key rule is:

- **Option A is the default production path**
- **Option C is experimental**
- **Inference is identical for both**

## 1. Option A: The Default SOTA Path

This is the recommended path when image-generation metrics are the top priority.

### Step 1: Run joint training with reconstruction disabled

Use:

- [launch/train_joint_mask.sh](/home/wenting/zr/gen_code_plan2_1/launch/train_joint_mask.sh:1)

Important:

- confirm the script contains `--lambda_recon 0.0`

Then run:

```bash
bash launch/train_joint_mask.sh
```

This keeps the main joint-training objective mathematically identical to the current best-performing run.

### Step 2: Post-train only the decoder

After the joint run finishes, train only the `MaskDecoder` using the saved `mask_modules.bin`.

Example command:

```bash
CKPT_DIR="results/joint_mask_cogen/checkpoints/0030000"

CUDA_VISIBLE_DEVICES=2 /home/wenting/miniconda3/envs/omnigen/bin/python train_mask_decoder_only.py \
  --mask_modules_path "${CKPT_DIR}/mask_modules.bin" \
  --json_file /home/wenting/zr/wt_dataset/LIDC_IDRI/anno/cxr_synth_anno_mask_train.jsonl \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --mixed_precision no
```

What this does:

- loads the saved `MaskEncoder` and `MaskDecoder`
- freezes the encoder
- trains only the decoder on GT masks
- writes the updated decoder back into the same `mask_modules.bin` by default

If you want to keep the original checkpoint untouched, add:

```bash
--output_mask_modules_path "${CKPT_DIR}/mask_modules_decoder_only.bin"
```

## 2. Option C: The Experimental Path

Use this only if you explicitly want to test a soft reconstruction regularizer during the main joint run.

### Step 1: Change `lambda_recon` in the launch script

Measured matched value from the full model probe:

- `lambda_recon = 0.04457831325301206`

Recommended practical value:

- `lambda_recon = 0.04458`

Edit [launch/train_joint_mask.sh](/home/wenting/zr/gen_code_plan2_1/launch/train_joint_mask.sh:1) and change:

```bash
--lambda_recon 0.0
```

to:

```bash
--lambda_recon 0.04458
```

Then run:

```bash
bash launch/train_joint_mask.sh
```

That is all.

There is no decoder-only post-processing step for Option C because the decoder is trained jointly during the main run.

## 3. Inference: Unified For Both Options

The inference path is exactly the same for Option A and Option C.

Use:

- [test_joint_mask.py](/home/wenting/zr/gen_code_plan2_1/test_joint_mask.py:1)
- [launch/test_joint_mask.sh](/home/wenting/zr/gen_code_plan2_1/launch/test_joint_mask.sh:1)

You only need to point inference to:

- the LoRA adapter directory for that run
- the corresponding `mask_modules.bin`

The script behavior does **not** change between options:

- it generates the edited image
- it writes the predicted `.npz` mask
- image evaluation remains image-only

## 4. Quick Decision Rule

Choose Option A when:

- image metrics are the main priority
- you want zero training-time risk to the current best image results
- you are willing to recover masks post-hoc with decoder-only fitting

Choose Option C when:

- you want to experiment with stronger mask recoverability during the main run
- you accept that any nonzero reconstruction term may alter the training dynamics
- you are actively doing ablations rather than protecting the best image checkpoint
