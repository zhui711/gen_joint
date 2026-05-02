# Forensic Report: Why the Inference Mask Is Garbage

## Trace Result

I ran `debug_inference_loop.py` for 5 ODE steps on one training sample (`LIDC-IDRI-0657_0001`) using the same Option A checkpoint and a control run with KV cache enabled.

Observed step trace:

```text
step 0 before  z_mask std=1.006210
step 0 after   z_mask std=1.006210
step 1 before  z_mask std=1.006210
step 1 after   z_mask std=1.006210
step 2 before  z_mask std=1.006210
step 2 after   z_mask std=1.006210
step 3 before  z_mask std=1.006210
step 3 after   z_mask std=1.006210
step 4 before  z_mask std=1.006210
step 4 after   z_mask std=1.006210
```

The key signal is the predicted mask update:

```text
pred_mask_std = 0.000000
```

for every traced step.

## What This Means

`z_mask` is not moving at all. The ODE solver is updating `z_img`, but the mask branch is effectively dead because `pred_mask` is zero. The inference failure is therefore upstream of decoding and saving.

This rules out the hypothesis that the saved mask is garbage because of a PNG/NPZ formatting mistake.

## Saving / Thresholding Logic

Current code in `OmniGen/pipeline.py`:

```python
decoded_masks = self.mask_decoder(mask_samples_clean)
binary_masks = (decoded_masks > mask_threshold).float()
output_masks = binary_masks.cpu()
```

Current code in `test_joint_mask.py`:

```python
np.savez_compressed(
    mask_save_path,
    mask=masks[idx_in_subbatch].cpu().numpy(),
)
```

This is not a save-format bug.

- The pipeline explicitly thresholds the decoder output before saving.
- The `.npz` file stores the thresholded binary tensor, not a raw PNG.
- So the mosaic is not coming from bad serialization.

## Exact Diagnosis

The bug is in the inference update path, not in saving.

Evidence:

1. `z_mask` std never changes from initialization.
2. `pred_mask_std` is exactly zero at each traced step.
3. `z_img` does update, so the loop itself is running.
4. `MaskDecoder(MaskEncoder(GT_mask))` works, so the decoder is not inherently broken.

Conclusion:

The joint inference path is failing to produce a non-zero mask velocity.

## Minimal Fix to Test

Do not change training or global scale yet.

Add a hard forensic assert in the joint ODE path:

```python
if pred_mask.std().item() == 0:
    raise RuntimeError("Mask branch is producing zero updates; inspect joint mask wiring/loading.")
```

Then inspect these possibilities in order:

1. `x_mask` is not actually reaching the mask branch in `forward_with_cfg`.
2. `mask_x_embedder` / `mask_final_layer` weights are not loaded as expected.
3. The guidance wrapper is collapsing the mask branch to zero.

The minimal code fix, once confirmed, is to restore a non-zero `pred_mask` from the mask branch before decoding. Thresholding and `.npz` saving do not need to change.

