# Zero Mask Verdict

## Result

The zero-output bug is a **weight loading failure in `mask_final_layer`**, not a slicing bug.

### Evidence

Loaded weight sums:

```text
final_layer abs_sum      = 148962.875847
mask_final_layer abs_sum = 0.000000
mask_x_embedder abs_sum  = 1081.389160
```

Forward trace:

```text
model_out_type = joint
img_out  std = 1.200601
mask_out std = 0.000000
pred_img std = 1.768736
pred_mask std = 0.000000
```

Interpretation:

- `mask_x_embedder` is loaded and active.
- The transformer produces a nonzero joint output for the image branch.
- The mask branch collapses exactly at `mask_final_layer`.
- Therefore the bug is not in tensor slicing from `output[:, -num_mask_tokens:]`.

## Exact Root Cause

`mask_modules.bin` contains the key:

```text
mask_final_layer.linear.weight
mask_final_layer.linear.bias
mask_final_layer.adaLN_modulation.1.weight
mask_final_layer.adaLN_modulation.1.bias
```

but the live model after loading has:

```text
mask_final_layer abs_sum = 0.000000
```

So the mask output head remained at its zero initialization, which means the checkpoint loading path silently failed to populate `mask_final_layer`.

## Minimal Fix

Inspect and correct the loading path in the joint-mask initialization helper used by inference:

```python
fl_state = {
    k.replace("mask_final_layer.", ""): v
    for k, v in state_dict.items()
    if k.startswith("mask_final_layer.")
}
if fl_state and inner.mask_final_layer is not None:
    inner.mask_final_layer.load_state_dict(fl_state)
```

The required fix is to ensure this block actually executes on the inference model instance and that the keys are not being dropped by wrapper unwrapping or by loading a stale checkpoint path.

## Conclusion

The mask mosaic is caused by the mask head staying zero, not by:

- ODE update failure
- tensor slicing
- `.npz` saving
- thresholding

