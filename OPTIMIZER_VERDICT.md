# Optimizer Verdict

## Finding

The custom joint-mask modules were effectively frozen and excluded from the optimizer in the `use_lora` training path.

## Evidence

Relevant training block in `train_joint_mask.py`:

```python
requires_grad(model, False)

# Re-enable gradients for new mask modules inside the model
if model.mask_x_embedder is not None:
    requires_grad(model.mask_x_embedder, True)
if model.mask_final_layer is not None:
    requires_grad(model.mask_final_layer, True)
if model.image_modality_embed is not None:
    model.image_modality_embed.requires_grad_(True)
if model.mask_modality_embed is not None:
    model.mask_modality_embed.requires_grad_(True)

transformer_lora_config = LoraConfig(...)
model.llm.enable_input_require_grads()
model = get_peft_model(model, transformer_lora_config)
```

Optimizer construction:

```python
trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
trainable_params += list(mask_encoder.parameters())
trainable_params += list(mask_decoder.parameters())
opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.adam_weight_decay)
```

PEFT behavior in the training environment (`peft 0.18.1`):

```python
def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if self.prefix not in n:
            p.requires_grad = False
```

That means `get_peft_model(...)` freezes every parameter that is not a LoRA adapter parameter.

## Verdict

Yes, the user’s deduction is correct.

- The custom mask modules are re-enabled **before** `get_peft_model(...)`.
- PEFT then freezes all non-adapter parameters again.
- The optimizer is built **after** PEFT wraps the model, so `filter(lambda p: p.requires_grad, model.parameters())` excludes the custom mask modules.
- `mask_encoder` and `mask_decoder` are still optimized because they are appended explicitly.

So `mask_x_embedder`, `mask_final_layer`, `image_modality_embed`, and `mask_modality_embed` were not trained in the LoRA path.

## Minimal Fix

Move the unfreeze block to **after** `model = get_peft_model(model, transformer_lora_config)`, and then build the optimizer.

Exact fix shape:

```python
model = get_peft_model(model, transformer_lora_config)

inner = _get_inner_omnigen_model(model)
if inner.mask_x_embedder is not None:
    requires_grad(inner.mask_x_embedder, True)
if inner.mask_final_layer is not None:
    requires_grad(inner.mask_final_layer, True)
if inner.image_modality_embed is not None:
    inner.image_modality_embed.requires_grad_(True)
if inner.mask_modality_embed is not None:
    inner.mask_modality_embed.requires_grad_(True)

trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
trainable_params += list(mask_encoder.parameters())
trainable_params += list(mask_decoder.parameters())
opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.adam_weight_decay)
```

## Launcher Check

`launch/test_joint_mask.sh` is correct. It passes:

```bash
--mask_modules_path "${MASK_MODULES_PATH}"
```

directly into `test_joint_mask.py`, so the failure is not in the bash launcher.

