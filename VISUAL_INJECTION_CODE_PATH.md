# OmniGen Visual Injection Code Path

This is a code-grounded trace of how source image, target image, and target mask information enters the LLM backbone in the current joint image-mask codebase.

## Final Verdict

- **Backbone type:** One Phi-3 decoder-only transformer stack (`Phi3Transformer(Phi3Model)`) driven by `inputs_embeds` and a custom self-attention mask.
- **Visual injection mechanism for source image:** Source images are VAE-encoded outside `OmniGen.forward`, patch-embedded by `input_x_embedder`, and replace placeholder token spans inside `condition_embeds`.
- **Visual injection mechanism for target image:** The current noisy target image latent `x` is patch-embedded by `x_embedder`, optionally gets `image_modality_embed`, and is concatenated after `condition_embeds` and `time_token`.
- **Visual injection mechanism for target mask:** The current noisy target mask latent `x_mask` is patch-embedded by `mask_x_embedder`, gets `mask_modality_embed`, and is concatenated after target image tokens in joint mode.
- **Final sequence layout:** Joint mode with conditioning uses `[condition_embeds, time_token, image_tokens, mask_tokens]`. Image-only mode uses `[condition_embeds, time_token, image_tokens]`. Without `input_ids`, the condition block is omitted.
- **Whether explicit cross-attention exists:** No project-defined explicit cross-attention path exists. The project passes one unified `inputs_embeds` tensor to a decoder layer stack. Installed `transformers==4.45.2` Phi-3 layers define `self_attn`, not image/context cross-attention.
- **Confidence level:** High. This is based on the actual call path in `train_joint_mask.py`, `loss_joint_mask.py`, `processor.py`, `model.py`, `transformer.py`, `scheduler.py`, plus symbol search for cross-attention names.

## Answer To The Specific Questions

### 1. Decoder-only unified sequence vs cross-attention

The project wrapper is a Phi-3 decoder model and feeds a single `inputs_embeds` sequence through decoder layers.

`OmniGen/transformer.py:25-29`

```python
class Phi3Transformer(Phi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]
    We only modified the attention mask
```

`OmniGen/transformer.py:129-165`

```python
hidden_states = inputs_embeds

# decoder layers
...
for decoder_layer in self.layers:
    ...
    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
```

The model entry point passes the assembled embedding sequence directly:

`OmniGen/model.py:453`

```python
output = self.llm(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model)
```

So the backbone sees one unified sequence. Visual tokens are not provided via a separate `encoder_hidden_states` or context argument.

The attention mask is also passed as a single sequence mask and converted to an additive decoder mask:

`OmniGen/transformer.py:118-129`

```python
if attention_mask is not None and attention_mask.dim() == 3:
    dtype = inputs_embeds.dtype
    min_dtype = torch.finfo(dtype).min
    attention_mask = (1 - attention_mask) * min_dtype
    attention_mask = attention_mask.unsqueeze(1).to(inputs_embeds.dtype)
else:
    raise Exception("attention_mask parameter was unavailable or invalid")

hidden_states = inputs_embeds
```

### 2. Source image injection

Source image placeholders are created in the text token sequence first. The processor splits on `<|image_N|>`, inserts `0` token placeholders, and records the span.

`OmniGen/processor.py:63-91`

```python
pattern = r"<\|image_\d+\|>"
prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]
...
for i in range(len(prompt_chunks)):
    all_input_ids.extend(prompt_chunks[i])
    if i != len(prompt_chunks) -1:
        start_inx = len(all_input_ids)
        size = input_images[i].size(-2) *  input_images[i].size(-1) // 16 // 16
        img_inx.append([start_inx, start_inx+size])
        all_input_ids.extend([0]*size)

return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}
```

The source image pixels are encoded to VAE latents before the model call.

Inference path, `OmniGen/pipeline.py:270-281`

```python
input_img_latents = []
...
for img in input_data['input_pixel_values']:
    img = self.vae_encode(img.to(self.device), dtype)
    input_img_latents.append(img)
```

Training path, `train_joint_mask.py:470-484`

```python
with torch.no_grad():
    output_images = data["output_images"]
    input_pixel_values = data["input_pixel_values"]
    ...
    if input_pixel_values is not None:
        input_pixel_values = input_pixel_values.to(device=device, dtype=torch.float32)
        input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
```

Inside `OmniGen.forward`, those source image latents are patch-embedded with `is_input_images=True`, using `input_x_embedder`.

`OmniGen/model.py:340-375`

```python
def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images:bool=False):
    ...
    if is_input_images:
        latents = self.input_x_embedder(latents)
    else:
        latents = self.x_embedder(latents)
    pos_embed = self.cropped_pos_embed(height, width)
    latents = latents + pos_embed
```

`OmniGen/model.py:430-440`

```python
if input_img_latents is not None:
    input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
if input_ids is not None:
    condition_embeds = self.llm.embed_tokens(input_ids).clone()
    input_img_inx = 0
    for b_inx in input_image_sizes.keys():
        for start_inx, end_inx in input_image_sizes[b_inx]:
            condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
            input_img_inx += 1
```

Verdict for source image: it is converted into patch/image tokens and inserted by replacing placeholder spans in the text/condition embedding sequence. It is not passed through a separate image encoder branch attended by cross-attention.

### 3. Target image and target mask injection during joint training

Training builds noisy image and mask states and calls the model once with both states.

`OmniGen/train_helper/loss_joint_mask.py:80-99`

```python
# Build noisy states
...
xt_img = t_ * x1_img + (1 - t_) * x0_img
ut_img = x1_img - x0_img

# Mask noisy state (always tensor, not list)
dims_mask = [1] * (len(x1_mask.size()) - 1)
t_mask = t.view(t.size(0), *dims_mask)
xt_mask = t_mask * x1_mask + (1 - t_mask) * x0_mask
ut_mask = x1_mask - x0_mask

model_output = model(xt_img, t, x_mask=xt_mask, **model_kwargs)
```

The GT mask is encoded to a mask latent before this loss call:

`train_joint_mask.py:493-496`

```python
# Map {0,1} -> [-1,1]
mask_cont = 2.0 * output_anatomy_masks.to(device=device, dtype=weight_dtype) - 1.0
# Encode through mask encoder (WITH gradients for mask_encoder training)
x1_mask = mask_encoder(mask_cont)
```

Target image tokens are produced at the start of `OmniGen.forward`:

`OmniGen/model.py:412-414`

```python
input_is_list = isinstance(x, list)
x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
time_token = self.time_token(timestep, dtype=x[0].dtype if isinstance(x, list) else x.dtype).unsqueeze(1)
```

Target mask tokens are produced only in joint mode:

`OmniGen/model.py:421-428`

```python
if joint_mask_mode:
    mask_tokens, num_mask_tokens, mask_shapes = self.patch_mask_latents(x_mask)
    # Add modality embeddings
    if isinstance(x, list):
        x = [xi + self.image_modality_embed for xi in x]
    else:
        x = x + self.image_modality_embed
    mask_tokens = mask_tokens + self.mask_modality_embed
```

The exact final transformer input assembly is here:

`OmniGen/model.py:432-453`

```python
if input_ids is not None:
    condition_embeds = self.llm.embed_tokens(input_ids).clone()
    input_img_inx = 0
    for b_inx in input_image_sizes.keys():
        for start_inx, end_inx in input_image_sizes[b_inx]:
            condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
            input_img_inx += 1
    if input_img_latents is not None:
        assert input_img_inx == len(input_latents)

    if joint_mask_mode:
        # Sequence: [condition_embeds, time_token, img_tokens, mask_tokens]
        input_emb = torch.cat([condition_embeds, time_token, x, mask_tokens], dim=1)
    else:
        input_emb = torch.cat([condition_embeds, time_token, x], dim=1)
else:
    if joint_mask_mode:
        input_emb = torch.cat([time_token, x, mask_tokens], dim=1)
    else:
        input_emb = torch.cat([time_token, x], dim=1)
```

The output suffix is split back into image-token and mask-token regions after the same backbone call:

`OmniGen/model.py:456-484`

```python
if joint_mask_mode:
    # Split output into image and mask portions
    # output suffix = [img_tokens, mask_tokens]
    total_gen_tokens = num_tokens + num_mask_tokens if not input_is_list else max(num_tokens) + num_mask_tokens

    if input_is_list:
        img_end = -num_mask_tokens
        image_embedding = output[:, -(max(num_tokens) + num_mask_tokens):img_end]
        mask_embedding = output[:, -num_mask_tokens:]
    else:
        image_embedding = output[:, -(num_tokens + num_mask_tokens):-num_mask_tokens]
        mask_embedding = output[:, -num_mask_tokens:]

    time_emb = self.t_embedder(timestep, dtype=image_embedding.dtype)

    # Image branch
    img_out = self.final_layer(image_embedding, time_emb)
    ...
    # Mask branch
    mask_out = self.mask_final_layer(mask_embedding, time_emb)
    mask_latents = self.unpatchify_mask(mask_out, mask_shapes[0], mask_shapes[1])
```

### 4. Exact code path for `condition_embeds`, `time_token`, image tokens, mask tokens, final `input_emb`

#### `condition_embeds`

`OmniGen/model.py:432-440`

```python
if input_ids is not None:
    condition_embeds = self.llm.embed_tokens(input_ids).clone()
    input_img_inx = 0
    for b_inx in input_image_sizes.keys():
        for start_inx, end_inx in input_image_sizes[b_inx]:
            condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
            input_img_inx += 1
    if input_img_latents is not None:
        assert input_img_inx == len(input_latents)
```

#### `time_token`

`OmniGen/model.py:413-414`

```python
x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
time_token = self.time_token(timestep, dtype=x[0].dtype if isinstance(x, list) else x.dtype).unsqueeze(1)
```

#### Target image tokens

`OmniGen/model.py:133-149`

```python
class PatchEmbedMR(nn.Module):
    ...
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

def forward(self, x):
    x = self.proj(x)
    x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    return x
```

`OmniGen/model.py:340-375`

```python
if is_input_images:
    latents = self.input_x_embedder(latents)
else:
    latents = self.x_embedder(latents)
pos_embed = self.cropped_pos_embed(height, width)
latents = latents + pos_embed
```

In `forward`, this patched image-token tensor is stored back into variable `x`:

`OmniGen/model.py:413`

```python
x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
```

#### Target mask tokens

Mask modules are initialized as a separate patch embedder and output head, not a separate attention branch.

`OmniGen/model.py:198-221`

```python
def init_mask_modules(self, mask_latent_channels: int = 4):
    ...
    self.mask_x_embedder = PatchEmbedMR(
        self.patch_size, mask_latent_channels, hidden_size, bias=True
    )
    ...
    self.mask_modality_embed = nn.Parameter(
        torch.zeros(1, 1, hidden_size)
    )
```

`OmniGen/model.py:377-394`

```python
def patch_mask_latents(self, mask_latents):
    ...
    height, width = mask_latents.shape[-2:]
    mask_tokens = self.mask_x_embedder(mask_latents)
    pos_embed = self.cropped_pos_embed(height, width)
    mask_tokens = mask_tokens + pos_embed
    num_mask_tokens = mask_tokens.size(1)
    mask_shapes = [height, width]
    return mask_tokens, num_mask_tokens, mask_shapes
```

#### Final `input_emb`

`OmniGen/model.py:442-453`

```python
if joint_mask_mode:
    # Sequence: [condition_embeds, time_token, img_tokens, mask_tokens]
    input_emb = torch.cat([condition_embeds, time_token, x, mask_tokens], dim=1)
else:
    input_emb = torch.cat([condition_embeds, time_token, x], dim=1)

output = self.llm(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model)
```

### 5. Cross-attention search and inherited Phi-3 check

Project search for:

```text
cross_attn
cross_attention
encoder_hidden_states
context_attn
attend_to_image
```

found no project code paths defining or calling such modules. The relevant hits were only `inputs_embeds`, `qkv_proj`, and `self_attns`.

The installed repo environment uses `transformers==4.45.2`. In that implementation, `Phi3Model` constructs decoder layers:

`transformers/models/phi3/modeling_phi3.py:942-959`

```python
class Phi3Model(Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]
    ...
    self.layers = nn.ModuleList(
        [Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )
```

`Phi3DecoderLayer` defines `self_attn`, not cross-attention:

`transformers/models/phi3/modeling_phi3.py:737-742`

```python
class Phi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
```

Its forward path calls only that self-attention module:

`transformers/models/phi3/modeling_phi3.py:789-798`

```python
# Self Attention
attn_outputs, self_attn_weights, present_key_value = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    output_attentions=output_attentions,
    use_cache=use_cache,
    cache_position=cache_position,
)
```

The attention module computes Q/K/V from the same `hidden_states` tensor:

`transformers/models/phi3/modeling_phi3.py:411-429`

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    ...
    qkv = self.qkv_proj(hidden_states)
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
```

There is no `encoder_hidden_states` argument in the project wrapper forward or in the Phi-3 attention signature used here.

## Sequence Sizing And Attention Mask

The collator sizes the sequence for `[condition_tokens, time_token, generated_tokens]`. In joint mode, generated tokens are `img_tokens + mask_tokens`.

`OmniGen/processor.py:269-303`

```python
def process_mllm_input_joint(self, mllm_inputs, target_img_size, num_mask_tokens):
    """Like process_mllm_input but adds mask tokens to the output sequence.

    The attention mask and position_ids are sized for:
      [condition_tokens, time_token, img_tokens, mask_tokens]

    The mask tokens are treated identically to image tokens in the attention
    pattern (full bidirectional attention among all generated tokens).
    """
    # Total generated tokens = img_tokens + mask_tokens
    num_tokens_for_output = []
    for img_size in target_img_size:
        img_tokens = img_size[0] * img_size[1] // 16 // 16
        num_tokens_for_output.append(img_tokens + num_mask_tokens)
    ...
    position_ids = self.create_position(attention_mask, num_tokens_for_output)
    attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output)
    attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)
```

Input-image placeholder spans are adjusted in the same attention mask, not routed through cross-attention:

`OmniGen/processor.py:209-214`

```python
def adjust_attention_for_input_images(self, attention_mask, image_sizes):
    for b_inx in image_sizes.keys():
        for start_inx, end_inx in image_sizes[b_inx]:
            attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1

    return attention_mask
```

Training uses this via `TrainDataCollator`, not directly via `OmniGenProcessor.__call__`:

`OmniGen/train_helper/data.py:95-113`

```python
class TrainDataCollator(OmniGenCollator):
    ...
    # For joint mask mode, inflate the output token count to include mask tokens.
    # The attention mask and position_ids must cover [condition, time, img_tokens, mask_tokens].
    if self.num_mask_tokens > 0:
        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes = self.process_mllm_input_joint(mllm_inputs, target_img_size, self.num_mask_tokens)
```

`train_joint_mask.py` hard-codes 256 mask tokens for 256x256 images:

`train_joint_mask.py:358-368`

```python
# Compute num_mask_tokens for the collator
# For 256x256 images: latent = (4, 32, 32), patch_size=2 -> 16x16 = 256 tokens
# This must match the mask latent spatial dims
num_mask_tokens = (256 // 8 // 2) * (256 // 8 // 2)  # = 256 for 256x256

collate_fn = TrainDataCollator(
    ...
    num_mask_tokens=num_mask_tokens,
)
```

## Pipeline Joint Inference Path

The inference pipeline computes the joint mask token count, creates random image and mask latent states, and calls the joint scheduler:

`OmniGen/pipeline.py:237-245`

```python
# Compute mask token count for joint mode
use_joint = self.model.use_joint_mask and self.mask_decoder is not None
if use_joint:
    _latent_h, _latent_w = height // 8, width // 8
    _num_mask_tokens = (_latent_h // self.model.patch_size) * (_latent_w // self.model.patch_size)
else:
    _num_mask_tokens = 0

input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=separate_cfg_infer, use_input_image_size_as_output=use_input_image_size_as_output, num_mask_tokens=_num_mask_tokens)
```

`OmniGen/pipeline.py:260-267`

```python
latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
latents = torch.cat([latents]*(1+num_cfg), 0).to(dtype)

# Initialize mask latents if in joint mode (use_joint already computed above)
mask_latents = None
if use_joint:
    mask_latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
    mask_latents = torch.cat([mask_latents]*(1+num_cfg), 0).to(dtype)
```

`OmniGen/pipeline.py:315-323`

```python
if use_joint:
    # Joint co-denoising
    samples, mask_samples = scheduler.__call_joint__(
        latents, mask_latents, func, model_kwargs,
        use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache
    )
else:
    # Standard image-only denoising
    samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache)
```

The joint scheduler passes those states into the same model function as `x` and `x_mask`:

`OmniGen/scheduler.py:198-210`

```python
# Total generated tokens = image tokens + mask tokens
num_img_tokens = z_img.size(-1) * z_img.size(-2) // 4
num_mask_tokens = z_mask.size(-1) * z_mask.size(-2) // 4
num_total_gen_tokens = num_img_tokens + num_mask_tokens
...
for i in tqdm(range(self.num_steps)):
    timesteps = torch.zeros(size=(len(z_img), )).to(z_img.device) + self.sigma[i]
    pred, cache = func(z_img, timesteps, past_key_values=cache, x_mask=z_mask, **model_kwargs)
```

## Important Distinction

Source image conditioning and target image/mask generation are injected differently:

- Source image: placeholder spans in `condition_embeds` are overwritten by patch embeddings from VAE-encoded input images.
- Target image: the noisy image state being denoised is patch-embedded as generated image tokens and appended after the time token.
- Target mask: the noisy mask state being denoised is patch-embedded as generated mask tokens and appended after target image tokens.

All three ultimately become token embeddings in one decoder-only self-attention sequence. None enters the LLM backbone through explicit cross-attention.
