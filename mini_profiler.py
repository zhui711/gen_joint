#!/usr/bin/env python3
"""Mini Profiler for Joint Image-Mask Co-Generation.

This script runs 1 batch of real data through the joint model to:
  1. Check the initial loss magnitude ratio between L_img and L_mask.
  2. Recommend an optimal lambda_mask.
  3. Verify the MaskEncoder/Decoder can produce reasonable latents.
  4. Check gradient flow through all new modules.

Usage:
  python mini_profiler.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --json_file /path/to/train.jsonl
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Shitao/OmniGen-v1")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--mask_latent_channels", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)

    # --- Load model ---
    print("[1/6] Loading OmniGen model...")
    from OmniGen import OmniGen, OmniGenProcessor
    from OmniGen.mask_autoencoder import MaskEncoder, MaskDecoder, MaskAutoencoder
    from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
    from OmniGen.train_helper.loss_joint_mask import training_losses_joint_mask
    from OmniGen.utils import crop_arr, vae_encode, vae_encode_list
    from diffusers.models import AutoencoderKL
    from peft import LoraConfig, get_peft_model

    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.init_mask_modules(mask_latent_channels=args.mask_latent_channels)
    model = model.to(device)

    # --- Load VAE ---
    print("[2/6] Loading VAE...")
    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    vae.requires_grad_(False)

    # --- Create mask autoencoder ---
    print("[3/6] Creating MaskEncoder/Decoder...")
    mask_encoder = MaskEncoder(in_channels=10, latent_channels=args.mask_latent_channels).to(device)
    mask_decoder = MaskDecoder(latent_channels=args.mask_latent_channels, out_channels=10).to(device)

    # Test mask autoencoder reconstruction
    print("\n--- Mask Autoencoder Sanity Check ---")
    dummy_mask = torch.randn(1, 10, 256, 256, device=device)  # random in [-1, 1]
    z_mask = mask_encoder(dummy_mask)
    recon_mask = mask_decoder(z_mask)
    recon_loss = nn.functional.mse_loss(recon_mask, dummy_mask)
    print(f"  Mask latent shape: {z_mask.shape}")
    print(f"  Mask latent stats: mean={z_mask.mean():.4f}, std={z_mask.std():.4f}")
    print(f"  Reconstruction MSE (random init): {recon_loss.item():.4f}")
    print(f"  MaskEncoder params: {sum(p.numel() for p in mask_encoder.parameters()):,}")
    print(f"  MaskDecoder params: {sum(p.numel() for p in mask_decoder.parameters()):,}")

    # --- Apply LoRA ---
    print("\n[4/6] Applying LoRA...")
    model.to(torch.bfloat16)
    from OmniGen.utils import requires_grad
    requires_grad(model, False)
    if model.mask_x_embedder is not None:
        requires_grad(model.mask_x_embedder, True)
    if model.mask_final_layer is not None:
        requires_grad(model.mask_final_layer, True)
    if model.image_modality_embed is not None:
        model.image_modality_embed.requires_grad_(True)
    if model.mask_modality_embed is not None:
        model.mask_modality_embed.requires_grad_(True)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model.llm.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA trainable params: {lora_params:,} / {total_params:,} ({100*lora_params/total_params:.2f}%)")

    # --- Load one batch ---
    print("\n[5/6] Loading one batch of real data...")
    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=1024,
        condition_dropout_prob=0.0,
        keep_raw_resolution=True,
    )
    # Must match the mask latent token count so attention mask/position_ids
    # cover [condition, time, img_tokens, mask_tokens].
    num_mask_tokens = (256 // 8 // 2) * (256 // 8 // 2)  # 256 for 256x256
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=3072,
        keep_raw_resolution=True,
        num_mask_tokens=num_mask_tokens,
    )
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_size=args.batch_size,
        shuffle=False, num_workers=0, drop_last=True,
    )
    data = next(iter(loader))

    # --- Encode ---
    weight_dtype = torch.bfloat16
    with torch.no_grad():
        output_images = data["output_images"]
        input_pixel_values = data["input_pixel_values"]
        if isinstance(output_images, list):
            output_images = [x.to(device) for x in output_images]
            if input_pixel_values is not None:
                input_pixel_values = [x.to(device) for x in input_pixel_values]
            output_images = vae_encode_list(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
        else:
            output_images = output_images.to(device)
            if input_pixel_values is not None:
                input_pixel_values = input_pixel_values.to(device)
            output_images = vae_encode(vae, output_images, weight_dtype)
            if input_pixel_values is not None:
                input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)

    output_anatomy_masks = data.get("output_anatomy_masks", None)
    if output_anatomy_masks is None:
        print("ERROR: No anatomy masks in data. Check JSONL file.")
        return

    mask_cont = 2.0 * output_anatomy_masks.to(device=device, dtype=weight_dtype) - 1.0
    mask_encoder = mask_encoder.to(weight_dtype)
    mask_decoder = mask_decoder.to(weight_dtype)
    x1_mask = mask_encoder(mask_cont)

    print(f"  Image latent: {output_images[0].shape if isinstance(output_images, list) else output_images.shape}")
    print(f"  Mask latent: {x1_mask.shape}")
    print(f"  Mask latent stats: mean={x1_mask.mean():.4f}, std={x1_mask.std():.4f}")

    # --- Compute loss ---
    print("\n[6/6] Computing joint loss (1 step)...")
    model.train()
    model_kwargs = dict(
        input_ids=data["input_ids"].to(device) if not isinstance(data["input_ids"], list) else data["input_ids"],
        input_img_latents=input_pixel_values,
        input_image_sizes=data["input_image_sizes"],
        attention_mask=data["attention_mask"].to(device) if not isinstance(data["attention_mask"], list) else data["attention_mask"],
        position_ids=data["position_ids"].to(device) if not isinstance(data["position_ids"], list) else data["position_ids"],
        padding_latent=data["padding_images"],
        past_key_values=None,
        return_past_key_values=False,
    )

    loss_dict = training_losses_joint_mask(
        model=model,
        x1_img=output_images,
        x1_mask=x1_mask,
        model_kwargs=model_kwargs,
        lambda_mask=1.0,  # Use 1.0 to see raw magnitudes
    )

    loss_img = loss_dict["loss_img"].item()
    loss_mask = loss_dict["loss_mask"].item()
    ratio = loss_img / (loss_mask + 1e-8)

    print("\n" + "=" * 60)
    print("  PROFILING RESULTS")
    print("=" * 60)
    print(f"  L_img  (image velocity MSE):  {loss_img:.6f}")
    print(f"  L_mask (mask velocity MSE):   {loss_mask:.6f}")
    print(f"  Ratio L_img / L_mask:         {ratio:.4f}")
    print()

    if ratio > 2.0:
        recommended_lambda = round(ratio * 0.5, 2)
        print(f"  -> L_img >> L_mask. Recommend lambda_mask = {recommended_lambda:.2f}")
        print(f"     (to bring mask loss to ~50% of image loss magnitude)")
    elif ratio < 0.5:
        recommended_lambda = round(ratio * 0.5, 2)
        print(f"  -> L_mask >> L_img. Recommend lambda_mask = {recommended_lambda:.2f}")
        print(f"     (to prevent mask from dominating)")
    else:
        print(f"  -> Losses are comparable. lambda_mask = 0.5 is a good starting point.")

    # --- Check gradient flow ---
    print("\n--- Gradient Flow Check ---")
    loss_dict["loss"].backward()

    grad_report = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "lora" in name:
                grad_report.setdefault("lora", []).append(grad_norm)
            elif "mask" in name:
                grad_report.setdefault("mask_model", []).append(grad_norm)
            elif "modality" in name:
                grad_report.setdefault("modality_embed", []).append(grad_norm)

    for name, param in mask_encoder.named_parameters():
        if param.grad is not None:
            grad_report.setdefault("mask_encoder", []).append(param.grad.norm().item())
    for name, param in mask_decoder.named_parameters():
        if param.grad is not None:
            grad_report.setdefault("mask_decoder", []).append(param.grad.norm().item())

    for group, norms in grad_report.items():
        avg = np.mean(norms)
        print(f"  {group:20s}: avg_grad_norm={avg:.6f} (n={len(norms)} params)")

    no_grad_modules = []
    for group in ["lora", "mask_model", "modality_embed", "mask_encoder", "mask_decoder"]:
        if group not in grad_report:
            no_grad_modules.append(group)
    if no_grad_modules:
        print(f"  WARNING: No gradients for: {no_grad_modules}")
    else:
        print("  All module groups have gradients flowing. OK.")

    print("\n" + "=" * 60)
    print("  Profiling complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
