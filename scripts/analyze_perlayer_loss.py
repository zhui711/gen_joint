"""
Per-Layer Anatomy Loss Analysis Script.

This script analyzes which encoder layer (2, 3, or 4) contributes most to the
anatomy loss during training. It simulates the training loop by:
1. Loading GT images and encoding them with VAE
2. Sampling timesteps and adding noise
3. Running the OmniGen model to predict velocity
4. Reconstructing x1_hat and decoding with VAE
5. Computing per-layer feature MSE

Usage:
    python scripts/analyze_perlayer_loss.py \
        --checkpoint_path results/cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16/checkpoints/0009000 \
        --num_samples 50
"""

import sys
import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add paths
sys.path.insert(0, "/home/wenting/zr/gen_code")
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")

from diffusers.models import AutoencoderKL
from OmniGen import OmniGen, OmniGenProcessor
from peft import PeftModel
import segmentation_models_pytorch as smp


def get_anatomy_model():
    """Create UNet with ResNet34 encoder."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )


def load_seg_model(ckpt_path, device):
    """Load frozen segmentation model."""
    model = get_anatomy_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def sample_timestep(batch_size, device):
    """Sample timestep from logistic-normal distribution."""
    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,))
    t = 1 / (1 + torch.exp(-u))
    return t.to(device)


def inverse_vae_scale(latents, vae):
    """Reverse the VAE scaling applied during encoding."""
    if vae.config.shift_factor is not None:
        return latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        return latents / vae.config.scaling_factor


def vae_encode(vae, image, weight_dtype):
    """Encode image to latent space."""
    with torch.no_grad():
        x = image.to(dtype=weight_dtype)
        x = vae.encode(x).latent_dist.sample()
        if vae.config.shift_factor is not None:
            x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x = x * vae.config.scaling_factor
    return x


def compute_perlayer_mse(seg_model, gen_images, gt_images):
    """
    Compute per-layer MSE between generated and GT image features.

    Returns dict with MSE for each layer index.
    """
    FEATURE_INDICES = [2, 3, 4]

    # GT features (no grad)
    with torch.no_grad():
        gt_features = seg_model.encoder(gt_images)

    # Gen features (with grad in training, but we're just measuring here)
    with torch.no_grad():
        gen_features = seg_model.encoder(gen_images)

    per_layer_mse = {}
    per_layer_stats = {}

    for idx in FEATURE_INDICES:
        gt_feat = gt_features[idx]
        gen_feat = gen_features[idx]

        # MSE
        mse = F.mse_loss(gen_feat, gt_feat, reduction='mean').item()
        per_layer_mse[f"layer_{idx}"] = mse

        # Additional stats
        per_layer_stats[f"layer_{idx}"] = {
            "mse": mse,
            "shape": list(gt_feat.shape),
            "gt_mean": gt_feat.mean().item(),
            "gt_std": gt_feat.std().item(),
            "gen_mean": gen_feat.mean().item(),
            "gen_std": gen_feat.std().item(),
            "diff_abs_mean": (gen_feat - gt_feat).abs().mean().item(),
        }

    avg_mse = sum(per_layer_mse.values()) / len(per_layer_mse)
    per_layer_mse["average"] = avg_mse

    return per_layer_mse, per_layer_stats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load segmentation model
    print(f"Loading segmentation model from {args.seg_model_ckpt}...")
    seg_model = load_seg_model(args.seg_model_ckpt, device)

    # Load VAE
    print("Loading VAE...")
    vae_path = os.path.join(args.model_name_or_path, "vae")
    if os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Load OmniGen model with LoRA
    print(f"Loading OmniGen model from {args.model_name_or_path}...")
    model = OmniGen.from_pretrained(args.model_name_or_path)

    if args.checkpoint_path:
        print(f"Loading LoRA weights from {args.checkpoint_path}...")
        model = PeftModel.from_pretrained(model, args.checkpoint_path)

    model = model.to(device)
    model.eval()

    # Load processor
    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    # Image transform
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load training data JSON
    print(f"Loading training data from {args.json_file}...")
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # Collect per-layer statistics
    all_layer_mse = {f"layer_{i}": [] for i in [2, 3, 4]}
    all_layer_mse["average"] = []
    all_timesteps = []

    # Also collect stats by timestep bins
    timestep_bins = {
        "t_0.0-0.2": {f"layer_{i}": [] for i in [2, 3, 4]},
        "t_0.2-0.4": {f"layer_{i}": [] for i in [2, 3, 4]},
        "t_0.4-0.6": {f"layer_{i}": [] for i in [2, 3, 4]},
        "t_0.6-0.8": {f"layer_{i}": [] for i in [2, 3, 4]},
        "t_0.8-1.0": {f"layer_{i}": [] for i in [2, 3, 4]},
    }

    print(f"\nAnalyzing {min(args.num_samples, len(data))} samples...")

    for i, item in enumerate(tqdm(data[:args.num_samples])):
        try:
            # Load GT image
            if args.image_path:
                img_path = os.path.join(args.image_path, item["output_image"])
            else:
                img_path = item["output_image"]

            if not os.path.exists(img_path):
                continue

            gt_img_pil = Image.open(img_path).convert("RGB")
            gt_img = image_transform(gt_img_pil).unsqueeze(0).to(device)

            # Encode GT to latent
            x1 = vae_encode(vae, gt_img, torch.float32)

            # Sample noise and timestep
            x0 = torch.randn_like(x1)
            t = sample_timestep(1, device)
            t_val = t.item()

            # Create noisy latent
            t_ = t.view(1, 1, 1, 1)
            xt = t_ * x1 + (1 - t_) * x0
            ut = x1 - x0  # target velocity

            # Model prediction (simplified - no text conditioning)
            with torch.no_grad():
                # For analysis, we'll use a simple forward pass
                # In full training, there would be text conditioning
                model_output = model(xt, t)

            # Reconstruct x1_hat
            x1_hat = xt + (1 - t_) * model_output

            # Decode to pixel space
            x1_hat_scaled = inverse_vae_scale(x1_hat.float(), vae)
            with torch.no_grad():
                gen_decoded = vae.decode(x1_hat_scaled).sample
            gen_decoded = gen_decoded.clamp(-1.0, 1.0)

            # Resize to 256x256 for seg model
            if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
                gen_decoded = F.interpolate(gen_decoded, size=(256, 256), mode="bilinear", align_corners=False)

            gt_256 = gt_img
            if gt_256.shape[-2] != 256 or gt_256.shape[-1] != 256:
                gt_256 = F.interpolate(gt_256, size=(256, 256), mode="bilinear", align_corners=False)

            # Compute per-layer MSE
            per_layer_mse, per_layer_stats = compute_perlayer_mse(seg_model, gen_decoded, gt_256)

            # Record
            for layer_key in [f"layer_{i}" for i in [2, 3, 4]]:
                all_layer_mse[layer_key].append(per_layer_mse[layer_key])
            all_layer_mse["average"].append(per_layer_mse["average"])
            all_timesteps.append(t_val)

            # Bin by timestep
            if t_val < 0.2:
                bin_key = "t_0.0-0.2"
            elif t_val < 0.4:
                bin_key = "t_0.2-0.4"
            elif t_val < 0.6:
                bin_key = "t_0.4-0.6"
            elif t_val < 0.8:
                bin_key = "t_0.6-0.8"
            else:
                bin_key = "t_0.8-1.0"

            for layer_key in [f"layer_{i}" for i in [2, 3, 4]]:
                timestep_bins[bin_key][layer_key].append(per_layer_mse[layer_key])

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Print results
    print("\n" + "="*80)
    print("PER-LAYER MSE ANALYSIS RESULTS")
    print("="*80)

    print("\n## Overall Statistics (across all timesteps)")
    print("-" * 60)
    print(f"{'Layer':<15} {'Mean MSE':<15} {'Std MSE':<15} {'Min':<12} {'Max':<12}")
    print("-" * 60)

    for layer_key in ["layer_2", "layer_3", "layer_4", "average"]:
        values = all_layer_mse[layer_key]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{layer_key:<15} {mean_val:<15.6f} {std_val:<15.6f} {min_val:<12.6f} {max_val:<12.6f}")

    print("\n## Per-Layer MSE by Timestep Bin")
    print("-" * 80)

    for bin_key in sorted(timestep_bins.keys()):
        bin_data = timestep_bins[bin_key]
        n_samples = len(bin_data["layer_2"])
        if n_samples == 0:
            continue

        print(f"\n### {bin_key} (n={n_samples})")
        print(f"{'Layer':<15} {'Mean MSE':<15} {'Contribution %':<15}")
        print("-" * 45)

        layer_means = {}
        for layer_key in ["layer_2", "layer_3", "layer_4"]:
            values = bin_data[layer_key]
            if values:
                layer_means[layer_key] = np.mean(values)

        total = sum(layer_means.values())
        for layer_key in ["layer_2", "layer_3", "layer_4"]:
            if layer_key in layer_means:
                mean_val = layer_means[layer_key]
                contrib = (mean_val / total * 100) if total > 0 else 0
                print(f"{layer_key:<15} {mean_val:<15.6f} {contrib:<15.1f}%")

    # Save detailed results
    results = {
        "overall": {
            layer_key: {
                "mean": float(np.mean(all_layer_mse[layer_key])),
                "std": float(np.std(all_layer_mse[layer_key])),
                "min": float(np.min(all_layer_mse[layer_key])),
                "max": float(np.max(all_layer_mse[layer_key])),
            }
            for layer_key in ["layer_2", "layer_3", "layer_4", "average"]
            if all_layer_mse[layer_key]
        },
        "by_timestep_bin": {
            bin_key: {
                layer_key: {
                    "mean": float(np.mean(values)) if values else None,
                    "n_samples": len(values),
                }
                for layer_key, values in bin_data.items()
            }
            for bin_key, bin_data in timestep_bins.items()
        },
        "raw_data": {
            "timesteps": all_timesteps,
            "layer_2_mse": all_layer_mse["layer_2"],
            "layer_3_mse": all_layer_mse["layer_3"],
            "layer_4_mse": all_layer_mse["layer_4"],
        }
    }

    output_path = os.path.join(os.path.dirname(args.checkpoint_path) if args.checkpoint_path else ".", "perlayer_mse_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/wenting/zr/gen_code/OmniGen-v1")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--seg_model_ckpt", type=str, default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth")
    parser.add_argument("--json_file", type=str, default="/home/wenting/zr/data/nih-cxr-lt/meta/train_prompts_balanced_conditions_v2.json")
    parser.add_argument("--image_path", type=str, default="/home/wenting/zr/data/nih-cxr-lt/images_rescaled_256")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    main(args)
