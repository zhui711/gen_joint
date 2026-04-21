"""
Per-Layer MSE Analysis Script for Anatomy Loss Debugging.

This script analyzes which encoder layer contributes most to the anatomy loss
by computing per-layer MSE values during simulated training steps.

Usage:
    python scripts/analyze_perlayer_mse.py \
        --checkpoint_path results/cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16/checkpoints/0009000 \
        --num_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# Add paths
sys.path.insert(0, "/home/wenting/zr/gen_code")
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")

from diffusers.models import AutoencoderKL
from peft import PeftModel
import segmentation_models_pytorch as smp

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.utils import crop_arr, vae_encode, vae_encode_list


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


def inverse_vae_scale(latents, vae):
    """Reverse the VAE scaling applied during encoding."""
    if vae.config.shift_factor is not None:
        return latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        return latents / vae.config.scaling_factor


def sample_timestep(batch_size, device):
    """Sample timestep from logistic-normal distribution."""
    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
    t = 1 / (1 + torch.exp(-u))
    return t


def compute_perlayer_mse(seg_model, gen_images, gt_images):
    """
    Compute per-layer MSE for encoder features.

    Returns:
        dict with per-layer MSE values and total
    """
    FEATURE_INDICES = [2, 3, 4]
    LAYER_NAMES = {
        0: "layer0_input (3ch, stride=1)",
        1: "layer1_conv (64ch, stride=2)",
        2: "layer2 (64ch, stride=4)",
        3: "layer3 (128ch, stride=8)",
        4: "layer4 (256ch, stride=16)",
        5: "layer5 (512ch, stride=32)",
    }

    with torch.no_grad():
        gt_features = seg_model.encoder(gt_images)

    with torch.no_grad():
        gen_features = seg_model.encoder(gen_images)

    results = {}
    total_loss = 0.0

    for idx in FEATURE_INDICES:
        gt_feat = gt_features[idx]
        gen_feat = gen_features[idx]

        mse = F.mse_loss(gen_feat, gt_feat, reduction='mean').item()
        results[f"layer{idx}"] = {
            "mse": mse,
            "shape": list(gt_feat.shape),
            "name": LAYER_NAMES[idx],
        }
        total_loss += mse

    results["average"] = total_loss / len(FEATURE_INDICES)
    results["total"] = total_loss

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load segmentation model
    print("Loading segmentation model...")
    seg_model = load_seg_model(args.seg_model_ckpt, device)

    # Load VAE
    print("Loading VAE...")
    from huggingface_hub import snapshot_download
    if args.vae_path is None:
        # Try to get VAE from the model folder
        model_path = args.model_name_or_path
        if not os.path.exists(model_path):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_path = snapshot_download(
                repo_id=args.model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
        vae_path = os.path.join(model_path, "vae")
        if not os.path.exists(vae_path):
            vae_path = "stabilityai/sdxl-vae"
    else:
        vae_path = args.vae_path
    print(f"VAE path: {vae_path}")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Load OmniGen model with LoRA
    print("Loading OmniGen model...")
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False

    if args.checkpoint_path:
        print(f"Loading LoRA weights from {args.checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path)

    model = model.to(device)
    model.eval()

    # Load processor
    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    # Setup data
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_arr(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = DatasetFromJson(
        json_file=args.json_file,
        image_path=args.image_path,
        processer=processor,
        image_transform=image_transform,
        max_input_length_limit=1024,
        condition_dropout_prob=0.0,
        keep_raw_resolution=True
    )

    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=True
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # Collect per-layer MSE statistics
    all_results = []
    layer2_mses = []
    layer3_mses = []
    layer4_mses = []

    print(f"\nAnalyzing {args.num_samples} samples...")
    print("=" * 70)

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, total=args.num_samples)):
            if i >= args.num_samples:
                break

            # Get GT pixel images
            output_images_pixel = data['output_images']

            # VAE encode
            output_images = data['output_images']
            input_pixel_values = data['input_pixel_values']

            if isinstance(output_images, list):
                output_images = [vae_encode(vae, img.to(device), torch.float32) for img in output_images]
                if input_pixel_values is not None:
                    input_pixel_values = [vae_encode(vae, img.to(device), torch.float32) for img in input_pixel_values]

            x1 = output_images
            B = len(x1)

            # Sample noise and timestep
            x0 = [torch.randn_like(img) for img in x1]
            t = sample_timestep(B, device)

            # Interpolate
            xt = [t[j] * x1[j] + (1 - t[j]) * x0[j] for j in range(B)]

            # Model forward
            model_kwargs = dict(
                input_ids=data['input_ids'].to(device),
                input_img_latents=input_pixel_values,
                input_image_sizes=data['input_image_sizes'],
                attention_mask=data['attention_mask'].to(device),
                position_ids=data['position_ids'].to(device),
                padding_latent=data['padding_images'],
                past_key_values=None,
                return_past_key_values=False,
            )

            model_output = model(xt, t, **model_kwargs)

            # Reconstruct x1_hat
            x1_hat = [xt[j] + (1 - t[j]) * model_output[j] for j in range(B)]

            # Decode predicted latent
            lat = x1_hat[0].float()
            if lat.dim() == 5:
                lat = lat.squeeze(1)
            lat_scaled = inverse_vae_scale(lat, vae)
            gen_decoded = vae.decode(lat_scaled).sample
            gen_decoded = gen_decoded.clamp(-1.0, 1.0)

            # Resize to 256x256
            if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
                gen_decoded = F.interpolate(gen_decoded, size=(256, 256), mode="bilinear", align_corners=False)

            # Get GT image
            gt_img = output_images_pixel[0]
            if gt_img.dim() == 3:
                gt_img = gt_img.unsqueeze(0)
            gt_img = gt_img.to(device=device, dtype=torch.float32)
            if gt_img.shape[-2] != 256 or gt_img.shape[-1] != 256:
                gt_img = F.interpolate(gt_img, size=(256, 256), mode="bilinear", align_corners=False)

            # Compute per-layer MSE
            result = compute_perlayer_mse(seg_model, gen_decoded, gt_img)
            result["timestep"] = t[0].item()
            result["sample_idx"] = i

            all_results.append(result)
            layer2_mses.append(result["layer2"]["mse"])
            layer3_mses.append(result["layer3"]["mse"])
            layer4_mses.append(result["layer4"]["mse"])

    # Print summary statistics
    print("\n" + "=" * 70)
    print("PER-LAYER MSE ANALYSIS SUMMARY")
    print("=" * 70)

    import numpy as np

    layer2_arr = np.array(layer2_mses)
    layer3_arr = np.array(layer3_mses)
    layer4_arr = np.array(layer4_mses)

    print(f"\n{'Layer':<40} {'Mean MSE':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 88)
    print(f"{'Layer 2 (64ch, stride=4, 1/4 res)':<40} {layer2_arr.mean():>12.6f} {layer2_arr.std():>12.6f} {layer2_arr.min():>12.6f} {layer2_arr.max():>12.6f}")
    print(f"{'Layer 3 (128ch, stride=8, 1/8 res)':<40} {layer3_arr.mean():>12.6f} {layer3_arr.std():>12.6f} {layer3_arr.min():>12.6f} {layer3_arr.max():>12.6f}")
    print(f"{'Layer 4 (256ch, stride=16, 1/16 res)':<40} {layer4_arr.mean():>12.6f} {layer4_arr.std():>12.6f} {layer4_arr.min():>12.6f} {layer4_arr.max():>12.6f}")
    print("-" * 88)

    avg_per_sample = (layer2_arr + layer3_arr + layer4_arr) / 3
    print(f"{'Average (what is logged as anat)':<40} {avg_per_sample.mean():>12.6f} {avg_per_sample.std():>12.6f} {avg_per_sample.min():>12.6f} {avg_per_sample.max():>12.6f}")

    # Compute contribution ratio
    total_per_sample = layer2_arr + layer3_arr + layer4_arr
    print(f"\n{'Layer Contribution Ratio (of total):':<40}")
    print(f"  Layer 2: {(layer2_arr.sum() / total_per_sample.sum()) * 100:.2f}%")
    print(f"  Layer 3: {(layer3_arr.sum() / total_per_sample.sum()) * 100:.2f}%")
    print(f"  Layer 4: {(layer4_arr.sum() / total_per_sample.sum()) * 100:.2f}%")

    # Analyze timestep dependency
    print("\n" + "=" * 70)
    print("TIMESTEP DEPENDENCY ANALYSIS")
    print("=" * 70)

    timesteps = np.array([r["timestep"] for r in all_results])

    # Bin by timestep
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"\n{'t range':<15} {'Count':>8} {'Avg L2':>12} {'Avg L3':>12} {'Avg L4':>12} {'Avg Total':>12}")
    print("-" * 75)

    for low, high in bins:
        mask = (timesteps >= low) & (timesteps < high)
        if mask.sum() > 0:
            print(f"[{low:.1f}, {high:.1f}){'':<7} {mask.sum():>8} {layer2_arr[mask].mean():>12.6f} {layer3_arr[mask].mean():>12.6f} {layer4_arr[mask].mean():>12.6f} {avg_per_sample[mask].mean():>12.6f}")

    # Save detailed results
    output_path = args.output_json
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "layer2": {"mean": float(layer2_arr.mean()), "std": float(layer2_arr.std())},
                    "layer3": {"mean": float(layer3_arr.mean()), "std": float(layer3_arr.std())},
                    "layer4": {"mean": float(layer4_arr.mean()), "std": float(layer4_arr.std())},
                    "average": {"mean": float(avg_per_sample.mean()), "std": float(avg_per_sample.std())},
                },
                "samples": all_results,
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="Shitao/OmniGen-v1")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--vae_path", type=str,
                        default=None,
                        help="VAE path, defaults to model_name_or_path/vae")
    parser.add_argument("--seg_model_ckpt", type=str,
                        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth")
    parser.add_argument("--json_file", type=str,
                        default="/home/wenting/zr/gen_code/datasets/cxr_train_data.json")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save detailed JSON results")

    args = parser.parse_args()
    main(args)
