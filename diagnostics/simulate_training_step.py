#!/usr/bin/env python3
"""
Dynamic Training Probe Script: Simulate Forward Pass and Track Layer-wise MSE

This script simulates the exact forward pass that occurs during anatomy-aware training:
1. Load GT CXR image -> encode to latent x_1
2. For each timestep t in [0.1, 0.5, 0.9]:
   - Sample noise x_0 ~ N(0, I)
   - Create noisy sample: x_t = t * x_1 + (1 - t) * x_0
   - Run OmniGen to predict velocity u_hat
   - Reconstruct clean image: x_1_hat = x_t + (1 - t) * u_hat
   - Decode through VAE
   - Compute layer-wise MSE (Layer 2, 3, 4) vs GT

This reveals:
- How reconstruction quality varies with noise level
- Which layers dominate the loss at different timesteps
- The exact feature gap behavior during training

Usage:
    python simulate_training_step.py \
        --checkpoint_step 10000 \
        --gt_image /path/to/gt_cxr.png \
        --seg_ckpt /path/to/best_anatomy_model.pth \
        --base_model_path /path/to/OmniGen-v1 \
        --results_dir /path/to/anatomy_training_results

Author: Claude (for Dr. Zhengrong Research Team)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add necessary paths
sys.path.insert(0, "/home/wenting/zr/gen_code")
sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")

from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from peft import PeftModel, LoraConfig

# Local imports
from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.transformer import Phi3Config
import segmentation_models_pytorch as smp


# =============================================================================
# Feature Layer Configuration
# =============================================================================

FEATURE_INDICES = [2, 3, 4]
FEATURE_INFO = {
    2: {"name": "Layer 2", "channels": 64, "spatial": "64x64", "stride": 4},
    3: {"name": "Layer 3", "channels": 128, "spatial": "32x32", "stride": 8},
    4: {"name": "Layer 4", "channels": 256, "spatial": "16x16", "stride": 16},
}


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_seg_model(ckpt_path, device):
    """Load frozen ResNet34-UNet segmentation model."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"[INFO] Loaded segmentation model from: {ckpt_path}")
    if "val_dice" in ckpt:
        print(f"[INFO] Model validation Dice: {ckpt['val_dice']:.4f}")

    return model


def load_omnigen_with_lora(base_model_path, adapter_path, device, dtype=torch.bfloat16):
    """
    Load OmniGen base model and apply LoRA adapter.

    Args:
        base_model_path: Path to base OmniGen model
        adapter_path: Path to LoRA adapter checkpoint directory
        device: torch device
        dtype: model dtype

    Returns:
        model: OmniGen model with LoRA applied
    """
    print(f"[INFO] Loading OmniGen base model from: {base_model_path}")
    model = OmniGen.from_pretrained(base_model_path)
    model.llm.config.use_cache = False

    # Load LoRA adapter
    print(f"[INFO] Loading LoRA adapter from: {adapter_path}")

    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        # Load using PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        print("[INFO] LoRA adapter loaded via PeftModel.from_pretrained")
    else:
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    return model


def load_vae(base_model_path, device):
    """Load VAE encoder/decoder."""
    vae_path = os.path.join(base_model_path, "vae")
    if os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path)
    else:
        print("[INFO] No VAE found in model, downloading stabilityai/sdxl-vae")
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)

    return vae


# =============================================================================
# VAE Operations
# =============================================================================

def vae_encode(vae, x):
    """Encode image to latent space."""
    with torch.no_grad():
        if vae.config.shift_factor is not None:
            latent = vae.encode(x).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
    return latent


def vae_decode(vae, latent):
    """Decode latent to image space."""
    if vae.config.shift_factor is not None:
        latent_scaled = latent / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latent_scaled = latent / vae.config.scaling_factor

    decoded = vae.decode(latent_scaled).sample
    decoded = decoded.clamp(-1.0, 1.0)
    return decoded


# =============================================================================
# Image Loading
# =============================================================================

def load_image_as_tensor(image_path, size=256, device="cuda"):
    """
    Load image and convert to [-1, 1] range tensor.

    Args:
        image_path: Path to image file
        size: Target size (square)
        device: torch device

    Returns:
        tensor: (1, 3, size, size) in [-1, 1]
        img_np: numpy array [0, 255]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    img_np = np.array(img)

    # Normalize to [-1, 1]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
    tensor = tensor.unsqueeze(0).to(device)

    return tensor, img_np


# =============================================================================
# Feature Matching Loss (Layer-wise)
# =============================================================================

def compute_layerwise_mse(seg_model, gen_images, gt_images):
    """
    Compute MSE for each layer separately.

    Args:
        seg_model: Frozen ResNet34-UNet
        gen_images: (B, 3, H, W) generated images in [-1, 1]
        gt_images: (B, 3, H, W) ground truth images in [-1, 1]

    Returns:
        dict with layer-wise MSE values
    """
    # Extract features
    with torch.no_grad():
        gt_features = seg_model.encoder(gt_images)

    with torch.no_grad():
        gen_features = seg_model.encoder(gen_images)

    # Compute MSE for each layer
    results = {}
    total_mse = 0.0

    for idx in FEATURE_INDICES:
        gt_feat = gt_features[idx]
        gen_feat = gen_features[idx]

        mse = F.mse_loss(gen_feat, gt_feat, reduction='mean').item()
        results[idx] = mse
        total_mse += mse

    results['average'] = total_mse / len(FEATURE_INDICES)

    return results


# =============================================================================
# Diffusion Simulation
# =============================================================================

def create_noisy_sample(x1, t):
    """
    Create noisy sample x_t from clean latent x_1.

    Rectified Flow formulation:
        x_t = t * x_1 + (1 - t) * x_0
        where x_0 ~ N(0, I)

    Args:
        x1: Clean latent (B, C, H, W)
        t: Timestep scalar [0, 1]

    Returns:
        x_t: Noisy latent
        x_0: Sampled noise
        u_t: Target velocity (x_1 - x_0)
    """
    x0 = torch.randn_like(x1)
    x_t = t * x1 + (1 - t) * x0
    u_t = x1 - x0

    return x_t, x0, u_t


def run_omnigen_forward(model, x_t, t_value, processor, device, dtype):
    """
    Run OmniGen forward pass to predict velocity.

    This simulates exactly what happens during training:
    - Model receives noisy latent x_t and timestep t
    - Model predicts velocity u_hat

    Args:
        model: OmniGen model with LoRA
        x_t: Noisy latent (1, C, H, W)
        t_value: Timestep scalar [0, 1]
        processor: OmniGenProcessor for text encoding
        device: torch device
        dtype: model dtype

    Returns:
        u_hat: Predicted velocity (1, C, H, W)
    """
    # Create timestep tensor
    t_tensor = torch.tensor([t_value], device=device, dtype=dtype)

    # For training simulation, we use minimal text prompt
    # This matches the CXR fine-tuning setup
    prompt = "Generate a chest X-ray image"

    # Process the prompt to get input_ids
    model_kwargs = processor.process_multi_modal_prompt(prompt, [])
    input_ids = model_kwargs.pop('input_ids').to(device)

    # Prepare input_img_latents (empty for generation)
    input_img_latents = None

    # Run model forward
    with torch.no_grad():
        # OmniGen forward expects x_t as list for variable resolution
        # or tensor for fixed resolution
        model_output = model(
            x=[x_t],  # List format for variable resolution
            timestep=t_tensor,
            input_ids=input_ids,
            input_img_latents=input_img_latents,
            input_image_sizes=None,
            attention_mask=model_kwargs.get('attention_mask', None).to(device),
            position_ids=model_kwargs.get('position_ids', None).to(device),
            padding_latent=None,
            past_key_values=None,
            return_past_key_values=False,
            offload_model=False,
        )

    # model_output is list for variable resolution
    if isinstance(model_output, list):
        u_hat = model_output[0]
    else:
        u_hat = model_output

    return u_hat


def simulate_training_step_simple(model, vae, x1, t_value, device, dtype):
    """
    Simplified training step simulation without text conditioning.

    For diagnostic purposes, we directly test the velocity prediction
    by bypassing the full text encoding pipeline.

    Args:
        model: OmniGen model
        vae: VAE for decoding
        x1: Clean latent (1, C, H, W)
        t_value: Timestep [0, 1]
        device: torch device
        dtype: model dtype

    Returns:
        x1_hat_decoded: Reconstructed image in [-1, 1]
        x_t: Noisy latent
        x0: Sampled noise
    """
    # Step 1: Create noisy sample
    x_t, x0, u_t = create_noisy_sample(x1, t_value)

    # Step 2: For this diagnostic, we'll use a simpler approach:
    # Instead of running full OmniGen forward (which requires complex text encoding),
    # we'll directly compute what the "perfect" reconstruction would be
    # and compare against a "noisy" reconstruction to understand behavior

    # Perfect reconstruction (if model predicted u_t perfectly):
    # x1_hat = x_t + (1 - t) * u_t = t*x1 + (1-t)*x0 + (1-t)*(x1-x0) = x1

    # For this diagnostic, we assume the model output matches training behavior
    # We'll decode x_t directly to see what the "input" looks like at each t

    # Decode noisy latent to see noise level
    x_t_decoded = vae_decode(vae, x_t.float())

    # Also decode clean latent for reference
    x1_decoded = vae_decode(vae, x1.float())

    return x1_decoded, x_t_decoded, x_t, x0


# =============================================================================
# Main Simulation Loop
# =============================================================================

def run_simulation(args):
    """Run the complete training step simulation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("=" * 70)
    print("TRAINING STEP SIMULATION: Layer-wise MSE Analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint Step: {args.checkpoint_step}")
    print()

    # -------------------------------------------------------------------------
    # Load Models
    # -------------------------------------------------------------------------
    print("[STEP 1] Loading models...")

    # Load segmentation model
    seg_model = load_seg_model(args.seg_ckpt, device)

    # Load VAE
    vae = load_vae(args.base_model_path, device)

    # Build adapter path from checkpoint step
    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        # Default anatomy training results path
        adapter_path = os.path.join(
            args.results_dir,
            "checkpoints",
            f"{args.checkpoint_step:07d}"
        )

    # Load OmniGen with LoRA (only if we need model forward pass)
    # For simplified diagnostic, we skip full model loading
    print(f"[INFO] Adapter path: {adapter_path}")

    # Load processor for potential model forward pass
    processor = OmniGenProcessor.from_pretrained(args.base_model_path)

    print()

    # -------------------------------------------------------------------------
    # Load GT Image
    # -------------------------------------------------------------------------
    print("[STEP 2] Loading GT image...")

    gt_tensor_256, gt_np = load_image_as_tensor(args.gt_image, size=256, device=device)
    print(f"GT image loaded: {args.gt_image}")
    print(f"GT tensor shape: {gt_tensor_256.shape}, range: [{gt_tensor_256.min():.2f}, {gt_tensor_256.max():.2f}]")

    # For VAE encoding, we need 1024x1024 if using standard SDXL VAE latent size
    # But for 256x256 images, we'll use the 256 directly and get 32x32 latent
    gt_tensor_vae = gt_tensor_256.float()

    print()

    # -------------------------------------------------------------------------
    # Encode to Latent Space
    # -------------------------------------------------------------------------
    print("[STEP 3] Encoding image to latent space...")

    with torch.no_grad():
        x1 = vae_encode(vae, gt_tensor_vae)

    print(f"Latent x_1 shape: {x1.shape}")  # Should be (1, 4, H/8, W/8)
    print()

    # -------------------------------------------------------------------------
    # Simulate Forward Pass at Different Timesteps
    # -------------------------------------------------------------------------
    print("[STEP 4] Simulating forward pass at different timesteps...")
    print()

    timesteps = [0.9, 0.5, 0.1]  # High noise to low noise

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    all_results = {}

    for t_val in timesteps:
        print(f"{'='*60}")
        print(f"=== Timestep t={t_val} ({'High Noise' if t_val > 0.7 else 'Mid Noise' if t_val > 0.3 else 'Low Noise'}) ===")
        print(f"{'='*60}")

        # Create noisy sample
        x_t, x0, u_t = create_noisy_sample(x1, t_val)

        # Reconstruct x1_hat assuming perfect velocity prediction
        # x1_hat = x_t + (1 - t) * u_hat
        # If u_hat = u_t = x1 - x0, then x1_hat = x1 (perfect reconstruction)
        # Here we simulate what happens when model predicts correctly
        x1_hat = x_t + (1 - t_val) * u_t  # Perfect case

        # Decode reconstructed latent
        x1_hat_decoded = vae_decode(vae, x1_hat.float())

        # Resize to 256x256 for seg model if needed
        if x1_hat_decoded.shape[-1] != 256:
            x1_hat_decoded = F.interpolate(
                x1_hat_decoded, size=(256, 256),
                mode='bilinear', align_corners=False
            )

        # Compute layer-wise MSE
        mse_results = compute_layerwise_mse(seg_model, x1_hat_decoded, gt_tensor_256)

        print(f"Layer 2 MSE: {mse_results[2]:.6f}")
        print(f"Layer 3 MSE: {mse_results[3]:.6f}")
        print(f"Layer 4 MSE: {mse_results[4]:.6f}")
        print(f"Average MSE: {mse_results['average']:.6f}")
        print()

        all_results[t_val] = mse_results

        # Also show MSE if we just decode x_t directly (to see noise impact)
        x_t_decoded = vae_decode(vae, x_t.float())
        if x_t_decoded.shape[-1] != 256:
            x_t_decoded = F.interpolate(
                x_t_decoded, size=(256, 256),
                mode='bilinear', align_corners=False
            )

        mse_noisy = compute_layerwise_mse(seg_model, x_t_decoded, gt_tensor_256)
        print(f"--- For reference: x_t (noisy input) vs GT ---")
        print(f"x_t Layer 2 MSE: {mse_noisy[2]:.6f}")
        print(f"x_t Layer 3 MSE: {mse_noisy[3]:.6f}")
        print(f"x_t Layer 4 MSE: {mse_noisy[4]:.6f}")
        print(f"x_t Average MSE: {mse_noisy['average']:.6f}")
        print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY: Perfect Reconstruction MSE (x1_hat = x1 case)")
    print("=" * 70)
    print()
    print("If model predicts velocity perfectly (u_hat = u_t = x1 - x0),")
    print("then x1_hat = x_t + (1-t) * u_hat = x1 exactly.")
    print()
    print("The MSE values above should be ~0 in this ideal case,")
    print("showing that the reconstruction formula is correct.")
    print()
    print("In actual training, u_hat != u_t, so x1_hat != x1,")
    print("and the layer-wise MSE will be non-zero based on model error.")
    print()

    # Show the baseline: GT vs GT (should be 0)
    print("-" * 60)
    print("BASELINE CHECK: GT vs GT (should be exactly 0)")
    print("-" * 60)
    baseline_mse = compute_layerwise_mse(seg_model, gt_tensor_256, gt_tensor_256)
    print(f"Layer 2 MSE: {baseline_mse[2]:.10f}")
    print(f"Layer 3 MSE: {baseline_mse[3]:.10f}")
    print(f"Layer 4 MSE: {baseline_mse[4]:.10f}")
    print()

    return all_results


def run_with_model_forward(args):
    """
    Run simulation with actual OmniGen model forward pass.

    This version loads the full model and runs actual inference
    to see real reconstruction quality.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("=" * 70)
    print("TRAINING STEP SIMULATION WITH MODEL FORWARD PASS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint Step: {args.checkpoint_step}")
    print()

    # -------------------------------------------------------------------------
    # Load Models
    # -------------------------------------------------------------------------
    print("[STEP 1] Loading models...")

    # Load segmentation model
    seg_model = load_seg_model(args.seg_ckpt, device)

    # Load VAE
    vae = load_vae(args.base_model_path, device)

    # Build adapter path
    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        adapter_path = os.path.join(
            args.results_dir,
            "checkpoints",
            f"{args.checkpoint_step:07d}"
        )

    # Load OmniGen with LoRA
    model = load_omnigen_with_lora(args.base_model_path, adapter_path, device, dtype)

    # Load processor
    processor = OmniGenProcessor.from_pretrained(args.base_model_path)

    print()

    # -------------------------------------------------------------------------
    # Load GT Image
    # -------------------------------------------------------------------------
    print("[STEP 2] Loading GT image...")

    gt_tensor_256, gt_np = load_image_as_tensor(args.gt_image, size=256, device=device)
    print(f"GT image loaded: {args.gt_image}")

    # Encode to latent
    gt_tensor_vae = gt_tensor_256.float()
    x1 = vae_encode(vae, gt_tensor_vae)
    print(f"Latent x_1 shape: {x1.shape}")
    print()

    # -------------------------------------------------------------------------
    # Run Forward Pass at Different Timesteps
    # -------------------------------------------------------------------------
    print("[STEP 3] Running model forward pass...")
    print()

    timesteps = [0.9, 0.5, 0.1]
    torch.manual_seed(42)

    for t_val in timesteps:
        print(f"{'='*60}")
        print(f"=== Timestep t={t_val} ===")
        print(f"{'='*60}")

        # Create noisy sample
        x_t, x0, u_t = create_noisy_sample(x1.to(dtype), t_val)

        # Run model forward to get u_hat
        try:
            u_hat = run_omnigen_forward(
                model, x_t, t_val, processor, device, dtype
            )

            # Reconstruct x1_hat
            x1_hat = x_t + (1 - t_val) * u_hat

            # Decode
            x1_hat_decoded = vae_decode(vae, x1_hat.float())

            # Resize if needed
            if x1_hat_decoded.shape[-1] != 256:
                x1_hat_decoded = F.interpolate(
                    x1_hat_decoded, size=(256, 256),
                    mode='bilinear', align_corners=False
                )

            # Compute layer-wise MSE
            mse_results = compute_layerwise_mse(seg_model, x1_hat_decoded, gt_tensor_256)

            print(f"Layer 2 MSE: {mse_results[2]:.6f}")
            print(f"Layer 3 MSE: {mse_results[3]:.6f}")
            print(f"Layer 4 MSE: {mse_results[4]:.6f}")
            print(f"Average MSE: {mse_results['average']:.6f}")

        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

        print()


# =============================================================================
# Entry Point
# =============================================================================

def compare_generated_vs_gt(args):
    """
    Directly compare generated images (from inference) against GT images.

    This mode is useful for analyzing already-generated outputs
    without needing to run the full model forward pass.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DIRECT COMPARISON: Generated Image vs GT Image")
    print("=" * 70)
    print()

    # Load segmentation model
    seg_model = load_seg_model(args.seg_ckpt, device)
    print()

    # Load images
    print("[INFO] Loading images...")
    gt_tensor, gt_np = load_image_as_tensor(args.gt_image, size=256, device=device)
    gen_tensor, gen_np = load_image_as_tensor(args.gen_image, size=256, device=device)

    print(f"GT image: {args.gt_image}")
    print(f"Generated image: {args.gen_image}")
    print()

    # Compute layer-wise MSE
    mse_results = compute_layerwise_mse(seg_model, gen_tensor, gt_tensor)

    print("=" * 60)
    print("LAYER-WISE MSE RESULTS")
    print("=" * 60)
    print(f"Layer 2 MSE: {mse_results[2]:.6f}  ({FEATURE_INFO[2]['channels']}ch, {FEATURE_INFO[2]['spatial']})")
    print(f"Layer 3 MSE: {mse_results[3]:.6f}  ({FEATURE_INFO[3]['channels']}ch, {FEATURE_INFO[3]['spatial']})")
    print(f"Layer 4 MSE: {mse_results[4]:.6f}  ({FEATURE_INFO[4]['channels']}ch, {FEATURE_INFO[4]['spatial']})")
    print("-" * 60)
    print(f"Average MSE: {mse_results['average']:.6f}")
    print()

    # Show which layer dominates
    layer_names = {2: "Layer 2", 3: "Layer 3", 4: "Layer 4"}
    max_layer = max(FEATURE_INDICES, key=lambda x: mse_results[x])
    min_layer = min(FEATURE_INDICES, key=lambda x: mse_results[x])

    print(f"Dominant Layer (highest MSE): {layer_names[max_layer]} ({mse_results[max_layer]:.4f})")
    print(f"Smallest Gap Layer: {layer_names[min_layer]} ({mse_results[min_layer]:.4f})")
    print()

    return mse_results


def batch_compare_images(args):
    """
    Compare multiple image pairs in batch mode.

    Expects --gt_dir and --gen_dir with matching filenames.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("BATCH COMPARISON MODE")
    print("=" * 70)
    print()

    # Load segmentation model
    seg_model = load_seg_model(args.seg_ckpt, device)
    print()

    # Find matching files
    import glob
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    gen_files = sorted(glob.glob(os.path.join(args.gen_dir, "*.png")))

    print(f"Found {len(gt_files)} GT files and {len(gen_files)} generated files")
    print()

    # Match by filename
    gt_basenames = {os.path.basename(f): f for f in gt_files}
    gen_basenames = {os.path.basename(f): f for f in gen_files}

    matched = set(gt_basenames.keys()) & set(gen_basenames.keys())
    print(f"Matched pairs: {len(matched)}")
    print()

    # Process each pair
    all_results = {2: [], 3: [], 4: [], 'average': []}

    for fname in sorted(matched)[:args.max_images]:
        gt_tensor, _ = load_image_as_tensor(gt_basenames[fname], size=256, device=device)
        gen_tensor, _ = load_image_as_tensor(gen_basenames[fname], size=256, device=device)

        mse = compute_layerwise_mse(seg_model, gen_tensor, gt_tensor)

        for key in all_results:
            all_results[key].append(mse[key])

    # Print statistics
    print("=" * 60)
    print(f"AGGREGATE STATISTICS (over {len(all_results[2])} image pairs)")
    print("=" * 60)

    for idx in FEATURE_INDICES:
        vals = all_results[idx]
        print(f"{FEATURE_INFO[idx]['name']}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    avg_vals = all_results['average']
    print(f"Average: mean={np.mean(avg_vals):.4f}, std={np.std(avg_vals):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate training step to observe layer-wise MSE"
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="simulate",
        choices=["simulate", "compare", "batch"],
        help="Mode: 'simulate' for forward pass simulation, 'compare' for image comparison, 'batch' for batch comparison"
    )

    parser.add_argument(
        "--checkpoint_step", type=int, default=10000,
        help="Training checkpoint step (e.g., 8000, 10000)"
    )
    parser.add_argument(
        "--gt_image", type=str,
        default="/home/wenting/zr/Segmentation/data/lidc_TotalSeg/LIDC-IDRI-0001/04_drr_256/cxr/000000.png",
        help="Path to ground truth CXR image"
    )
    parser.add_argument(
        "--gen_image", type=str, default=None,
        help="Path to generated image (for 'compare' mode)"
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Directory with GT images (for 'batch' mode)"
    )
    parser.add_argument(
        "--gen_dir", type=str, default=None,
        help="Directory with generated images (for 'batch' mode)"
    )
    parser.add_argument(
        "--max_images", type=int, default=100,
        help="Maximum images to process in batch mode"
    )
    parser.add_argument(
        "--seg_ckpt", type=str,
        default="/home/wenting/zr/Segmentation/checkpoints/train_Seg/best_anatomy_model.pth",
        help="Path to segmentation model checkpoint"
    )
    parser.add_argument(
        "--base_model_path", type=str,
        default="/home/wenting/.cache/huggingface/hub/models--Shitao--OmniGen-v1/snapshots/58e249c7c7634423c0ba41c34a774af79aa87889",
        help="Path to base OmniGen model"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/wenting/zr/gen_code/results/cxr_finetune_lora_30ksteps_feature_lamda0.005_subbatch16",
        help="Path to anatomy training results directory"
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Direct path to LoRA adapter (overrides results_dir/checkpoints)"
    )
    parser.add_argument(
        "--use_model_forward", action="store_true",
        help="Run actual model forward pass (requires more VRAM)"
    )

    args = parser.parse_args()

    # Run based on mode
    if args.mode == "compare":
        if args.gen_image is None:
            parser.error("--gen_image is required for 'compare' mode")
        compare_generated_vs_gt(args)
    elif args.mode == "batch":
        if args.gt_dir is None or args.gen_dir is None:
            parser.error("--gt_dir and --gen_dir are required for 'batch' mode")
        batch_compare_images(args)
    elif args.use_model_forward:
        run_with_model_forward(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
