"""
Anatomy-Aware Loss v2: Mask-Weighted Feature Matching

================================================================================
APPROACH: Instead of pixel-level BCE+Dice on segmentation logits, we perform
MASK-WEIGHTED FEATURE MATCHING at a single intermediate encoder layer
(1/4 resolution = features[2], 64x64 for 256x256 input).

This provides:
  - Spatial tolerance (protects SSIM/PSNR vs hard pixel-level constraints)
  - Decoupled per-organ loss (10 anatomical channels)
  - Focus on anatomically relevant regions only

MATHEMATICAL FLOW:
  1. Forward Pass: X_gt, X_gen -> seg_model.encoder -> F_gt, F_gen (64x64)
     Also: seg_model.decoder + segmentation_head on gen encoder features
           -> Mask_gen (10ch, 256x256, sigmoid)
     (Mask_gt is loaded from .npz dataloader)
  2. Downsample: Mask_gt, Mask_gen -> 64x64 via avg_pool2d
  3. Per-Organ MSE: For each channel c in [0,9]:
       OrganFeature_gen^c = M_gen^c * F_gen
       OrganFeature_gt^c  = M_gt^c  * F_gt
       MSE_c = MSE(OrganFeature_gen^c, OrganFeature_gt^c)
  4. Sum: Loss_anatomy = sum(MSE_c)

LAYER SELECTION (ResNet34 encoder, 256x256 input):
    - Feature 0: (B, 3, 256, 256)   stride=1  - original input, SKIP
    - Feature 1: (B, 64, 128, 128)  stride=2  - initial conv, SKIP
    - Feature 2: (B, 64, 64, 64)    stride=4  - 1/4 res, USE <- TARGET
    - Feature 3: (B, 128, 32, 32)   stride=8  - 1/8 res, SKIP
    - Feature 4: (B, 256, 16, 16)   stride=16 - 1/16 res, SKIP
    - Feature 5: (B, 512, 8, 8)     stride=32 - too high-level, SKIP

GOLDEN RULE COMPLIANCE:
  - The main diffusion logic (u_hat, loss_diffusion) is NEVER modified
  - All tensor reshaping happens ONLY on cloned/sub-batched tensors
  - loss_diffusion remains mathematically identical to original OmniGen
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Anatomical Channel Names (for reference/logging)
# =============================================================================
ANATOMY_CHANNELS = [
    "Lung_Left",      # Channel 0
    "Lung_Right",     # Channel 1
    "Heart",          # Channel 2
    "Aorta",          # Channel 3
    "Liver",          # Channel 4
    "Stomach",        # Channel 5
    "Trachea",        # Channel 6
    "Ribs",           # Channel 7
    "Vertebrae",      # Channel 8
    "Upper_Skeleton", # Channel 9
]


# =============================================================================
# Core Utility Functions (shared with v1)
# =============================================================================

def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sample_x0(x1):
    """Sample Gaussian noise matching shape of x1."""
    if isinstance(x1, (list, tuple)):
        return [torch.randn_like(img) for img in x1]
    return torch.randn_like(x1)


def sample_timestep(x1):
    """Sample timestep from logistic-normal distribution."""
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def inverse_vae_scale(latents, vae):
    """Reverse the VAE scaling applied during encoding."""
    if vae.config.shift_factor is not None:
        return latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        return latents / vae.config.scaling_factor


def _ensure_4d_latent(lat):
    """Ensure latent tensor is exactly 4D: (B, C, H, W)."""
    ndim = lat.dim()
    if ndim == 5:
        if lat.shape[1] == 1:
            lat = lat.squeeze(1)
        else:
            raise ValueError(
                f"5D latent with NumImages > 1 not supported. Got shape {lat.shape}"
            )
    elif ndim == 3:
        lat = lat.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(
            f"Expected latent with 3, 4, or 5 dimensions, got {ndim}D with shape {lat.shape}"
        )
    return lat


def _ensure_4d_image(img):
    """Ensure image tensor is exactly 4D: (B, C, H, W)."""
    ndim = img.dim()
    if ndim == 3:
        img = img.unsqueeze(0)
    elif ndim != 4:
        raise ValueError(
            f"Expected image with 3 or 4 dimensions, got {ndim}D with shape {img.shape}"
        )
    return img


# =============================================================================
# Mask-Weighted Feature Matching Loss (CORE)
# =============================================================================

def compute_mask_weighted_feature_loss(
    seg_model,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    mask_gt: torch.Tensor,
    feature_layer_idx: int = 2,
    use_gen_mask: bool = True,
) -> torch.Tensor:
    """
    Compute mask-weighted feature matching loss.

    Extracts encoder features at the specified layer, then computes per-organ
    MSE weighted by anatomical masks. Efficient: encoder features are reused
    for mask prediction (decoder+head only, no redundant encoder pass).

    Args:
        seg_model: Frozen smp.Unet (eval mode, requires_grad=False)
        gen_images: (B, 3, H, W) in [-1, 1], WITH autograd graph
        gt_images:  (B, 3, H, W) in [-1, 1], no grad needed
        mask_gt:    (B, 10, 256, 256) ground-truth binary masks in [0, 1]
        feature_layer_idx: Encoder layer index (2 = 1/4 res for ResNet34)
        use_gen_mask: If True, predict mask from gen image for weighting gen features;
                      If False, use GT mask for both (more stable early training)

    Returns:
        loss_anatomy: Scalar tensor with gradient attached
    """
    device = gen_images.device
    gt_images = gt_images.to(device=device, dtype=torch.float32)
    mask_gt = mask_gt.to(device=device, dtype=torch.float32)

    # === Step 1: Extract encoder features ===
    # GT: no grad needed (frozen reference)
    with torch.no_grad():
        gt_features_all = seg_model.encoder(gt_images)
        F_gt = gt_features_all[feature_layer_idx]  # (B, C_feat, H_feat, W_feat)

    # Gen: WITH grad — gradients flow through frozen seg encoder back to gen_images
    gen_features_all = seg_model.encoder(gen_images)
    F_gen = gen_features_all[feature_layer_idx]  # (B, C_feat, H_feat, W_feat)

    _, C_feat, H_feat, W_feat = F_gen.shape

    # === Step 2: Get masks ===
    if use_gen_mask:
        # Reuse already-computed encoder features: only run decoder + segmentation_head
        # This avoids a redundant full forward pass through the encoder
        decoder_output = seg_model.decoder(gen_features_all)
        logits_gen = seg_model.segmentation_head(decoder_output)  # (B, 10, 256, 256)
        mask_gen = torch.sigmoid(logits_gen)
    else:
        mask_gen = mask_gt

    # === Step 3: Downsample masks to feature resolution ===
    input_H = mask_gt.shape[-2]  # 256
    if H_feat != input_H:
        kernel_size = input_H // H_feat  # 256 // 64 = 4
        mask_gt_down = F.avg_pool2d(mask_gt, kernel_size=kernel_size)    # (B, 10, 64, 64)
        mask_gen_down = F.avg_pool2d(mask_gen, kernel_size=kernel_size)  # (B, 10, 64, 64)
    else:
        mask_gt_down = mask_gt
        mask_gen_down = mask_gen

    # === Step 4: Per-channel mask-weighted MSE ===
    # CRITICAL: Each organ channel computed independently, then summed (NOT averaged).
    loss_total = torch.tensor(0.0, device=device)

    for c in range(10):
        # (B, 1, H_feat, W_feat) — broadcasts with (B, C_feat, H_feat, W_feat)
        M_gt_c = mask_gt_down[:, c:c+1, :, :]
        M_gen_c = mask_gen_down[:, c:c+1, :, :]

        OrganFeature_gt_c = M_gt_c * F_gt    # (B, C_feat, H_feat, W_feat)
        OrganFeature_gen_c = M_gen_c * F_gen  # (B, C_feat, H_feat, W_feat)

        mse_c = F.mse_loss(OrganFeature_gen_c, OrganFeature_gt_c, reduction='mean')
        loss_total = loss_total + mse_c

    return loss_total


# =============================================================================
# Main Training Loss Function
# =============================================================================

def training_losses_with_anatomy_v2(
    model,
    x1,
    model_kwargs,
    output_images_pixel,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy: float = 0.1,
    anatomy_subbatch_size: int = 4,
    feature_layer_idx: int = 2,
    use_gen_mask: bool = True,
):
    """
    Combined rectified-flow + mask-weighted anatomy-aware loss.

    GOLDEN RULE: loss_diffusion is IDENTICAL to original OmniGen.

    Args:
        model: OmniGen model (with LoRA)
        x1: VAE-encoded target latent, (B, C, H, W) tensor OR list of (1, C, H, W)
        model_kwargs: dict for model forward pass
        output_images_pixel: GT images in pixel space, [-1, 1] range.
            - List of (1, 3, H, W) for variable-resolution, or (B, 3, H, W) tensor
        output_anatomy_masks: (B, 10, 256, 256) float32 binary masks from dataloader
        vae: Frozen AutoencoderKL (fp32, requires_grad=False)
        seg_model: Frozen smp.Unet segmentation model
        lambda_anatomy: Weight for anatomy loss
        anatomy_subbatch_size: Max samples to decode per step (VRAM safety)
        feature_layer_idx: Encoder layer index (2 = 1/4 res)
        use_gen_mask: Whether to predict mask for gen images (True) or use GT (False)

    Returns:
        dict: loss_total, loss_diffusion, loss_anatomy, anatomy_subbatch_size_actual,
              feature_layer_idx
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = len(x1)

    # ==========================================================================
    # GOLDEN RULE: Rectified Flow — IDENTICAL to original OmniGen
    # ==========================================================================
    x0 = sample_x0(x1)
    t = sample_timestep(x1)

    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    # ==========================================================================
    # GOLDEN RULE: Model forward — IDENTICAL to original OmniGen
    # ==========================================================================
    model_output = model(xt, t, **model_kwargs)

    # ==========================================================================
    # GOLDEN RULE: Diffusion loss (MSE) — IDENTICAL to original OmniGen
    # ==========================================================================
    if isinstance(x1, (list, tuple)):
        loss_diffusion = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        ).mean()
    else:
        loss_diffusion = mean_flat((model_output - ut) ** 2).mean()

    # ==========================================================================
    # AUXILIARY ANATOMY BRANCH (Mask-Weighted Feature Matching)
    # Operates on CLONED/INDEXED tensors only. Main tensors untouched.
    # ==========================================================================

    n = min(anatomy_subbatch_size, B)

    if isinstance(x1, (list, tuple)):
        # --- Variable-resolution path ---
        x1_hat = [xt[i] + (1 - t[i]) * model_output[i] for i in range(B)]
        idx = torch.randperm(B, device=model_output[0].device)[:n]

        anatomy_losses = []
        for i_idx in idx:
            i = i_idx.item()

            # Decode predicted latent -> generated image (WITH grad)
            lat = _ensure_4d_latent(x1_hat[i]).float()
            lat_scaled = inverse_vae_scale(lat, vae)
            gen_decoded = vae.decode(lat_scaled).sample.clamp(-1.0, 1.0)

            if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
                gen_decoded = F.interpolate(
                    gen_decoded, size=(256, 256),
                    mode="bilinear", align_corners=False
                )

            # Get GT image in pixel space
            gt_img = _ensure_4d_image(output_images_pixel[i])
            gt_img = gt_img.to(device=gen_decoded.device, dtype=torch.float32)
            if gt_img.shape[-2] != 256 or gt_img.shape[-1] != 256:
                gt_img = F.interpolate(
                    gt_img, size=(256, 256),
                    mode="bilinear", align_corners=False
                )

            mask_gt_i = output_anatomy_masks[i:i+1].to(
                device=gen_decoded.device, dtype=torch.float32
            )

            loss_i = compute_mask_weighted_feature_loss(
                seg_model, gen_decoded, gt_img, mask_gt_i,
                feature_layer_idx=feature_layer_idx,
                use_gen_mask=use_gen_mask,
            )
            anatomy_losses.append(loss_i)

        loss_anatomy = torch.stack(anatomy_losses).mean()

    else:
        # --- Fixed-resolution path ---
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        x1_hat = xt + (1 - t_) * model_output

        idx = torch.randperm(B, device=x1_hat.device)[:n]

        x1_hat_sub = _ensure_4d_latent(x1_hat[idx]).float()
        x1_hat_sub_scaled = inverse_vae_scale(x1_hat_sub, vae)
        gen_decoded = vae.decode(x1_hat_sub_scaled).sample.clamp(-1.0, 1.0)

        if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
            gen_decoded = F.interpolate(
                gen_decoded, size=(256, 256),
                mode="bilinear", align_corners=False
            )

        gt_imgs = output_images_pixel[idx].to(
            device=gen_decoded.device, dtype=torch.float32
        )
        if gt_imgs.shape[-2] != 256 or gt_imgs.shape[-1] != 256:
            gt_imgs = F.interpolate(
                gt_imgs, size=(256, 256),
                mode="bilinear", align_corners=False
            )

        masks_gt_sub = output_anatomy_masks[idx].to(
            device=gen_decoded.device, dtype=torch.float32
        )

        loss_anatomy = compute_mask_weighted_feature_loss(
            seg_model, gen_decoded, gt_imgs, masks_gt_sub,
            feature_layer_idx=feature_layer_idx,
            use_gen_mask=use_gen_mask,
        )

    # ==========================================================================
    # Combined Loss
    # ==========================================================================
    loss_total = loss_diffusion + lambda_anatomy * loss_anatomy

    return {
        "loss_total": loss_total,
        "loss_diffusion": loss_diffusion,
        "loss_anatomy": loss_anatomy,
        "anatomy_subbatch_size_actual": n,
        "feature_layer_idx": feature_layer_idx,
    }
