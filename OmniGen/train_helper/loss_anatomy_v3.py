"""
Anatomy-Aware Loss v3: Conservative Refinement for Stable Training

================================================================================
CHANGES FROM v2 (Root Cause Fixes):

1. TIMESTEP GATING (Critical):
   - Only compute anatomy loss when t > t_threshold (default 0.5)
   - At low t, x1_hat is OOD/blurry -> penalizing features destroys generative prior
   - If no samples meet threshold, return 0.0 (gradient-attached for DDP safety)

2. FEATURE L2 NORMALIZATION:
   - Normalize F_gt and F_gen along channel dimension before masking
   - Prevents high-activation channels from dominating the loss
   - Ensures balanced contribution from all anatomical features

3. AREA-NORMALIZED MSE:
   - Instead of reduction='mean' over all elements, compute:
     sum((F_gen - F_gt)^2 * mask) / (mask_area * n_channels)
   - Prevents large organs (lungs) from dominating small ones (trachea)
   - Clamp denominator >= 1.0 to avoid division by zero

4. REMOVE VAE DECODE CLAMP:
   - Changed vae.decode().sample.clamp(-1,1) to just vae.decode().sample
   - Clamp kills gradients for OOD predictions during early training

5. DEFAULT use_gen_mask=False:
   - Always use GT mask as spatial anchor
   - Predicted masks on blurry x1_hat create chaotic self-referential gradients

MATHEMATICAL FORMULATION:
  For each organ c in [0, 9]:
    F_gt_norm = F_gt / ||F_gt||_2   (along channel dim)
    F_gen_norm = F_gen / ||F_gen||_2

    OrganFeature_gt  = M_c * F_gt_norm
    OrganFeature_gen = M_c * F_gen_norm

    squared_diff = (OrganFeature_gen - OrganFeature_gt)^2
    mask_area = sum(M_c)  # Number of active spatial locations

    loss_c = sum(squared_diff) / max(mask_area * C_feat, 1.0)

  Loss_anatomy = sum(loss_c for c in [0,9])

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
# Core Utility Functions
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
# Feature Normalization Helper
# =============================================================================

def l2_normalize_features(features, eps=1e-6):
    """
    L2 normalize features along the channel dimension.

    Args:
        features: (B, C, H, W) tensor
        eps: Small value to prevent division by zero

    Returns:
        Normalized features with unit L2 norm along channel dimension at each spatial location.
    """
    # Compute L2 norm along channel dimension: (B, 1, H, W)
    norm = torch.norm(features, p=2, dim=1, keepdim=True) + eps
    return features / norm


# =============================================================================
# Mask-Weighted Feature Matching Loss (CORE - v3 with fixes)
# =============================================================================

def compute_mask_weighted_feature_loss_v3(
    seg_model,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    mask_gt: torch.Tensor,
    feature_layer_idx: int = 2,
    use_gen_mask: bool = False,  # CHANGED DEFAULT: False for stability
) -> torch.Tensor:
    """
    Compute mask-weighted feature matching loss with v3 fixes.

    KEY CHANGES from v2:
    1. L2 normalize features before masking (balanced channel contribution)
    2. Area-normalized MSE (prevents large organs from dominating)
    3. Default use_gen_mask=False (stable GT mask anchor)

    Args:
        seg_model: Frozen smp.Unet (eval mode, requires_grad=False)
        gen_images: (B, 3, H, W) in [-1, 1] or wider, WITH autograd graph
        gt_images:  (B, 3, H, W) in [-1, 1], no grad needed
        mask_gt:    (B, 10, 256, 256) ground-truth binary masks in [0, 1]
        feature_layer_idx: Encoder layer index (2 = 1/4 res for ResNet34)
        use_gen_mask: If True, predict mask from gen image; If False (default), use GT

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

    # Gen: WITH grad - gradients flow through frozen seg encoder back to gen_images
    gen_features_all = seg_model.encoder(gen_images)
    F_gen = gen_features_all[feature_layer_idx]  # (B, C_feat, H_feat, W_feat)

    B, C_feat, H_feat, W_feat = F_gen.shape

    # === FIX 1: L2 Normalize features along channel dimension ===
    # This prevents high-activation channels from dominating the MSE
    # 不归一化特征了，防止特征空间扭曲
    # F_gt_norm = l2_normalize_features(F_gt)
    # F_gen_norm = l2_normalize_features(F_gen)

    # === Step 2: Get masks ===
    if use_gen_mask:
        # Predict mask from gen image (less stable, use with caution)
        decoder_output = seg_model.decoder(gen_features_all)
        logits_gen = seg_model.segmentation_head(decoder_output)
        mask_gen = torch.sigmoid(logits_gen)
    else:
        # Use GT mask for both - more stable, recommended
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

    # === Step 4: Per-channel area-normalized MSE ===
    # FIX 2: Area normalization - prevents large organs from dominating
    loss_total = torch.tensor(0.0, device=device, requires_grad=True)

    for c in range(10):
        # (B, 1, H_feat, W_feat) - broadcasts with (B, C_feat, H_feat, W_feat)
        M_gt_c = mask_gt_down[:, c:c+1, :, :]
        M_gen_c = mask_gen_down[:, c:c+1, :, :]

        # Apply mask to normalized features
        # OrganFeature_gt_c = M_gt_c * F_gt_norm    # (B, C_feat, H_feat, W_feat)
        # OrganFeature_gen_c = M_gen_c * F_gen_norm  # (B, C_feat, H_feat, W_feat)

        # Compute squared difference
        # squared_diff = (OrganFeature_gen_c - OrganFeature_gt_c) ** 2  # (B, C_feat, H_feat, W_feat)

        if use_gen_mask:
            # 如果依然开启了预测掩码，退回老逻辑——不推荐
            OrganFeature_gt_c = M_gt_c * F_gt
            OrganFeature_gen_c = M_gen_c * F_gen
            squared_diff = (OrganFeature_gen_c - OrganFeature_gt_c) ** 2
        else:
            # 提取到平方外面
            squared_diff = M_gt_c * ((F_gen - F_gt) ** 2)

        # FIX 2: Area-normalized loss
        # Sum the squared diff, divide by (mask_area * n_channels) for each sample
        # mask_area = sum of mask values (can be fractional due to avg_pool2d)
        mask_area = M_gt_c.sum(dim=(2, 3))  # (B, 1) - sum over spatial dims

        # Total active elements = mask_area * C_feat (features at each masked location)
        # We average over the batch, but normalize within each sample
        sum_sq_diff = squared_diff.sum(dim=(1, 2, 3))  # (B,) - sum over C, H, W

        # Denominator: clamp to >= 1.0 to avoid div-by-zero when organ is absent
        denominator = (mask_area.squeeze(1) * C_feat).clamp(min=1.0)  # (B,)

        # Per-sample normalized loss, then mean over batch
        loss_c = (sum_sq_diff / denominator).mean()

        loss_total = loss_total + loss_c

    return loss_total


# =============================================================================
# Main Training Loss Function (v3)
# =============================================================================

def training_losses_with_anatomy_v3(
    model,
    x1,
    model_kwargs,
    output_images_pixel,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy: float = 0.02,  # CHANGED: Lower default for gentle recovery
    anatomy_subbatch_size: int = 4,
    feature_layer_idx: int = 2,
    use_gen_mask: bool = False,   # CHANGED: Default to False for stability
    t_threshold: float = 0.5,     # NEW: Timestep gating threshold
):
    """
    Combined rectified-flow + mask-weighted anatomy-aware loss (v3 with fixes).

    KEY CHANGES from v2:
    1. Timestep gating: Only compute anatomy loss when t > t_threshold
    2. Feature L2 normalization: Balanced channel contributions
    3. Area-normalized MSE: Prevents large organ dominance
    4. No VAE decode clamp: Preserves gradients for OOD predictions
    5. Default use_gen_mask=False: Stable GT mask anchor

    Args:
        model: OmniGen model (with LoRA)
        x1: VAE-encoded target latent, (B, C, H, W) tensor OR list of (1, C, H, W)
        model_kwargs: dict for model forward pass
        output_images_pixel: GT images in pixel space, [-1, 1] range
        output_anatomy_masks: (B, 10, 256, 256) float32 binary masks
        vae: Frozen AutoencoderKL (fp32, requires_grad=False)
        seg_model: Frozen smp.Unet segmentation model
        lambda_anatomy: Weight for anatomy loss (default: 0.02)
        anatomy_subbatch_size: Max samples to decode per step (VRAM safety)
        feature_layer_idx: Encoder layer index (2 = 1/4 res)
        use_gen_mask: Whether to predict mask for gen images (default: False)
        t_threshold: Only compute anatomy loss when t > this value (default: 0.5)

    Returns:
        dict: loss_total, loss_diffusion, loss_anatomy, n_anatomy_valid, t_threshold
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = len(x1)
    device = x1[0].device if isinstance(x1, (list, tuple)) else x1.device

    # ==========================================================================
    # GOLDEN RULE: Rectified Flow - IDENTICAL to original OmniGen
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
    # GOLDEN RULE: Model forward - IDENTICAL to original OmniGen
    # ==========================================================================
    model_output = model(xt, t, **model_kwargs)

    # ==========================================================================
    # GOLDEN RULE: Diffusion loss (MSE) - IDENTICAL to original OmniGen
    # ==========================================================================
    if isinstance(x1, (list, tuple)):
        loss_diffusion = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        ).mean()
    else:
        loss_diffusion = mean_flat((model_output - ut) ** 2).mean()

    # ==========================================================================
    # FIX 1: TIMESTEP GATING
    # Only compute anatomy loss for samples where t > t_threshold
    # At low t, x1_hat is OOD/blurry -> features are meaningless
    # ==========================================================================
    valid_mask = t > t_threshold  # (B,) boolean tensor
    valid_indices = torch.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        # No samples have t > threshold - return zero loss (gradient-attached for DDP)
        loss_anatomy = 0.0 * loss_diffusion  # Attached to graph to avoid DDP sync issues
        loss_total = loss_diffusion + lambda_anatomy * loss_anatomy
        return {
            "loss_total": loss_total,
            "loss_diffusion": loss_diffusion,
            "loss_anatomy": loss_anatomy,
            "n_anatomy_valid": 0,
            "t_threshold": t_threshold,
        }

    # ==========================================================================
    # AUXILIARY ANATOMY BRANCH (only for valid timesteps)
    # ==========================================================================

    # Limit to subbatch size
    n = min(anatomy_subbatch_size, n_valid)
    # Randomly select from valid indices
    perm = torch.randperm(n_valid, device=device)[:n]
    idx = valid_indices[perm]

    if isinstance(x1, (list, tuple)):
        # --- Variable-resolution path ---
        x1_hat = [xt[i] + (1 - t[i]) * model_output[i] for i in range(B)]

        anatomy_losses = []
        for i_idx in idx:
            i = i_idx.item()

            # FIX 3: Decode WITHOUT clamp - preserves gradients for OOD predictions
            lat = _ensure_4d_latent(x1_hat[i]).float()
            lat_scaled = inverse_vae_scale(lat, vae)
            gen_decoded = vae.decode(lat_scaled).sample  # NO CLAMP

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

            loss_i = compute_mask_weighted_feature_loss_v3(
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

        # FIX 3: Decode WITHOUT clamp
        x1_hat_sub = _ensure_4d_latent(x1_hat[idx]).float()
        x1_hat_sub_scaled = inverse_vae_scale(x1_hat_sub, vae)
        gen_decoded = vae.decode(x1_hat_sub_scaled).sample  # NO CLAMP

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

        loss_anatomy = compute_mask_weighted_feature_loss_v3(
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
        "n_anatomy_valid": n,
        "t_threshold": t_threshold,
    }
