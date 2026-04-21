"""
Anatomy-Aware Loss for OmniGen LoRA Fine-tuning.

Computes:
  1. Standard rectified-flow MSE loss (loss_diffusion)
  2. Anatomy segmentation loss via frozen UNet on decoded sub-batch (loss_anatomy)

Total: loss_total = loss_diffusion + lambda_anatomy * loss_anatomy

================================================================================
GOLDEN RULE COMPLIANCE:
- The main diffusion logic (u_hat, loss_diffusion) is NEVER modified by
  the anatomy branch. All tensor reshaping happens ONLY on cloned/sub-batched
  tensors flowing into the auxiliary branch.
- loss_diffusion remains mathematically identical to original OmniGen.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def dice_loss_fn(logits, targets, smooth=0.0, eps=1e-7):
    """
    Multilabel Dice loss operating on raw logits.
    logits: (B, C, H, W) raw logits
    targets: (B, C, H, W) float32 binary masks
    """
    probs = torch.sigmoid(logits)
    # Flatten spatial dims
    probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)   # (B, C, N)
    targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)

    intersection = (probs_flat * targets_flat).sum(dim=-1)  # (B, C)
    cardinality = probs_flat.sum(dim=-1) + targets_flat.sum(dim=-1)  # (B, C)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth + eps)
    return (1.0 - dice_score).mean()


def compute_anatomy_loss(seg_model, decoded_images, mask_sub):
    """
    Run frozen segmentation model on decoded images and compute anatomy loss.

    Args:
        seg_model: Frozen ResNet34-UNet (eval mode, requires_grad=False)
        decoded_images: (n, 3, H, W) in [-1, 1], with autograd graph attached
        mask_sub: (n, 10, H, W) float32 ground-truth binary masks

    Returns:
        loss_anatomy: scalar tensor with grad
    """
    # The seg model was trained on [-1, 1] inputs (Normalize(0.5, 0.5) on [0,255]).
    # The VAE outputs [-1, 1]. So pass directly, no extra normalization needed.
    logits = seg_model(decoded_images)  # (n, 10, H, W)

    bce_loss = F.binary_cross_entropy_with_logits(logits, mask_sub)
    d_loss = dice_loss_fn(logits, mask_sub)

    return 0.5 * bce_loss + 0.5 * d_loss


def _ensure_4d_latent(lat):
    """
    ============================================================================
    DATA FLOW FIX #1: 5D -> 4D Conversion (Before VAE Decode)
    ============================================================================
    OmniGen's variable-resolution mode produces latents with shape:
      - x1[i]: (1, C, H, W)  when from vae_encode_list (batched single images)
      - or potentially (N, C, H, W) for N>1 in edge cases

    This helper ensures we always have exactly (B, C, H, W) for VAE decode.
    If the tensor has shape (1, 1, C, H, W) due to extra squeeze, remove it.
    If it has shape (C, H, W), add a batch dimension.

    GOLDEN RULE: This operates on a COPY flowing to the aux branch only.
    The original tensor in the main diffusion path is NEVER touched.
    """
    ndim = lat.dim()

    if ndim == 5:
        # Shape: (B, NumImages, C, H, W) -> squeeze NumImages if it's 1
        # This handles the OmniGen (B, 1, 4, 32, 32) case
        if lat.shape[1] == 1:
            lat = lat.squeeze(1)  # -> (B, C, H, W)
        else:
            raise ValueError(
                f"5D latent with NumImages > 1 not supported for anatomy branch. "
                f"Got shape {lat.shape}"
            )
    elif ndim == 3:
        # Shape: (C, H, W) -> add batch dimension
        lat = lat.unsqueeze(0)  # -> (1, C, H, W)
    elif ndim != 4:
        raise ValueError(
            f"Expected latent with 3, 4, or 5 dimensions, got {ndim}D with shape {lat.shape}"
        )

    return lat


def training_losses_with_anatomy(
    model,
    x1,
    model_kwargs,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy=0.1,
    anatomy_subbatch_size=4,
):
    """
    Combined rectified-flow + anatomy-aware loss.

    ============================================================================
    GOLDEN RULE COMPLIANCE:
    - loss_diffusion is computed EXACTLY as in the original OmniGen loss.py
    - u_hat (model_output) is NEVER modified in-place
    - All reshaping/squeezing happens ONLY on tensors flowing to the aux branch
    ============================================================================

    Args:
        model: OmniGen model (with LoRA)
        x1: VAE-encoded target latent, (B, C, H, W) tensor OR list of (1, C, H, W)
        model_kwargs: dict for model forward pass
        output_anatomy_masks: (B, 10, 256, 256) float32 binary masks
        vae: Frozen AutoencoderKL (kept in fp32, requires_grad=False)
        seg_model: Frozen ResNet34-UNet segmentation model
        lambda_anatomy: weight for anatomy loss
        anatomy_subbatch_size: max samples to decode per step (VRAM safety)

    Returns:
        dict with loss_total, loss_diffusion, loss_anatomy
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = len(x1)

    # ==========================================================================
    # GOLDEN RULE: Rectified Flow sampling - IDENTICAL to original OmniGen
    # ==========================================================================
    x0 = sample_x0(x1)
    t = sample_timestep(x1)

    if isinstance(x1, (list, tuple)):
        # Variable resolution path (keep_raw_resolution=True)
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        # Fixed resolution path (keep_raw_resolution=False)
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    # ==========================================================================
    # GOLDEN RULE: Model forward pass - IDENTICAL to original OmniGen
    # ==========================================================================
    model_output = model(xt, t, **model_kwargs)

    # ==========================================================================
    # GOLDEN RULE: Diffusion loss (MSE) - IDENTICAL to original OmniGen
    # This loss remains UNCHANGED regardless of anatomy branch.
    # ==========================================================================
    if isinstance(x1, (list, tuple)):
        loss_diffusion = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        ).mean()
    else:
        loss_diffusion = mean_flat((model_output - ut) ** 2).mean()

    # ==========================================================================
    # AUXILIARY ANATOMY BRANCH
    # Everything below operates on CLONED/INDEXED tensors only.
    # The main diffusion tensors (xt, ut, model_output) are never modified.
    # ==========================================================================

    if isinstance(x1, (list, tuple)):
        # ----------------------------------------------------------------------
        # Variable-resolution path (keep_raw_resolution=True)
        # x1, xt, model_output are all LISTS of tensors with shape (1, C, H, W)
        # ----------------------------------------------------------------------

        # Reconstruct predicted clean latent for each sample
        # x1_hat[i] = x_t[i] + (1 - t[i]) * u_hat[i]
        # Note: t[i] is a scalar, xt[i] and model_output[i] are (1, C, H, W)
        x1_hat = [xt[i] + (1 - t[i]) * model_output[i] for i in range(B)]

        # Sub-batch selection for VRAM safety
        n = min(anatomy_subbatch_size, B)
        idx = torch.randperm(B, device=model_output[0].device)[:n]

        anatomy_losses = []
        for i_idx in idx:
            i = i_idx.item()

            # ==================================================================
            # DATA FLOW FIX #2: Proper 5D -> 4D handling
            # ==================================================================
            # x1_hat[i] shape: typically (1, C, H, W) from OmniGen variable-res
            # The old code did `.unsqueeze(0)` which created (1, 1, C, H, W) - WRONG!
            #
            # Now we use _ensure_4d_latent to safely get (B, C, H, W) format
            # without adding unnecessary dimensions.
            lat = _ensure_4d_latent(x1_hat[i])  # -> (1, C, H, W) guaranteed
            lat = lat.float()  # Cast to fp32 for stable VAE decode

            # ==================================================================
            # DATA FLOW FIX #3: Differentiable Decode (NO torch.no_grad!)
            # ==================================================================
            # Inverse-scale the latent (reverse VAE encoding transform)
            lat_scaled = inverse_vae_scale(lat, vae)

            # Decode through frozen VAE - gradients flow THROUGH, not TO
            # VAE weights are frozen (requires_grad=False), but the computation
            # graph is preserved for backprop to model_output -> LoRA params
            decoded = vae.decode(lat_scaled).sample  # (1, 3, H_dec, W_dec)

            # Clamp to valid image range (maintains differentiability)
            decoded = decoded.clamp(-1.0, 1.0)

            # ==================================================================
            # DATA FLOW FIX #4: Spatial Alignment (Before Seg Model)
            # ==================================================================
            # Decoded resolution depends on latent resolution (H_dec = H_lat * 8)
            # With dynamic/variable resolution, this may not be 256x256.
            # The segmentation model and masks expect (B, C, 256, 256).
            # Use bilinear interpolation for differentiable spatial alignment.
            if decoded.shape[-2] != 256 or decoded.shape[-1] != 256:
                decoded = F.interpolate(
                    decoded,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False
                )

            # ==================================================================
            # DATA FLOW FIX #5: Loss Type Matching
            # ==================================================================
            # Ensure mask is on same device and in float32
            mask_i = output_anatomy_masks[i:i+1].to(
                device=decoded.device,
                dtype=torch.float32
            )  # (1, 10, 256, 256)

            # Compute anatomy loss (BCE + Dice on seg model logits vs mask)
            loss_i = compute_anatomy_loss(seg_model, decoded, mask_i)
            anatomy_losses.append(loss_i)

        loss_anatomy = torch.stack(anatomy_losses).mean()

    else:
        # ----------------------------------------------------------------------
        # Fixed-resolution path (keep_raw_resolution=False)
        # x1, xt, model_output are all tensors with shape (B, C, H, W)
        # ----------------------------------------------------------------------

        # Reconstruct predicted clean latent
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        x1_hat = xt + (1 - t_) * model_output  # (B, C, H, W)

        # Sub-batch selection for VRAM safety
        n = min(anatomy_subbatch_size, B)
        idx = torch.randperm(B, device=x1_hat.device)[:n]

        # ==================================================================
        # DATA FLOW FIX #2: Handle potential 5D tensor in fixed-res path too
        # ==================================================================
        x1_hat_sub = x1_hat[idx]  # (n, ...) - slice the sub-batch
        x1_hat_sub = _ensure_4d_latent(x1_hat_sub)  # Ensure (n, C, H, W)

        # Cast to fp32 for stable VAE decode
        x1_hat_sub_fp32 = x1_hat_sub.float()

        # ==================================================================
        # DATA FLOW FIX #3: Differentiable Decode
        # ==================================================================
        x1_hat_sub_scaled = inverse_vae_scale(x1_hat_sub_fp32, vae)
        decoded = vae.decode(x1_hat_sub_scaled).sample  # (n, 3, H_dec, W_dec)
        decoded = decoded.clamp(-1.0, 1.0)

        # ==================================================================
        # DATA FLOW FIX #4: Spatial Alignment
        # ==================================================================
        if decoded.shape[-2] != 256 or decoded.shape[-1] != 256:
            decoded = F.interpolate(
                decoded,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

        # ==================================================================
        # DATA FLOW FIX #5: Loss Type Matching
        # ==================================================================
        mask_sub = output_anatomy_masks[idx].to(
            device=decoded.device,
            dtype=torch.float32
        )  # (n, 10, 256, 256)

        loss_anatomy = compute_anatomy_loss(seg_model, decoded, mask_sub)

    # ==========================================================================
    # Combined Loss
    # ==========================================================================
    loss_total = loss_diffusion + lambda_anatomy * loss_anatomy

    return {
        "loss_total": loss_total,
        "loss_diffusion": loss_diffusion,
        "loss_anatomy": loss_anatomy,
        "anatomy_subbatch_size_actual": n,
    }
