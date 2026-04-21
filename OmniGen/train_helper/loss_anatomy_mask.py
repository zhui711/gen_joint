"""
Mask-based anatomy-aware loss for OmniGen LoRA fine-tuning.

This module implements Plan 1:
  1. Keep the original rectified-flow diffusion loss unchanged.
  2. Decode a sub-batch of predicted clean latents with the frozen VAE.
  3. Run the frozen segmentation model on decoded images.
  4. Convert segmentation logits to probabilities with sigmoid.
  5. Compute per-channel MSE against the 10-channel GT anatomy mask.
  6. Sum the 10 channel losses and add them to the diffusion loss.

The key gradient rule is preserved:
  - Gradients must flow through seg_model(input), through VAE decode, and back to
    OmniGen's predicted latent path.
  - Therefore the generated-image branch must NOT be wrapped in torch.no_grad().
  - We also avoid clamping decoded images to preserve gradients outside [-1, 1].
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sample_x0(x1):
    """Sample Gaussian noise matching the target latent shape."""
    if isinstance(x1, (list, tuple)):
        return [torch.randn_like(img) for img in x1]
    return torch.randn_like(x1)


def sample_timestep(x1):
    """Sample timesteps from the logistic-normal distribution used by OmniGen."""
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def inverse_vae_scale(latents, vae):
    """Undo the VAE latent scaling applied during encoding."""
    if vae.config.shift_factor is not None:
        return latents / vae.config.scaling_factor + vae.config.shift_factor
    return latents / vae.config.scaling_factor


def _ensure_4d_latent(lat):
    """Ensure the latent tensor is shaped as (B, C, H, W) for VAE decode."""
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


def _mask_to_debug_vis(mask_tensor):
    """Convert a multi-channel anatomy mask into a single-channel image for saving."""
    if mask_tensor.dim() != 3:
        raise ValueError(f"Expected a 3D mask tensor, got shape {mask_tensor.shape}")

    if mask_tensor.shape[0] == 1:
        mask_vis = mask_tensor
    else:
        mask_vis = mask_tensor.sum(dim=0, keepdim=True)

    mask_vis = mask_vis.float()
    return mask_vis / mask_vis.amax().clamp(min=1e-6)


def _maybe_save_debug_visuals(
    gen_decoded,
    mask_gen,
    mask_gt,
    t_value,
    global_step=None,
    sample_idx=0,
    debug_vis_dir="./debug_vis",
):
    """Save decoded image and predicted/GT masks for extreme timesteps."""
    if not (t_value < 0.2 or t_value > 0.8):
        return

    debug_dir = Path(debug_vis_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    step_str = "unknown" if global_step is None else str(global_step)
    t_str = f"{t_value:.4f}"
    prefix = debug_dir / f"step_{step_str}_sample_{sample_idx}_t_{t_str}"

    decoded_vis = ((gen_decoded.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    pred_mask_vis = _mask_to_debug_vis(mask_gen.detach().cpu())
    gt_mask_vis = _mask_to_debug_vis(mask_gt.detach().cpu())

    save_image(decoded_vis, str(prefix) + "_decoded.png")
    save_image(pred_mask_vis, str(prefix) + "_pred_mask.png")
    save_image(gt_mask_vis, str(prefix) + "_gt_mask.png")


def compute_mask_mse_loss(seg_model, gen_images, mask_gt, return_mask_gen=False):
    """
    Compute mask-based anatomy loss.

    Args:
        seg_model: Frozen ResUNet34-style segmentation model.
        gen_images: (B, 3, H, W) decoded generated images with autograd graph.
        mask_gt:    (B, 10, 256, 256) float32 ground-truth masks in {0, 1}.

    Returns:
        Scalar tensor containing the summed per-channel MSE loss.
        If return_mask_gen=True, also returns the sigmoid mask prediction.
    """
    if mask_gt.dim() != 4:
        raise ValueError(f"Expected mask_gt to be 4D, got shape {mask_gt.shape}")
    if mask_gt.shape[1] != 10:
        raise ValueError(
            f"Expected 10 anatomy channels in mask_gt, got shape {mask_gt.shape}"
        )

    mask_gt = mask_gt.to(device=gen_images.device, dtype=torch.float32)

    # The segmentation head outputs raw logits because activation=None is used
    # when constructing the model. For MSE against GT masks in {0, 1}, we must
    # compare probabilities rather than unconstrained logits.
    logits_gen = seg_model(gen_images)  # (B, 10, 256, 256)
    mask_gen = torch.sigmoid(logits_gen)

    loss_anatomy = torch.tensor(0.0, device=gen_images.device)
    for c in range(10):
        loss_anatomy = loss_anatomy + F.mse_loss(
            mask_gen[:, c:c + 1],
            mask_gt[:, c:c + 1],
            reduction="mean",
        )

    if return_mask_gen:
        return loss_anatomy, mask_gen

    return loss_anatomy


def training_losses_with_anatomy_mask(
    model,
    x1,
    model_kwargs,
    output_anatomy_masks,
    vae,
    seg_model,
    lambda_anatomy=0.1,
    anatomy_subbatch_size=4,
    anatomy_alpha=4.0,
    debug_vis=False,
    debug_global_step=None,
    debug_vis_dir="./debug_vis",
):
    """
    Combined rectified-flow + mask-based anatomy MSE loss.

    Args:
        model: OmniGen model (with or without LoRA).
        x1: VAE-encoded GT target latent. Either:
            - tensor (B, C, H, W) for fixed-resolution training, or
            - list of tensors shaped (1, C, H, W) for variable-resolution training.
        model_kwargs: Keyword arguments for the OmniGen forward pass.
        output_anatomy_masks: (B, 10, 256, 256) GT masks from the dataloader.
        vae: Frozen VAE used for differentiable decode.
        seg_model: Frozen segmentation model used to predict anatomy masks.
        lambda_anatomy: Weight applied to the anatomy loss.
        anatomy_subbatch_size: Max number of decoded samples per training step.
        anatomy_alpha: Polynomial power for timestep weighting of anatomy loss.
        debug_vis: If True, save decoded images and masks for extreme timesteps.

    Returns:
        Dict containing total loss and individual loss terms.
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = len(x1)

    # Keep the diffusion path mathematically identical to the existing OmniGen
    # training loss so the only experimental change is the auxiliary mask loss.
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

    model_output = model(xt, t, **model_kwargs)

    if isinstance(x1, (list, tuple)):
        loss_diffusion = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        ).mean()
    else:
        loss_diffusion = mean_flat((model_output - ut) ** 2).mean()

    n = min(anatomy_subbatch_size, B)

    if isinstance(x1, (list, tuple)):
        x1_hat = [xt[i] + (1 - t[i]) * model_output[i] for i in range(B)]
        idx = torch.randperm(B, device=model_output[0].device)[:n]

        anatomy_losses = []
        for i_idx in idx:
            i = i_idx.item()

            # Decode WITH gradient tracking. Do not wrap this branch in
            # torch.no_grad(), because anatomy loss gradients must flow back to
            # the predicted latent and then to OmniGen.
            lat = _ensure_4d_latent(x1_hat[i]).float()
            lat_scaled = inverse_vae_scale(lat, vae)
            gen_decoded = vae.decode(lat_scaled).sample

            if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
                gen_decoded = F.interpolate(
                    gen_decoded,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )

            mask_gt_i = output_anatomy_masks[i:i + 1].to(
                device=gen_decoded.device,
                dtype=torch.float32,
            )
            t_i = t[i].to(device=gen_decoded.device, dtype=gen_decoded.dtype)
            if debug_vis:
                loss_i, mask_gen_i = compute_mask_mse_loss(
                    seg_model,
                    gen_decoded,
                    mask_gt_i,
                    return_mask_gen=True,
                )
                t_value = float(t_i.detach().item())
                _maybe_save_debug_visuals(
                    gen_decoded=gen_decoded[0],
                    mask_gen=mask_gen_i[0],
                    mask_gt=mask_gt_i[0],
                    t_value=t_value,
                    global_step=debug_global_step,
                    sample_idx=i,
                    debug_vis_dir=debug_vis_dir,
                )
            else:
                loss_i = compute_mask_mse_loss(seg_model, gen_decoded, mask_gt_i)

            weighted_loss_i = loss_i * (t_i ** anatomy_alpha)
            anatomy_losses.append(weighted_loss_i)

        loss_anatomy = torch.stack(anatomy_losses).mean()

    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        x1_hat = xt + (1 - t_) * model_output

        idx = torch.randperm(B, device=x1_hat.device)[:n]
        x1_hat_sub = _ensure_4d_latent(x1_hat[idx]).float()
        x1_hat_sub_scaled = inverse_vae_scale(x1_hat_sub, vae)
        gen_decoded = vae.decode(x1_hat_sub_scaled).sample

        if gen_decoded.shape[-2] != 256 or gen_decoded.shape[-1] != 256:
            gen_decoded = F.interpolate(
                gen_decoded,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )

        mask_sub = output_anatomy_masks[idx].to(
            device=gen_decoded.device,
            dtype=torch.float32,
        )

        anatomy_losses = []
        idx_list = idx.detach().cpu().tolist()
        for local_idx, batch_idx in enumerate(idx_list):
            gen_decoded_i = gen_decoded[local_idx:local_idx + 1]
            mask_gt_i = mask_sub[local_idx:local_idx + 1]
            t_i = t[batch_idx].to(device=gen_decoded.device, dtype=gen_decoded.dtype)

            if debug_vis:
                loss_i, mask_gen_i = compute_mask_mse_loss(
                    seg_model,
                    gen_decoded_i,
                    mask_gt_i,
                    return_mask_gen=True,
                )
                t_value = float(t_i.detach().item())
                _maybe_save_debug_visuals(
                    gen_decoded=gen_decoded_i[0],
                    mask_gen=mask_gen_i[0],
                    mask_gt=mask_gt_i[0],
                    t_value=t_value,
                    global_step=debug_global_step,
                    sample_idx=batch_idx,
                    debug_vis_dir=debug_vis_dir,
                )
            else:
                loss_i = compute_mask_mse_loss(seg_model, gen_decoded_i, mask_gt_i)

            weighted_loss_i = loss_i * (t_i ** anatomy_alpha)
            anatomy_losses.append(weighted_loss_i)

        loss_anatomy = torch.stack(anatomy_losses).mean()

    loss = loss_diffusion + lambda_anatomy * loss_anatomy

    return {
        "loss": loss,
        "loss_diffusion": loss_diffusion,
        "loss_anatomy": loss_anatomy,
        "anatomy_subbatch_size_actual": n,
    }
