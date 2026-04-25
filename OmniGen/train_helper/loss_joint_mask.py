"""Joint Image-Mask Latent-Space Flow Matching Loss.

This module implements the co-generation training loss for Plan 2:
  1. Encode GT image to image latent via frozen VAE.
  2. Encode GT mask to mask latent via the trainable MaskEncoder.
  3. Sample shared noise and timestep for both modalities.
  4. Compute noisy states for both image and mask latents.
  5. Run the joint model forward (predicts velocities for both).
  6. Compute MSE loss on predicted velocities vs GT velocities.
  7. Optionally add a lightweight reconstruction regularizer.
  8. Return weighted combination: L_img + lambda_mask * L_mask + lambda_recon * L_recon.

No frozen segmentation model is needed. No VAE decode in the loss path.
"""

import torch
import torch.nn.functional as F


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
    t = t.to(x1[0] if isinstance(x1, (list, tuple)) else x1)
    return t


def training_losses_joint_mask(
    model,
    x1_img,
    x1_mask,
    model_kwargs,
    lambda_mask=0.5,
    mask_decoder=None,
    gt_mask_cont=None,
    lambda_recon=0.0,
):
    """
    Joint rectified-flow loss for image and mask latents.

    Args:
        model: OmniGen model with joint mask modules initialized.
        x1_img: VAE-encoded GT target image latent.
            - tensor (B, 4, H, W) for fixed-resolution, or
            - list of tensors (1, 4, H, W) for variable-resolution.
        x1_mask: Mask-encoder output GT mask latent (B, 4, 32, 32).
        model_kwargs: Keyword arguments for the OmniGen forward pass.
        lambda_mask: Weight for the mask velocity loss.
        mask_decoder: Optional MaskDecoder used for auxiliary reconstruction.
        gt_mask_cont: Optional GT masks in [-1, 1], shape (B, 10, H, W).
        lambda_recon: Weight for the optional reconstruction loss.

    Returns:
        Dict with 'loss', 'loss_img', 'loss_mask', 'loss_recon'.
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = len(x1_img)

    # Sample noise for both modalities
    x0_img = sample_x0(x1_img)
    x0_mask = torch.randn_like(x1_mask)

    # Shared timestep for coupled co-evolution
    t = sample_timestep(x1_img)

    # Build noisy states
    if isinstance(x1_img, (list, tuple)):
        xt_img = [t[i] * x1_img[i] + (1 - t[i]) * x0_img[i] for i in range(B)]
        ut_img = [x1_img[i] - x0_img[i] for i in range(B)]
    else:
        dims = [1] * (len(x1_img.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt_img = t_ * x1_img + (1 - t_) * x0_img
        ut_img = x1_img - x0_img

    # Mask noisy state (always tensor, not list)
    dims_mask = [1] * (len(x1_mask.size()) - 1)
    t_mask = t.view(t.size(0), *dims_mask)
    xt_mask = t_mask * x1_mask + (1 - t_mask) * x0_mask
    ut_mask = x1_mask - x0_mask

    # Keep exactly one joint forward per iteration. Splitting image/mask
    # predictions into separate forwards can break DDP hook accounting when
    # LoRA and gradient checkpointing are both enabled.
    model_output = model(xt_img, t, x_mask=xt_mask, **model_kwargs)

    # model_output is (img_pred, mask_pred) in joint mode
    if not (isinstance(model_output, (tuple, list)) and len(model_output) == 2):
        raise RuntimeError(
            "Joint mask training expects a single model forward that returns "
            "(pred_img, pred_mask)."
        )
    pred_img, pred_mask = model_output

    # Image velocity loss
    if isinstance(x1_img, (list, tuple)):
        loss_img = torch.stack(
            [((ut_img[i] - pred_img[i]) ** 2).mean() for i in range(B)],
            dim=0,
        ).mean()
    else:
        loss_img = mean_flat((pred_img - ut_img) ** 2).mean()

    # Mask velocity loss
    loss_mask = mean_flat((pred_mask - ut_mask) ** 2).mean()

    # Optional reconstruction loss.
    #
    # When lambda_recon == 0.0 we must remain mathematically equivalent to the
    # current SOTA run, so we bypass MaskDecoder entirely.
    if lambda_recon == 0.0:
        loss_recon = loss_img.new_zeros(())
    else:
        if mask_decoder is None:
            raise ValueError("mask_decoder must be provided when lambda_recon > 0.")
        if gt_mask_cont is None:
            raise ValueError("gt_mask_cont must be provided when lambda_recon > 0.")
        recon_mask = mask_decoder(x1_mask)
        loss_recon = F.mse_loss(recon_mask.float(), gt_mask_cont.float())

    # Combined loss
    loss = loss_img + lambda_mask * loss_mask + lambda_recon * loss_recon

    return {
        "loss": loss,
        "loss_img": loss_img,
        "loss_mask": loss_mask,
        "loss_recon": loss_recon,
    }
