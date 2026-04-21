"""
loss_anatomy_alternatives.py - Alternative Anatomy Loss Implementations

This module provides drop-in replacement functions for compute_feature_matching_loss()
in OmniGen/train_helper/loss_anatomy.py that address the identified issues:

1. Thresholded Logits Loss: Only penalize if difference exceeds margin
2. Blurred Feature Matching: Apply Gaussian blur before MSE

Usage:
    In OmniGen/train_helper/loss_anatomy.py, replace:

        loss_i = compute_feature_matching_loss(seg_model, gen_decoded, gt_img)

    With one of:

        from diagnostics.loss_anatomy_alternatives import compute_thresholded_logits_loss
        loss_i = compute_thresholded_logits_loss(seg_model, gen_decoded, gt_img, margin=2.0)

    Or:

        from diagnostics.loss_anatomy_alternatives import compute_blurred_feature_loss
        loss_i = compute_blurred_feature_loss(seg_model, gen_decoded, gt_img, blur_sigma=3.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ===========================================================================
# Alternative 1: Thresholded Logits Loss
# ===========================================================================

def compute_thresholded_logits_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Thresholded Logits Loss: Only penalize if logit difference exceeds margin.

    This allows the generator freedom for texture variation as long as
    the anatomical structure is approximately correct.

    L = mean(max(0, |gen_logit - gt_logit| - margin)^2)

    Args:
        seg_model: Frozen ResNet34-UNet (eval mode, requires_grad=False)
        gen_images: (B, 3, H, W) generated images in [-1, 1], WITH grad
        gt_images: (B, 3, H, W) ground truth images in [-1, 1]
        margin: Tolerance margin in logit space
                - margin=1.0: Tight (prob diff ~27%)
                - margin=2.0: Moderate (prob diff ~47%) [RECOMMENDED]
                - margin=3.0: Loose (prob diff ~59%)

    Returns:
        Scalar loss tensor

    Mathematical Properties:
        - When |diff| <= margin: loss = 0, gradient = 0
        - When |diff| > margin: loss = (|diff| - margin)^2
        - Gradient only flows for "large" errors
    """
    # GT logits: no grad needed (frozen target)
    with torch.no_grad():
        gt_logits = seg_model(gt_images)  # (B, C, H, W)

    # Gen logits: WITH grad (this is what we're training)
    gen_logits = seg_model(gen_images)  # (B, C, H, W)

    # Compute absolute difference
    diff = torch.abs(gen_logits - gt_logits)  # (B, C, H, W)

    # Apply margin threshold: max(0, |diff| - margin)
    thresholded_diff = F.relu(diff - margin)  # (B, C, H, W)

    # Squared loss on thresholded differences
    loss = (thresholded_diff ** 2).mean()

    return loss


def compute_asymmetric_thresholded_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    margin_over: float = 2.0,
    margin_under: float = 1.5,
    under_weight: float = 2.0,
) -> torch.Tensor:
    """
    Asymmetric Thresholded Loss: Different margins for over/under-prediction.

    Rationale:
        - Under-predicting anatomy (missing structures) is worse than over-predicting
        - Use smaller margin for false negatives, larger margin for false positives

    Args:
        margin_over: Margin when gen_logit > gt_logit (over-prediction)
        margin_under: Margin when gen_logit < gt_logit (under-prediction)
        under_weight: Weight multiplier for under-prediction loss
    """
    with torch.no_grad():
        gt_logits = seg_model(gt_images)

    gen_logits = seg_model(gen_images)

    diff = gen_logits - gt_logits

    # Over-prediction: gen > gt (positive diff)
    over_pred = F.relu(diff - margin_over)

    # Under-prediction: gen < gt (negative diff)
    under_pred = F.relu(-diff - margin_under)

    # Weight under-prediction more heavily
    loss = (over_pred ** 2).mean() + under_weight * (under_pred ** 2).mean()

    return loss


# ===========================================================================
# Alternative 2: Blurred Feature Matching
# ===========================================================================

class GaussianBlur2d(nn.Module):
    """
    Differentiable 2D Gaussian blur using separable convolutions.

    This is more efficient than a full 2D convolution.
    """
    def __init__(self, channels: int, kernel_size: int = 11, sigma: float = 3.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable kernels
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)

        self.register_buffer("kernel_h", kernel_h)
        self.register_buffer("kernel_v", kernel_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to input tensor."""
        x = F.conv2d(x, self.kernel_h, padding=(0, self.padding), groups=self.channels)
        x = F.conv2d(x, self.kernel_v, padding=(self.padding, 0), groups=self.channels)
        return x


# Global cache for blur modules to avoid recreation
_blur_cache = {}


def get_blur_module(channels: int, sigma: float, device: torch.device) -> GaussianBlur2d:
    """Get or create a cached blur module."""
    key = (channels, sigma, str(device))
    if key not in _blur_cache:
        kernel_size = int(sigma * 4) | 1  # Ensure odd
        _blur_cache[key] = GaussianBlur2d(channels, kernel_size, sigma).to(device)
    return _blur_cache[key]


def compute_blurred_feature_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    blur_sigma: float = 3.0,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Blurred Feature Matching Loss: Align macro-structures, ignore high-frequency.

    Strategy:
        1. Get logits from both generated and GT images
        2. Apply Gaussian blur to both
        3. Compute MSE on blurred logits

    This forces the model to align large-scale anatomical shapes without
    overfitting to pixel-level variations or VAE artifacts.

    Args:
        seg_model: Frozen ResNet34-UNet
        gen_images: (B, 3, H, W) generated images, WITH grad
        gt_images: (B, 3, H, W) GT images
        blur_sigma: Gaussian sigma
                    - 2.0: Light blur, preserves edges
                    - 3.0: Moderate blur [RECOMMENDED]
                    - 5.0: Heavy blur, only coarse shapes
        num_classes: Number of segmentation classes (default 10)

    Returns:
        Scalar loss tensor
    """
    device = gen_images.device
    blur = get_blur_module(num_classes, blur_sigma, device)

    # GT logits: blurred, no grad
    with torch.no_grad():
        gt_logits = seg_model(gt_images)
        gt_logits_blurred = blur(gt_logits)

    # Gen logits: WITH grad
    gen_logits = seg_model(gen_images)
    gen_logits_blurred = blur(gen_logits)

    # MSE on blurred logits
    loss = F.mse_loss(gen_logits_blurred, gt_logits_blurred)

    return loss


def compute_multiscale_feature_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 5.0),
    weights: Tuple[float, ...] = (0.2, 0.5, 0.3),
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Multi-Scale Feature Matching: Combine multiple blur levels.

    Captures both coarse structure (large sigma) and finer detail (small sigma).

    Args:
        sigmas: Tuple of blur sigmas for each scale
        weights: Tuple of weights for each scale (should sum to 1.0)
    """
    assert len(sigmas) == len(weights)
    device = gen_images.device

    # Get logits once
    with torch.no_grad():
        gt_logits = seg_model(gt_images)
    gen_logits = seg_model(gen_images)

    total_loss = 0.0

    for sigma, weight in zip(sigmas, weights):
        blur = get_blur_module(num_classes, sigma, device)

        with torch.no_grad():
            gt_blurred = blur(gt_logits)
        gen_blurred = blur(gen_logits)

        scale_loss = F.mse_loss(gen_blurred, gt_blurred)
        total_loss = total_loss + weight * scale_loss

    return total_loss


# ===========================================================================
# Alternative 3: Combined Approach (Blurred + Thresholded)
# ===========================================================================

def compute_blurred_thresholded_loss(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    blur_sigma: float = 2.0,
    margin: float = 1.5,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Combined: Blur THEN threshold.

    This provides maximum robustness:
    1. Blur removes high-frequency artifacts
    2. Thresholding provides margin tolerance

    Recommended settings:
        blur_sigma=2.0 (lighter blur since thresholding adds robustness)
        margin=1.5 (smaller margin since blur already smooths)
    """
    device = gen_images.device
    blur = get_blur_module(num_classes, blur_sigma, device)

    with torch.no_grad():
        gt_logits = seg_model(gt_images)
        gt_blurred = blur(gt_logits)

    gen_logits = seg_model(gen_images)
    gen_blurred = blur(gen_logits)

    diff = torch.abs(gen_blurred - gt_blurred)
    thresholded_diff = F.relu(diff - margin)
    loss = (thresholded_diff ** 2).mean()

    return loss


# ===========================================================================
# Wrapper for Easy Integration
# ===========================================================================

def compute_feature_matching_loss_v2(
    seg_model: torch.nn.Module,
    gen_images: torch.Tensor,
    gt_images: torch.Tensor,
    method: str = "blurred",
    **kwargs,
) -> torch.Tensor:
    """
    Unified interface for all alternative loss methods.

    Args:
        method: One of "raw", "thresholded", "blurred", "multiscale", "combined"
        **kwargs: Method-specific parameters

    Usage:
        loss = compute_feature_matching_loss_v2(
            seg_model, gen_images, gt_images,
            method="blurred",
            blur_sigma=3.0
        )
    """
    if method == "raw":
        # Original MSE without modifications
        with torch.no_grad():
            gt_logits = seg_model(gt_images)
        gen_logits = seg_model(gen_images)
        return F.mse_loss(gen_logits, gt_logits)

    elif method == "thresholded":
        margin = kwargs.get("margin", 2.0)
        return compute_thresholded_logits_loss(seg_model, gen_images, gt_images, margin)

    elif method == "asymmetric":
        return compute_asymmetric_thresholded_loss(
            seg_model, gen_images, gt_images,
            margin_over=kwargs.get("margin_over", 2.0),
            margin_under=kwargs.get("margin_under", 1.5),
            under_weight=kwargs.get("under_weight", 2.0),
        )

    elif method == "blurred":
        return compute_blurred_feature_loss(
            seg_model, gen_images, gt_images,
            blur_sigma=kwargs.get("blur_sigma", 3.0),
            num_classes=kwargs.get("num_classes", 10),
        )

    elif method == "multiscale":
        return compute_multiscale_feature_loss(
            seg_model, gen_images, gt_images,
            sigmas=kwargs.get("sigmas", (1.5, 3.0, 5.0)),
            weights=kwargs.get("weights", (0.2, 0.5, 0.3)),
            num_classes=kwargs.get("num_classes", 10),
        )

    elif method == "combined":
        return compute_blurred_thresholded_loss(
            seg_model, gen_images, gt_images,
            blur_sigma=kwargs.get("blur_sigma", 2.0),
            margin=kwargs.get("margin", 1.5),
            num_classes=kwargs.get("num_classes", 10),
        )

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: raw, thresholded, asymmetric, blurred, multiscale, combined")


# ===========================================================================
# Testing
# ===========================================================================

if __name__ == "__main__":
    print("Testing alternative loss implementations...")

    # Create dummy inputs
    B, C, H, W = 2, 3, 256, 256
    num_classes = 10

    gen_images = torch.randn(B, C, H, W, requires_grad=True)
    gt_images = torch.randn(B, C, H, W)

    # Mock seg model
    class MockSegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, num_classes, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    seg_model = MockSegModel()
    seg_model.eval()

    # Test each method
    methods = ["raw", "thresholded", "blurred", "multiscale", "combined"]

    for method in methods:
        loss = compute_feature_matching_loss_v2(
            seg_model, gen_images, gt_images, method=method
        )
        print(f"Method '{method}': loss = {loss.item():.4f}")

        # Verify gradient flows
        loss.backward(retain_graph=True)
        if gen_images.grad is not None:
            grad_norm = gen_images.grad.norm().item()
            print(f"  Gradient norm: {grad_norm:.4f}")
            gen_images.grad.zero_()

    print("\nAll tests passed!")
