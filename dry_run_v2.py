#!/usr/bin/env python3
"""
Dry-run validation for loss_anatomy_v2 (Mask-Weighted Feature Matching).

Tests:
  1. Encoder feature extraction at all layers — verify shapes
  2. Mask downsampling — verify resolution matching
  3. Per-organ MSE computation — verify gradient flow and broadcasting
  4. Full loss function — verify correct output dict structure
  5. Gradient flow check — verify gradients reach the input (proxy for LoRA params)
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/home/wenting/zr/Segmentation/segmentation_models_pytorch")
import segmentation_models_pytorch as smp

# Import the v2 loss module
sys.path.insert(0, "/home/wenting/zr/gen_code")
from OmniGen.train_helper.loss_anatomy_v2 import (
    compute_mask_weighted_feature_loss,
    ANATOMY_CHANNELS,
)


def test_encoder_features():
    """Test 1: Verify encoder feature shapes at all layers."""
    print("=" * 70)
    print("TEST 1: Encoder Feature Shapes (ResNet34, input 256x256)")
    print("=" * 70)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        features = model.encoder(x)

    expected = [
        (2, 3, 256, 256),    # Feature 0: identity (stride 1)
        (2, 64, 128, 128),   # Feature 1: conv1 (stride 2)
        (2, 64, 64, 64),     # Feature 2: layer1 (stride 4) <- TARGET
        (2, 128, 32, 32),    # Feature 3: layer2 (stride 8)
        (2, 256, 16, 16),    # Feature 4: layer3 (stride 16)
        (2, 512, 8, 8),      # Feature 5: layer4 (stride 32)
    ]

    all_ok = True
    for i, feat in enumerate(features):
        shape = tuple(feat.shape)
        status = "OK" if shape == expected[i] else "FAIL"
        if status == "FAIL":
            all_ok = False
        stride = 256 // feat.shape[-1]
        marker = " <-- TARGET (1/4 res)" if i == 2 else ""
        print(f"  Feature {i}: {str(shape):30s} stride={stride:3d}  [{status}]{marker}")

    print(f"\n  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_mask_downsampling():
    """Test 2: Verify mask downsampling via avg_pool2d."""
    print("\n" + "=" * 70)
    print("TEST 2: Mask Downsampling (256 -> 64 via avg_pool2d)")
    print("=" * 70)

    mask = torch.zeros(1, 10, 256, 256)
    # Create a square mask for channel 2 (Heart)
    mask[0, 2, 64:192, 80:176] = 1.0

    kernel_size = 256 // 64  # = 4
    mask_down = F.avg_pool2d(mask, kernel_size=kernel_size)

    ok = mask_down.shape == (1, 10, 64, 64)
    print(f"  Input shape:  {tuple(mask.shape)}")
    print(f"  Output shape: {tuple(mask_down.shape)}  [{'OK' if ok else 'FAIL'}]")

    # Verify non-zero region is preserved
    heart_down = mask_down[0, 2]
    nonzero = heart_down.sum().item()
    print(f"  Heart mask sum (downsampled): {nonzero:.1f} (should be > 0)  "
          f"[{'OK' if nonzero > 0 else 'FAIL'}]")

    # Verify other channels are zero
    other_sum = mask_down[0, [0, 1, 3, 4, 5, 6, 7, 8, 9]].sum().item()
    print(f"  Other channels sum: {other_sum:.1f} (should be 0)  "
          f"[{'OK' if other_sum == 0 else 'FAIL'}]")

    ok = ok and nonzero > 0 and other_sum == 0
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_broadcasting():
    """Test 3: Verify mask-feature broadcasting."""
    print("\n" + "=" * 70)
    print("TEST 3: Mask-Feature Broadcasting")
    print("=" * 70)

    B, C_feat, H, W = 2, 64, 64, 64
    F_gen = torch.randn(B, C_feat, H, W)
    M_c = torch.rand(B, 1, H, W)  # single-channel mask

    # Broadcasting: (B, 1, H, W) * (B, 64, H, W) -> (B, 64, H, W)
    result = M_c * F_gen
    ok = result.shape == (B, C_feat, H, W)
    print(f"  Mask shape:    {tuple(M_c.shape)}")
    print(f"  Feature shape: {tuple(F_gen.shape)}")
    print(f"  Result shape:  {tuple(result.shape)}  [{'OK' if ok else 'FAIL'}]")

    # Verify that masking zeroes out where mask is zero
    M_zero = torch.zeros(B, 1, H, W)
    result_zero = M_zero * F_gen
    zero_ok = result_zero.abs().sum().item() == 0
    print(f"  Zero mask -> zero features: [{'OK' if zero_ok else 'FAIL'}]")

    ok = ok and zero_ok
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradient_flow():
    """Test 4: Verify gradients flow through frozen seg model to input."""
    print("\n" + "=" * 70)
    print("TEST 4: Gradient Flow Through Frozen Seg Model")
    print("=" * 70)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # gen_images requires grad (proxy for LoRA output)
    gen_images = torch.randn(1, 3, 256, 256, requires_grad=True)
    gt_images = torch.randn(1, 3, 256, 256)
    mask_gt = torch.rand(1, 10, 256, 256)

    loss = compute_mask_weighted_feature_loss(
        model, gen_images, gt_images, mask_gt,
        feature_layer_idx=2, use_gen_mask=True,
    )

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss requires_grad: {loss.requires_grad}  [{'OK' if loss.requires_grad else 'FAIL'}]")

    loss.backward()

    has_grad = gen_images.grad is not None and gen_images.grad.abs().sum().item() > 0
    print(f"  gen_images.grad exists and non-zero: [{'OK' if has_grad else 'FAIL'}]")

    # Verify seg model params have NO grad
    seg_has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Seg model params have no grad: [{'OK' if not seg_has_grad else 'FAIL'}]")

    ok = loss.requires_grad and has_grad and not seg_has_grad
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_decoder_reuse():
    """Test 5: Verify decoder+head reuse gives same result as full forward."""
    print("\n" + "=" * 70)
    print("TEST 5: Decoder Reuse (no redundant encoder pass)")
    print("=" * 70)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        # Full forward
        logits_full = model(x)  # (1, 10, 256, 256)

        # Reuse encoder features
        features = model.encoder(x)
        decoder_output = model.decoder(features)
        logits_reuse = model.segmentation_head(decoder_output)

    diff = (logits_full - logits_reuse).abs().max().item()
    ok = diff < 1e-5
    print(f"  Max diff between full forward and reuse: {diff:.2e}  [{'OK' if ok else 'FAIL'}]")
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_per_channel_loss():
    """Test 6: Verify per-channel loss is computed independently."""
    print("\n" + "=" * 70)
    print("TEST 6: Per-Channel Independence")
    print("=" * 70)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    gen_images = torch.randn(1, 3, 256, 256)
    gt_images = gen_images.clone()  # identical -> loss should be ~0 for matching masks
    mask_gt = torch.zeros(1, 10, 256, 256)

    # Only activate channel 2 (Heart)
    mask_gt[0, 2, 64:192, 80:176] = 1.0

    loss = compute_mask_weighted_feature_loss(
        model, gen_images, gt_images, mask_gt,
        feature_layer_idx=2, use_gen_mask=False,  # use GT mask for both
    )

    # With identical images and same mask: loss should be ~0
    print(f"  Loss with identical images: {loss.item():.8f}")
    ok = loss.item() < 1e-5
    print(f"  Loss near zero: [{'OK' if ok else 'FAIL'}]")

    # With different images: loss should be > 0
    gen_different = torch.randn(1, 3, 256, 256)
    loss_diff = compute_mask_weighted_feature_loss(
        model, gen_different, gt_images, mask_gt,
        feature_layer_idx=2, use_gen_mask=False,
    )
    print(f"  Loss with different images: {loss_diff.item():.6f}")
    ok2 = loss_diff.item() > 0.001
    print(f"  Loss > 0: [{'OK' if ok2 else 'FAIL'}]")

    ok = ok and ok2
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_use_gen_mask_flag():
    """Test 7: Verify use_gen_mask=True produces predicted mask, False uses GT."""
    print("\n" + "=" * 70)
    print("TEST 7: use_gen_mask Flag")
    print("=" * 70)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=10,
        activation=None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    gen = torch.randn(1, 3, 256, 256)
    gt = torch.randn(1, 3, 256, 256)
    mask_gt = torch.rand(1, 10, 256, 256)

    loss_with_gen = compute_mask_weighted_feature_loss(
        model, gen, gt, mask_gt, feature_layer_idx=2, use_gen_mask=True)
    loss_with_gt = compute_mask_weighted_feature_loss(
        model, gen, gt, mask_gt, feature_layer_idx=2, use_gen_mask=False)

    # Losses should differ because masks differ
    differ = abs(loss_with_gen.item() - loss_with_gt.item()) > 1e-6
    print(f"  Loss (use_gen_mask=True):  {loss_with_gen.item():.6f}")
    print(f"  Loss (use_gen_mask=False): {loss_with_gt.item():.6f}")
    print(f"  Losses differ: [{'OK' if differ else 'FAIL'}]")

    print(f"\n  Result: {'PASS' if differ else 'FAIL'}")
    return differ


if __name__ == "__main__":
    print("Dry-Run Validation: loss_anatomy_v2 (Mask-Weighted Feature Matching)")
    print("=" * 70)

    results = []
    results.append(("Encoder Features", test_encoder_features()))
    results.append(("Mask Downsampling", test_mask_downsampling()))
    results.append(("Broadcasting", test_broadcasting()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Decoder Reuse", test_decoder_reuse()))
    results.append(("Per-Channel Loss", test_per_channel_loss()))
    results.append(("use_gen_mask Flag", test_use_gen_mask_flag()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:30s}  [{status}]")
        if not ok:
            all_pass = False

    print("=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)
