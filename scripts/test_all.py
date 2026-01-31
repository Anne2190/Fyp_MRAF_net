"""
MRAF-Net Test Script
Tests all components to ensure they work correctly

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_metrics():
    """Test the metrics module."""
    print("\n" + "=" * 60)
    print("TESTING METRICS MODULE")
    print("=" * 60)
    
    from src.utils.metrics import (
        compute_dice, compute_hausdorff95, compute_sensitivity,
        compute_specificity, compute_iou, compute_metrics, MetricTracker
    )
    
    # Test 1: Basic shapes
    print("\n1. Testing basic metric computations...")
    
    pred = np.zeros((64, 64, 64), dtype=np.int64)
    target = np.zeros((64, 64, 64), dtype=np.int64)
    
    # Add tumor regions
    pred[25:45, 25:45, 25:45] = 1
    pred[30:40, 30:40, 30:40] = 2
    pred[33:37, 33:37, 33:37] = 3
    
    target[23:47, 23:47, 23:47] = 1
    target[28:42, 28:42, 28:42] = 2
    target[31:39, 31:39, 31:39] = 3
    
    # Compute all metrics
    metrics = compute_metrics(pred, target)
    
    print(f"  Dice WT: {metrics['dice_wt']:.4f}")
    print(f"  Dice TC: {metrics['dice_tc']:.4f}")
    print(f"  Dice ET: {metrics['dice_et']:.4f}")
    print(f"  Dice Mean: {metrics['dice_mean']:.4f}")
    print(f"  HD95 WT: {metrics['hd95_wt']:.2f}")
    print(f"  HD95 TC: {metrics['hd95_tc']:.2f}")
    print(f"  HD95 ET: {metrics['hd95_et']:.2f}")
    
    assert 0 <= metrics['dice_wt'] <= 1, "Dice WT out of range"
    assert 0 <= metrics['dice_tc'] <= 1, "Dice TC out of range"
    assert 0 <= metrics['dice_et'] <= 1, "Dice ET out of range"
    print("  ✓ Basic metrics test passed")
    
    # Test 2: Edge case - empty prediction
    print("\n2. Testing empty prediction...")
    
    empty_pred = np.zeros((32, 32, 32), dtype=np.int64)
    small_target = np.zeros((32, 32, 32), dtype=np.int64)
    small_target[10:20, 10:20, 10:20] = 1
    
    metrics_empty = compute_metrics(empty_pred, small_target)
    print(f"  Dice WT (empty pred): {metrics_empty['dice_wt']:.4f}")
    print(f"  HD95 WT (empty pred): {metrics_empty['hd95_wt']}")
    print("  ✓ Empty prediction test passed")
    
    # Test 3: Edge case - identical masks
    print("\n3. Testing identical masks...")
    
    identical = np.zeros((32, 32, 32), dtype=np.int64)
    identical[10:20, 10:20, 10:20] = 1
    identical[12:18, 12:18, 12:18] = 3
    
    metrics_identical = compute_metrics(identical.copy(), identical.copy())
    print(f"  Dice WT (identical): {metrics_identical['dice_wt']:.4f}")
    print(f"  Dice Mean (identical): {metrics_identical['dice_mean']:.4f}")
    
    assert metrics_identical['dice_wt'] == 1.0, "Dice should be 1.0 for identical masks"
    print("  ✓ Identical masks test passed")
    
    # Test 4: MetricTracker
    print("\n4. Testing MetricTracker...")
    
    tracker = MetricTracker()
    for _ in range(5):
        tracker.update(metrics)
    
    avg = tracker.get_average()
    print(f"  Tracked dice_mean: {avg['dice_mean']:.4f}")
    assert abs(avg['dice_mean'] - metrics['dice_mean']) < 1e-6
    print("  ✓ MetricTracker test passed")
    
    # Test 5: Large volume (memory test)
    print("\n5. Testing with larger volume (memory test)...")
    
    large_pred = np.zeros((128, 128, 128), dtype=np.int64)
    large_target = np.zeros((128, 128, 128), dtype=np.int64)
    
    large_pred[30:100, 30:100, 30:100] = 1
    large_pred[40:90, 40:90, 40:90] = 2
    large_pred[50:80, 50:80, 50:80] = 3
    
    large_target[28:102, 28:102, 28:102] = 1
    large_target[38:92, 38:92, 38:92] = 2
    large_target[48:82, 48:82, 48:82] = 3
    
    try:
        metrics_large = compute_metrics(large_pred, large_target)
        print(f"  Large volume Dice Mean: {metrics_large['dice_mean']:.4f}")
        print(f"  Large volume HD95 Mean: {metrics_large['hd95_mean']:.2f}")
        print("  ✓ Large volume test passed")
    except MemoryError:
        print("  ✗ Memory error on large volume - this should not happen!")
        return False
    
    print("\n✓ All metrics tests passed!")
    return True


def test_model():
    """Test the model module."""
    print("\n" + "=" * 60)
    print("TESTING MODEL MODULE")
    print("=" * 60)
    
    from src.models.mraf_net import MRAFNet, create_model
    
    print("\n1. Testing model creation...")
    
    model = MRAFNet(
        in_channels=4,
        num_classes=4,
        base_features=32,
        deep_supervision=True
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Test forward pass with small input
    print("\n2. Testing forward pass...")
    
    x = torch.randn(1, 4, 32, 32, 32)  # Small input for testing
    
    model.eval()
    with torch.no_grad():
        output, ds_outputs = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == (1, 4, 32, 32, 32), f"Unexpected output shape: {output.shape}"
    print("  ✓ Forward pass test passed")
    
    # Test gradient checkpointing
    print("\n3. Testing gradient checkpointing...")
    
    model.enable_gradient_checkpointing()
    assert model.use_checkpointing == True
    print("  ✓ Gradient checkpointing enabled")
    
    model.disable_gradient_checkpointing()
    assert model.use_checkpointing == False
    print("  ✓ Gradient checkpointing disabled")
    
    print("\n✓ All model tests passed!")
    return True


def test_losses():
    """Test the losses module."""
    print("\n" + "=" * 60)
    print("TESTING LOSSES MODULE")
    print("=" * 60)
    
    from src.losses.losses import DiceLoss, DiceCELoss, FocalLoss, DeepSupervisionLoss
    
    # Create test data
    B, C, D, H, W = 2, 4, 16, 16, 16
    pred = torch.randn(B, C, D, H, W)
    target = torch.randint(0, C, (B, D, H, W))
    
    print("\n1. Testing DiceLoss...")
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"  DiceLoss: {loss.item():.4f}")
    assert 0 <= loss.item() <= 1, "Dice loss should be in [0, 1]"
    print("  ✓ DiceLoss test passed")
    
    print("\n2. Testing DiceCELoss...")
    dice_ce_loss = DiceCELoss()
    loss = dice_ce_loss(pred, target)
    print(f"  DiceCELoss: {loss.item():.4f}")
    assert loss.item() > 0, "Combined loss should be positive"
    print("  ✓ DiceCELoss test passed")
    
    print("\n3. Testing FocalLoss...")
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    print(f"  FocalLoss: {loss.item():.4f}")
    assert loss.item() > 0, "Focal loss should be positive"
    print("  ✓ FocalLoss test passed")
    
    print("\n4. Testing DeepSupervisionLoss...")
    ds_outputs = [
        torch.randn(B, C, D//2, H//2, W//2),
        torch.randn(B, C, D//4, H//4, W//4)
    ]
    ds_loss = DeepSupervisionLoss(DiceCELoss())
    loss = ds_loss((pred, ds_outputs), target)
    print(f"  DeepSupervisionLoss: {loss.item():.4f}")
    assert loss.item() > 0, "DS loss should be positive"
    print("  ✓ DeepSupervisionLoss test passed")
    
    print("\n✓ All loss tests passed!")
    return True


def test_preprocessing():
    """Test preprocessing functions."""
    print("\n" + "=" * 60)
    print("TESTING PREPROCESSING MODULE")
    print("=" * 60)
    
    from src.data.preprocessing import (
        normalize_intensity, crop_to_foreground,
        remove_small_connected_components
    )
    
    print("\n1. Testing intensity normalization...")
    
    images = np.random.randn(4, 64, 64, 64).astype(np.float32)
    images = np.abs(images) + 1  # Make positive
    
    # Add background (zero region)
    images[:, :10, :, :] = 0
    
    normalized = normalize_intensity(images.copy(), method='zscore')
    
    mask = normalized[0] != 0
    mean = normalized[0][mask].mean()
    std = normalized[0][mask].std()
    
    print(f"  After zscore: mean={mean:.4f}, std={std:.4f}")
    assert abs(mean) < 0.1, "Mean should be close to 0"
    assert abs(std - 1.0) < 0.1, "Std should be close to 1"
    print("  ✓ Normalization test passed")
    
    print("\n2. Testing foreground cropping...")
    
    images_crop = np.zeros((4, 100, 100, 100), dtype=np.float32)
    images_crop[:, 20:80, 20:80, 20:80] = 1.0
    
    label_crop = np.zeros((100, 100, 100), dtype=np.int64)
    label_crop[30:70, 30:70, 30:70] = 1
    
    cropped_img, cropped_label = crop_to_foreground(images_crop, label_crop, margin=5)
    
    print(f"  Original: {images_crop.shape[1:]}, Cropped: {cropped_img.shape[1:]}")
    assert cropped_img.shape[1] < images_crop.shape[1], "Should be cropped"
    print("  ✓ Foreground cropping test passed")
    
    print("\n3. Testing small component removal...")
    
    pred = np.zeros((50, 50, 50), dtype=np.int64)
    pred[10:40, 10:40, 10:40] = 1  # Large component
    pred[0:3, 0:3, 0:3] = 1  # Small component (27 voxels)
    
    cleaned = remove_small_connected_components(pred, min_size=100)
    
    # Small component should be removed
    assert cleaned[0, 0, 0] == 0, "Small component should be removed"
    assert cleaned[20, 20, 20] == 1, "Large component should remain"
    print("  ✓ Small component removal test passed")
    
    print("\n✓ All preprocessing tests passed!")
    return True


def test_helpers():
    """Test helper utilities."""
    print("\n" + "=" * 60)
    print("TESTING HELPERS MODULE")
    print("=" * 60)
    
    from src.utils.helpers import AverageMeter, EarlyStopping, set_seed
    
    print("\n1. Testing AverageMeter...")
    
    meter = AverageMeter('loss')
    for i in range(10):
        meter.update(i * 0.1)
    
    assert abs(meter.avg - 0.45) < 0.01, "Average should be 0.45"
    print(f"  Average: {meter.avg:.4f}")
    print("  ✓ AverageMeter test passed")
    
    print("\n2. Testing EarlyStopping...")
    
    es = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.55, 0.54, 0.53]  # Improves, then stagnates
    
    for i, score in enumerate(scores):
        should_stop = es(score)
        print(f"  Score: {score:.2f}, Counter: {es.counter}, Stop: {should_stop}")
    
    assert should_stop == True, "Should trigger early stopping"
    print("  ✓ EarlyStopping test passed")
    
    print("\n3. Testing set_seed...")
    
    set_seed(42)
    val1 = np.random.rand()
    
    set_seed(42)
    val2 = np.random.rand()
    
    assert val1 == val2, "Same seed should give same random value"
    print("  ✓ set_seed test passed")
    
    print("\n✓ All helper tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MRAF-NET COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_metrics()
    except Exception as e:
        print(f"\n✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_losses()
    except Exception as e:
        print(f"\n✗ Losses test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_preprocessing()
    except Exception as e:
        print(f"\n✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_helpers()
    except Exception as e:
        print(f"\n✗ Helpers test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_model()
    except Exception as e:
        print(f"\n✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
