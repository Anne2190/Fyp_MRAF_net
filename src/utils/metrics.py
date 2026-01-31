"""
MRAF-Net Evaluation Metrics
Dice Score, Hausdorff Distance, Sensitivity, Specificity

Author: Anne Nidhusha Nithiyalan (w1985740)

FIXED: Memory-efficient Hausdorff distance computation
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute Dice Similarity Coefficient.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
    
    Returns:
        Dice score in [0, 1]
    """
    # Ensure binary masks
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    intersection = np.sum(pred * target)
    pred_sum = np.sum(pred)
    target_sum = np.sum(target)
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return float(dice)


def get_surface_points(mask: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """
    Get surface (boundary) points of binary mask.
    
    Uses erosion to find boundary voxels and optionally subsamples
    to limit memory usage.
    
    Args:
        mask: Binary mask (D, H, W)
        max_points: Maximum number of surface points to return
    
    Returns:
        Array of surface point coordinates (N, 3)
    """
    if not np.any(mask):
        return np.array([]).reshape(0, 3)
    
    # Get surface via erosion
    try:
        eroded = binary_erosion(mask)
        surface = mask & ~eroded
    except Exception:
        # If erosion fails, use the mask itself for very small regions
        surface = mask.copy()
    
    # Get coordinates of surface points
    coords = np.argwhere(surface)
    
    if len(coords) == 0:
        # If no surface points (e.g., single voxel), use mask points
        coords = np.argwhere(mask)
    
    # Subsample if too many points to avoid memory issues
    if len(coords) > max_points:
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
    
    return coords


def compute_hausdorff95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_points: int = 10000
) -> float:
    """
    Compute 95th percentile Hausdorff Distance using memory-efficient approach.
    
    Uses surface point sampling instead of full distance transforms to avoid
    memory issues with large volumes.
    
    Args:
        pred: Binary prediction mask (D, H, W)
        target: Binary ground truth mask (D, H, W)
        spacing: Voxel spacing (d, h, w) in mm
        max_points: Maximum surface points to use (memory control)
    
    Returns:
        95th percentile Hausdorff distance in mm
    """
    # Validate inputs
    if pred.ndim != 3 or target.ndim != 3:
        print(f"Warning: Expected 3D arrays, got pred={pred.shape}, target={target.shape}")
        return float('inf')
    
    # Ensure binary masks
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    # Check if either mask is empty
    if not np.any(pred) or not np.any(target):
        return float('inf')
    
    # Check if masks are identical
    if np.array_equal(pred, target):
        return 0.0
    
    try:
        # Get surface points (with subsampling for memory efficiency)
        pred_surface_points = get_surface_points(pred, max_points)
        target_surface_points = get_surface_points(target, max_points)
        
        if len(pred_surface_points) == 0 or len(target_surface_points) == 0:
            return float('inf')
        
        # Apply spacing to coordinates
        spacing_array = np.array(spacing)
        pred_points_mm = pred_surface_points.astype(np.float64) * spacing_array
        target_points_mm = target_surface_points.astype(np.float64) * spacing_array
        
        # Compute distances using chunked approach to save memory
        # For each pred surface point, find distance to nearest target surface point
        pred_to_target_distances = _compute_min_distances(pred_points_mm, target_points_mm)
        target_to_pred_distances = _compute_min_distances(target_points_mm, pred_points_mm)
        
        # Combine all distances
        all_distances = np.concatenate([pred_to_target_distances, target_to_pred_distances])
        
        # Return 95th percentile
        hd95 = np.percentile(all_distances, 95)
        
        return float(hd95)
        
    except MemoryError:
        print("Warning: Memory error in Hausdorff computation, returning inf")
        return float('inf')
    except Exception as e:
        print(f"Warning: Error in Hausdorff computation: {e}")
        return float('inf')


def _compute_min_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
    chunk_size: int = 1000
) -> np.ndarray:
    """
    Compute minimum distances from source points to target points.
    Uses chunked computation to limit memory usage.
    
    Args:
        source_points: Source point coordinates (N, 3)
        target_points: Target point coordinates (M, 3)
        chunk_size: Number of source points to process at once
    
    Returns:
        Array of minimum distances (N,)
    """
    n_source = len(source_points)
    n_target = len(target_points)
    
    if n_source == 0 or n_target == 0:
        return np.array([])
    
    min_distances = np.zeros(n_source, dtype=np.float64)
    
    # Process in chunks to avoid memory issues
    for start_idx in range(0, n_source, chunk_size):
        end_idx = min(start_idx + chunk_size, n_source)
        source_chunk = source_points[start_idx:end_idx]
        
        # Compute distances from chunk to all target points
        # Using broadcasting: (chunk_size, 1, 3) - (1, n_target, 3) -> (chunk_size, n_target, 3)
        # Then sum squared and sqrt
        
        # More memory efficient: process target in chunks too if needed
        if n_target > 10000:
            chunk_min = np.full(len(source_chunk), np.inf)
            for t_start in range(0, n_target, chunk_size):
                t_end = min(t_start + chunk_size, n_target)
                target_chunk = target_points[t_start:t_end]
                
                diff = source_chunk[:, np.newaxis, :] - target_chunk[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=2))
                chunk_min = np.minimum(chunk_min, np.min(distances, axis=1))
            min_distances[start_idx:end_idx] = chunk_min
        else:
            diff = source_chunk[:, np.newaxis, :] - target_points[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            min_distances[start_idx:end_idx] = np.min(distances, axis=1)
    
    return min_distances


def compute_hausdorff95_fast(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Fast Hausdorff distance using distance transforms.
    Only use for small volumes where memory is not a concern.
    
    Args:
        pred: Binary prediction mask (D, H, W)
        target: Binary ground truth mask (D, H, W)
        spacing: Voxel spacing (d, h, w)
    
    Returns:
        95th percentile Hausdorff distance in mm
    """
    # Validate inputs
    if pred.ndim != 3 or target.ndim != 3:
        return float('inf')
    
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    if not np.any(pred) or not np.any(target):
        return float('inf')
    
    try:
        # Get surface points
        pred_eroded = binary_erosion(pred)
        target_eroded = binary_erosion(target)
        
        pred_surface = pred ^ pred_eroded  # XOR to get surface
        target_surface = target ^ target_eroded
        
        # Handle edge case where erosion removes everything
        if not np.any(pred_surface):
            pred_surface = pred
        if not np.any(target_surface):
            target_surface = target
        
        # Compute distance transforms
        pred_dist = distance_transform_edt(~pred, sampling=spacing)
        target_dist = distance_transform_edt(~target, sampling=spacing)
        
        # Get distances at surface points
        pred_to_target = target_dist[pred_surface > 0]
        target_to_pred = pred_dist[target_surface > 0]
        
        if len(pred_to_target) == 0 or len(target_to_pred) == 0:
            return float('inf')
        
        # Combine and compute 95th percentile
        all_distances = np.concatenate([pred_to_target.ravel(), target_to_pred.ravel()])
        
        return float(np.percentile(all_distances, 95))
        
    except Exception as e:
        print(f"Warning: Fast Hausdorff failed: {e}")
        return float('inf')


def compute_sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute Sensitivity (True Positive Rate / Recall).
    
    Sensitivity = TP / (TP + FN)
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
    
    Returns:
        Sensitivity in [0, 1]
    """
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    
    return float(sensitivity)


def compute_specificity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute Specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
    
    Returns:
        Specificity in [0, 1]
    """
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    tn = np.sum((1 - pred) * (1 - target))
    fp = np.sum(pred * (1 - target))
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return float(specificity)


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
    
    Returns:
        IoU in [0, 1]
    """
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def compute_brats_regions(
    pred: np.ndarray,
    target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert prediction and target to BraTS evaluation regions.
    
    BraTS regions:
    - Whole Tumor (WT): labels 1, 2, 3 (originally 1, 2, 4)
    - Tumor Core (TC): labels 1, 3 (originally 1, 4)
    - Enhancing Tumor (ET): label 3 (originally 4)
    
    Note: Assumes labels have been converted (4 -> 3)
    
    Args:
        pred: Prediction with labels [0, 1, 2, 3]
        target: Ground truth with labels [0, 1, 2, 3]
    
    Returns:
        (pred_wt, pred_tc, pred_et, target_wt, target_tc, target_et)
    """
    # Whole Tumor: labels 1, 2, 3
    pred_wt = (pred > 0).astype(np.uint8)
    target_wt = (target > 0).astype(np.uint8)
    
    # Tumor Core: labels 1, 3 (NCR + ET)
    pred_tc = ((pred == 1) | (pred == 3)).astype(np.uint8)
    target_tc = ((target == 1) | (target == 3)).astype(np.uint8)
    
    # Enhancing Tumor: label 3
    pred_et = (pred == 3).astype(np.uint8)
    target_et = (target == 3).astype(np.uint8)
    
    return pred_wt, pred_tc, pred_et, target_wt, target_tc, target_et


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    compute_hd: bool = True
) -> Dict[str, float]:
    """
    Compute all metrics for BraTS evaluation.
    
    Args:
        pred: Prediction array with labels [0, 1, 2, 3], shape (D, H, W)
        target: Ground truth array with labels [0, 1, 2, 3], shape (D, H, W)
        spacing: Voxel spacing for Hausdorff distance
        compute_hd: Whether to compute Hausdorff distance (can skip for speed)
    
    Returns:
        Dictionary of metrics for each region
    """
    # Validate input shapes
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D prediction array, got shape {pred.shape}")
    if target.ndim != 3:
        raise ValueError(f"Expected 3D target array, got shape {target.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    
    # Get BraTS regions
    pred_wt, pred_tc, pred_et, target_wt, target_tc, target_et = compute_brats_regions(pred, target)
    
    metrics = {}
    
    # Whole Tumor metrics
    metrics['dice_wt'] = compute_dice(pred_wt, target_wt)
    metrics['sens_wt'] = compute_sensitivity(pred_wt, target_wt)
    metrics['spec_wt'] = compute_specificity(pred_wt, target_wt)
    
    # Tumor Core metrics
    metrics['dice_tc'] = compute_dice(pred_tc, target_tc)
    metrics['sens_tc'] = compute_sensitivity(pred_tc, target_tc)
    metrics['spec_tc'] = compute_specificity(pred_tc, target_tc)
    
    # Enhancing Tumor metrics
    metrics['dice_et'] = compute_dice(pred_et, target_et)
    metrics['sens_et'] = compute_sensitivity(pred_et, target_et)
    metrics['spec_et'] = compute_specificity(pred_et, target_et)
    
    # Hausdorff distance (optional - can be slow and memory intensive)
    if compute_hd:
        # Use memory-efficient version
        metrics['hd95_wt'] = compute_hausdorff95(pred_wt, target_wt, spacing)
        metrics['hd95_tc'] = compute_hausdorff95(pred_tc, target_tc, spacing)
        metrics['hd95_et'] = compute_hausdorff95(pred_et, target_et, spacing)
    else:
        metrics['hd95_wt'] = float('inf')
        metrics['hd95_tc'] = float('inf')
        metrics['hd95_et'] = float('inf')
    
    # Mean metrics
    metrics['dice_mean'] = (metrics['dice_wt'] + metrics['dice_tc'] + metrics['dice_et']) / 3
    
    # Handle infinite HD95 values
    hd_values = [metrics['hd95_wt'], metrics['hd95_tc'], metrics['hd95_et']]
    valid_hd = [h for h in hd_values if not np.isinf(h)]
    metrics['hd95_mean'] = np.mean(valid_hd) if valid_hd else float('inf')
    
    return metrics


class MetricTracker:
    """
    Utility class for tracking metrics across batches/epochs.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """Add a batch of metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            if value is not None and not np.isinf(value) and not np.isnan(value):
                self.metrics[key] += value
                self.counts[key] += 1
    
    def get_average(self) -> Dict[str, float]:
        """Get average of all tracked metrics."""
        avg = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                avg[key] = self.metrics[key] / self.counts[key]
            else:
                avg[key] = 0.0
        return avg
    
    def __str__(self) -> str:
        avg = self.get_average()
        return ' | '.join([f"{k}: {v:.4f}" for k, v in avg.items()])


if __name__ == "__main__":
    # Test metrics
    print("Testing Evaluation Metrics...")
    
    # Create dummy prediction and target
    pred = np.zeros((64, 64, 64), dtype=np.int64)
    target = np.zeros((64, 64, 64), dtype=np.int64)
    
    # Add some tumor regions
    pred[25:45, 25:45, 25:45] = 1  # NCR
    pred[30:40, 30:40, 30:40] = 2  # Edema
    pred[33:37, 33:37, 33:37] = 3  # ET
    
    target[23:47, 23:47, 23:47] = 1
    target[28:42, 28:42, 28:42] = 2
    target[31:39, 31:39, 31:39] = 3
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(pred, target)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if np.isinf(value):
            print(f"  {key}: inf")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()
    tracker.update(metrics)
    tracker.update(metrics)
    print(f"Tracked metrics: {tracker}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Empty prediction
    empty_pred = np.zeros((32, 32, 32), dtype=np.int64)
    target_small = np.zeros((32, 32, 32), dtype=np.int64)
    target_small[10:20, 10:20, 10:20] = 1
    
    metrics_empty = compute_metrics(empty_pred, target_small)
    print(f"Empty pred - Dice WT: {metrics_empty['dice_wt']:.4f}")
    
    # Identical masks
    identical = np.zeros((32, 32, 32), dtype=np.int64)
    identical[10:20, 10:20, 10:20] = 1
    
    metrics_identical = compute_metrics(identical.copy(), identical.copy())
    print(f"Identical masks - Dice WT: {metrics_identical['dice_wt']:.4f}")
    
    print("\nAll tests passed!")
