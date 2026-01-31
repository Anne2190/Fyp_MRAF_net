"""
MRAF-Net MRI Preprocessing Utilities
Intensity normalization and preprocessing for brain MRI

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def normalize_intensity(
    images: np.ndarray,
    method: str = 'zscore',
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize MRI intensity values.
    
    Args:
        images: Input images of shape (C, D, H, W) where C is modalities
        method: Normalization method ('zscore', 'minmax', 'percentile')
        clip_range: Optional range to clip values after normalization
    
    Returns:
        Normalized images of same shape
    """
    images = images.astype(np.float32)
    
    for c in range(images.shape[0]):
        # Get brain mask (non-zero voxels)
        mask = images[c] > 0
        
        if mask.sum() == 0:
            continue
        
        if method == 'zscore':
            # Z-score normalization within brain mask
            mean = images[c][mask].mean()
            std = images[c][mask].std()
            if std > 0:
                images[c] = (images[c] - mean) / std
            images[c][~mask] = 0
            
        elif method == 'minmax':
            # Min-max normalization within brain mask
            min_val = images[c][mask].min()
            max_val = images[c][mask].max()
            if max_val > min_val:
                images[c] = (images[c] - min_val) / (max_val - min_val)
            images[c][~mask] = 0
            
        elif method == 'percentile':
            # Percentile-based normalization (robust to outliers)
            p1 = np.percentile(images[c][mask], 1)
            p99 = np.percentile(images[c][mask], 99)
            images[c] = np.clip(images[c], p1, p99)
            images[c] = (images[c] - p1) / (p99 - p1 + 1e-8)
            images[c][~mask] = 0
    
    if clip_range is not None:
        images = np.clip(images, clip_range[0], clip_range[1])
    
    return images


def preprocess_volume(
    images: np.ndarray,
    label: Optional[np.ndarray] = None,
    target_spacing: Tuple[float, float, float] = None,
    current_spacing: Tuple[float, float, float] = None,
    crop_foreground: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Complete preprocessing pipeline for a single volume.
    
    Args:
        images: Input images of shape (C, H, W, D)
        label: Optional segmentation label of shape (H, W, D)
        target_spacing: Target voxel spacing (optional resampling)
        current_spacing: Current voxel spacing
        crop_foreground: Whether to crop to foreground
        normalize: Whether to normalize intensities
    
    Returns:
        Tuple of (preprocessed_images, preprocessed_label)
    """
    # Resample if needed
    if target_spacing is not None and current_spacing is not None:
        images, label = resample_volume(
            images, label, current_spacing, target_spacing
        )
    
    # Crop to foreground
    if crop_foreground:
        images, label = crop_to_foreground(images, label, margin=5)
    
    # Normalize intensities
    if normalize:
        images = normalize_intensity(images, method='zscore')
    
    return images, label


def resample_volume(
    images: np.ndarray,
    label: Optional[np.ndarray],
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Resample volume to target spacing.
    
    Args:
        images: Input images of shape (C, H, W, D)
        label: Optional label of shape (H, W, D)
        current_spacing: Current voxel spacing (h, w, d)
        target_spacing: Target voxel spacing (h, w, d)
    
    Returns:
        Resampled (images, label)
    """
    # Calculate zoom factors
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    
    # Resample each modality
    resampled_images = []
    for c in range(images.shape[0]):
        resampled = ndimage.zoom(
            images[c], zoom_factors, order=3, mode='nearest'
        )
        resampled_images.append(resampled)
    
    images = np.stack(resampled_images, axis=0)
    
    # Resample label with nearest neighbor
    if label is not None:
        label = ndimage.zoom(
            label.astype(np.float32), zoom_factors, order=0, mode='nearest'
        ).astype(np.int64)
    
    return images, label


def crop_to_foreground(
    images: np.ndarray,
    label: Optional[np.ndarray],
    margin: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Crop volume to foreground region.
    
    Args:
        images: Input images of shape (C, H, W, D)
        label: Optional label of shape (H, W, D)
        margin: Margin to add around foreground
    
    Returns:
        Cropped (images, label)
    """
    # Find foreground mask (any modality > 0)
    mask = np.any(images > 0, axis=0)
    
    if not np.any(mask):
        return images, label
    
    # Get bounding box
    nonzero = np.where(mask)
    h_min, h_max = nonzero[0].min(), nonzero[0].max() + 1
    w_min, w_max = nonzero[1].min(), nonzero[1].max() + 1
    d_min, d_max = nonzero[2].min(), nonzero[2].max() + 1
    
    # Add margin
    h_min = max(0, h_min - margin)
    h_max = min(images.shape[1], h_max + margin)
    w_min = max(0, w_min - margin)
    w_max = min(images.shape[2], w_max + margin)
    d_min = max(0, d_min - margin)
    d_max = min(images.shape[3], d_max + margin)
    
    # Crop
    images = images[:, h_min:h_max, w_min:w_max, d_min:d_max].copy()
    
    if label is not None:
        label = label[h_min:h_max, w_min:w_max, d_min:d_max].copy()
    
    return images, label


def pad_or_crop_to_size(
    images: np.ndarray,
    label: Optional[np.ndarray],
    target_size: Tuple[int, int, int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Pad or crop volume to target size.
    
    Args:
        images: Input images of shape (C, H, W, D)
        label: Optional label of shape (H, W, D)
        target_size: Target size (H, W, D)
    
    Returns:
        Resized (images, label)
    """
    current_size = images.shape[1:]
    
    # Calculate padding/cropping for each dimension
    deltas = [t - c for t, c in zip(target_size, current_size)]
    
    # Process each dimension
    for dim, delta in enumerate(deltas):
        if delta > 0:
            # Need to pad
            pad_before = delta // 2
            pad_after = delta - pad_before
            
            # Create padding configuration
            pad_width = [(0, 0)] + [(0, 0)] * dim + [(pad_before, pad_after)] + [(0, 0)] * (2 - dim)
            images = np.pad(images, pad_width, mode='constant', constant_values=0)
            
            if label is not None:
                label_pad = [(0, 0)] * dim + [(pad_before, pad_after)] + [(0, 0)] * (2 - dim)
                label = np.pad(label, label_pad, mode='constant', constant_values=0)
                
        elif delta < 0:
            # Need to crop
            crop_before = (-delta) // 2
            crop_after = current_size[dim] - ((-delta) - crop_before)
            
            slices = [slice(None)] + [slice(None)] * dim + [slice(crop_before, crop_after)] + [slice(None)] * (2 - dim)
            images = images[tuple(slices)]
            
            if label is not None:
                label_slices = [slice(None)] * dim + [slice(crop_before, crop_after)] + [slice(None)] * (2 - dim)
                label = label[tuple(label_slices)]
    
    return images, label


def remove_small_connected_components(
    prediction: np.ndarray,
    min_size: int = 200
) -> np.ndarray:
    """
    Post-processing: Remove small connected components.
    
    Args:
        prediction: Predicted segmentation of shape (H, W, D)
        min_size: Minimum component size to keep
    
    Returns:
        Cleaned prediction
    """
    from scipy.ndimage import label as nd_label
    
    cleaned = np.zeros_like(prediction)
    
    for class_idx in np.unique(prediction):
        if class_idx == 0:
            continue
        
        # Get binary mask for this class
        binary = (prediction == class_idx).astype(np.int32)
        
        # Find connected components
        labeled, num_features = nd_label(binary)
        
        # Keep only components larger than threshold
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() >= min_size:
                cleaned[component] = class_idx
    
    return cleaned


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing Preprocessing Functions...")
    
    # Create dummy data
    images = np.random.randn(4, 240, 240, 155).astype(np.float32)
    images[:, 20:220, 20:220, 10:145] = np.abs(images[:, 20:220, 20:220, 10:145]) + 1
    
    label = np.zeros((240, 240, 155), dtype=np.int64)
    label[100:140, 100:140, 70:90] = 1
    label[105:135, 105:135, 75:85] = 2
    label[110:130, 110:130, 77:83] = 4
    
    print(f"Original image shape: {images.shape}")
    print(f"Original label shape: {label.shape}")
    
    # Test normalization
    norm_images = normalize_intensity(images.copy(), method='zscore')
    print(f"After normalization - mean: {norm_images[0][norm_images[0] > 0].mean():.4f}, std: {norm_images[0][norm_images[0] > 0].std():.4f}")
    
    # Test crop to foreground
    cropped_images, cropped_label = crop_to_foreground(images.copy(), label.copy(), margin=5)
    print(f"After crop - image: {cropped_images.shape}, label: {cropped_label.shape}")
    
    # Test full preprocessing
    processed_images, processed_label = preprocess_volume(
        images.copy(), label.copy(),
        crop_foreground=True, normalize=True
    )
    print(f"After full preprocessing - image: {processed_images.shape}, label: {processed_label.shape}")
    
    # Test connected component removal
    cleaned = remove_small_connected_components(label, min_size=100)
    print(f"After CC removal - unique labels: {np.unique(cleaned)}")
    
    print("All tests passed!")
