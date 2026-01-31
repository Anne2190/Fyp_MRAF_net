"""
MRAF-Net Data Transforms
Data augmentation for brain tumor segmentation

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import random
from typing import Dict, Tuple, Callable, List

import numpy as np
from scipy import ndimage


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomFlip:
    """Random flip along each axis."""
    
    def __init__(self, prob: float = 0.5, axes: Tuple[int, ...] = (0, 1, 2)):
        self.prob = prob
        self.axes = axes
    
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        label = sample['label']
        
        for axis in self.axes:
            if random.random() < self.prob:
                # Flip along axis (add 1 for channel dim in image)
                image = np.flip(image, axis=axis + 1).copy()
                label = np.flip(label, axis=axis).copy()
        
        sample['image'] = image
        sample['label'] = label
        return sample


class RandomRotate90:
    """Random 90-degree rotation in the axial plane."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            label = sample['label']
            
            k = random.randint(1, 3)  # Number of 90-degree rotations
            
            # Rotate in the last two axes (H, W plane)
            image = np.rot90(image, k, axes=(2, 3)).copy()
            label = np.rot90(label, k, axes=(1, 2)).copy()
            
            sample['image'] = image
            sample['label'] = label
        
        return sample


class RandomScaleIntensity:
    """Random intensity scaling."""
    
    def __init__(self, prob: float = 0.5, scale_range: Tuple[float, float] = (-0.1, 0.1)):
        self.prob = prob
        self.scale_range = scale_range
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            
            # Apply random scale to each channel
            for c in range(image.shape[0]):
                scale = 1.0 + random.uniform(*self.scale_range)
                image[c] = image[c] * scale
            
            sample['image'] = image
        
        return sample


class RandomShiftIntensity:
    """Random intensity shift."""
    
    def __init__(self, prob: float = 0.5, shift_range: Tuple[float, float] = (-0.1, 0.1)):
        self.prob = prob
        self.shift_range = shift_range
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            
            # Apply random shift to each channel
            for c in range(image.shape[0]):
                shift = random.uniform(*self.shift_range)
                image[c] = image[c] + shift
            
            sample['image'] = image
        
        return sample


class RandomGaussianNoise:
    """Add random Gaussian noise."""
    
    def __init__(self, prob: float = 0.2, std_range: Tuple[float, float] = (0.0, 0.1)):
        self.prob = prob
        self.std_range = std_range
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            
            std = random.uniform(*self.std_range)
            noise = np.random.randn(*image.shape).astype(np.float32) * std
            image = image + noise
            
            sample['image'] = image
        
        return sample


class RandomGaussianBlur:
    """Apply random Gaussian blur."""
    
    def __init__(self, prob: float = 0.2, sigma_range: Tuple[float, float] = (0.5, 1.5)):
        self.prob = prob
        self.sigma_range = sigma_range
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            
            sigma = random.uniform(*self.sigma_range)
            
            # Apply blur to each channel
            for c in range(image.shape[0]):
                image[c] = ndimage.gaussian_filter(image[c], sigma=sigma)
            
            sample['image'] = image
        
        return sample


class RandomGamma:
    """Apply random gamma correction."""
    
    def __init__(self, prob: float = 0.3, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        self.prob = prob
        self.gamma_range = gamma_range
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            
            gamma = random.uniform(*self.gamma_range)
            
            # Apply gamma to each channel
            for c in range(image.shape[0]):
                # Shift to positive range
                min_val = image[c].min()
                image[c] = image[c] - min_val + 1e-8
                # Apply gamma
                image[c] = np.power(image[c], gamma)
                # Normalize
                image[c] = (image[c] - image[c].mean()) / (image[c].std() + 1e-8)
            
            sample['image'] = image
        
        return sample


class RandomElasticDeformation:
    """Apply random elastic deformation (expensive, use sparingly)."""
    
    def __init__(self, prob: float = 0.1, alpha: float = 100, sigma: float = 10):
        self.prob = prob
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            image = sample['image']
            label = sample['label']
            
            shape = image.shape[1:]  # (D, H, W)
            
            # Create displacement fields
            dx = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dz = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            
            # Create mesh grid
            d, h, w = shape
            z, y, x = np.meshgrid(
                np.arange(d), np.arange(h), np.arange(w), indexing='ij'
            )
            
            indices = [
                np.clip(z + dz, 0, d - 1).astype(int),
                np.clip(y + dy, 0, h - 1).astype(int),
                np.clip(x + dx, 0, w - 1).astype(int)
            ]
            
            # Apply deformation
            for c in range(image.shape[0]):
                image[c] = image[c][indices[0], indices[1], indices[2]]
            label = label[indices[0], indices[1], indices[2]]
            
            sample['image'] = image
            sample['label'] = label
        
        return sample


class CropForeground:
    """Crop to foreground region (brain mask)."""
    
    def __init__(self, margin: int = 0):
        self.margin = margin
    
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        label = sample['label']
        
        # Find foreground (non-zero) region across all modalities
        mask = np.any(image > 0, axis=0)
        
        # Get bounding box
        nonzero = np.where(mask)
        
        if len(nonzero[0]) > 0:
            d_min, d_max = nonzero[0].min(), nonzero[0].max() + 1
            h_min, h_max = nonzero[1].min(), nonzero[1].max() + 1
            w_min, w_max = nonzero[2].min(), nonzero[2].max() + 1
            
            # Add margin
            d_min = max(0, d_min - self.margin)
            d_max = min(image.shape[1], d_max + self.margin)
            h_min = max(0, h_min - self.margin)
            h_max = min(image.shape[2], h_max + self.margin)
            w_min = max(0, w_min - self.margin)
            w_max = min(image.shape[3], w_max + self.margin)
            
            # Crop
            image = image[:, d_min:d_max, h_min:h_max, w_min:w_max].copy()
            label = label[d_min:d_max, h_min:h_max, w_min:w_max].copy()
        
        sample['image'] = image
        sample['label'] = label
        
        return sample


class Identity:
    """Identity transform (no operation)."""
    
    def __call__(self, sample: Dict) -> Dict:
        return sample


def get_train_transforms() -> Compose:
    """Get training data transforms with augmentation."""
    return Compose([
        RandomFlip(prob=0.5, axes=(0, 1, 2)),
        RandomRotate90(prob=0.5),
        RandomScaleIntensity(prob=0.5, scale_range=(-0.1, 0.1)),
        RandomShiftIntensity(prob=0.5, shift_range=(-0.1, 0.1)),
        RandomGaussianNoise(prob=0.2, std_range=(0.0, 0.1)),
        RandomGaussianBlur(prob=0.2, sigma_range=(0.5, 1.0)),
        RandomGamma(prob=0.3, gamma_range=(0.8, 1.2)),
    ])


def get_val_transforms() -> Identity:
    """Get validation data transforms (no augmentation)."""
    return Identity()


if __name__ == "__main__":
    # Test transforms
    print("Testing Data Transforms...")
    
    # Create dummy sample
    sample = {
        'image': np.random.randn(4, 96, 96, 96).astype(np.float32),
        'label': np.random.randint(0, 4, (96, 96, 96)).astype(np.int64)
    }
    
    # Test individual transforms
    transforms_to_test = [
        RandomFlip(prob=1.0),
        RandomRotate90(prob=1.0),
        RandomScaleIntensity(prob=1.0),
        RandomShiftIntensity(prob=1.0),
        RandomGaussianNoise(prob=1.0),
        RandomGaussianBlur(prob=1.0),
        RandomGamma(prob=1.0),
    ]
    
    for transform in transforms_to_test:
        result = transform(sample.copy())
        print(f"{transform.__class__.__name__}: image={result['image'].shape}, label={result['label'].shape}")
    
    # Test composed transforms
    train_transforms = get_train_transforms()
    result = train_transforms(sample.copy())
    print(f"Train transforms: image={result['image'].shape}, label={result['label'].shape}")
    
    val_transforms = get_val_transforms()
    result = val_transforms(sample.copy())
    print(f"Val transforms: image={result['image'].shape}, label={result['label'].shape}")
    
    print("All tests passed!")
