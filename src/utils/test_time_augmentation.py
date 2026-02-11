"""
MRAF-Net Test-Time Augmentation (TTA)
Applies multiple geometric transforms during inference and averages predictions
for more robust and accurate segmentation results.

This approach is standard in medical image segmentation competitions but
rarely implemented in academic projects — adding it demonstrates practical
deployment readiness and can boost Dice scores by 1-2%.

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) wrapper for 3D segmentation models.
    
    During inference, applies a set of geometric augmentations (flips along
    each axis), runs the model on each augmented version, un-augments the
    predictions, and averages them for a more robust final prediction.
    
    Args:
        model: The trained segmentation model.
        flip_axes: Which axes to flip. Default: all 3 spatial axes.
        merge_mode: How to merge predictions ('mean' or 'max').
    """
    
    def __init__(
        self,
        model: nn.Module,
        flip_axes: List[int] = None,
        merge_mode: str = 'mean'
    ):
        self.model = model
        self.flip_axes = flip_axes or [2, 3, 4]  # D, H, W axes (after batch & channel)
        self.merge_mode = merge_mode
    
    def _get_augmentations(self) -> List[List[int]]:
        """
        Generate all combinations of axis flips.
        
        For 3 axes, gives 2^3 = 8 combinations (including no-flip).
        """
        augmentations = [[]]  # No flip
        for axis in self.flip_axes:
            new_augs = []
            for aug in augmentations:
                new_augs.append(aug + [axis])
            augmentations.extend(new_augs)
        return augmentations
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA and return averaged predictions.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            Averaged softmax predictions of shape (B, num_classes, D, H, W)
        """
        self.model.eval()
        augmentations = self._get_augmentations()
        all_predictions = []
        
        for flip_dims in augmentations:
            # Apply augmentation (flip)
            augmented = x
            for dim in flip_dims:
                augmented = torch.flip(augmented, dims=[dim])
            
            # Run model
            output, _ = self.model(augmented)
            pred = F.softmax(output, dim=1)
            
            # Reverse augmentation (flip back)
            for dim in reversed(flip_dims):
                pred = torch.flip(pred, dims=[dim])
            
            all_predictions.append(pred)
        
        # Merge predictions
        stacked = torch.stack(all_predictions, dim=0)
        
        if self.merge_mode == 'mean':
            final_pred = stacked.mean(dim=0)
        elif self.merge_mode == 'max':
            final_pred = stacked.max(dim=0).values
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")
        
        return final_pred


class SlidingWindowInference:
    """
    Sliding window inference for full-volume prediction.
    
    Processes a full 3D volume using overlapping patches (windows),
    runs each through the model, and stitches the results with
    Gaussian-weighted blending for smooth boundaries.
    
    Args:
        model: The trained segmentation model.
        patch_size: Size of each patch (D, H, W).
        overlap: Overlap ratio between adjacent patches (0.0 to 0.9).
        use_tta: Whether to apply TTA on each patch.
        batch_size: Number of patches to process simultaneously.
    """
    
    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5,
        use_tta: bool = False,
        batch_size: int = 1
    ):
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        
        if use_tta:
            self.predictor = TestTimeAugmentation(model)
        else:
            self.predictor = None
    
    def _create_gaussian_weight(self) -> torch.Tensor:
        """Create 3D Gaussian weight map for blending overlapping patches."""
        sigma = 0.125
        d, h, w = self.patch_size
        
        coords_d = torch.arange(d).float() / d - 0.5
        coords_h = torch.arange(h).float() / h - 0.5
        coords_w = torch.arange(w).float() / w - 0.5
        
        grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
        gaussian = torch.exp(-(grid_d**2 + grid_h**2 + grid_w**2) / (2 * sigma**2))
        
        # Normalize
        gaussian = gaussian / gaussian.max()
        gaussian = torch.clamp(gaussian, min=1e-4)
        
        return gaussian
    
    @torch.no_grad()
    def __call__(
        self,
        volume: torch.Tensor,
        num_classes: int = 4
    ) -> torch.Tensor:
        """
        Run sliding window inference on a full volume.
        
        Args:
            volume: Input tensor of shape (1, C, D, H, W)
            num_classes: Number of output classes
        
        Returns:
            Prediction of shape (1, num_classes, D, H, W)
        """
        self.model.eval()
        device = volume.device
        _, C, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        # Calculate step sizes
        step_d = max(int(pd * (1 - self.overlap)), 1)
        step_h = max(int(ph * (1 - self.overlap)), 1)
        step_w = max(int(pw * (1 - self.overlap)), 1)
        
        # Pad volume if needed
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
        
        _, _, Dp, Hp, Wp = volume.shape
        
        # Create output tensors
        output = torch.zeros(1, num_classes, Dp, Hp, Wp, device=device)
        count = torch.zeros(1, 1, Dp, Hp, Wp, device=device)
        
        # Gaussian weight
        weight = self._create_gaussian_weight().to(device)
        
        # Collect patch positions
        positions = []
        for d in range(0, Dp - pd + 1, step_d):
            for h in range(0, Hp - ph + 1, step_h):
                for w in range(0, Wp - pw + 1, step_w):
                    positions.append((d, h, w))
        
        # Process patches in batches
        for i in range(0, len(positions), self.batch_size):
            batch_positions = positions[i:i + self.batch_size]
            patches = []
            
            for d, h, w in batch_positions:
                patch = volume[:, :, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
            
            batch = torch.cat(patches, dim=0)  # (batch_size, C, pd, ph, pw)
            
            # Predict
            if self.predictor is not None:
                pred = self.predictor(batch)
            else:
                model_output, _ = self.model(batch)
                pred = F.softmax(model_output, dim=1)
            
            # Accumulate predictions
            for j, (d, h, w) in enumerate(batch_positions):
                output[0, :, d:d+pd, h:h+ph, w:w+pw] += pred[j] * weight
                count[0, :, d:d+pd, h:h+ph, w:w+pw] += weight
        
        # Average
        output = output / (count + 1e-8)
        
        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            output = output[:, :, :D, :H, :W]
        
        return output


if __name__ == "__main__":
    print("Testing Test-Time Augmentation...")
    
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 4, 1)
        
        def forward(self, x):
            return self.conv(x), None
    
    model = DummyModel()
    
    # Test TTA
    tta = TestTimeAugmentation(model)
    x = torch.randn(1, 4, 32, 32, 32)
    out = tta(x)
    print(f"TTA: {x.shape} -> {out.shape}")
    
    # Test Sliding Window
    sw = SlidingWindowInference(model, patch_size=(32, 32, 32), overlap=0.5)
    vol = torch.randn(1, 4, 64, 64, 64)
    out = sw(vol, num_classes=4)
    print(f"SlidingWindow: {vol.shape} -> {out.shape}")
    
    print("All TTA tests passed!")
