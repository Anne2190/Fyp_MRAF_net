"""
MRAF-Net Prediction Script
Run inference on single MRI cases

Author: Anne Nidhusha Nithiyalan (w1985740)

Usage:
    python scripts/predict.py --checkpoint checkpoints/best_model.pth --input path/to/case
    python scripts/predict.py --checkpoint checkpoints/best_model.pth --input path/to/case --output predictions/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mraf_net import create_model
from src.data.preprocessing import normalize_intensity, remove_small_connected_components
from src.utils.helpers import load_config, get_device


class Predictor:
    """
    MRAF-Net Predictor for single case inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: torch.device = None
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            device: Compute device
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_device()
        
        # Load config
        if config_path is None:
            config_path = self.checkpoint_path.parent.parent / 'config.yaml'
        
        if Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Load model
        self._load_model()
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'data': {
                'in_channels': 4,
                'num_classes': 4,
                'modalities': ['flair', 't1', 't1ce', 't2']
            },
            'model': {
                'base_features': 32,
                'use_attention': True,
                'use_cross_modal_fusion': True,
                'use_aspp': True,
                'deep_supervision': False,
                'dropout': 0.0
            },
            'inference': {
                'roi_size': [128, 128, 128],
                'sw_batch_size': 4,
                'overlap': 0.5,
                'mode': 'gaussian'
            }
        }
    
    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_case(self, case_dir: str) -> tuple:
        """
        Load MRI case from directory.
        
        Args:
            case_dir: Path to case directory
        
        Returns:
            (images, affine, header, case_id)
        """
        case_path = Path(case_dir)
        case_id = case_path.name
        
        modalities = self.config['data'].get('modalities', ['flair', 't1', 't1ce', 't2'])
        
        images = []
        affine = None
        header = None
        
        for mod in modalities:
            # Try different naming conventions
            possible_names = [
                f"{case_id}_{mod}.nii.gz",
                f"{case_id}_{mod}.nii",
                f"{mod}.nii.gz",
                f"{mod}.nii"
            ]
            
            found = False
            for name in possible_names:
                img_path = case_path / name
                if img_path.exists():
                    img_nii = nib.load(str(img_path))
                    images.append(img_nii.get_fdata().astype(np.float32))
                    
                    if affine is None:
                        affine = img_nii.affine
                        header = img_nii.header
                    
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"Could not find {mod} image in {case_path}")
        
        images = np.stack(images, axis=0)  # (C, H, W, D)
        
        return images, affine, header, case_id
    
    def predict(
        self,
        images: np.ndarray,
        roi_size: tuple = None,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Run prediction on images.
        
        Args:
            images: Input images of shape (C, H, W, D)
            roi_size: Size of sliding window
            overlap: Overlap between windows
        
        Returns:
            Prediction of shape (H, W, D)
        """
        # Normalize
        images = normalize_intensity(images)
        
        # Transpose to (C, D, H, W)
        images = np.transpose(images, (0, 3, 1, 2))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(images).float().unsqueeze(0).to(self.device)
        
        # Get inference config
        if roi_size is None:
            roi_size = tuple(self.config['inference'].get('roi_size', [128, 128, 128]))
        
        sw_batch_size = self.config['inference'].get('sw_batch_size', 4)
        mode = self.config['inference'].get('mode', 'gaussian')
        
        # Run sliding window inference
        pred_probs = self._sliding_window_inference(
            image_tensor, roi_size, sw_batch_size, overlap, mode
        )
        
        # Get prediction
        pred = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
        
        # Transpose back to (H, W, D)
        pred = np.transpose(pred, (1, 2, 0))
        
        # Post-processing
        pred = remove_small_connected_components(pred, min_size=200)
        
        # Convert label 3 back to 4 for BraTS format
        pred[pred == 3] = 4
        
        return pred
    
    def _sliding_window_inference(
        self,
        image: torch.Tensor,
        roi_size: tuple,
        sw_batch_size: int,
        overlap: float,
        mode: str
    ) -> torch.Tensor:
        """Perform sliding window inference."""
        _, C, D, H, W = image.shape
        num_classes = self.config['data']['num_classes']
        
        # Calculate step size
        step_d = int(roi_size[0] * (1 - overlap))
        step_h = int(roi_size[1] * (1 - overlap))
        step_w = int(roi_size[2] * (1 - overlap))
        
        step_d = max(1, step_d)
        step_h = max(1, step_h)
        step_w = max(1, step_w)
        
        # Pad if needed
        pad_d = max(0, roi_size[0] - D)
        pad_h = max(0, roi_size[1] - H)
        pad_w = max(0, roi_size[2] - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
            _, _, D, H, W = image.shape
        
        # Output tensors
        output_sum = torch.zeros((1, num_classes, D, H, W), device=self.device)
        count_map = torch.zeros((1, 1, D, H, W), device=self.device)
        
        # Importance map
        if mode == 'gaussian':
            importance_map = self._create_gaussian_map(roi_size).to(self.device)
        else:
            importance_map = torch.ones((1, 1) + roi_size, device=self.device)
        
        # Collect positions
        positions = []
        for d in range(0, max(1, D - roi_size[0] + 1), step_d):
            for h in range(0, max(1, H - roi_size[1] + 1), step_h):
                for w in range(0, max(1, W - roi_size[2] + 1), step_w):
                    positions.append((d, h, w))
        
        # Process
        print(f"Processing {len(positions)} windows...")
        
        for i in range(0, len(positions), sw_batch_size):
            batch_positions = positions[i:i + sw_batch_size]
            
            windows = []
            for d, h, w in batch_positions:
                window = image[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]]
                windows.append(window)
            
            batch = torch.cat(windows, dim=0)
            
            with autocast():
                with torch.no_grad():
                    pred = self.model(batch)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pred = F.softmax(pred, dim=1)
            
            for j, (d, h, w) in enumerate(batch_positions):
                output_sum[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]] += \
                    pred[j:j+1] * importance_map
                count_map[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]] += \
                    importance_map
        
        output = output_sum / (count_map + 1e-8)
        output = output[:, :, :D-pad_d or None, :H-pad_h or None, :W-pad_w or None]
        
        return output
    
    def _create_gaussian_map(self, size: tuple) -> torch.Tensor:
        """Create Gaussian importance map."""
        sigma = 0.125
        coords = [torch.linspace(-1, 1, s) for s in size]
        grid = torch.meshgrid(*coords, indexing='ij')
        distance = sum(g ** 2 for g in grid)
        importance = torch.exp(-distance / (2 * sigma ** 2))
        return importance.unsqueeze(0).unsqueeze(0)
    
    def predict_case(
        self,
        case_dir: str,
        output_dir: str = None,
        visualize: bool = True
    ) -> np.ndarray:
        """
        Predict on a single case.
        
        Args:
            case_dir: Path to case directory
            output_dir: Path to save outputs
            visualize: Whether to create visualization
        
        Returns:
            Prediction array
        """
        print(f"\nProcessing case: {case_dir}")
        
        # Load case
        images, affine, header, case_id = self.load_case(case_dir)
        print(f"  - Input shape: {images.shape}")
        
        # Predict
        prediction = self.predict(images)
        print(f"  - Prediction shape: {prediction.shape}")
        print(f"  - Unique labels: {np.unique(prediction)}")
        
        # Save prediction
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save NIfTI
            pred_nii = nib.Nifti1Image(prediction.astype(np.int16), affine, header)
            pred_path = output_path / f"{case_id}_pred.nii.gz"
            nib.save(pred_nii, str(pred_path))
            print(f"  - Saved prediction to {pred_path}")
            
            # Save visualization
            if visualize:
                self._save_visualization(images, prediction, output_path, case_id)
        
        return prediction
    
    def _save_visualization(
        self,
        images: np.ndarray,
        prediction: np.ndarray,
        output_dir: Path,
        case_id: str
    ):
        """Create and save visualization."""
        # Get middle slice
        mid_slice = prediction.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show FLAIR and prediction
        axes[0, 0].imshow(images[0, :, :, mid_slice].T, cmap='gray')
        axes[0, 0].set_title('FLAIR')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(images[1, :, :, mid_slice].T, cmap='gray')
        axes[0, 1].set_title('T1')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(images[2, :, :, mid_slice].T, cmap='gray')
        axes[0, 2].set_title('T1ce')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(images[3, :, :, mid_slice].T, cmap='gray')
        axes[1, 0].set_title('T2')
        axes[1, 0].axis('off')
        
        # Show prediction
        axes[1, 1].imshow(images[0, :, :, mid_slice].T, cmap='gray')
        pred_mask = np.ma.masked_where(prediction[:, :, mid_slice].T == 0, prediction[:, :, mid_slice].T)
        axes[1, 1].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Prediction Overlay')
        axes[1, 1].axis('off')
        
        # Show prediction only
        axes[1, 2].imshow(prediction[:, :, mid_slice].T, cmap='nipy_spectral')
        axes[1, 2].set_title('Prediction')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Case: {case_id}')
        plt.tight_layout()
        
        fig_path = output_dir / f"{case_id}_visualization.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - Saved visualization to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='MRAF-Net Single Case Prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input case directory')
    parser.add_argument('--output', type=str, default='outputs/predictions',
                        help='Path to save outputs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Skip visualization')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Run prediction
    predictor.predict_case(
        case_dir=args.input,
        output_dir=args.output,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()
