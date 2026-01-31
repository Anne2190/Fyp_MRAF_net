"""
MRAF-Net Evaluation Script
Evaluate trained model on validation/test data

Author: Anne Nidhusha Nithiyalan (w1985740)

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/brats2020/validation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mraf_net import MRAFNet, create_model
from src.data.dataset import BraTSDataset, get_case_ids
from src.data.preprocessing import normalize_intensity, remove_small_connected_components
from src.utils.helpers import load_config, load_checkpoint, get_device, set_seed
from src.utils.metrics import compute_metrics, MetricTracker


class Evaluator:
    """
    MRAF-Net Evaluator class.
    
    Evaluates trained model on validation/test data using sliding window inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: torch.device = None
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (if not in checkpoint dir)
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
            # Use default config
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
                'mode': 'gaussian',
                'use_tta': False
            }
        }
    
    def _load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Create model
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"  - Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"  - Checkpoint metrics: {checkpoint['metrics']}")
    
    def sliding_window_inference(
        self,
        image: torch.Tensor,
        roi_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = 'gaussian'
    ) -> torch.Tensor:
        """
        Perform sliding window inference.
        
        Args:
            image: Input image of shape (1, C, D, H, W)
            roi_size: Size of sliding window
            sw_batch_size: Number of windows per batch
            overlap: Overlap between windows
            mode: Blending mode ('gaussian' or 'constant')
        
        Returns:
            Predicted segmentation of shape (1, num_classes, D, H, W)
        """
        _, C, D, H, W = image.shape
        num_classes = self.config['data']['num_classes']
        
        # Calculate step size
        step_d = int(roi_size[0] * (1 - overlap))
        step_h = int(roi_size[1] * (1 - overlap))
        step_w = int(roi_size[2] * (1 - overlap))
        
        # Ensure at least 1 step
        step_d = max(1, step_d)
        step_h = max(1, step_h)
        step_w = max(1, step_w)
        
        # Pad image if needed
        pad_d = max(0, roi_size[0] - D)
        pad_h = max(0, roi_size[1] - H)
        pad_w = max(0, roi_size[2] - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
            _, _, D, H, W = image.shape
        
        # Create output tensors
        output_sum = torch.zeros((1, num_classes, D, H, W), device=self.device)
        count_map = torch.zeros((1, 1, D, H, W), device=self.device)
        
        # Create importance map for blending
        if mode == 'gaussian':
            importance_map = self._create_gaussian_importance_map(roi_size)
            importance_map = importance_map.to(self.device)
        else:
            importance_map = torch.ones((1, 1) + roi_size, device=self.device)
        
        # Collect all window positions
        positions = []
        for d in range(0, D - roi_size[0] + 1, step_d):
            for h in range(0, H - roi_size[1] + 1, step_h):
                for w in range(0, W - roi_size[2] + 1, step_w):
                    positions.append((d, h, w))
        
        # Add edge positions
        if D > roi_size[0]:
            for h in range(0, H - roi_size[1] + 1, step_h):
                for w in range(0, W - roi_size[2] + 1, step_w):
                    positions.append((D - roi_size[0], h, w))
        
        # Process in batches
        for i in range(0, len(positions), sw_batch_size):
            batch_positions = positions[i:i + sw_batch_size]
            
            # Extract windows
            windows = []
            for d, h, w in batch_positions:
                window = image[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]]
                windows.append(window)
            
            # Stack and predict
            batch = torch.cat(windows, dim=0)
            
            with autocast():
                with torch.no_grad():
                    pred = self.model(batch)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pred = F.softmax(pred, dim=1)
            
            # Accumulate predictions
            for j, (d, h, w) in enumerate(batch_positions):
                output_sum[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]] += \
                    pred[j:j+1] * importance_map
                count_map[:, :, d:d+roi_size[0], h:h+roi_size[1], w:w+roi_size[2]] += \
                    importance_map
        
        # Average predictions
        output = output_sum / (count_map + 1e-8)
        
        # Remove padding
        original_shape = image.shape[2:]
        output = output[:, :, :D-pad_d or None, :H-pad_h or None, :W-pad_w or None]
        
        return output
    
    def _create_gaussian_importance_map(self, size: tuple) -> torch.Tensor:
        """Create Gaussian importance map for blending."""
        sigma = 0.125
        
        coords = [
            torch.linspace(-1, 1, s) for s in size
        ]
        
        grid = torch.meshgrid(*coords, indexing='ij')
        distance = sum(g ** 2 for g in grid)
        
        importance = torch.exp(-distance / (2 * sigma ** 2))
        importance = importance.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        return importance
    
    def evaluate_volume(
        self,
        image: np.ndarray,
        label: np.ndarray = None
    ) -> Dict:
        """
        Evaluate a single volume.
        
        Args:
            image: Input image of shape (C, H, W, D)
            label: Ground truth of shape (H, W, D)
        
        Returns:
            Dictionary with prediction and metrics
        """
        # Normalize
        image = normalize_intensity(image)
        
        # Transpose to (C, D, H, W)
        image = np.transpose(image, (0, 3, 1, 2))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        # Get inference config
        inf_config = self.config.get('inference', {})
        roi_size = tuple(inf_config.get('roi_size', [128, 128, 128]))
        sw_batch_size = inf_config.get('sw_batch_size', 4)
        overlap = inf_config.get('overlap', 0.5)
        mode = inf_config.get('mode', 'gaussian')
        
        # Run inference
        pred_probs = self.sliding_window_inference(
            image_tensor, roi_size, sw_batch_size, overlap, mode
        )
        
        # Test time augmentation
        if inf_config.get('use_tta', False):
            pred_probs = self._apply_tta(image_tensor, pred_probs, roi_size, sw_batch_size, overlap, mode)
        
        # Get final prediction
        pred = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
        
        # Transpose back to (H, W, D)
        pred = np.transpose(pred, (1, 2, 0))
        
        # Post-processing
        pred = remove_small_connected_components(pred, min_size=200)
        
        # Convert label 3 back to 4 for BraTS format
        pred_brats = pred.copy()
        pred_brats[pred_brats == 3] = 4
        
        result = {
            'prediction': pred_brats,
            'probabilities': pred_probs.cpu().numpy()
        }
        
        # Compute metrics if label provided
        if label is not None:
            label_converted = label.copy()
            label_converted[label_converted == 4] = 3
            
            metrics = compute_metrics(pred, label_converted)
            result['metrics'] = metrics
        
        return result
    
    def _apply_tta(
        self,
        image: torch.Tensor,
        base_pred: torch.Tensor,
        roi_size: tuple,
        sw_batch_size: int,
        overlap: float,
        mode: str
    ) -> torch.Tensor:
        """Apply test time augmentation with flips."""
        preds = [base_pred]
        
        # Flip along each axis
        for axis in [2, 3, 4]:  # D, H, W
            flipped = torch.flip(image, dims=[axis])
            pred = self.sliding_window_inference(flipped, roi_size, sw_batch_size, overlap, mode)
            pred = torch.flip(pred, dims=[axis])
            preds.append(pred)
        
        # Average predictions
        return torch.stack(preds, dim=0).mean(dim=0)
    
    def evaluate_dataset(
        self,
        data_dir: str,
        output_dir: str = None,
        save_predictions: bool = True
    ) -> Dict:
        """
        Evaluate on entire dataset.
        
        Args:
            data_dir: Path to dataset
            output_dir: Path to save predictions
            save_predictions: Whether to save predictions as NIfTI
        
        Returns:
            Dictionary of aggregate metrics
        """
        print(f"Evaluating on dataset: {data_dir}")
        
        # Get case IDs
        case_ids = get_case_ids(data_dir)
        print(f"Found {len(case_ids)} cases")
        
        # Output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Track metrics
        metric_tracker = MetricTracker()
        results = []
        
        modalities = self.config['data'].get('modalities', ['flair', 't1', 't1ce', 't2'])
        
        for case_id in tqdm(case_ids, desc="Evaluating"):
            case_path = Path(data_dir) / case_id
            
            # Load images
            images = []
            for mod in modalities:
                img_path = case_path / f"{case_id}_{mod}.nii.gz"
                img = nib.load(str(img_path))
                images.append(img.get_fdata().astype(np.float32))
            
            images = np.stack(images, axis=0)
            
            # Load label if exists
            seg_path = case_path / f"{case_id}_seg.nii.gz"
            if seg_path.exists():
                label_nii = nib.load(str(seg_path))
                label = label_nii.get_fdata().astype(np.int64)
            else:
                label = None
            
            # Evaluate
            result = self.evaluate_volume(images, label)
            
            # Save prediction
            if save_predictions and output_dir:
                pred_path = output_path / f"{case_id}_pred.nii.gz"
                pred_nii = nib.Nifti1Image(result['prediction'].astype(np.int16), 
                                           label_nii.affine if label is not None else np.eye(4))
                nib.save(pred_nii, str(pred_path))
            
            # Track metrics
            if label is not None:
                metric_tracker.update(result['metrics'])
                results.append({
                    'case_id': case_id,
                    **result['metrics']
                })
        
        # Get aggregate metrics
        avg_metrics = metric_tracker.get_average()
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Number of cases: {len(case_ids)}")
        print(f"\nDice Scores:")
        print(f"  Whole Tumor (WT): {avg_metrics.get('dice_wt', 0):.4f}")
        print(f"  Tumor Core (TC):  {avg_metrics.get('dice_tc', 0):.4f}")
        print(f"  Enhancing (ET):   {avg_metrics.get('dice_et', 0):.4f}")
        print(f"  Mean:             {avg_metrics.get('dice_mean', 0):.4f}")
        print(f"\nHausdorff Distance 95%:")
        print(f"  WT: {avg_metrics.get('hd95_wt', 0):.2f} mm")
        print(f"  TC: {avg_metrics.get('hd95_tc', 0):.2f} mm")
        print(f"  ET: {avg_metrics.get('hd95_et', 0):.2f} mm")
        print("=" * 60)
        
        # Save results
        if output_dir:
            results_path = output_path / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump({
                    'aggregate': avg_metrics,
                    'per_case': results
                }, f, indent=2)
            print(f"\nResults saved to {results_path}")
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MRAF-Net Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Path to save predictions')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions as NIfTI files')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Get data directory from config if not specified
    if args.data_dir is None:
        args.data_dir = evaluator.config['data'].get('data_dir', 
            'data/brats2020/MICCAI_BraTS2020_TrainingData')
    
    # Run evaluation
    evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions
    )


if __name__ == '__main__':
    main()
