"""
MRAF-Net BraTS Dataset Loader
PyTorch Dataset for Brain Tumor Segmentation Challenge Data

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_train_transforms, get_val_transforms
from .preprocessing import normalize_intensity


def find_file(case_path: Path, case_id: str, suffix: str) -> Optional[Path]:
    """
    Find a file with given suffix, checking both .nii.gz and .nii extensions.
    
    Args:
        case_path: Path to case directory
        case_id: Case ID (e.g., 'BraTS20_Training_001')
        suffix: File suffix (e.g., 'flair', 't1', 'seg')
    
    Returns:
        Path to file if found, None otherwise
    """
    # Try different extensions
    extensions = ['.nii.gz', '.nii']
    
    for ext in extensions:
        file_path = case_path / f"{case_id}_{suffix}{ext}"
        if file_path.exists():
            return file_path
    
    return None


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS (Brain Tumor Segmentation) Challenge data.
    
    Loads multimodal MRI scans (FLAIR, T1, T1ce, T2) and segmentation masks.
    Supports both .nii and .nii.gz file formats.
    
    BraTS Label Convention:
        0: Background
        1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
        2: Peritumoral Edema (ED)
        4: GD-Enhancing Tumor (ET)
    
    Evaluation Regions:
        Whole Tumor (WT): Labels 1, 2, 4
        Tumor Core (TC): Labels 1, 4
        Enhancing Tumor (ET): Label 4
    
    Args:
        data_dir: Path to BraTS dataset directory
        case_ids: List of case IDs to include
        patch_size: Size of random patches to extract (D, H, W)
        is_training: Whether this is for training (enables augmentation)
        samples_per_volume: Number of patches to extract per volume during training
        modalities: List of modality names in order
    """
    
    def __init__(
        self,
        data_dir: str,
        case_ids: List[str],
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        is_training: bool = True,
        samples_per_volume: int = 4,
        modalities: List[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.is_training = is_training
        self.samples_per_volume = samples_per_volume if is_training else 1
        
        # Default modality order: FLAIR, T1, T1ce, T2
        self.modalities = modalities or ['flair', 't1', 't1ce', 't2']
        
        # Get transforms
        if is_training:
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()
        
        # Verify all cases exist
        self._verify_cases()
    
    def _verify_cases(self):
        """Verify that all case directories exist and contain required files."""
        valid_cases = []
        
        for case_id in self.case_ids:
            case_path = self.data_dir / case_id
            
            if not case_path.exists():
                print(f"Warning: Case directory not found: {case_path}")
                continue
            
            # Check for all required modality files (try both .nii and .nii.gz)
            all_modalities_exist = True
            for mod in self.modalities:
                if find_file(case_path, case_id, mod) is None:
                    all_modalities_exist = False
                    break
            
            # Check for segmentation file
            seg_file = find_file(case_path, case_id, 'seg')
            
            if all_modalities_exist and seg_file is not None:
                valid_cases.append(case_id)
            else:
                print(f"Warning: Missing files for case: {case_id}")
        
        self.case_ids = valid_cases
        print(f"Found {len(self.case_ids)} valid cases")
    
    def __len__(self) -> int:
        return len(self.case_ids) * self.samples_per_volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'image' (4, D, H, W) and 'label' (D, H, W)
        """
        # Get case index
        case_idx = idx // self.samples_per_volume
        case_id = self.case_ids[case_idx]
        case_path = self.data_dir / case_id
        
        # Load all modalities
        images = []
        for mod in self.modalities:
            img_path = find_file(case_path, case_id, mod)
            img = nib.load(str(img_path)).get_fdata().astype(np.float32)
            images.append(img)
        
        # Stack modalities: (4, H, W, D)
        images = np.stack(images, axis=0)
        
        # Load segmentation
        seg_path = find_file(case_path, case_id, 'seg')
        label = nib.load(str(seg_path)).get_fdata().astype(np.int64)
        
        # Convert label 4 to 3 for continuous labels (0, 1, 2, 3)
        label[label == 4] = 3
        
        # Normalize each modality
        images = normalize_intensity(images)
        
        # Transpose to (C, D, H, W) format - BraTS is (H, W, D)
        images = np.transpose(images, (0, 3, 1, 2))  # (4, D, H, W)
        label = np.transpose(label, (2, 0, 1))        # (D, H, W)
        
        # Extract patch if training
        # if self.is_training:
        images, label = self._extract_patch(images, label)
        
        # Apply transforms
        sample = {'image': images, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        # Convert to tensors
        sample['image'] = torch.from_numpy(sample['image'].copy()).float()
        sample['label'] = torch.from_numpy(sample['label'].copy()).long()
        
        return sample
    
    def _extract_patch(
        self,
        images: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a random patch from the volume.
        
        Uses tumor-centered sampling 50% of the time to handle class imbalance.
        """
        C, D, H, W = images.shape
        pD, pH, pW = self.patch_size
        
        # Ensure patch fits in volume
        pD = min(pD, D)
        pH = min(pH, H)
        pW = min(pW, W)
        
        # 50% chance to center on tumor region
        if random.random() < 0.5 and np.any(label > 0):
            # Get tumor voxel coordinates
            tumor_coords = np.where(label > 0)
            
            if len(tumor_coords[0]) > 0:
                # Random tumor voxel
                rand_idx = random.randint(0, len(tumor_coords[0]) - 1)
                center_d = tumor_coords[0][rand_idx]
                center_h = tumor_coords[1][rand_idx]
                center_w = tumor_coords[2][rand_idx]
            else:
                # Fallback to random
                center_d = random.randint(pD // 2, D - pD // 2)
                center_h = random.randint(pH // 2, H - pH // 2)
                center_w = random.randint(pW // 2, W - pW // 2)
        else:
            # Random center
            center_d = random.randint(pD // 2, max(pD // 2, D - pD // 2))
            center_h = random.randint(pH // 2, max(pH // 2, H - pH // 2))
            center_w = random.randint(pW // 2, max(pW // 2, W - pW // 2))
        
        # Calculate patch bounds
        d_start = max(0, center_d - pD // 2)
        d_end = min(D, d_start + pD)
        d_start = d_end - pD
        
        h_start = max(0, center_h - pH // 2)
        h_end = min(H, h_start + pH)
        h_start = h_end - pH
        
        w_start = max(0, center_w - pW // 2)
        w_end = min(W, w_start + pW)
        w_start = w_end - pW
        
        # Extract patch
        images_patch = images[:, d_start:d_end, h_start:h_end, w_start:w_end]
        label_patch = label[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return images_patch, label_patch


def get_case_ids(data_dir: str) -> List[str]:
    """
    Get all case IDs from the dataset directory.
    
    Args:
        data_dir: Path to BraTS dataset
    
    Returns:
        List of case IDs (directory names)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Get all subdirectories that match BraTS naming pattern
    case_ids = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('BraTS'):
            case_ids.append(item.name)
    
    case_ids.sort()
    return case_ids


def split_dataset(
    case_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split case IDs into training and validation sets.
    
    Args:
        case_ids: List of all case IDs
        train_ratio: Fraction for training
        seed: Random seed for reproducibility
    
    Returns:
        (train_ids, val_ids)
    """
    random.seed(seed)
    shuffled = case_ids.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:]
    
    return train_ids, val_ids


def get_data_loaders(
    data_dir: str,
    batch_size: int = 1,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    train_ratio: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    samples_per_volume: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Path to BraTS dataset
        batch_size: Batch size
        patch_size: Size of patches to extract
        train_ratio: Fraction of data for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        samples_per_volume: Patches per volume in training
        seed: Random seed
    
    Returns:
        (train_loader, val_loader)
    """
    # Get and split case IDs
    case_ids = get_case_ids(data_dir)
    train_ids, val_ids = split_dataset(case_ids, train_ratio, seed)
    
    print(f"Training cases: {len(train_ids)}")
    print(f"Validation cases: {len(val_ids)}")
    
    # Create datasets
    train_dataset = BraTSDataset(
        data_dir=data_dir,
        case_ids=train_ids,
        patch_size=patch_size,
        is_training=True,
        samples_per_volume=samples_per_volume
    )
    
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        case_ids=val_ids,
        patch_size=patch_size,
        is_training=False,
        samples_per_volume=1
    )
    
    # Check if we have valid samples
    if len(train_dataset.case_ids) == 0:
        raise ValueError("No valid training cases found! Check your data directory structure.")
    
    if len(val_dataset.case_ids) == 0:
        raise ValueError("No valid validation cases found! Check your data directory structure.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full volume for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing BraTS Dataset...")
    
    # This would need actual data to run
    data_dir = "data/brats2020/MICCAI_BraTS2020_TrainingData"
    
    if os.path.exists(data_dir):
        case_ids = get_case_ids(data_dir)
        print(f"Found {len(case_ids)} cases")
        
        if len(case_ids) > 0:
            # Test dataset
            dataset = BraTSDataset(
                data_dir=data_dir,
                case_ids=case_ids[:5],
                patch_size=(96, 96, 96),
                is_training=True
            )
            
            print(f"Dataset length: {len(dataset)}")
            
            # Get a sample
            sample = dataset[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Label shape: {sample['label'].shape}")
            print(f"Label unique values: {torch.unique(sample['label'])}")
    else:
        print(f"Dataset directory not found: {data_dir}")
        print("Please download the BraTS dataset first.")
