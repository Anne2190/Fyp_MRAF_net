"""
MRAF-Net Data Preparation Script
Verify and prepare BraTS dataset for training

Author: Anne Nidhusha Nithiyalan (w1985740)

Usage:
    python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData
    python scripts/prepare_data.py --data_dir data/brats2020/MICCAI_BraTS2020_TrainingData --verify_only
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def verify_case(case_path: Path, modalities: List[str]) -> Dict:
    """
    Verify a single case has all required files.
    Supports both .nii and .nii.gz extensions.
    
    Args:
        case_path: Path to case directory
        modalities: List of required modality names
    
    Returns:
        Dictionary with verification results
    """
    case_id = case_path.name
    result = {
        'case_id': case_id,
        'valid': True,
        'issues': [],
        'shape': None,
        'has_label': False,
        'file_extension': None
    }
    
    # Check for modality files (try both .nii and .nii.gz)
    for mod in modalities:
        mod_file = find_file(case_path, case_id, mod)
        if mod_file is None:
            result['valid'] = False
            result['issues'].append(f"Missing {mod} file")
        elif result['file_extension'] is None:
            # Record the extension being used
            result['file_extension'] = mod_file.suffix if mod_file.suffix != '.gz' else ''.join(mod_file.suffixes)
    
    # Check for segmentation (try both .nii and .nii.gz)
    seg_file = find_file(case_path, case_id, 'seg')
    if seg_file is not None:
        result['has_label'] = True
    else:
        result['issues'].append("Missing segmentation (may be validation data)")
    
    # Verify shapes match
    if result['valid']:
        shapes = []
        for mod in modalities:
            mod_file = find_file(case_path, case_id, mod)
            if mod_file:
                try:
                    img = nib.load(str(mod_file))
                    shapes.append(img.shape)
                except Exception as e:
                    result['valid'] = False
                    result['issues'].append(f"Error loading {mod}: {e}")
        
        if len(set(shapes)) > 1:
            result['valid'] = False
            result['issues'].append(f"Shape mismatch: {shapes}")
        elif shapes:
            result['shape'] = shapes[0]
    
    return result


def analyze_dataset(data_dir: str, modalities: List[str] = None) -> Dict:
    """
    Analyze the entire dataset.
    
    Args:
        data_dir: Path to dataset directory
        modalities: List of modality names
    
    Returns:
        Analysis results
    """
    if modalities is None:
        modalities = ['flair', 't1', 't1ce', 't2']
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Get all case directories
    case_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('BraTS')])
    
    print(f"Found {len(case_dirs)} case directories")
    
    # Check file naming in first case to detect extension
    if case_dirs:
        first_case = case_dirs[0]
        first_case_id = first_case.name
        
        # Check what extension is being used
        sample_file = find_file(first_case, first_case_id, 'flair')
        if sample_file:
            ext = ''.join(sample_file.suffixes)
            print(f"Detected file extension: {ext}")
        else:
            print("Warning: Could not detect file extension from first case")
    
    # Verify each case
    results = {
        'total_cases': len(case_dirs),
        'valid_cases': 0,
        'cases_with_labels': 0,
        'invalid_cases': [],
        'shapes': [],
        'label_distribution': {},
        'cases': []
    }
    
    for case_path in tqdm(case_dirs, desc="Verifying cases"):
        case_result = verify_case(case_path, modalities)
        results['cases'].append(case_result)
        
        if case_result['valid']:
            results['valid_cases'] += 1
            if case_result['shape']:
                results['shapes'].append(case_result['shape'])
        else:
            results['invalid_cases'].append(case_result)
        
        if case_result['has_label']:
            results['cases_with_labels'] += 1
    
    # Analyze label distribution
    if results['cases_with_labels'] > 0:
        print("\nAnalyzing label distribution...")
        label_counts = {0: 0, 1: 0, 2: 0, 4: 0}
        
        valid_cases_with_labels = [c for c in results['cases'] if c['valid'] and c['has_label']]
        sample_size = min(50, len(valid_cases_with_labels))
        
        for case in tqdm(valid_cases_with_labels[:sample_size], desc="Sampling labels"):
            case_path = data_path / case['case_id']
            seg_file = find_file(case_path, case['case_id'], 'seg')
            
            if seg_file:
                try:
                    seg = nib.load(str(seg_file)).get_fdata()
                    unique, counts = np.unique(seg.astype(int), return_counts=True)
                    for u, c in zip(unique, counts):
                        if u in label_counts:
                            label_counts[u] += c
                except Exception as e:
                    print(f"Warning: Error loading {seg_file}: {e}")
        
        results['label_distribution'] = label_counts
    
    return results


def print_analysis_report(analysis: Dict):
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\nTotal cases: {analysis['total_cases']}")
    print(f"Valid cases: {analysis['valid_cases']}")
    print(f"Cases with labels: {analysis['cases_with_labels']}")
    
    if analysis['invalid_cases']:
        print(f"\nInvalid cases ({len(analysis['invalid_cases'])}):")
        for case in analysis['invalid_cases'][:5]:
            print(f"  - {case['case_id']}: {', '.join(case['issues'])}")
        if len(analysis['invalid_cases']) > 5:
            print(f"  ... and {len(analysis['invalid_cases']) - 5} more")
    
    if analysis['shapes']:
        unique_shapes = list(set(analysis['shapes']))
        print(f"\nImage shapes: {unique_shapes}")
    
    if analysis['label_distribution']:
        print(f"\nLabel distribution (sampled):")
        total = sum(analysis['label_distribution'].values())
        for label, count in sorted(analysis['label_distribution'].items()):
            pct = 100 * count / total if total > 0 else 0
            label_name = {0: 'Background', 1: 'NCR/NET', 2: 'Edema', 4: 'Enhancing'}
            print(f"  Label {label} ({label_name.get(label, 'Unknown')}): {pct:.2f}%")
    
    print("\n" + "=" * 60)
    
    if analysis['valid_cases'] == analysis['total_cases']:
        print("✓ Dataset is ready for training!")
    elif analysis['valid_cases'] > 0:
        print(f"✓ Found {analysis['valid_cases']} valid cases for training!")
    else:
        print("⚠ No valid cases found. Check your data directory structure.")
    
    print("=" * 60)


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    output_file: str = None
) -> Tuple[List[str], List[str]]:
    """
    Create train/validation splits.
    
    Args:
        data_dir: Path to dataset
        train_ratio: Fraction for training
        seed: Random seed
        output_file: Path to save split file
    
    Returns:
        (train_ids, val_ids)
    """
    import random
    
    data_path = Path(data_dir)
    case_ids = sorted([d.name for d in data_path.iterdir() 
                      if d.is_dir() and d.name.startswith('BraTS')])
    
    # Check which cases have labels (check both .nii and .nii.gz)
    cases_with_labels = []
    modalities = ['flair', 't1', 't1ce', 't2']
    
    for case_id in case_ids:
        case_path = data_path / case_id
        
        # Check all modalities exist
        all_exist = all(find_file(case_path, case_id, mod) is not None for mod in modalities)
        
        # Check segmentation exists
        has_seg = find_file(case_path, case_id, 'seg') is not None
        
        if all_exist and has_seg:
            cases_with_labels.append(case_id)
    
    print(f"Cases with all files: {len(cases_with_labels)}")
    
    # Shuffle and split
    random.seed(seed)
    shuffled = cases_with_labels.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:]
    
    print(f"Training cases: {len(train_ids)}")
    print(f"Validation cases: {len(val_ids)}")
    
    # Save splits
    if output_file:
        splits = {
            'train': train_ids,
            'val': val_ids,
            'seed': seed,
            'train_ratio': train_ratio
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Saved splits to {output_file}")
    
    return train_ids, val_ids


def compute_dataset_statistics(data_dir: str, modalities: List[str] = None, num_samples: int = 50):
    """
    Compute intensity statistics for normalization.
    
    Args:
        data_dir: Path to dataset
        modalities: List of modality names
        num_samples: Number of cases to sample
    """
    if modalities is None:
        modalities = ['flair', 't1', 't1ce', 't2']
    
    data_path = Path(data_dir)
    case_ids = sorted([d.name for d in data_path.iterdir() 
                      if d.is_dir() and d.name.startswith('BraTS')])
    
    # Filter to valid cases only
    valid_cases = []
    for case_id in case_ids:
        case_path = data_path / case_id
        if all(find_file(case_path, case_id, mod) is not None for mod in modalities):
            valid_cases.append(case_id)
    
    # Sample cases
    import random
    random.seed(42)
    sample_ids = random.sample(valid_cases, min(num_samples, len(valid_cases)))
    
    print(f"Computing statistics from {len(sample_ids)} cases...")
    
    stats = {mod: {'means': [], 'stds': []} for mod in modalities}
    
    for case_id in tqdm(sample_ids, desc="Computing stats"):
        case_path = data_path / case_id
        
        for mod in modalities:
            mod_file = find_file(case_path, case_id, mod)
            if mod_file:
                img = nib.load(str(mod_file)).get_fdata()
                mask = img > 0
                
                if mask.sum() > 0:
                    mean_val = img[mask].mean()
                    std_val = img[mask].std()
                    stats[mod]['means'].append(mean_val)
                    stats[mod]['stds'].append(std_val)
    
    print("\nDataset Statistics (per modality):")
    print("-" * 40)
    for mod in modalities:
        if stats[mod]['means']:
            mean_of_means = np.mean(stats[mod]['means'])
            mean_of_stds = np.mean(stats[mod]['stds'])
            print(f"{mod.upper():6s}: mean={mean_of_means:.2f}, std={mean_of_stds:.2f}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description='Prepare BraTS Dataset for MRAF-Net')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BraTS dataset directory')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify dataset without creating splits')
    parser.add_argument('--create_splits', action='store_true',
                        help='Create train/val splits')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')
    parser.add_argument('--compute_stats', action='store_true',
                        help='Compute intensity statistics')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for split files')
    
    args = parser.parse_args()
    
    # Verify dataset
    print("Analyzing dataset...")
    analysis = analyze_dataset(args.data_dir)
    print_analysis_report(analysis)
    
    # Save analysis
    analysis_path = Path(args.output_dir) / 'dataset_analysis.json'
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert shapes to lists for JSON serialization
    serializable_analysis = analysis.copy()
    serializable_analysis['shapes'] = [list(s) for s in analysis['shapes']]
    
    # Convert numpy int64 to Python int for JSON serialization
    if 'label_distribution' in serializable_analysis:
        serializable_analysis['label_distribution'] = {
            int(k): int(v) for k, v in serializable_analysis['label_distribution'].items()
        }
    
    with open(analysis_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")
    
    if args.verify_only:
        return
    
    # Create splits
    if args.create_splits:
        splits_path = Path(args.output_dir) / 'data_splits.json'
        create_data_splits(
            args.data_dir,
            train_ratio=args.train_ratio,
            output_file=str(splits_path)
        )
    
    # Compute statistics
    if args.compute_stats:
        compute_dataset_statistics(args.data_dir)


if __name__ == '__main__':
    main()
