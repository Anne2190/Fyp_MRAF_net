"""
MRAF-Net Helper Utilities
Configuration loading, checkpointing, and other utilities

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import os
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Current metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load to
    
    Returns:
        Checkpoint metadata (epoch, metrics, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', None)
    }


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get the compute device.
    
    Args:
        gpu_id: GPU device ID to use
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like Dice
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> Path:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for experiment
    
    Returns:
        Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'predictions').mkdir(exist_ok=True)
    
    return exp_dir


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 4, 96, 96, 96)):
    """
    Print model summary including layer shapes and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, D, H, W)
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Input Size: {input_size}")
    print("=" * 60)
    
    # Try to get output shape
    try:
        device = next(model.parameters()).device
        x = torch.randn(input_size).to(device)
        model.eval()
        with torch.no_grad():
            output = model(x)
        if isinstance(output, tuple):
            print(f"Output Shape: {output[0].shape}")
            print(f"Deep Supervision Outputs: {len(output[1])}")
        else:
            print(f"Output Shape: {output.shape}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test helpers
    print("Testing Helper Utilities...")
    
    # Test set_seed
    set_seed(42)
    
    # Test get_device
    device = get_device()
    
    # Test AverageMeter
    meter = AverageMeter('loss')
    for i in range(10):
        meter.update(i * 0.1)
    print(f"AverageMeter: {meter}")
    
    # Test EarlyStopping
    es = EarlyStopping(patience=3, mode='min')
    scores = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]
    for score in scores:
        should_stop = es(score)
        print(f"Score: {score:.2f}, Should Stop: {should_stop}")
    
    print("All tests passed!")
