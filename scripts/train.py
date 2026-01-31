"""
MRAF-Net Training Script
Main entry point for training the brain tumor segmentation model

Author: Anne Nidhusha Nithiyalan (w1985740)

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --config config/config.yaml --mode laptop
"""

import os
import sys
import gc
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mraf_net import MRAFNet, create_model
from src.data.dataset import get_data_loaders, get_case_ids
from src.losses.losses import get_loss_function
from src.utils.helpers import (
    load_config, save_config, save_checkpoint, load_checkpoint,
    set_seed, get_device, count_parameters, get_lr,
    AverageMeter, EarlyStopping, create_experiment_dir
)
from src.utils.metrics import compute_metrics, MetricTracker


class Trainer:
    """MRAF-Net Trainer class."""
    
    def __init__(self, config: dict, mode: str = 'standard'):
        self.config = config
        self.mode = mode
        
        if mode == 'laptop':
            self._adjust_for_laptop()
        
        set_seed(config['data'].get('seed', 42))
        self.device = get_device()
        
        exp_name = f"mraf_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = create_experiment_dir('experiments', exp_name)
        save_config(config, self.exp_dir / 'config.yaml')
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        self._setup_logging()
        
        print(f"\nExperiment directory: {self.exp_dir}")
        print(f"Training mode: {mode}")
        print(f"Model parameters: {count_parameters(self.model):,}")
    
    def _adjust_for_laptop(self):
        print("\n*** LAPTOP MODE: Optimizing for limited GPU memory ***\n")
        
        self.config['training']['batch_size'] = 1
        self.config['training']['patch_size'] = [64, 64, 64]
        self.config['training']['use_amp'] = True
        self.config['training']['gradient_checkpointing'] = True
        self.config['training']['num_workers'] = 2
        self.config['training']['samples_per_volume'] = 2
        
        print("Laptop mode settings:")
        print(f"  - Batch size: {self.config['training']['batch_size']}")
        print(f"  - Patch size: {self.config['training']['patch_size']}")
        print(f"  - Mixed precision (AMP): Enabled")
        print(f"  - Gradient checkpointing: Enabled")
        print()
    
    def _setup_model(self):
        print("Setting up model...")
        
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        if self.config['training'].get('gradient_checkpointing', False):
            self.model.enable_gradient_checkpointing()
            print("  - Gradient checkpointing enabled")
    
    def _setup_data(self):
        print("Setting up data loaders...")
        
        data_config = self.config['data']
        train_config = self.config['training']
        
        self.train_loader, self.val_loader = get_data_loaders(
            data_dir=data_config['data_dir'],
            batch_size=train_config['batch_size'],
            patch_size=tuple(train_config['patch_size']),
            train_ratio=data_config.get('train_val_split', 0.8),
            num_workers=train_config.get('num_workers', 4),
            pin_memory=train_config.get('pin_memory', True),
            samples_per_volume=train_config.get('samples_per_volume', 4),
            seed=data_config.get('seed', 42)
        )
        
        print(f"  - Training samples: {len(self.train_loader.dataset)}")
        print(f"  - Validation samples: {len(self.val_loader.dataset)}")
    
    def _setup_training(self):
        print("Setting up training components...")
        
        train_config = self.config['training']
        
        self.criterion = get_loss_function(self.config)
        
        optimizer_name = train_config.get('optimizer', 'adamw').lower()
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.99, weight_decay=weight_decay, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        scheduler_name = train_config.get('scheduler', 'cosine').lower()
        self.epochs = train_config.get('epochs', 300)
        
        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-7)
        elif scheduler_name == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)
        else:
            self.scheduler = None
        
        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        patience = train_config.get('early_stopping_patience', 50)
        self.early_stopping = EarlyStopping(patience=patience, mode='max') if patience > 0 else None
        
        self.start_epoch = 0
        self.best_metric = 0.0
        
        resume_path = self.config['checkpoint'].get('resume')
        if resume_path and Path(resume_path).exists():
            checkpoint_info = load_checkpoint(resume_path, self.model, self.optimizer, self.scheduler, self.device)
            self.start_epoch = checkpoint_info['epoch'] + 1
            self.best_metric = checkpoint_info['metrics'].get('dice_mean', 0.0)
            print(f"  - Resumed from epoch {self.start_epoch}")
        
        print(f"  - Optimizer: {optimizer_name}")
        print(f"  - Scheduler: {scheduler_name}")
        print(f"  - Mixed precision: {self.use_amp}")
    
    def _setup_logging(self):
        log_dir = self.exp_dir / 'logs'
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"  - TensorBoard logs: {log_dir}")
    
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        loss_meter = AverageMeter('loss')
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs, ds_outputs = self.model(images)
                    loss = self.criterion((outputs, ds_outputs), labels)
                
                self.scaler.scale(loss).backward()
                
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, ds_outputs = self.model(images)
                loss = self.criterion((outputs, ds_outputs), labels)
                loss.backward()
                
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                
                self.optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'lr': f'{get_lr(self.optimizer):.2e}'})
            
            step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['logging'].get('print_frequency', 10) == 0:
                self.writer.add_scalar('train/loss', loss.item(), step)
                self.writer.add_scalar('train/lr', get_lr(self.optimizer), step)
        
        return {'loss': loss_meter.avg}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model with proper error handling."""
        self.model.eval()
        
        metric_tracker = MetricTracker()
        loss_meter = AverageMeter('val_loss')
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_amp:
                    with autocast('cuda'):
                        outputs, _ = self.model(images)
                        loss = F.cross_entropy(outputs, labels) + self._dice_loss(outputs, labels)
                else:
                    outputs, _ = self.model(images)
                    loss = F.cross_entropy(outputs, labels) + self._dice_loss(outputs, labels)
                
                loss_meter.update(loss.item())
                
                # Get predictions
                pred = torch.argmax(outputs, dim=1)
                
                # Compute metrics for each sample in batch
                for i in range(pred.shape[0]):
                    try:
                        pred_np = pred[i].cpu().numpy()
                        label_np = labels[i].cpu().numpy()
                        
                        # Validate shapes before computing metrics
                        if pred_np.ndim != 3 or label_np.ndim != 3:
                            print(f"Warning: Unexpected shapes - pred: {pred_np.shape}, label: {label_np.shape}")
                            continue
                        
                        # Compute metrics with optional HD computation
                        # Skip HD for speed during early epochs or if memory is tight
                        compute_hd = (epoch >= 10) or (batch_idx == 0)  # Only compute HD after epoch 10 or first batch
                        metrics = compute_metrics(pred_np, label_np, compute_hd=compute_hd)
                        metric_tracker.update(metrics)
                        
                    except Exception as e:
                        print(f"Warning: Error computing metrics for sample {i}: {e}")
                        continue
                
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
                
                # Clear cache periodically to prevent memory buildup
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Warning: Error in validation batch {batch_idx}: {e}")
                continue
        
        avg_metrics = metric_tracker.get_average()
        avg_metrics['val_loss'] = loss_meter.avg
        
        # Ensure required keys exist
        for key in ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']:
            if key not in avg_metrics:
                avg_metrics[key] = 0.0
        
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return avg_metrics
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss for validation."""
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_soft.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice[:, 1:].mean()
    
    def train(self):
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_frequency = self.config['training'].get('val_frequency', 5)
            if (epoch + 1) % val_frequency == 0 or epoch == self.epochs - 1:
                try:
                    val_metrics = self.validate(epoch)
                    
                    print(f"\nEpoch {epoch}/{self.epochs}")
                    print(f"  Train Loss: {train_metrics['loss']:.4f}")
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                    print(f"  Dice WT: {val_metrics.get('dice_wt', 0):.4f}")
                    print(f"  Dice TC: {val_metrics.get('dice_tc', 0):.4f}")
                    print(f"  Dice ET: {val_metrics.get('dice_et', 0):.4f}")
                    print(f"  Dice Mean: {val_metrics.get('dice_mean', 0):.4f}")
                    
                    current_metric = val_metrics.get('dice_mean', 0)
                    is_best = current_metric > self.best_metric
                    
                    if is_best:
                        self.best_metric = current_metric
                        print(f"  *** New best model! ***")
                    
                    # Save checkpoint
                    save_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        save_path=save_path,
                        is_best=is_best
                    )
                    
                    # Early stopping
                    if self.early_stopping is not None:
                        if self.early_stopping(current_metric):
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                            break
                            
                except Exception as e:
                    print(f"Warning: Validation failed at epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    try:
                        self.scheduler.step(val_metrics.get('dice_mean', 0))
                    except:
                        pass
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            print(f"  Epoch time: {epoch_time / 60:.1f} min")
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best Dice Mean: {self.best_metric:.4f}")
        print(f"Best model saved to: {self.exp_dir / 'checkpoints' / 'best_model.pth'}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MRAF-Net for Brain Tumor Segmentation')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'laptop'], help='Training mode')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.resume:
        config['checkpoint']['resume'] = args.resume
    
    trainer = Trainer(config, mode=args.mode)
    trainer.train()


if __name__ == '__main__':
    main()
