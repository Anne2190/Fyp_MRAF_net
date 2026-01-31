"""
MRAF-Net Loss Functions
Dice Loss, Cross-Entropy, Focal Loss and combinations

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        include_background: Whether to include background in loss computation
        smooth: Smoothing factor to prevent division by zero
        squared_pred: Use squared prediction in denominator
    """
    
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-5,
        squared_pred: bool = True
    ):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.squared_pred = squared_pred
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, D, H, W) with class indices
        
        Returns:
            Dice loss value
        """
        num_classes = pred.shape[1]
        
        # Softmax predictions
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Optionally exclude background
        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        
        # Flatten spatial dimensions
        pred = pred.flatten(2)  # (B, C, N)
        target_one_hot = target_one_hot.flatten(2)  # (B, C, N)
        
        # Compute intersection and cardinalities
        intersection = (pred * target_one_hot).sum(dim=2)
        
        if self.squared_pred:
            pred_sum = (pred * pred).sum(dim=2)
            target_sum = (target_one_hot * target_one_hot).sum(dim=2)
        else:
            pred_sum = pred.sum(dim=2)
            target_sum = target_one_hot.sum(dim=2)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Average across classes and batch
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross-Entropy Loss.
    
    This is the standard loss for brain tumor segmentation (used by nnU-Net).
    
    Args:
        include_background: Whether to include background in Dice
        dice_weight: Weight for Dice loss
        ce_weight: Weight for Cross-Entropy loss
        ce_weights: Optional class weights for CE loss
    """
    
    def __init__(
        self,
        include_background: bool = False,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ce_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.dice_loss = DiceLoss(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights)
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, D, H, W)
        
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL = -α(1-p)^γ * log(p)
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class balancing weight
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, D, H, W)
            target: Ground truth of shape (B, D, H, W)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Get predicted probabilities
        pred_soft = F.softmax(pred, dim=1)
        
        # Gather probabilities for correct class
        target_expanded = target.unsqueeze(1)  # (B, 1, D, H, W)
        p_t = pred_soft.gather(1, target_expanded).squeeze(1)  # (B, D, H, W)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal Loss.
    
    Good for highly imbalanced datasets like brain tumor segmentation.
    
    Args:
        include_background: Whether to include background in Dice
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss
        focal_gamma: Gamma parameter for Focal loss
    """
    
    def __init__(
        self,
        include_background: bool = False,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.dice_loss = DiceLoss(include_background=include_background)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for multi-scale outputs.
    
    Applies the base loss at multiple scales with decreasing weights.
    
    Args:
        base_loss: The base loss function to use
        weights: Weights for each supervision level
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: List[float] = [1.0, 0.5, 0.25, 0.125]
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(
        self,
        outputs: tuple,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: Tuple of (main_output, [ds_output1, ds_output2, ...])
            target: Ground truth of shape (B, D, H, W)
        
        Returns:
            Total loss value
        """
        if isinstance(outputs, tuple) and len(outputs) == 2:
            main_output, ds_outputs = outputs
        else:
            # No deep supervision
            return self.base_loss(outputs, target)
        
        # Main output loss
        total_loss = self.weights[0] * self.base_loss(main_output, target)
        
        # Deep supervision losses
        for i, (ds_output, weight) in enumerate(zip(ds_outputs, self.weights[1:])):
            # Resize target if needed
            if ds_output.shape[2:] != target.shape[1:]:
                target_ds = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=ds_output.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            else:
                target_ds = target
            
            total_loss += weight * self.base_loss(ds_output, target_ds)
        
        return total_loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Loss function module
    """
    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'dice_ce')
    
    if loss_name == 'dice':
        base_loss = DiceLoss()
    elif loss_name == 'dice_ce':
        base_loss = DiceCELoss(
            dice_weight=loss_config.get('dice_weight', 1.0),
            ce_weight=loss_config.get('ce_weight', 1.0)
        )
    elif loss_name == 'focal':
        base_loss = FocalLoss(gamma=loss_config.get('focal_gamma', 2.0))
    elif loss_name == 'dice_focal':
        base_loss = DiceFocalLoss(
            dice_weight=loss_config.get('dice_weight', 1.0),
            focal_weight=loss_config.get('focal_weight', 1.0),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    # Wrap with deep supervision if enabled
    if config.get('model', {}).get('deep_supervision', False):
        weights = loss_config.get('ds_weights', [1.0, 0.5, 0.25, 0.125])
        return DeepSupervisionLoss(base_loss, weights)
    
    return base_loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Create dummy predictions and targets
    B, C, D, H, W = 2, 4, 32, 32, 32
    pred = torch.randn(B, C, D, H, W)
    target = torch.randint(0, C, (B, D, H, W))
    
    # Test DiceLoss
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"DiceLoss: {loss.item():.4f}")
    
    # Test DiceCELoss
    dice_ce_loss = DiceCELoss()
    loss = dice_ce_loss(pred, target)
    print(f"DiceCELoss: {loss.item():.4f}")
    
    # Test FocalLoss
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    print(f"FocalLoss: {loss.item():.4f}")
    
    # Test DiceFocalLoss
    dice_focal_loss = DiceFocalLoss()
    loss = dice_focal_loss(pred, target)
    print(f"DiceFocalLoss: {loss.item():.4f}")
    
    # Test DeepSupervisionLoss
    ds_outputs = [
        torch.randn(B, C, D//2, H//2, W//2),
        torch.randn(B, C, D//4, H//4, W//4),
        torch.randn(B, C, D//8, H//8, W//8)
    ]
    ds_loss = DeepSupervisionLoss(DiceCELoss())
    loss = ds_loss((pred, ds_outputs), target)
    print(f"DeepSupervisionLoss: {loss.item():.4f}")
    
    print("All tests passed!")
