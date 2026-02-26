"""
MRAF-Net: Multi-Resolution Aligned and Robust Fusion Network
Main Model Architecture for Brain Tumor Segmentation

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Optional, Dict

from .encoder import MultiScaleEncoder, ModalityEncoder
from .decoder import AttentionDecoder
from .fusion import CrossModalityFusion
from .swin_transformer import SwinBottleneck3D


class MRAFNet(nn.Module):
    """
    Multi-Resolution Aligned and Robust Fusion Network for Brain Tumor Segmentation.
    
    Architecture Overview:
        1. Modality-specific encoders extract features from each MRI modality
        2. Cross-modality fusion combines complementary information
        3. Multi-scale encoder captures hierarchical features
        4. Attention-enhanced decoder produces segmentation mask
        5. Deep supervision aids training (optional)
    
    Args:
        in_channels: Number of input modalities (default: 4 for FLAIR, T1, T1ce, T2)
        num_classes: Number of output classes (default: 4 for BraTS)
        base_features: Base number of features (default: 32)
        deep_supervision: Whether to use deep supervision (default: True)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_features: int = 32,
        deep_supervision: bool = True,
        dropout: float = 0.0,
        use_swin_bottleneck: bool = True,
        swin_depth: int = 2,
        swin_heads: int = 8,
        swin_window_size: Tuple[int, int, int] = (2, 2, 2)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_features = base_features
        self.deep_supervision = deep_supervision
        self.use_checkpointing = False
        
        # Modality-specific encoders (one per MRI modality)
        self.modality_encoders = nn.ModuleList([
            ModalityEncoder(in_channels=1, base_features=base_features)
            for _ in range(in_channels)
        ])
        
        # Cross-modality fusion
        fused_channels = base_features * in_channels  # 32 * 4 = 128
        self.cross_modal_fusion = CrossModalityFusion(
            channels=fused_channels,
            reduction=8
        )
        
        # Multi-scale encoder
        encoder_features = [
            base_features * 2,   # 64
            base_features * 4,   # 128
            base_features * 8,   # 256
            base_features * 10,  # 320
            base_features * 10   # 320
        ]
        self.encoder = MultiScaleEncoder(
            in_channels=fused_channels,
            features=encoder_features,
            use_aspp=True
        )
        
        # Swin Transformer bottleneck for global context (NOVEL)
        self.use_swin_bottleneck = use_swin_bottleneck
        if use_swin_bottleneck:
            self.swin_bottleneck = SwinBottleneck3D(
                dim=encoder_features[-1],
                depth=swin_depth,
                num_heads=swin_heads,
                window_size=swin_window_size,
                mlp_ratio=4.0
            )
        
        # Attention-enhanced decoder
        self.decoder = AttentionDecoder(
            encoder_features=encoder_features,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_checkpointing = False

    def get_parameter_count(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of MRAF-Net.
        
        Args:
            x: Input tensor of shape (B, 4, D, H, W) containing 4 MRI modalities
               Channel order: [FLAIR, T1, T1ce, T2]
        
        Returns:
            If deep_supervision:
                (main_output, [ds_outputs]): Main segmentation + deep supervision outputs
            Else:
                main_output: Segmentation output of shape (B, num_classes, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Extract modality-specific features
        modal_features = []
        for i in range(self.in_channels):
            modality_input = x[:, i:i+1, :, :, :]  # (B, 1, D, H, W)
            
            if self.use_checkpointing and self.training:
                feat = checkpoint(
                    self.modality_encoders[i],
                    modality_input,
                    use_reentrant=False
                )
            else:
                feat = self.modality_encoders[i](modality_input)
            
            modal_features.append(feat)
        
        # Cross-modality fusion
        if self.use_checkpointing and self.training:
            fused = checkpoint(
                self.cross_modal_fusion,
                modal_features,
                use_reentrant=False
            )
        else:
            fused = self.cross_modal_fusion(modal_features)
        
        # Multi-scale encoding
        if self.use_checkpointing and self.training:
            encoder_features, bottleneck = checkpoint(
                self.encoder,
                fused,
                use_reentrant=False
            )
        else:
            encoder_features, bottleneck = self.encoder(fused)
        
        # Apply Swin Transformer at bottleneck for global context
        if self.use_swin_bottleneck:
            if self.use_checkpointing and self.training:
                bottleneck = checkpoint(
                    self.swin_bottleneck,
                    bottleneck,
                    use_reentrant=False
                )
            else:
                bottleneck = self.swin_bottleneck(bottleneck)
        
        # Decoding with attention
        if self.deep_supervision and self.training:
            main_output, ds_outputs = self.decoder(encoder_features, bottleneck)
            return main_output, ds_outputs
        else:
            main_output = self.decoder(encoder_features, bottleneck)
            if isinstance(main_output, tuple):
                return main_output[0], None
            return main_output, None


def create_model(config: Dict) -> MRAFNet:
    """
    Create MRAF-Net model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Initialized MRAFNet model
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model = MRAFNet(
        in_channels=data_config.get('in_channels', 4),
        num_classes=data_config.get('num_classes', 4),
        base_features=model_config.get('base_features', 32),
        deep_supervision=model_config.get('deep_supervision', True),
        dropout=model_config.get('dropout', 0.0),
        use_swin_bottleneck=model_config.get('use_swin_bottleneck', False),
    )
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing MRAF-Net...")
    
    # Create model
    model = MRAFNet(
        in_channels=4,
        num_classes=4,
        base_features=32,
        deep_supervision=True
    )
    
    # Test input (laptop mode: 64x64x64 patches)
    x = torch.randn(1, 4, 64, 64, 64)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, ds_outputs = model(x)
    
    print(f"Output shape: {output.shape}")
    if ds_outputs:
        print(f"Deep supervision outputs: {len(ds_outputs)}")
        for i, ds in enumerate(ds_outputs):
            print(f"  DS {i}: {ds.shape}")
    
    print("\nModel test passed!")
