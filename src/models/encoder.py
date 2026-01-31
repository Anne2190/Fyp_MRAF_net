"""
MRAF-Net Encoder Module
Multi-Resolution Feature Extraction with ASPP for Brain Tumor Segmentation

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock3D(nn.Module):
    """Basic 3D Convolution Block: Conv -> Norm -> Activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """3D Residual Block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class ASPP3D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 3D volumes.
    
    IMPORTANT: Global pooling branch does NOT use InstanceNorm because
    after pooling to 1x1x1, InstanceNorm fails (requires >1 spatial elements).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Tuple[int, ...] = (1, 2, 3)
    ):
        super().__init__()
        
        # Branch 1: 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Branch 2-4: Dilated convolutions
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            self.dilated_convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.01, inplace=True)
            ))
        
        # Branch 5: Global average pooling (NO InstanceNorm - would fail with 1x1x1!)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Output projection
        num_branches = 2 + len(dilations)
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        out1 = self.conv1x1(x)
        dilated_outs = [conv(x) for conv in self.dilated_convs]
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='trilinear', align_corners=False)
        
        out = torch.cat([out1] + dilated_outs + [global_feat], dim=1)
        return self.project(out)


class ModalityEncoder(nn.Module):
    """Encoder for a single MRI modality."""
    
    def __init__(self, in_channels: int = 1, base_features: int = 32):
        super().__init__()
        
        self.conv1 = ConvBlock3D(in_channels, base_features)
        self.conv2 = ConvBlock3D(base_features, base_features)
        self.residual = ResidualBlock3D(base_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        return x


class EncoderLevel(nn.Module):
    """Single level of the multi-scale encoder."""
    
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        
        stride = 2 if downsample else 1
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv = ConvBlock3D(out_channels, out_channels)
        self.residual = ResidualBlock3D(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        x = self.residual(x)
        return x


class MultiScaleEncoder(nn.Module):
    """
    Multi-Scale Encoder for hierarchical feature extraction.
    
    For 64x64x64 input with default features [64, 128, 256, 320, 320]:
        Level 0: 64x64x64 -> 64 channels
        Level 1: 32x32x32 -> 128 channels
        Level 2: 16x16x16 -> 256 channels
        Level 3: 8x8x8 -> 320 channels
        Level 4: 4x4x4 -> 320 channels
        ASPP: 4x4x4 -> 320 channels (multi-scale context)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        features: List[int] = None,
        use_aspp: bool = True
    ):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 320, 320]
        
        self.features = features
        self.use_aspp = use_aspp
        
        # Encoder levels
        self.level0 = EncoderLevel(in_channels, features[0], downsample=False)
        self.level1 = EncoderLevel(features[0], features[1], downsample=True)
        self.level2 = EncoderLevel(features[1], features[2], downsample=True)
        self.level3 = EncoderLevel(features[2], features[3], downsample=True)
        self.level4 = EncoderLevel(features[3], features[4], downsample=True)
        
        # ASPP at bottleneck (use small dilations for 4x4x4 feature maps)
        if use_aspp:
            self.aspp = ASPP3D(features[4], features[4], dilations=(1, 2, 3))
        else:
            self.aspp = None
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        e0 = self.level0(x)   # Full res
        e1 = self.level1(e0)  # 1/2
        e2 = self.level2(e1)  # 1/4
        e3 = self.level3(e2)  # 1/8
        e4 = self.level4(e3)  # 1/16
        
        bottleneck = self.aspp(e4) if self.use_aspp else e4
        
        return [e0, e1, e2, e3], bottleneck


if __name__ == "__main__":
    print("Testing Encoder...")
    
    # Test with 64x64x64 input
    x = torch.randn(1, 128, 64, 64, 64)
    encoder = MultiScaleEncoder(in_channels=128, use_aspp=True)
    features, bottleneck = encoder(x)
    
    print(f"Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"Level {i}: {f.shape}")
    print(f"Bottleneck: {bottleneck.shape}")
    print("Encoder test passed!")
