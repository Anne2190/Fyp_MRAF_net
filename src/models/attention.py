"""
MRAF-Net Attention Modules
Memory-Efficient Attention Mechanisms for 3D Medical Image Segmentation

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        return self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention using channel pooling."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(spatial))


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module for 3D."""
    
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class AttentionGate3D(nn.Module):
    """Attention Gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels, affine=True)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels, affine=True)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1, 1)
        return x * y


if __name__ == "__main__":
    print("Testing Attention Modules...")
    
    x = torch.randn(1, 64, 32, 32, 32)
    
    # Test ChannelAttention
    ca = ChannelAttention(64)
    out = ca(x)
    print(f"ChannelAttention: {x.shape} -> {out.shape}")
    
    # Test SpatialAttention
    sa = SpatialAttention()
    out = sa(x)
    print(f"SpatialAttention: {x.shape} -> {out.shape}")
    
    # Test CBAM3D
    cbam = CBAM3D(64)
    out = cbam(x)
    print(f"CBAM3D: {x.shape} -> {out.shape}")
    
    # Test AttentionGate3D
    g = torch.randn(1, 128, 16, 16, 16)
    skip = torch.randn(1, 64, 32, 32, 32)
    ag = AttentionGate3D(128, 64)
    out = ag(g, skip)
    print(f"AttentionGate3D: g={g.shape}, skip={skip.shape} -> {out.shape}")
    
    print("All attention tests passed!")
