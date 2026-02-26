"""
MRAF-Net Cross-Modality Fusion Module
Memory-Efficient Fusion of T1, T1ce, T2, FLAIR MRI Modalities

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention using global pooling (memory efficient)."""
    
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
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * attention


class CoordinateAttention3D(nn.Module):
    """
    3D Coordinate Attention (Novel).
    
    Encodes long-range spatial dependencies along each of the three spatial
    axes (D, H, W) into channel attention maps. This provides richer
    positional information than standard channel attention (SE blocks),
    enabling more precise cross-modality alignment.
    
    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.
    Extended here to 3D for volumetric medical image segmentation.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        mid_channels = max(channels // reduction, 8)
        
        # Shared transform
        self.shared_conv = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Per-axis attention heads
        self.conv_d = nn.Conv1d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_h = nn.Conv1d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv1d(mid_channels, channels, kernel_size=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W)"""
        B, C, D, H, W = x.shape
        
        # Pool along each axis to get 1D descriptors
        # Average over (H, W) -> (B, C, D)
        pool_d = x.mean(dim=[3, 4])  # (B, C, D)
        # Average over (D, W) -> (B, C, H)
        pool_h = x.mean(dim=[2, 4])  # (B, C, H)
        # Average over (D, H) -> (B, C, W)
        pool_w = x.mean(dim=[2, 3])  # (B, C, W)
        
        # Shared transform
        y_d = self.shared_conv(pool_d)  # (B, mid, D)
        y_h = self.shared_conv(pool_h)  # (B, mid, H)
        y_w = self.shared_conv(pool_w)  # (B, mid, W)
        
        # Per-axis attention
        a_d = self.sigmoid(self.conv_d(y_d)).unsqueeze(3).unsqueeze(4)  # (B, C, D, 1, 1)
        a_h = self.sigmoid(self.conv_h(y_h)).unsqueeze(2).unsqueeze(4)  # (B, C, 1, H, 1)
        a_w = self.sigmoid(self.conv_w(y_w)).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1, W)
        
        return x * a_d * a_h * a_w


class ModalityAttention(nn.Module):
    """Attention for modality groups (channel + spatial, matches trained checkpoints)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels // 4, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        spatial_attn = self.spatial_conv(x)
        return x * spatial_attn


class CrossModalityGate(nn.Module):
    """Memory-efficient cross-modality gating (no attention matrices)."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.Sigmoid()
        )
        self.feat_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate = self.gate_conv(torch.cat([x1, x2], dim=1))
        gated = x1 * gate + x2 * (1 - gate)
        return self.feat_conv(gated)


class CrossModalityFusion(nn.Module):
    """
    Memory-Efficient Cross-Modality Fusion.
    
    Groups modalities by clinical relevance:
    - Anatomical: T1 + T1ce (structural details)
    - Pathological: T2 + FLAIR (edema/lesions)
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.mod_channels = channels // 4
        
        # Intra-group attention
        self.anatomical_attn = ModalityAttention(self.mod_channels * 2, reduction)
        self.pathological_attn = ModalityAttention(self.mod_channels * 2, reduction)
        
        # Cross-group gating (memory efficient)
        self.cross_gate_a2p = CrossModalityGate(self.mod_channels * 2)
        self.cross_gate_p2a = CrossModalityGate(self.mod_channels * 2)
        
        # Final fusion: anatomical + pathological + cross = 6x mod_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.mod_channels * 6, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, modality_features: list) -> torch.Tensor:
        """
        Args:
            modality_features: [flair, t1, t1ce, t2], each (B, mod_channels, D, H, W)
        
        Returns:
            Fused features (B, channels, D, H, W)
        """
        flair, t1, t1ce, t2 = modality_features
        
        anatomical = torch.cat([t1, t1ce], dim=1)
        pathological = torch.cat([t2, flair], dim=1)
        
        anatomical_attn = self.anatomical_attn(anatomical)
        pathological_attn = self.pathological_attn(pathological)
        
        cross_a2p = self.cross_gate_a2p(anatomical_attn, pathological_attn)
        cross_p2a = self.cross_gate_p2a(pathological_attn, anatomical_attn)
        
        fused = torch.cat([
            anatomical_attn,
            pathological_attn,
            cross_a2p + cross_p2a
        ], dim=1)
        
        return self.fusion_conv(fused)


class SimpleFusion(nn.Module):
    """Simple concatenation fusion (most memory efficient)."""
    
    def __init__(self, channels: int, num_modalities: int = 4):
        super().__init__()
        
        self.channel_attn = ChannelAttention(channels, reduction=16)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, modality_features: list) -> torch.Tensor:
        x = torch.cat(modality_features, dim=1)
        x = self.channel_attn(x)
        return self.conv(x)


if __name__ == "__main__":
    print("Testing Fusion...")
    
    B, C, D, H, W = 1, 32, 64, 64, 64
    modality_features = [torch.randn(B, C, D, H, W) for _ in range(4)]
    
    fusion = CrossModalityFusion(channels=C * 4)
    output = fusion(modality_features)
    
    print(f"Input: 4 x {modality_features[0].shape}")
    print(f"Output: {output.shape}")
    print("Fusion test passed!")
