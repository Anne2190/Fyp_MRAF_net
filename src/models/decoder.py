"""
MRAF-Net Decoder Module
Attention-Enhanced Decoder for Brain Tumor Segmentation

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features."""
    
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
        """
        Args:
            g: Gating signal from decoder (coarser)
            x: Skip connection from encoder (finer)
        """
        # Upsample g to match x size if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection and attention."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Upsample
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Attention gate
        if use_attention:
            self.attention = AttentionGate(
                gate_channels=in_channels // 2,
                skip_channels=skip_channels
            )
        
        # Convolution after concatenation
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels, dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        if self.use_attention:
            skip = self.attention(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionDecoder(nn.Module):
    """
    Attention-Enhanced Decoder with optional deep supervision.
    
    Takes encoder features [e0, e1, e2, e3] and bottleneck, produces segmentation.
    
    For 64x64x64 input with encoder features [64, 128, 256, 320]:
        Bottleneck: 4x4x4, 320 channels
        Up1: 8x8x8, 256 channels
        Up2: 16x16x16, 128 channels
        Up3: 32x32x32, 64 channels
        Up4: 64x64x64, 32 channels
        Output: 64x64x64, num_classes channels
    """
    
    def __init__(
        self,
        encoder_features: List[int] = None,
        num_classes: int = 4,
        deep_supervision: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if encoder_features is None:
            encoder_features = [64, 128, 256, 320, 320]
        
        self.deep_supervision = deep_supervision
        
        # Decoder levels (reverse of encoder)
        # e0=64, e1=128, e2=256, e3=320, bottleneck=320
        self.up1 = UpBlock(encoder_features[4], encoder_features[3], 256, use_attention=True, dropout=dropout)
        self.up2 = UpBlock(256, encoder_features[2], 128, use_attention=True, dropout=dropout)
        self.up3 = UpBlock(128, encoder_features[1], 64, use_attention=True, dropout=dropout)
        self.up4 = UpBlock(64, encoder_features[0], 32, use_attention=True, dropout=dropout)
        
        # Final segmentation head
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.ds1 = nn.Conv3d(256, num_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(128, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        bottleneck: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass of decoder.
        
        Args:
            encoder_features: [e0, e1, e2, e3] from encoder
            bottleneck: Bottleneck features from ASPP
        
        Returns:
            main_output: Final segmentation
            ds_outputs: Deep supervision outputs (if enabled)
        """
        e0, e1, e2, e3 = encoder_features
        
        # Decoder path
        d1 = self.up1(bottleneck, e3)  # 4->8
        d2 = self.up2(d1, e2)          # 8->16
        d3 = self.up3(d2, e1)          # 16->32
        d4 = self.up4(d3, e0)          # 32->64
        
        # Final output
        output = self.final_conv(d4)
        
        # Deep supervision
        if self.deep_supervision and self.training:
            ds1_out = F.interpolate(self.ds1(d1), size=output.shape[2:], mode='trilinear', align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), size=output.shape[2:], mode='trilinear', align_corners=False)
            ds3_out = F.interpolate(self.ds3(d3), size=output.shape[2:], mode='trilinear', align_corners=False)
            return output, [ds1_out, ds2_out, ds3_out]
        
        return output, None


class LightweightDecoder(nn.Module):
    """Lightweight decoder for memory-constrained environments."""
    
    def __init__(
        self,
        encoder_features: List[int] = None,
        num_classes: int = 4
    ):
        super().__init__()
        
        if encoder_features is None:
            encoder_features = [64, 128, 256, 320, 320]
        
        # Simple upsampling path
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(encoder_features[4], 256, kernel_size=2, stride=2),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256 + encoder_features[3], 128, kernel_size=2, stride=2),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(128 + encoder_features[2], 64, kernel_size=2, stride=2),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(64 + encoder_features[1], 32, kernel_size=2, stride=2),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        self.final = nn.Conv3d(32 + encoder_features[0], num_classes, kernel_size=1)
    
    def forward(
        self,
        encoder_features: List[torch.Tensor],
        bottleneck: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        e0, e1, e2, e3 = encoder_features
        
        x = self.up1(bottleneck)
        x = F.interpolate(x, size=e3.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, e3], dim=1)
        
        x = self.up2(x)
        x = F.interpolate(x, size=e2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        
        x = self.up3(x)
        x = F.interpolate(x, size=e1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        
        x = self.up4(x)
        x = F.interpolate(x, size=e0.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, e0], dim=1)
        
        return self.final(x), None


if __name__ == "__main__":
    print("Testing Decoder...")
    
    # Create dummy encoder outputs (for 64x64x64 input)
    e0 = torch.randn(1, 64, 64, 64, 64)
    e1 = torch.randn(1, 128, 32, 32, 32)
    e2 = torch.randn(1, 256, 16, 16, 16)
    e3 = torch.randn(1, 320, 8, 8, 8)
    bottleneck = torch.randn(1, 320, 4, 4, 4)
    
    encoder_features = [e0, e1, e2, e3]
    
    # Test AttentionDecoder
    decoder = AttentionDecoder(
        encoder_features=[64, 128, 256, 320, 320],
        num_classes=4,
        deep_supervision=True
    )
    decoder.train()
    output, ds_outputs = decoder(encoder_features, bottleneck)
    
    print(f"Main output: {output.shape}")
    if ds_outputs:
        for i, ds in enumerate(ds_outputs):
            print(f"DS {i}: {ds.shape}")
    
    print("Decoder test passed!")
