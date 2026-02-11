"""
MRAF-Net Swin Transformer 3D Block
Lightweight 3D Swin Transformer for capturing long-range dependencies at the bottleneck.

This module introduces a novel hybrid CNN-Transformer architecture by placing
a window-based self-attention block at the bottleneck of the encoder-decoder network.
This allows the model to capture global context that pure CNNs miss, while keeping
computational cost manageable through windowed attention.

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-Head Self-Attention (W-MSA).
    
    Computes self-attention within non-overlapping 3D windows.
    Uses relative position bias for spatial awareness.
    
    Args:
        dim: Number of input channels.
        window_size: Size of the 3D attention window (D, H, W).
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to Q, K, V.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int] = (2, 2, 2),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # (3, D, H, W)
        coords_flatten = torch.flatten(coords, 1)  # (3, D*H*W)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_windows * B, window_volume, C)
        
        Returns:
            Output tensor of same shape
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B_, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP3D(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition a 3D volume into non-overlapping windows.
    
    Args:
        x: (B, D, H, W, C)
        window_size: (wD, wH, wW)
    
    Returns:
        windows: (num_windows * B, wD * wH * wW, C)
    """
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
    
    x = x.view(B, D // wD, wD, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wD * wH * wW, C)
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int], D: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows * B, wD * wH * wW, C)
        window_size: (wD, wH, wW)
        D, H, W: Original spatial dimensions
    
    Returns:
        x: (B, D, H, W, C)
    """
    wD, wH, wW = window_size
    num_windows_per_sample = (D // wD) * (H // wH) * (W // wW)
    B = windows.shape[0] // num_windows_per_sample
    
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class SwinTransformerBlock3D(nn.Module):
    """
    A single Swin Transformer block for 3D data.
    
    Uses Window-based Multi-Head Self-Attention followed by a Feed-Forward Network.
    Layer normalization and residual connections are applied.
    
    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Size of the 3D attention window.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to Q, K, V.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (2, 2, 2),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP3D(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Reshape to (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # Pad if needed so spatial dims are divisible by window_size
        wD, wH, wW = self.window_size
        pad_d = (wD - D % wD) % wD
        pad_h = (wH - H % wH) % wH
        pad_w = (wW - W % wW) % wW
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
        
        # Window-based self-attention
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition_3d(x, self.window_size)  # (num_windows*B, window_volume, C)
        attn_windows = self.attn(x_windows)
        x = window_reverse_3d(attn_windows, self.window_size, Dp, Hp, Wp)
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        
        # Reshape back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class SwinBottleneck3D(nn.Module):
    """
    Stacked Swin Transformer blocks to serve as the bottleneck in a U-Net style network.
    
    This is the key novelty component: a hybrid CNN-Transformer architecture where
    the Transformer captures global context at the deepest feature resolution, 
    complementing the CNN's local receptive field.
    
    Args:
        dim: Number of input/output channels.
        depth: Number of Swin Transformer blocks to stack.
        num_heads: Number of attention heads.
        window_size: Size of the 3D attention window.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (2, 2, 2),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        
        Returns:
            (B, C, D, H, W),  same shape as input
        """
        for block in self.blocks:
            x = block(x)
        
        # Apply final norm
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        return x


if __name__ == "__main__":
    print("Testing Swin Transformer 3D...")
    
    # Simulate bottleneck features: 320 channels, 4x4x4 spatial
    x = torch.randn(1, 320, 4, 4, 4)
    
    swin = SwinBottleneck3D(dim=320, depth=2, num_heads=8, window_size=(2, 2, 2))
    out = swin(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in swin.parameters()):,}")
    print("Swin Transformer test passed!")
