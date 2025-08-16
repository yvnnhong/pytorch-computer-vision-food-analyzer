"""
Channel and spatial attention mechanisms (CBAM family).
Implements channel attention, spatial attention, CBAM, and Squeeze-Excitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base_attention import BaseAttention


class ChannelAttention(BaseAttention):
    """
    Channel Attention Module for emphasizing important feature channels.
    
    Mathematical formulation:
    A_c = σ(W_2 * ReLU(W_1 * GAP(F)) + W_2 * ReLU(W_1 * GMP(F)))
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Shared MLP for both average and max pooling paths
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, in_channels)
        )
        
        # Global pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # Global Average Pooling path
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.shared_mlp(avg_pool)
        
        # Global Max Pooling path
        max_pool = self.max_pool(x).view(batch_size, channels)
        max_out = self.shared_mlp(max_pool)
        
        # Combine and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        channel_attention = channel_attention.view(batch_size, channels, 1, 1)
        
        return x * channel_attention


class SpatialAttention(BaseAttention):
    """
    Spatial Attention Module for emphasizing important spatial locations.
    
    Mathematical formulation:
    A_s = σ(conv([AvgPool_c(F); MaxPool_c(F)]))
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid
        spatial_attention = self.sigmoid(self.conv(pooled))
        
        return x * spatial_attention


class CBAM(BaseAttention):
    """
    Convolutional Block Attention Module (CBAM).
    Combines both channel and spatial attention sequentially.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x


class SqueezeExcitation(BaseAttention):
    """
    Squeeze-and-Excitation mechanism for efficient channel attention.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SqueezeExcitation, self).__init__()
        
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        squeezed = self.squeeze(x).view(batch_size, channels)
        
        # Excitation: FC layers with sigmoid
        excited = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        
        return x * excited