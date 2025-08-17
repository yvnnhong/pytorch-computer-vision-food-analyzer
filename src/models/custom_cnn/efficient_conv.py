"""
Efficient convolution implementations for mobile deployment.
Implements depthwise separable convolutions and mobile-optimized blocks.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Efficient depthwise separable convolution for mobile-friendly food analysis.
    
    Mathematical decomposition:
    Standard Conv: H × W × C_in × C_out × K × K
    Depthwise Sep: H × W × C_in × K × K + H × W × C_in × C_out × 1 × 1
    
    Computational savings: ~8-9x reduction in parameters and FLOPs
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class EfficientFoodBlock(nn.Module):
    """
    Efficient version of food analysis block using depthwise separable convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EfficientFoodBlock, self).__init__()
        
        self.efficient_conv = DepthwiseSeparableConv(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficient_conv(x)