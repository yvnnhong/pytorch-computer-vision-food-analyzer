"""
Ultra-efficient FoodNet variant for mobile and edge deployment.
Optimized for speed and memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .efficient_conv import DepthwiseSeparableConv


class MobileFoodNet(nn.Module):
    """
    Ultra-efficient FoodNet variant for mobile and edge deployment.
    
    Optimizations:
    - Depthwise separable convolutions throughout
    - Reduced channel dimensions
    - Quantization-friendly design
    """
    
    def __init__(self, 
                 num_food_classes: int = 101,
                 num_cuisine_classes: int = 10,
                 nutrition_dim: int = 4,
                 width_multiplier: float = 1.0):
        super(MobileFoodNet, self).__init__()
        
        # Calculate channel dimensions with width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        channels = [
            make_divisible(32 * width_multiplier),
            make_divisible(64 * width_multiplier),
            make_divisible(128 * width_multiplier),
            make_divisible(256 * width_multiplier),
            make_divisible(512 * width_multiplier)
        ]
        
        # Mobile-optimized architecture
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv(channels[0], channels[1], stride=1),
            DepthwiseSeparableConv(channels[1], channels[1], stride=2),
            
            DepthwiseSeparableConv(channels[1], channels[2], stride=1),
            DepthwiseSeparableConv(channels[2], channels[2], stride=2),
            
            DepthwiseSeparableConv(channels[2], channels[3], stride=1),
            DepthwiseSeparableConv(channels[3], channels[3], stride=2),
            
            DepthwiseSeparableConv(channels[3], channels[4], stride=1),
            DepthwiseSeparableConv(channels[4], channels[4], stride=1),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[4], num_food_classes + num_cuisine_classes + nutrition_dim)
        )
        
        self.num_food_classes = num_food_classes
        self.num_cuisine_classes = num_cuisine_classes
        self.nutrition_dim = nutrition_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # Combined output
        combined_output = self.classifier(x)
        
        # Split outputs
        food_logits = combined_output[:, :self.num_food_classes]
        cuisine_logits = combined_output[:, self.num_food_classes:self.num_food_classes + self.num_cuisine_classes]
        nutrition_values = combined_output[:, -self.nutrition_dim:]
        
        # Ensure positive nutrition values
        nutrition_values = F.relu(nutrition_values)
        
        return food_logits, cuisine_logits, nutrition_values