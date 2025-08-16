"""
Domain-specific attention mechanisms for food image analysis.
Implements food-specific attention that understands texture, color, and shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_attention import BaseAttention
from .channel_spatial import ChannelAttention, SpatialAttention


class FoodSpecificAttention(BaseAttention):
    """
    Food-specific attention mechanism designed for food image analysis.
    
    Combines multiple attention mechanisms optimized for food characteristics:
    - Texture attention (for food surface details)
    - Color attention (for ingredient identification)  
    - Shape attention (for food structure recognition)
    """
    
    def __init__(self, in_channels: int, num_food_classes: int = 101):
        super(FoodSpecificAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_food_classes = num_food_classes
        
        # Texture attention - focuses on surface patterns
        self.texture_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Color attention - emphasizes color information
        self.color_attention = ChannelAttention(in_channels, reduction_ratio=8)
        
        # Shape attention - focuses on structural elements
        self.shape_attention = SpatialAttention(kernel_size=7)
        
        # Food-class specific weighting
        self.class_weights = nn.Parameter(torch.ones(num_food_classes, 3))  # 3 attention types
        
        # Fusion layer
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, food_class_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply different attention mechanisms
        texture_attended = x * self.texture_attention(x)
        color_attended = self.color_attention(x)
        shape_attended = self.shape_attention(x)
        
        # Adaptive weighting based on predicted food class
        if food_class_logits is not None:
            batch_size = x.size(0)
            
            # Get class probabilities
            class_probs = F.softmax(food_class_logits, dim=1)
            
            # Weight attention mechanisms by predicted class
            attention_weights = torch.matmul(class_probs, F.softmax(self.class_weights, dim=1))
            
            # Apply weights
            texture_weight = attention_weights[:, 0].view(batch_size, 1, 1, 1)
            color_weight = attention_weights[:, 1].view(batch_size, 1, 1, 1)
            shape_weight = attention_weights[:, 2].view(batch_size, 1, 1, 1)
            
            # Weighted combination
            combined = (texture_weight * texture_attended + 
                       color_weight * color_attended + 
                       shape_weight * shape_attended)
        else:
            # Equal weighting if no class information
            combined = (texture_attended + color_attended + shape_attended) / 3
        
        # Final fusion
        output = self.fusion(combined)
        
        return output
    
    def get_attention_components(self, x: torch.Tensor) -> dict:
        """Get individual attention components for analysis."""
        with torch.no_grad():
            texture_map = self.texture_attention(x)
            color_attended = self.color_attention(x)
            shape_attended = self.shape_attention(x)
            
            return {
                'texture_attention': texture_map,
                'color_attention': color_attended,
                'shape_attention': shape_attended
            }