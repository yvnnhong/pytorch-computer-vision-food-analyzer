"""
Domain-specific building blocks for food image analysis.
Implements texture, color, and structure analysis components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FoodTextureBlock(nn.Module):
    """
    Specialized block for analyzing food textures and surface patterns.
    
    Uses multiple kernel sizes to capture different texture scales:
    - 3x3: Fine texture details (seeds, seasonings, small patterns)
    - 5x5: Medium texture (meat grain, vegetable surfaces)
    - 7x7: Coarse texture (bread texture, overall surface patterns)
    """
    
    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 4):
        super(FoodTextureBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Multi-scale texture convolutions
        self.texture_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )
        
        self.texture_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )
        
        self.texture_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )
        
        # Texture attention mechanism
        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction_ratio, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract textures at different scales
        texture_fine = self.texture_3x3(x)
        texture_medium = self.texture_5x5(x)
        texture_coarse = self.texture_7x7(x)
        
        # Combine textures
        combined_texture = torch.cat([texture_fine, texture_medium, texture_coarse], dim=1)
        
        # Apply texture attention
        attention_weights = self.texture_attention(combined_texture)
        attended_texture = combined_texture * attention_weights
        
        # Residual connection
        residual = self.residual(x)
        
        return attended_texture + residual


class ColorAnalysisBlock(nn.Module):
    """
    Specialized block for food color analysis and ingredient identification.
    
    Focuses on color variations that indicate:
    - Ripeness and freshness
    - Cooking level (browning, caramelization)
    - Ingredient composition
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ColorAnalysisBlock, self).__init__()
        
        # Color-sensitive convolutions
        self.color_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # HSV-inspired color space transformations
        self.hue_branch = nn.Conv2d(out_channels, out_channels // 3, kernel_size=3, padding=1)
        self.saturation_branch = nn.Conv2d(out_channels, out_channels // 3, kernel_size=3, padding=1)
        self.value_branch = nn.Conv2d(out_channels, out_channels // 3, kernel_size=3, padding=1)
        
        # Color harmony attention
        self.color_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract color features
        color_features = self.color_conv(x)
        
        # Analyze different color aspects
        hue_features = self.hue_branch(color_features)
        saturation_features = self.saturation_branch(color_features)
        value_features = self.value_branch(color_features)
        
        # Combine color aspects
        combined_color = torch.cat([hue_features, saturation_features, value_features], dim=1)
        
        # Apply color attention
        batch_size, channels, height, width = combined_color.shape
        attention_weights = self.color_attention(combined_color).view(batch_size, channels, 1, 1)
        
        return combined_color * attention_weights


class FoodStructureBlock(nn.Module):
    """
    Block for analyzing food structure, shape, and composition.
    
    Captures:
    - Food arrangement and plating
    - Portion sizes and proportions
    - Geometric patterns and shapes
    """
    
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: List[int] = [1, 2, 4]):
        super(FoodStructureBlock, self).__init__()
        
        self.dilation_rates = dilation_rates
        
        # Multi-scale dilated convolutions for structure analysis
        self.structure_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(dilation_rates), 
                         kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels // len(dilation_rates)),
                nn.ReLU(inplace=True)
            ) for rate in dilation_rates
        ])
        
        # Global structure understanding
        self.global_structure = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # Preserve spatial structure
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Structure fusion
        self.structure_fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        
        # Extract multi-scale structural features
        structure_features = []
        for conv in self.structure_convs:
            structure_features.append(conv(x))
        
        # Combine multi-scale features
        combined_structure = torch.cat(structure_features, dim=1)
        
        # Global structure analysis
        global_structure = self.global_structure(combined_structure)
        global_structure = F.interpolate(global_structure, size=(height, width), 
                                       mode='bilinear', align_corners=False)
        
        # Fuse local and global structure
        fused_structure = torch.cat([combined_structure, global_structure], dim=1)
        output = self.structure_fusion(fused_structure)
        
        return output


class FoodCNNBlock(nn.Module):
    """
    Comprehensive food analysis block combining texture, color, and structure analysis.
    """
    
    def __init__(self, in_channels: int, out_channels: int, block_type: str = "full"):
        super(FoodCNNBlock, self).__init__()
        
        self.block_type = block_type
        
        if block_type == "full":
            # Full analysis with all components
            self.texture_block = FoodTextureBlock(in_channels, out_channels // 3)
            self.color_block = ColorAnalysisBlock(in_channels, out_channels // 3)
            self.structure_block = FoodStructureBlock(in_channels, out_channels // 3)
            
            # Feature fusion
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        elif block_type == "texture_focus":
            # Focus on texture analysis
            self.texture_block = FoodTextureBlock(in_channels, out_channels)
            
        elif block_type == "color_focus":
            # Focus on color analysis
            self.color_block = ColorAnalysisBlock(in_channels, out_channels)
            
        else:
            raise ValueError(f"Unknown block type: {block_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == "full":
            texture_features = self.texture_block(x)
            color_features = self.color_block(x)
            structure_features = self.structure_block(x)
            
            # Combine all features
            combined = torch.cat([texture_features, color_features, structure_features], dim=1)
            return self.fusion(combined)
            
        elif self.block_type == "texture_focus":
            return self.texture_block(x)
            
        elif self.block_type == "color_focus":
            return self.color_block(x)