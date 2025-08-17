"""
Main FoodNet architectures for food image analysis.
Implements standard, efficient, and deep variants.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .food_blocks import FoodCNNBlock
from .efficient_conv import EfficientFoodBlock


class FoodNet(nn.Module):
    """
    Custom CNN architecture specifically designed for food image analysis.
    
    Architecture overview:
    1. Early layers focus on low-level food features (textures, colors)
    2. Middle layers analyze food structure and composition
    3. Late layers perform high-level food categorization
    4. Multi-task heads for classification and regression
    """
    
    def __init__(self, 
                 num_food_classes: int = 101,
                 num_cuisine_classes: int = 10,
                 nutrition_dim: int = 4,
                 architecture: str = "standard",
                 dropout_rate: float = 0.3):
        super(FoodNet, self).__init__()
        
        self.num_food_classes = num_food_classes
        self.num_cuisine_classes = num_cuisine_classes
        self.nutrition_dim = nutrition_dim
        self.architecture = architecture
        
        # Input preprocessing
        self.input_norm = nn.BatchNorm2d(3)
        
        if architecture == "standard":
            self._build_standard_architecture(dropout_rate)
        elif architecture == "efficient":
            self._build_efficient_architecture(dropout_rate)
        elif architecture == "deep":
            self._build_deep_architecture(dropout_rate)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_standard_architecture(self, dropout_rate: float):
        """Build standard FoodNet architecture."""
        
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: Low-level food features (56x56)
        self.stage1 = nn.Sequential(
            FoodCNNBlock(32, 64, "texture_focus"),
            FoodCNNBlock(64, 64, "color_focus"),
        )
        
        # Stage 2: Mid-level food analysis (28x28)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FoodCNNBlock(64, 128, "full"),
            FoodCNNBlock(128, 128, "full"),
        )
        
        # Stage 3: High-level food understanding (14x14)
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FoodCNNBlock(128, 256, "full"),
            FoodCNNBlock(256, 256, "full"),
            FoodCNNBlock(256, 256, "full"),
        )
        
        # Stage 4: Food categorization (7x7)
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            EfficientFoodBlock(256, 512),
            EfficientFoodBlock(512, 512),
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-task heads
        self._build_task_heads(512, dropout_rate)
    
    def _build_efficient_architecture(self, dropout_rate: float):
        """Build efficient FoodNet for mobile deployment."""
        
        # Efficient stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Efficient stages using depthwise separable convolutions
        self.stage1 = nn.Sequential(
            EfficientFoodBlock(32, 64),
            EfficientFoodBlock(64, 64),
        )
        
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            EfficientFoodBlock(64, 128),
            EfficientFoodBlock(128, 128),
        )
        
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            EfficientFoodBlock(128, 256),
            EfficientFoodBlock(256, 256),
        )
        
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            EfficientFoodBlock(256, 512),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self._build_task_heads(512, dropout_rate)
    
    def _build_deep_architecture(self, dropout_rate: float):
        """Build deep FoodNet for maximum accuracy."""
        
        # Deep stem with more initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Deep stages with more blocks
        self.stage1 = self._make_stage(64, 128, 3, "full")
        self.stage2 = self._make_stage(128, 256, 4, "full")  
        self.stage3 = self._make_stage(256, 512, 6, "full")
        self.stage4 = self._make_stage(512, 1024, 3, "efficient")
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self._build_task_heads(1024, dropout_rate)
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, block_type: str):
        """Helper to create a stage with multiple blocks."""
        layers = []
        
        # First block with downsampling
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if block_type == "full":
            layers.append(FoodCNNBlock(in_channels, out_channels, block_type))
        else:
            layers.append(EfficientFoodBlock(in_channels, out_channels))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            if block_type == "full":
                layers.append(FoodCNNBlock(out_channels, out_channels, block_type))
            else:
                layers.append(EfficientFoodBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _build_task_heads(self, feature_dim: int, dropout_rate: float):
        """Build multi-task classification and regression heads."""
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Food classification head
        self.food_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_food_classes)
        )
        
        # Cuisine classification head
        self.cuisine_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.num_cuisine_classes)
        )
        
        # Nutrition regression head
        self.nutrition_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.nutrition_dim),
            nn.ReLU()  # Ensure positive nutrition values
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through FoodNet.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Tuple of (food_logits, cuisine_logits, nutrition_values)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Shared processing
        shared_features = self.shared_fc(x)
        
        # Multi-task predictions
        food_logits = self.food_classifier(shared_features)
        cuisine_logits = self.cuisine_classifier(shared_features)
        nutrition_values = self.nutrition_regressor(shared_features)
        
        return food_logits, cuisine_logits, nutrition_values
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization and analysis.
        """
        feature_maps = {}
        
        x = self.input_norm(x)
        feature_maps['input_norm'] = x
        
        x = self.stem(x)
        feature_maps['stem'] = x
        
        x = self.stage1(x)
        feature_maps['stage1'] = x
        
        x = self.stage2(x)
        feature_maps['stage2'] = x
        
        x = self.stage3(x)
        feature_maps['stage3'] = x
        
        x = self.stage4(x)
        feature_maps['stage4'] = x
        
        return feature_maps