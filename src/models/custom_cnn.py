"""
Custom CNN Architectures Optimized for Food Image Analysis

implements domain-specific CNN architectures designed for food recognition
tasks
specific optimizations: 
- Food texture recognition (surface patterns, ingredient visibility)
- Color variation analysis (ripeness, cooking level, freshness)
- Shape and structure understanding (plating, portion size, composition)
- Multi-scale feature extraction (ingredient details to overall presentation)

Math: 
- Depthwise Separable Convolutions for efficiency
- Dilated Convolutions for multi-scale receptive fields
- Attention-guided feature selection
- Residual connections with food-specific skip patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict


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
    - Visual appeal and presentation
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
    - Spatial relationships between ingredients
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
            nn.Upsample(scale_factor=None, mode='bilinear', align_corners=False)
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
        global_structure = self.global_structure[:-1](combined_structure)  # All except upsample
        global_structure = F.interpolate(global_structure, size=(height, width), 
                                       mode='bilinear', align_corners=False)
        
        # Fuse local and global structure
        fused_structure = torch.cat([combined_structure, global_structure], dim=1)
        output = self.structure_fusion(fused_structure)
        
        return output


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
            
        elif block_type == "efficient":
            # Efficient version using depthwise separable convolutions
            self.efficient_conv = DepthwiseSeparableConv(in_channels, out_channels)
            
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
            
        elif self.block_type == "efficient":
            return self.efficient_conv(x)
            
        elif self.block_type == "texture_focus":
            return self.texture_block(x)
            
        elif self_block_type == "color_focus":
            return self.color_block(x)


class FoodNet(nn.Module):
    """
    Custom CNN architecture specifically designed for food image analysis.
    Architecture overview 
    1. Early layers focus on low-level food features (textures, colors)
    2. Middle layers analyze food structure and composition
    3. Late layers perform high-level food categorization
    4. Multi-task heads for classification and regression
    
    Key Features:
    - Food-specific inductive biases
    - Efficient computation for mobile deployment
    - Multi-scale feature extraction
    - Attention mechanisms for important feature selection
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
            FoodCNNBlock(256, 512, "efficient"),
            FoodCNNBlock(512, 512, "efficient"),
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-task heads
        self._build_task_heads(512, dropout_rate)
    
    def _build_efficient_architecture(self, dropout_rate: float):
        """Build efficient FoodNet for mobile deployment."""
        
        # Efficient stem
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Efficient stages using depthwise separable convolutions
        self.stage1 = nn.Sequential(
            FoodCNNBlock(32, 64, "efficient"),
            FoodCNNBlock(64, 64, "efficient"),
        )
        
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FoodCNNBlock(64, 128, "efficient"),
            FoodCNNBlock(128, 128, "efficient"),
        )
        
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FoodCNNBlock(128, 256, "efficient"),
            FoodCNNBlock(256, 256, "efficient"),
        )
        
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            FoodCNNBlock(256, 512, "efficient"),
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
        layers.append(FoodCNNBlock(in_channels, out_channels, block_type))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(FoodCNNBlock(out_channels, out_channels, block_type))
        
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
        
        Args:
            x: Input tensor
            
        Returns:
            Dict of feature maps from different stages
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


class MobileFoodNet(nn.Module):
    """
    Ultra-efficient FoodNet variant for mobile and edge deployment.
    
    Optimizations:
    - Depthwise separable convolutions throughout
    - Reduced channel dimensions
    - Efficient attention mechanisms
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


def create_food_cnn(architecture: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create different FoodNet variants.
    
    Args:
        architecture: Architecture type ('standard', 'efficient', 'deep', 'mobile')
        **kwargs: Additional arguments for model creation
        
    Returns:
        nn.Module: FoodNet model
    """
    
    if architecture == "mobile":
        return MobileFoodNet(**kwargs)
    else:
        return FoodNet(architecture=architecture, **kwargs)


def calculate_model_efficiency(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate efficiency metrics for food CNN models.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict with efficiency metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Count different layer types
    conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    depthwise_layers = len([m for m in model.modules() if isinstance(m, DepthwiseSeparableConv)])
    
    # Efficiency score (higher is more efficient)
    efficiency_score = 100 / (1 + math.log10(total_params / 100000))
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'conv_layers': conv_layers,
        'depthwise_layers': depthwise_layers,
        'efficiency_score': efficiency_score,
        'mobile_friendly': depthwise_layers > conv_layers / 2
    }


if __name__ == "__main__":
    print("Testing Custom Food CNN Architectures...")
    print("=" * 60)
    
    # Test different architectures
    architectures = {
        'Standard FoodNet': create_food_cnn('standard'),
        'Efficient FoodNet': create_food_cnn('efficient'),
        'Deep FoodNet': create_food_cnn('deep'),
        'Mobile FoodNet': create_food_cnn('mobile', width_multiplier=1.0),
        'Mobile FoodNet 0.5x': create_food_cnn('mobile', width_multiplier=0.5)
    }
    
    # Test input
    test_input = torch.randn(2, 3, 224, 224)
    
    print(f"Input shape: {test_input.shape}")
    print("-" * 60)
    
    for name, model in architectures.items():
        try:
            # Forward pass
            food_logits, cuisine_logits, nutrition_values = model(test_input)
            
            # Calculate efficiency metrics
            efficiency = calculate_model_efficiency(model)
            
            print(f"{name}:")
            print(f"  Output shapes: Food{food_logits.shape}, Cuisine{cuisine_logits.shape}, Nutrition{nutrition_values.shape}")
            print(f"  Parameters: {efficiency['total_parameters']:,}")
            print(f"  Model size: {efficiency['model_size_mb']:.2f} MB")
            print(f"  Conv layers: {efficiency['conv_layers']}")
            print(f"  Depthwise layers: {efficiency['depthwise_layers']}")
            print(f"  Efficiency score: {efficiency['efficiency_score']:.1f}")
            print(f"  Mobile friendly: {efficiency['mobile_friendly']}")
            print()
            
        except Exception as e:
            print(f"{name}: Error - {e}")
            print()
    
    # Test feature extraction
    print("Testing Feature Extraction...")
    print("-" * 30)
    
    standard_model = create_food_cnn('standard')
    feature_maps = standard_model.get_feature_maps(test_input)
    
    for stage, features in feature_maps.items():
        print(f"  {stage}: {features.shape}")
    
    print("\n" + "=" * 60)
    print("Custom Food CNN Testing Complete!")