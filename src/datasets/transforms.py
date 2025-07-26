"""
Advanced image transformation and augmentation pipeline for food classification.
Includes both standard and food-specific augmentation strategies.
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Union, Tuple, List
import random
import numpy as np
from PIL import Image, ImageFilter
import math
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class FoodSpecificAugmentation:
    """Food-specific augmentation strategies based on food image characteristics"""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            # Apply one of several food-specific augmentations
            augmentation = random.choice([
                self._add_steam_effect,
                self._simulate_lighting_variation,
                self._add_texture_enhancement
            ])
            img = augmentation(img)
        return img
    
    def _add_steam_effect(self, img: Image.Image) -> Image.Image:
        """Simulate steam/heat effect for hot foods"""
        # Slight blur to simulate steam
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img
    
    def _simulate_lighting_variation(self, img: Image.Image) -> Image.Image:
        """Simulate restaurant/kitchen lighting variations"""
        # Adjust brightness and color temperature
        enhancer = transforms.ColorJitter(brightness=0.3, contrast=0.2)
        return enhancer(img)
    
    def _add_texture_enhancement(self, img: Image.Image) -> Image.Image:
        """Enhance food texture details"""
        # Slight sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        return img


class AdaptiveCrop:
    """Adaptive cropping that focuses on food regions"""
    
    def __init__(self, size: Union[int, Tuple[int, int]], scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # For now, use center-biased random crop
        # In practice, this could use attention maps or food detection
        width, height = img.size
        
        # Bias towards center for food crops
        scale_factor = random.uniform(*self.scale)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Center-biased random position
        max_x = width - new_width
        max_y = height - new_height
        
        # Bias towards center (weighted random)
        center_bias = 0.7
        x = int(random.betavariate(2, 2) * max_x * center_bias + 
                random.uniform(0, max_x * (1 - center_bias)))
        y = int(random.betavariate(2, 2) * max_y * center_bias + 
                random.uniform(0, max_y * (1 - center_bias)))
        
        # Crop and resize
        img = img.crop((x, y, x + new_width, y + new_height))
        if isinstance(self.size, int):
            img = img.resize((self.size, self.size))
        else:
            img = img.resize(self.size)
        
        return img


class MixUp:
    """MixUp augmentation for improved generalization"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, batch_data):
        """Apply MixUp to a batch of data"""
        images, food_labels, cuisine_labels, nutrition_targets = batch_data
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        # For classification targets, we'll handle mixing in the loss function
        # For regression targets (nutrition), we can mix directly
        mixed_nutrition = lam * nutrition_targets + (1 - lam) * nutrition_targets[index]
        
        return mixed_images, food_labels, cuisine_labels, mixed_nutrition, index, lam


def get_transforms(split: str = 'train', input_size: int = 224, 
                  augmentation_strength: str = 'medium') -> transforms.Compose:
    """
    Get optimized image transforms for different training phases.
    
    Args:
        split: 'train', 'val', 'test'
        input_size: Target image size
        augmentation_strength: 'light', 'medium', 'heavy'
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    
    # Define augmentation parameters based on strength
    aug_params = {
        'light': {
            'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05,
            'rotation': 5, 'scale': (0.9, 1.0), 'ratio': (0.9, 1.1)
        },
        'medium': {
            'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1,
            'rotation': 10, 'scale': (0.8, 1.0), 'ratio': (0.8, 1.2)
        },
        'heavy': {
            'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.15,
            'rotation': 15, 'scale': (0.7, 1.0), 'ratio': (0.7, 1.3)
        }
    }
    
    params = aug_params.get(augmentation_strength, aug_params['medium'])
    
    if split == 'train':
        # Training transforms with progressive augmentation
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            
            # Geometric augmentations
            transforms.RandomResizedCrop(
                input_size, 
                scale=params['scale'],
                ratio=params['ratio']
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=params['rotation']),
            
            # Color augmentations
            transforms.ColorJitter(
                brightness=params['brightness'],
                contrast=params['contrast'],
                saturation=params['saturation'],
                hue=params['hue']
            ),
            
            # Advanced augmentations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2)
            ], p=0.3),
            
            # Food-specific augmentation
            FoodSpecificAugmentation(prob=0.4),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            
            # Advanced normalization strategies
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            ),
            
            # Random erasing for regularization
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ]
        
    elif split in ['val', 'test']:
        # Validation/test transforms (deterministic)
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
    elif split == 'tta':  # Test Time Augmentation
        # Multiple augmented versions for ensemble prediction
        transform_list = [
            transforms.Resize((input_size + 16, input_size + 16)),
            transforms.FiveCrop(input_size),  # 5 crops: 4 corners + center
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(transforms.ToTensor()(crop)) for crop in crops
            ]))
        ]
    
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return transforms.Compose(transform_list)


def get_food_specific_transforms(food_category: str, input_size: int = 224) -> transforms.Compose:
    """
    Get transforms optimized for specific food categories.
    
    Args:
        food_category: Type of food ('dessert', 'main_dish', 'appetizer', etc.)
        input_size: Target image size
        
    Returns:
        transforms.Compose: Category-optimized transforms
    """
    
    base_transforms = [
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size)
    ]
    
    # Category-specific augmentations
    if food_category in ['dessert', 'cake', 'ice_cream']:
        # Desserts: preserve colors, gentle augmentation
        category_transforms = [
            transforms.RandomHorizontalFlip(p=0.3),  # Less flipping
            transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.15),
            transforms.RandomRotation(degrees=5)  # Minimal rotation
        ]
    
    elif food_category in ['salad', 'vegetables']:
        # Vegetables: enhance colors, more geometric variation
        category_transforms = [
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
            transforms.RandomRotation(degrees=15)
        ]
    
    elif food_category in ['meat', 'steak', 'chicken']:
        # Meat: focus on texture, moderate augmentation
        category_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.1),
            transforms.RandomRotation(degrees=8)
        ]
    
    else:
        # Default: balanced augmentation
        category_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10)
        ]
    
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(base_transforms + category_transforms + final_transforms)


def denormalize_tensor(tensor: torch.Tensor, 
                      mean: List[float] = [0.485, 0.456, 0.406],
                      std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor (C, H, W)
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_augmentations(image_path: str, num_samples: int = 8, 
                           augmentation_strength: str = 'medium'):
    """
    Visualize different augmentations applied to an image.
    
    Args:
        image_path: Path to input image
        num_samples: Number of augmented samples to generate
        augmentation_strength: Strength of augmentation
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms('train', augmentation_strength=augmentation_strength)
    
    # Generate augmented samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        augmented = transform(image)
        # Denormalize for visualization
        augmented = denormalize_tensor(augmented)
        augmented = augmented.permute(1, 2, 0)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmentation {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test transform configurations
    print("Testing Transform Configurations...")
    
    # Test different transform types
    train_transform = get_transforms('train', augmentation_strength='medium')
    val_transform = get_transforms('val')
    
    print(f"Train transforms: {len(train_transform.transforms)} steps")
    print(f"Val transforms: {len(val_transform.transforms)} steps")
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (256, 256), color='red')
    
    try:
        train_result = train_transform(dummy_image)
        val_result = val_transform(dummy_image)
        
        print(f"Train output shape: {train_result.shape}")
        print(f"Val output shape: {val_result.shape}")
        print("Transform test successful!")
        
    except Exception as e:
        print(f"Transform test failed: {e}")