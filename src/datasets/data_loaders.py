"""
DataLoader configurations and utilities for multi-task food classification.
Handles batch creation, data loading optimization, and dataset splitting.
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import Counter
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.dataset import MultiTaskFoodDataset
from src.datasets.transforms import get_transforms


def create_dataloaders(
    batch_size: int = 32, 
    num_workers: int = 4, 
    subset_size: Optional[int] = None,
    root_dir: str = './data/raw/food-101',
    val_split: float = 0.2,
    use_weighted_sampling: bool = False,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create optimized training and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        subset_size: If specified, use subset for quick testing
        root_dir: Path to Food-101 dataset
        val_split: Fraction of training data to use for validation
        use_weighted_sampling: Whether to use weighted sampling for class balance
        pin_memory: Whether to pin memory for GPU transfer optimization
        
    Returns:
        tuple: (train_loader, val_loader, class_info)
    """
    
    # Get transforms
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    # Create datasets
    train_dataset = MultiTaskFoodDataset(
        root_dir=root_dir,
        split='train', 
        transform=train_transform,
        subset_size=subset_size
    )
    
    val_dataset = MultiTaskFoodDataset(
        root_dir=root_dir,
        split='test',  # Food-101 uses 'test' for validation
        transform=val_transform,
        subset_size=subset_size//4 if subset_size else None
    )
    
    # Optional: Create weighted sampler for balanced training
    train_sampler = None
    if use_weighted_sampling:
        train_sampler = create_weighted_sampler(train_dataset)
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batches for stable training
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Fixed: None when num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None  # Fixed: None when num_workers=0
    )
    
    # Get comprehensive class information
    class_info = get_comprehensive_class_info(train_dataset, val_dataset)
    
    print(f"\nDataLoader Configuration:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Total train samples: {len(train_dataset)}")
    print(f"Total val samples: {len(val_dataset)}")
    print(f"Weighted sampling: {use_weighted_sampling}")
    print(f"Workers: {num_workers}")
    print(f"Pin memory: {pin_memory and torch.cuda.is_available()}")
    
    return train_loader, val_loader, class_info


def create_weighted_sampler(dataset: MultiTaskFoodDataset) -> WeightedRandomSampler:
    """
    Create weighted random sampler for balanced training across food classes.
    
    Args:
        dataset: Training dataset
        
    Returns:
        WeightedRandomSampler: Sampler that balances class distribution
    """
    
    # Count samples per food class
    food_labels = []
    for i in range(len(dataset)):
        _, food_label, _, _ = dataset[i]
        food_labels.append(food_label)
    
    # Calculate class weights (inverse frequency)
    class_counts = Counter(food_labels)
    total_samples = len(food_labels)
    num_classes = len(class_counts)
    
    # Weight = 1 / (class_frequency)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in food_labels]
    
    print(f"Created weighted sampler with {len(class_weights)} classes")
    print(f"Weight range: {min(sample_weights):.3f} - {max(sample_weights):.3f}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_stratified_split(
    dataset: MultiTaskFoodDataset, 
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Create stratified train/val split maintaining class distribution.
    
    Args:
        dataset: Full dataset to split
        val_split: Fraction for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_subset, val_subset)
    """
    
    torch.manual_seed(random_seed)
    
    # Get all food labels for stratification
    food_labels = []
    for i in range(len(dataset)):
        _, food_label, _, _ = dataset[i]
        food_labels.append(food_label)
    
    # Create stratified indices
    train_indices = []
    val_indices = []
    
    class_indices = {}
    for idx, label in enumerate(food_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Split each class proportionally
    for class_label, indices in class_indices.items():
        np.random.shuffle(indices)
        val_size = int(len(indices) * val_split)
        
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Stratified split: {len(train_indices)} train, {len(val_indices)} val")
    
    return train_subset, val_subset


def get_comprehensive_class_info(
    train_dataset: MultiTaskFoodDataset, 
    val_dataset: MultiTaskFoodDataset
) -> Dict[str, Any]:
    """
    Get comprehensive information about dataset classes and distribution.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        dict: Comprehensive class and distribution information
    """
    
    # Basic class info
    class_info = train_dataset.get_class_info()
    
    # Calculate class distributions
    train_food_dist = Counter()
    train_cuisine_dist = Counter()
    
    for i in range(min(1000, len(train_dataset))):  # Sample for efficiency
        _, food_label, cuisine_label, _ = train_dataset[i]
        train_food_dist[food_label] += 1
        train_cuisine_dist[cuisine_label] += 1
    
    # Add distribution statistics
    class_info.update({
        'train_food_distribution': dict(train_food_dist),
        'train_cuisine_distribution': dict(train_cuisine_dist),
        'food_class_balance': {
            'min_samples': min(train_food_dist.values()) if train_food_dist else 0,
            'max_samples': max(train_food_dist.values()) if train_food_dist else 0,
            'mean_samples': np.mean(list(train_food_dist.values())) if train_food_dist else 0,
            'std_samples': np.std(list(train_food_dist.values())) if train_food_dist else 0
        },
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'total': len(train_dataset) + len(val_dataset)
        }
    })
    
    return class_info


def create_test_dataloader(
    batch_size: int = 1,
    num_workers: int = 0,
    root_dir: str = './data/raw/food-101'
) -> DataLoader:
    """
    Create dataloader for testing/inference with minimal batching.
    
    Args:
        batch_size: Usually 1 for inference
        num_workers: Usually 0 for inference
        root_dir: Path to Food-101 dataset
        
    Returns:
        DataLoader: Test dataloader optimized for inference
    """
    
    test_transform = get_transforms('test')
    
    test_dataset = MultiTaskFoodDataset(
        root_dir=root_dir,
        split='test',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Not needed for inference
        drop_last=False   # Keep all samples for testing
    )
    
    print(f"Test dataloader: {len(test_loader)} batches")
    
    return test_loader


if __name__ == "__main__":
    # Test dataloader configurations
    print("Testing DataLoader Configurations...")
    
    try:
        # Test basic configuration
        train_loader, val_loader, class_info = create_dataloaders(
            batch_size=4,
            subset_size=20,
            num_workers=0,
            use_weighted_sampling=True
        )
        
        print(f"\nClass Info Summary:")
        print(f"Food classes: {class_info['num_food_classes']}")
        print(f"Cuisine classes: {class_info['num_cuisine_classes']}")
        print(f"Train samples: {class_info['dataset_sizes']['train']}")
        print(f"Val samples: {class_info['dataset_sizes']['val']}")
        
        # Test batch loading
        print(f"\nTesting batch loading...")
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}: {images.shape}")
            if batch_idx >= 1:
                break
        
        print("\nDataLoader test complete!")
        
    except Exception as e:
        print(f"DataLoader test failed: {e}")