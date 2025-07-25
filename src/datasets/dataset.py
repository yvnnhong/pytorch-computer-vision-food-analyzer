# Multi-task PyTorch Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path

class MultiTaskFoodDataset(Dataset):
    """
    Multi-task dataset for Food-101 that provides:
    1. Food classification labels (101 classes)
    2. Cuisine classification labels (based on mappings)  
    3. Nutrition regression targets (calories, protein, carbs, fat)
    
    Loads images from Food-101 and creates corresponding multi-task labels.
    """
    
    def __init__(self, root_dir='./data/raw/food-101', split='train', 
                 cuisine_mapping_path='./data/cuisine_mappings.json',
                 nutrition_db_path='./data/nutrition_db.json',
                 transform=None, subset_size=None):
        """
        Args:
            root_dir: Path to Food-101 dataset
            split: 'train' or 'test'
            cuisine_mapping_path: Path to food->cuisine mapping JSON
            nutrition_db_path: Path to nutrition database JSON
            transform: Image transforms to apply
            subset_size: If specified, only use this many samples (for quick testing)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load mappings and nutrition data
        self.cuisine_mapping = self._load_json(cuisine_mapping_path)
        self.nutrition_db = self._load_json(nutrition_db_path)
        
        # Create class mappings
        self.food_classes = sorted(self.cuisine_mapping.keys())
        self.cuisine_classes = sorted(list(set(self.cuisine_mapping.values())))
        
        self.food_to_idx = {food: idx for idx, food in enumerate(self.food_classes)}
        self.cuisine_to_idx = {cuisine: idx for idx, cuisine in enumerate(self.cuisine_classes)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Apply subset if specified (useful for quick testing)
        if subset_size and subset_size < len(self.samples):
            self.samples = self.samples[:subset_size]
            print(f"Using subset of {subset_size} samples for {split}")
        
        print(f"Loaded {len(self.samples)} {split} samples")
        print(f"Food classes: {len(self.food_classes)}")
        print(f"Cuisine classes: {len(self.cuisine_classes)}")
    
    def _load_json(self, path):
        """Load JSON file with error handling"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not load {path}")
            return {}
    
    def _load_samples(self):
        """Load all image paths and create labels"""
        samples = []
        
        # Food-101 structure: images/food_class/image_name.jpg
        images_dir = self.root_dir / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Food-101 images directory not found: {images_dir}")
        
        # Load based on Food-101 train/test split files
        split_file = self.root_dir / 'meta' / f'{self.split}.txt'
        
        if split_file.exists():
            # Use official Food-101 splits
            with open(split_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            
            for image_name in image_names:
                food_class = image_name.split('/')[0]
                if food_class in self.food_to_idx:
                    image_path = images_dir / f"{image_name}.jpg"
                    if image_path.exists():
                        samples.append({
                            'image_path': str(image_path),
                            'food_class': food_class,
                            'food_idx': self.food_to_idx[food_class]
                        })
        else:
            # Fallback: scan directory structure
            print(f"Split file not found, scanning directory...")
            for food_dir in images_dir.iterdir():
                if food_dir.is_dir() and food_dir.name in self.food_to_idx:
                    food_class = food_dir.name
                    for img_path in food_dir.glob("*.jpg"):
                        samples.append({
                            'image_path': str(img_path),
                            'food_class': food_class,
                            'food_idx': self.food_to_idx[food_class]
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample with multi-task labels.
        
        Returns:
            tuple: (image_tensor, food_label, cuisine_label, nutrition_target)
        """
        sample = self.samples[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, 224, 224)
        
        # Get food label
        food_label = sample['food_idx']
        food_class = sample['food_class']
        
        # Get cuisine label
        cuisine_name = self.cuisine_mapping.get(food_class, 'Fusion')  # Default to 'Fusion'
        cuisine_label = self.cuisine_to_idx.get(cuisine_name, 0)  # Default to first class
        
        # Get nutrition targets
        nutrition_data = self.nutrition_db.get(food_class, {
            'calories': 250, 'protein': 15, 'carbs': 30, 'fat': 10
        })
        
        nutrition_target = torch.tensor([
            nutrition_data['calories'],
            nutrition_data['protein'], 
            nutrition_data['carbs'],
            nutrition_data['fat']
        ], dtype=torch.float32)
        
        return image, food_label, cuisine_label, nutrition_target
    
    def get_class_info(self):
        """Return information about classes for model initialization"""
        return {
            'num_food_classes': len(self.food_classes),
            'num_cuisine_classes': len(self.cuisine_classes),
            'food_classes': self.food_classes,
            'cuisine_classes': self.cuisine_classes
        }


def get_transforms(split='train', input_size=224):
    """
    Get image transforms for training/validation.
    
    Args:
        split: 'train' or 'val'/'test'
        input_size: Target image size
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    
    if split == 'train':
        # Training transforms with data augmentation
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),  # Slightly larger
            transforms.RandomCrop(input_size),                       # Random crop
            transforms.RandomHorizontalFlip(p=0.5),                 # 50% chance flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2,    # Color variation
                                 saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),                  # Slight rotation
            transforms.ToTensor(),                                  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet normalization
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(batch_size=32, num_workers=4, subset_size=None, 
                      root_dir='./data/raw/food-101'):
    """
    Create training and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        subset_size: If specified, use subset for quick testing
        root_dir: Path to Food-101 dataset
        
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
        subset_size=subset_size//4 if subset_size else None  # Smaller val set
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class information
    class_info = train_dataset.get_class_info()
    
    print(f"\nDataLoader Summary:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Total train samples: {len(train_dataset)}")
    print(f"Total val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, class_info


if __name__ == "__main__":
    # Quick test of the dataset
    print("Testing Multi-Task Food Dataset...")
    
    # Test with small subset for speed
    try:
        train_loader, val_loader, class_info = create_dataloaders(
            batch_size=4, 
            subset_size=20,  # Very small for testing
            num_workers=0    # Avoid multiprocessing issues on some systems
        )
        
        print(f"\nClass Info:")
        print(f"Food classes: {class_info['num_food_classes']}")
        print(f"Cuisine classes: {class_info['num_cuisine_classes']}")
        
        # Test loading one batch
        print(f"\nTesting batch loading...")
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Food labels: {food_labels}")
            print(f"  Cuisine labels: {cuisine_labels}")
            print(f"  Nutrition targets shape: {nutrition_targets.shape}")
            print(f"  Sample nutrition: {nutrition_targets[0]}")
            
            if batch_idx >= 2:  # Only test a few batches
                break
        
        print("\nDataset test complete!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("Make sure Food-101 dataset is downloaded in ./data/raw/food-101/")