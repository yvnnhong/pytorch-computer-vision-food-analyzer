"""
Multi-task dataset class for Food-101 with proper separation of concerns.
Focused solely on dataset logic without transforms or dataloader (see data_loaders.py 
and transforms.py)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any


class MultiTaskFoodDataset(Dataset):
    """
    Multi-task dataset for Food-101 that provides:
    1. Food classification labels (101 classes)
    2. Cuisine classification labels (based on mappings)  
    3. Nutrition regression targets (calories, protein, carbs, fat)
    
    This class handles only core dataset functionality - image loading and label creation.
    Transforms and DataLoader configurations are handled by separate modules.
    """
    
    def __init__(self, 
                 root_dir: str = './data/raw/food-101', 
                 split: str = 'train',
                 cuisine_mapping_path: str = './data/cuisine_mappings.json',
                 nutrition_db_path: str = './data/nutrition_db.json',
                 transform: Optional[Any] = None, 
                 subset_size: Optional[int] = None):
        """
        Initialize the multi-task food dataset.
        
        Args:
            root_dir: Path to Food-101 dataset
            split: 'train' or 'test'
            cuisine_mapping_path: Path to food->cuisine mapping JSON
            nutrition_db_path: Path to nutrition database JSON
            transform: Transform pipeline to apply to images
            subset_size: If specified, only use this many samples (for quick testing)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load mappings and nutrition data
        self.cuisine_mapping = self._load_json(cuisine_mapping_path)
        self.nutrition_db = self._load_json(nutrition_db_path)
        
        # Validate data integrity
        self._validate_data_integrity()
        
        # Create class mappings
        self._create_class_mappings()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Apply subset if specified
        self._apply_subset(subset_size)
        
        # Print dataset summary
        self._print_dataset_summary()
    
    def _load_json(self, path: str) -> Dict:
        """Load JSON file with comprehensive error handling"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if not data:
                raise ValueError(f"Empty JSON file: {path}")
            return data
        except FileNotFoundError:
            print(f"Warning: Could not load {path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {path}: {e}")
            return {}
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return {}
    
    def _validate_data_integrity(self):
        """Validate that required data files and mappings are consistent"""
        if not self.cuisine_mapping:
            raise ValueError("Cuisine mapping is empty or could not be loaded")
        
        if not self.nutrition_db:
            print("Warning: Nutrition database is empty, using default values")
        
        # Check that nutrition DB covers all foods in cuisine mapping
        missing_nutrition = set(self.cuisine_mapping.keys()) - set(self.nutrition_db.keys())
        if missing_nutrition:
            print(f"Warning: {len(missing_nutrition)} foods missing nutrition data")
    
    def _create_class_mappings(self):
        """Create bidirectional mappings between class names and indices"""
        # Food class mappings
        self.food_classes = sorted(self.cuisine_mapping.keys())
        self.food_to_idx = {food: idx for idx, food in enumerate(self.food_classes)}
        self.idx_to_food = {idx: food for food, idx in self.food_to_idx.items()}
        
        # Cuisine class mappings
        self.cuisine_classes = sorted(list(set(self.cuisine_mapping.values())))
        self.cuisine_to_idx = {cuisine: idx for idx, cuisine in enumerate(self.cuisine_classes)}
        self.idx_to_cuisine = {idx: cuisine for cuisine, idx in self.cuisine_to_idx.items()}
        
        # Validate mappings
        assert len(self.food_classes) == len(self.food_to_idx), "Food mapping inconsistency"
        assert len(self.cuisine_classes) == len(self.cuisine_to_idx), "Cuisine mapping inconsistency"
    
    def _load_samples(self) -> list:
        """Load all image paths and create corresponding labels"""
        samples = []
        images_dir = self.root_dir / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Food-101 images directory not found: {images_dir}")
        
        # Try to use official Food-101 splits first
        split_file = self.root_dir / 'meta' / f'{self.split}.txt'
        
        if split_file.exists():
            samples = self._load_from_official_split(split_file, images_dir)
        else:
            print(f"Official split file not found, scanning directory structure...")
            samples = self._load_from_directory_scan(images_dir)
        
        if not samples:
            raise ValueError(f"No valid samples found for {self.split} split")
        
        return samples
    
    def _load_from_official_split(self, split_file: Path, images_dir: Path) -> list:
        """Load samples using official Food-101 train/test split files"""
        samples = []
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        for image_name in image_names:
            food_class = image_name.split('/')[0]
            
            # Only include foods that we have mappings for
            if food_class in self.food_to_idx:
                image_path = images_dir / f"{image_name}.jpg"
                
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'food_class': food_class,
                        'food_idx': self.food_to_idx[food_class]
                    })
                else:
                    print(f"Warning: Image not found: {image_path}")
        
        return samples
    
    def _load_from_directory_scan(self, images_dir: Path) -> list:
        """Fallback: Load samples by scanning directory structure"""
        samples = []
        
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
    
    def _apply_subset(self, subset_size: Optional[int]):
        """Apply subset limitation if specified"""
        if subset_size and subset_size < len(self.samples):
            # Stratified sampling to maintain class distribution
            self.samples = self._stratified_subsample(self.samples, subset_size)
            print(f"Applied stratified subset: {subset_size} samples for {self.split}")
    
    def _stratified_subsample(self, samples: list, target_size: int) -> list:
        """Create stratified subsample maintaining class distribution"""
        from collections import defaultdict
        import random
        
        # Group samples by food class
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample['food_class']].append(sample)
        
        # Calculate samples per class
        num_classes = len(class_samples)
        base_samples_per_class = target_size // num_classes
        remaining_samples = target_size % num_classes
        
        stratified_samples = []
        classes = list(class_samples.keys())
        random.shuffle(classes)
        
        for i, food_class in enumerate(classes):
            class_sample_list = class_samples[food_class]
            
            # Some classes get one extra sample
            samples_for_this_class = base_samples_per_class
            if i < remaining_samples:
                samples_for_this_class += 1
            
            # Sample from this class
            if len(class_sample_list) >= samples_for_this_class:
                sampled = random.sample(class_sample_list, samples_for_this_class)
            else:
                sampled = class_sample_list  # Use all available
            
            stratified_samples.extend(sampled)
        
        return stratified_samples
    
    def _print_dataset_summary(self):
        """Print comprehensive dataset summary"""
        print(f"\n=== {self.split.upper()} Dataset Summary ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"Food classes: {len(self.food_classes)}")
        print(f"Cuisine classes: {len(self.cuisine_classes)}")
        print(f"Transform: {'Yes' if self.transform else 'None'}")
        
        # Class distribution summary
        if len(self.samples) < 10000:  # Only for smaller datasets
            from collections import Counter
            food_dist = Counter(sample['food_class'] for sample in self.samples)
            print(f"Samples per class: {min(food_dist.values())} - {max(food_dist.values())}")
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        """
        Get a single sample with multi-task labels.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, food_label, cuisine_label, nutrition_target)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = self._load_image(sample['image_path'])
        
        # Get labels
        food_label = sample['food_idx']
        cuisine_label = self._get_cuisine_label(sample['food_class'])
        nutrition_target = self._get_nutrition_target(sample['food_class'])
        
        return image, food_label, cuisine_label, nutrition_target
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image with error handling"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default: convert to tensor
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            if self.transform:
                # Try to infer expected size from transform
                fallback_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                return self.transform(fallback_image)
            else:
                return torch.zeros(3, 224, 224)
    
    def _get_cuisine_label(self, food_class: str) -> int:
        """Get cuisine label for a food class"""
        cuisine_name = self.cuisine_mapping.get(food_class, 'Fusion')
        return self.cuisine_to_idx.get(cuisine_name, 0)  # Default to first class
    
    def _get_nutrition_target(self, food_class: str) -> torch.Tensor:
        """Get nutrition target values for a food class"""
        nutrition_data = self.nutrition_db.get(food_class, {
            'calories': 250, 'protein': 15, 'carbs': 30, 'fat': 10  # Default values
        })
        
        return torch.tensor([
            float(nutrition_data.get('calories', 250)),
            float(nutrition_data.get('protein', 15)),
            float(nutrition_data.get('carbs', 30)),
            float(nutrition_data.get('fat', 10))
        ], dtype=torch.float32)
    
    def get_class_info(self) -> Dict[str, Any]:
        """Return comprehensive information about dataset classes"""
        return {
            'num_food_classes': len(self.food_classes),
            'num_cuisine_classes': len(self.cuisine_classes),
            'food_classes': self.food_classes,
            'cuisine_classes': self.cuisine_classes,
            'food_to_idx': self.food_to_idx,
            'cuisine_to_idx': self.cuisine_to_idx,
            'idx_to_food': self.idx_to_food,
            'idx_to_cuisine': self.idx_to_cuisine
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.samples[idx]
        food_class = sample['food_class']
        
        return {
            'index': idx,
            'image_path': sample['image_path'],
            'food_class': food_class,
            'food_idx': sample['food_idx'],
            'cuisine_class': self.cuisine_mapping.get(food_class, 'Unknown'),
            'cuisine_idx': self._get_cuisine_label(food_class),
            'nutrition_data': self.nutrition_db.get(food_class, {})
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset integrity and return diagnostic information"""
        validation_results = {
            'total_samples': len(self.samples),
            'accessible_images': 0,
            'missing_images': 0,
            'corrupted_images': 0,
            'class_distribution': {},
            'errors': []
        }
        
        from collections import Counter
        food_dist = Counter()
        
        print("Validating dataset...")
        for i, sample in enumerate(self.samples):
            if i % 1000 == 0:
                print(f"Validated {i}/{len(self.samples)} samples")
            
            try:
                # Check if image exists and can be loaded
                image_path = sample['image_path']
                if not os.path.exists(image_path):
                    validation_results['missing_images'] += 1
                    validation_results['errors'].append(f"Missing: {image_path}")
                    continue
                
                # Try to load image
                Image.open(image_path).convert('RGB')
                validation_results['accessible_images'] += 1
                food_dist[sample['food_class']] += 1
                
            except Exception as e:
                validation_results['corrupted_images'] += 1
                validation_results['errors'].append(f"Corrupted {image_path}: {e}")
        
        validation_results['class_distribution'] = dict(food_dist)
        
        print(f"Dataset validation complete:")
        print(f"  Accessible: {validation_results['accessible_images']}")
        print(f"  Missing: {validation_results['missing_images']}")
        print(f"  Corrupted: {validation_results['corrupted_images']}")
        
        return validation_results


if __name__ == "__main__":
    # Test the clean dataset implementation
    print("Testing Clean Multi-Task Food Dataset...")
    
    try:
        # Test basic functionality
        dataset = MultiTaskFoodDataset(
            subset_size=20,
            split='train'
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Sample count: {len(dataset)}")
        
        # Test sample access
        sample_image, food_label, cuisine_label, nutrition_target = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {sample_image.shape}")
        print(f"  Food label: {food_label}")
        print(f"  Cuisine label: {cuisine_label}")
        print(f"  Nutrition: {nutrition_target}")
        
        # Test class info
        class_info = dataset.get_class_info()
        print(f"\nClass info:")
        print(f"  Food classes: {class_info['num_food_classes']}")
        print(f"  Cuisine classes: {class_info['num_cuisine_classes']}")
        
        # Test sample info
        sample_info = dataset.get_sample_info(0)
        print(f"\nSample 0 detailed info:")
        print(f"  Food: {sample_info['food_class']}")
        print(f"  Cuisine: {sample_info['cuisine_class']}")
        
        print("\nClean dataset test successful!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()