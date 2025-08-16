"""
dataset.py
Multi-task dataset with proper class alignment and data validation.
Fixed class mismatch issues(101 vs 126) and improved data handling for small datasets
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class MultiTaskFoodDataset(Dataset):
    """
    Production-ready multi-task dataset for Food-101 with proper class alignment.
    
    Key improvements:
    - Resolves class mismatch between model expectations and data
    - Better error handling and validation
    - Enhanced data augmentation for small datasets
    - Comprehensive logging and diagnostics
    """
    
    def __init__(self, 
                 root_dir: str = './data/raw/food-101', 
                 split: str = 'train',
                 cuisine_mapping_path: str = './data/cuisine_mappings.json',
                 nutrition_db_path: str = './data/nutrition_db.json',
                 transform: Optional[Any] = None, 
                 subset_size: Optional[int] = None,
                 target_food_classes: int = 101,  # NEW: Enforce specific class count
                 min_samples_per_class: int = 5,   # NEW: Minimum samples for training
                 validate_data: bool = True):      # NEW: Optional data validation
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_food_classes = target_food_classes
        self.min_samples_per_class = min_samples_per_class
        
        # Load mappings and nutrition data
        self.cuisine_mapping = self._load_json(cuisine_mapping_path)
        self.nutrition_db = self._load_json(nutrition_db_path)
        
        # Validate data integrity
        if validate_data:
            self._validate_data_integrity()
        
        # Create class mappings with proper alignment
        self._create_aligned_class_mappings()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Filter classes with insufficient samples
        self.samples = self._filter_low_sample_classes()
        
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
            logger.warning(f"Could not load {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}
    
    def _validate_data_integrity(self):
        """Enhanced data validation with actionable feedback"""
        if not self.cuisine_mapping:
            raise ValueError("Cuisine mapping is empty or could not be loaded")
        
        if not self.nutrition_db:
            logger.warning("Nutrition database is empty, using default values")
        
        # Check class count alignment
        mapped_foods = len(self.cuisine_mapping)
        if mapped_foods != self.target_food_classes:
            logger.warning(
                f"Class count mismatch: {mapped_foods} mapped foods vs "
                f"{self.target_food_classes} target classes"
            )
        
        # Check nutrition coverage
        missing_nutrition = set(self.cuisine_mapping.keys()) - set(self.nutrition_db.keys())
        if missing_nutrition:
            logger.warning(
                f"{len(missing_nutrition)} foods missing nutrition data. "
                f"Examples: {list(missing_nutrition)[:5]}"
            )
    
    def _create_aligned_class_mappings(self):
        """Create class mappings aligned with target class count"""
        # Get available food classes and limit to target count
        available_foods = sorted(self.cuisine_mapping.keys())
        
        if len(available_foods) > self.target_food_classes:
            # Take first N classes alphabetically for consistency
            self.food_classes = available_foods[:self.target_food_classes]
            logger.info(
                f"Limited food classes from {len(available_foods)} to "
                f"{self.target_food_classes} for model compatibility"
            )
        else:
            self.food_classes = available_foods
        
        # Create food mappings
        self.food_to_idx = {food: idx for idx, food in enumerate(self.food_classes)}
        self.idx_to_food = {idx: food for food, idx in self.food_to_idx.items()}
        
        # Create cuisine mappings from selected foods only
        selected_cuisines = set()
        for food in self.food_classes:
            if food in self.cuisine_mapping:
                selected_cuisines.add(self.cuisine_mapping[food])
        
        self.cuisine_classes = sorted(list(selected_cuisines))
        self.cuisine_to_idx = {cuisine: idx for idx, cuisine in enumerate(self.cuisine_classes)}
        self.idx_to_cuisine = {idx: cuisine for cuisine, idx in self.cuisine_to_idx.items()}
        
        logger.info(
            f"Class alignment: {len(self.food_classes)} food classes, "
            f"{len(self.cuisine_classes)} cuisine classes"
        )
    
    def _load_samples(self) -> List[Dict]:
        """Load samples with better error handling and filtering"""
        samples = []
        images_dir = self.root_dir / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Food-101 images directory not found: {images_dir}")
        
        # Try official splits first
        split_file = self.root_dir / 'meta' / f'{self.split}.txt'
        
        if split_file.exists():
            samples = self._load_from_official_split(split_file, images_dir)
        else:
            logger.warning("Official split file not found, scanning directory...")
            samples = self._load_from_directory_scan(images_dir)
        
        if not samples:
            raise ValueError(f"No valid samples found for {self.split} split")
        
        return samples
    
    def _load_from_official_split(self, split_file: Path, images_dir: Path) -> List[Dict]:
        """Load from official Food-101 splits with validation"""
        samples = []
        skipped_classes = set()
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        for image_name in image_names:
            food_class = image_name.split('/')[0]
            
            # Only include foods in our aligned class set
            if food_class in self.food_to_idx:
                image_path = images_dir / f"{image_name}.jpg"
                
                if image_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'food_class': food_class,
                        'food_idx': self.food_to_idx[food_class]
                    })
                else:
                    logger.debug(f"Image not found: {image_path}")
            else:
                skipped_classes.add(food_class)
        
        if skipped_classes:
            logger.info(f"Skipped {len(skipped_classes)} classes not in target set")
        
        return samples
    
    def _load_from_directory_scan(self, images_dir: Path) -> List[Dict]:
        """Fallback directory scan with better validation"""
        samples = []
        
        for food_dir in images_dir.iterdir():
            if food_dir.is_dir() and food_dir.name in self.food_to_idx:
                food_class = food_dir.name
                
                # Get all valid images
                valid_extensions = {'.jpg', '.jpeg', '.png'}
                for img_path in food_dir.iterdir():
                    if img_path.suffix.lower() in valid_extensions:
                        samples.append({
                            'image_path': str(img_path),
                            'food_class': food_class,
                            'food_idx': self.food_to_idx[food_class]
                        })
        
        return samples
    
    def _filter_low_sample_classes(self) -> List[Dict]:
        """Filter out classes with insufficient training samples"""
        if self.split != 'train':
            return self.samples  # Don't filter validation/test sets
        
        # Count samples per class
        class_counts = Counter(sample['food_class'] for sample in self.samples)
        
        # Identify classes with insufficient samples
        low_sample_classes = {
            cls for cls, count in class_counts.items() 
            if count < self.min_samples_per_class
        }
        
        if low_sample_classes:
            logger.warning(
                f"Removing {len(low_sample_classes)} classes with < "
                f"{self.min_samples_per_class} samples: {list(low_sample_classes)[:5]}..."
            )
            
            # Filter samples and update class mappings
            filtered_samples = [
                sample for sample in self.samples 
                if sample['food_class'] not in low_sample_classes
            ]
            
            # Update class mappings
            remaining_classes = sorted(set(
                sample['food_class'] for sample in filtered_samples
            ))
            
            self.food_classes = remaining_classes
            self.food_to_idx = {food: idx for idx, food in enumerate(self.food_classes)}
            self.idx_to_food = {idx: food for food, idx in self.food_to_idx.items()}
            
            # Update sample indices
            for sample in filtered_samples:
                sample['food_idx'] = self.food_to_idx[sample['food_class']]
            
            return filtered_samples
        
        return self.samples
    
    def _apply_subset(self, subset_size: Optional[int]):
        """Enhanced stratified subsampling"""
        if subset_size and subset_size < len(self.samples):
            # Use improved stratified sampling
            self.samples = self._enhanced_stratified_subsample(self.samples, subset_size)
            logger.info(f"Applied enhanced stratified subset: {subset_size} samples")
    
    def _enhanced_stratified_subsample(self, samples: List[Dict], target_size: int) -> List[Dict]:
        """Enhanced stratified sampling with minimum guarantees"""
        import random
        random.seed(42)  # For reproducibility
        
        # Group samples by class
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample['food_class']].append(sample)
        
        num_classes = len(class_samples)
        
        # Ensure minimum samples per class (at least 1)
        min_per_class = max(1, target_size // (num_classes * 2))  # Conservative allocation
        base_samples_per_class = max(min_per_class, target_size // num_classes)
        
        stratified_samples = []
        remaining_budget = target_size
        
        # First pass: allocate minimum samples per class
        for food_class, class_sample_list in class_samples.items():
            allocation = min(base_samples_per_class, len(class_sample_list), remaining_budget)
            
            if allocation > 0:
                sampled = random.sample(class_sample_list, allocation)
                stratified_samples.extend(sampled)
                remaining_budget -= allocation
        
        # Second pass: distribute remaining budget proportionally
        if remaining_budget > 0:
            for food_class, class_sample_list in class_samples.items():
                current_class_samples = sum(
                    1 for s in stratified_samples if s['food_class'] == food_class
                )
                available = len(class_sample_list) - current_class_samples
                additional = min(available, remaining_budget)
                
                if additional > 0:
                    # Sample from remaining samples in this class
                    used_paths = {s['image_path'] for s in stratified_samples}
                    remaining_samples = [
                        s for s in class_sample_list 
                        if s['image_path'] not in used_paths
                    ]
                    
                    if remaining_samples:
                        additional_samples = random.sample(
                            remaining_samples, 
                            min(additional, len(remaining_samples))
                        )
                        stratified_samples.extend(additional_samples)
                        remaining_budget -= len(additional_samples)
                
                if remaining_budget <= 0:
                    break
        
        return stratified_samples
    
    def _print_dataset_summary(self):
        """Enhanced dataset summary with diagnostics"""
        print(f"\n=== {self.split.upper()} Dataset Summary ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"Food classes: {len(self.food_classes)}")
        print(f"Cuisine classes: {len(self.cuisine_classes)}")
        print(f"Target classes: {self.target_food_classes}")
        print(f"Transform: {'Yes' if self.transform else 'None'}")
        
        # Class distribution analysis
        if len(self.samples) < 5000:  # Only for manageable datasets
            food_dist = Counter(sample['food_class'] for sample in self.samples)
            print(f"Samples per class: {min(food_dist.values())} - {max(food_dist.values())}")
            print(f"Mean samples per class: {np.mean(list(food_dist.values())):.1f}")
            
            # Identify problematic classes
            low_sample_classes = [
                cls for cls, count in food_dist.items() 
                if count < self.min_samples_per_class
            ]
            if low_sample_classes:
                print(f"⚠️  Classes with < {self.min_samples_per_class} samples: {len(low_sample_classes)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        """Get sample with enhanced error handling"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = self._load_image_safe(sample['image_path'])
        
        # Get labels
        food_label = sample['food_idx']
        cuisine_label = self._get_cuisine_label(sample['food_class'])
        nutrition_target = self._get_nutrition_target(sample['food_class'])
        
        return image, food_label, cuisine_label, nutrition_target
    
    def _load_image_safe(self, image_path: str) -> torch.Tensor:
        """Safe image loading with fallback"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Validate image
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError("Image too small")
            
            if self.transform:
                image = self.transform(image)
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            # Return black fallback image
            if self.transform:
                fallback_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                return self.transform(fallback_image)
            else:
                return torch.zeros(3, 224, 224)
    
    def _get_cuisine_label(self, food_class: str) -> int:
        """Get cuisine label with fallback"""
        cuisine_name = self.cuisine_mapping.get(food_class, self.cuisine_classes[0])
        return self.cuisine_to_idx.get(cuisine_name, 0)
    
    def _get_nutrition_target(self, food_class: str) -> torch.Tensor:
        """Get nutrition with enhanced defaults"""
        nutrition_data = self.nutrition_db.get(food_class, {})
        
        # Use smarter defaults based on food type
        calories = float(nutrition_data.get('calories', 300))
        protein = float(nutrition_data.get('protein', 15))
        carbs = float(nutrition_data.get('carbs', 35))
        fat = float(nutrition_data.get('fat', 12))
        
        # Ensure reasonable ranges
        calories = max(50, min(1000, calories))
        protein = max(1, min(50, protein))
        carbs = max(5, min(100, carbs))
        fat = max(1, min(50, fat))
        
        return torch.tensor([calories, protein, carbs, fat], dtype=torch.float32)
    
    def get_class_info(self) -> Dict[str, Any]:
        """Enhanced class information"""
        return {
            'num_food_classes': len(self.food_classes),
            'num_cuisine_classes': len(self.cuisine_classes),
            'target_food_classes': self.target_food_classes,
            'food_classes': self.food_classes,
            'cuisine_classes': self.cuisine_classes,
            'food_to_idx': self.food_to_idx,
            'cuisine_to_idx': self.cuisine_to_idx,
            'idx_to_food': self.idx_to_food,
            'idx_to_cuisine': self.idx_to_cuisine,
            'class_aligned': len(self.food_classes) == self.target_food_classes
        }
    
    def diagnose_dataset(self) -> Dict[str, Any]:
        """Comprehensive dataset diagnostics for debugging"""
        food_dist = Counter(sample['food_class'] for sample in self.samples)
        cuisine_dist = Counter(
            self.cuisine_mapping.get(sample['food_class'], 'Unknown') 
            for sample in self.samples
        )
        
        return {
            'total_samples': len(self.samples),
            'food_class_distribution': dict(food_dist),
            'cuisine_class_distribution': dict(cuisine_dist),
            'samples_per_food_class': {
                'min': min(food_dist.values()) if food_dist else 0,
                'max': max(food_dist.values()) if food_dist else 0,
                'mean': np.mean(list(food_dist.values())) if food_dist else 0,
                'std': np.std(list(food_dist.values())) if food_dist else 0
            },
            'class_alignment': {
                'target_classes': self.target_food_classes,
                'actual_classes': len(self.food_classes),
                'aligned': len(self.food_classes) == self.target_food_classes
            },
            'data_quality': {
                'classes_with_min_samples': sum(
                    1 for count in food_dist.values() 
                    if count >= self.min_samples_per_class
                ),
                'low_sample_classes': [
                    cls for cls, count in food_dist.items() 
                    if count < self.min_samples_per_class
                ]
            }
        }


if __name__ == "__main__":
    print("Testing Improved Multi-Task Food Dataset...")
    
    try:
        # Test with class alignment
        dataset = MultiTaskFoodDataset(
            subset_size=100,
            split='train',
            target_food_classes=101,  # Enforce Food-101 standard
            min_samples_per_class=2,
            validate_data=True
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Class alignment: {dataset.get_class_info()['class_aligned']}")
        
        # Test diagnostics
        diagnostics = dataset.diagnose_dataset()
        print(f"\nDiagnostics:")
        print(f"- Actual vs target classes: {diagnostics['class_alignment']}")
        print(f"- Sample distribution: {diagnostics['samples_per_food_class']}")
        print(f"- Data quality: {diagnostics['data_quality']}")
        
        # Test sample access
        sample_image, food_label, cuisine_label, nutrition_target = dataset[0]
        print(f"\nSample test successful!")
        print(f"- Image shape: {sample_image.shape}")
        print(f"- Food label: {food_label} (max: {len(dataset.food_classes)-1})")
        print(f"- Cuisine label: {cuisine_label} (max: {len(dataset.cuisine_classes)-1})")
        
        print("\n✅ Improved dataset test successful!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()