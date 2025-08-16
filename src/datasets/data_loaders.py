"""
data_loaders.py
Enhanced DataLoader configurations optimized for small dataset training.
Includes advanced sampling strategies and batch optimization.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import Counter, defaultdict
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.dataset import MultiTaskFoodDataset
from src.datasets.transforms import get_transforms

logger = logging.getLogger(__name__)


class SmartBatchSampler:
    """
    Smart batch sampler that ensures balanced representation for small datasets.
    Addresses the challenge of having only 1-3 samples per class.
    """
    
    def __init__(self, dataset: MultiTaskFoodDataset, batch_size: int = 32, 
                 strategy: str = 'balanced'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        
        # Analyze dataset structure
        self.class_indices = self._build_class_indices()
        self.num_classes = len(self.class_indices)
        
        # Calculate sampling weights
        self.sample_weights = self._calculate_smart_weights()
        
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping from class labels to sample indices"""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            _, food_label, _, _ = self.dataset[idx]
            class_indices[food_label].append(idx)
        
        return dict(class_indices)
    
    def _calculate_smart_weights(self) -> List[float]:
        """Calculate sampling weights optimized for small datasets"""
        sample_weights = []
        
        # Count samples per class
        class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        total_samples = sum(class_counts.values())
        
        # For very small datasets, use more aggressive balancing
        if total_samples < 1000:
            # Use inverse square root for less aggressive balancing
            max_count = max(class_counts.values())
            for idx in range(len(self.dataset)):
                _, food_label, _, _ = self.dataset[idx]
                class_count = class_counts[food_label]
                # Less aggressive weight calculation for small datasets
                weight = np.sqrt(max_count / class_count)
                sample_weights.append(weight)
        else:
            # Standard inverse frequency weighting for larger datasets
            for idx in range(len(self.dataset)):
                _, food_label, _, _ = self.dataset[idx]
                class_count = class_counts[food_label]
                weight = total_samples / (self.num_classes * class_count)
                sample_weights.append(weight)
        
        return sample_weights
    
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler with optimized parameters"""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True  # Essential for small datasets
        )


def create_enhanced_dataloaders(
    batch_size: int = 32, 
    num_workers: int = 4, 
    subset_size: Optional[int] = None,
    root_dir: str = './data/raw/food-101',
    sampling_strategy: str = 'smart_weighted',
    pin_memory: bool = True,
    persistent_workers: bool = True,
    target_food_classes: int = 101,
    augmentation_strength: str = 'medium'
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create enhanced dataloaders optimized for small dataset training.
    
    Args:
        batch_size: Training batch size
        num_workers: Number of data loading workers
        subset_size: Optional subset size for testing
        root_dir: Path to Food-101 dataset
        sampling_strategy: 'smart_weighted', 'standard', 'class_balanced'
        pin_memory: Pin memory for GPU transfer
        persistent_workers: Keep workers alive between epochs
        target_food_classes: Target number of food classes (for alignment)
        augmentation_strength: 'light', 'medium', 'heavy'
        
    Returns:
        tuple: (train_loader, val_loader, comprehensive_info)
    """
    
    # Get optimized transforms
    train_transform = get_transforms('train', augmentation_strength=augmentation_strength)
    val_transform = get_transforms('val')
    
    # Create enhanced datasets with class alignment
    train_dataset = MultiTaskFoodDataset(
        root_dir=root_dir,
        split='train', 
        transform=train_transform,
        subset_size=subset_size,
        target_food_classes=target_food_classes,
        min_samples_per_class=2,  # Minimum for training
        validate_data=True
    )
    
    val_dataset = MultiTaskFoodDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        subset_size=subset_size//4 if subset_size else None,
        target_food_classes=target_food_classes,
        min_samples_per_class=1,  # More lenient for validation
        validate_data=False  # Skip validation for speed
    )
    
    # Create smart sampler based on strategy
    train_sampler = None
    shuffle_train = True
    
    if sampling_strategy == 'smart_weighted':
        smart_sampler = SmartBatchSampler(
            train_dataset, 
            batch_size=batch_size, 
            strategy='balanced'
        )
        train_sampler = smart_sampler.create_weighted_sampler()
        shuffle_train = False
        logger.info("Using smart weighted sampling for class balance")
        
    elif sampling_strategy == 'class_balanced':
        train_sampler = create_class_balanced_sampler(train_dataset)
        shuffle_train = False
        logger.info("Using class-balanced sampling")
    
    # Optimize batch size for small datasets
    effective_batch_size = min(batch_size, len(train_dataset) // 4)
    if effective_batch_size != batch_size:
        logger.warning(
            f"Reduced batch size from {batch_size} to {effective_batch_size} "
            f"for small dataset ({len(train_dataset)} samples)"
        )
        batch_size = effective_batch_size
    
    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Important for batch norm stability
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn  # For reproducibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 2),  # Fewer workers for validation
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,  # Keep all validation samples
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Comprehensive information for monitoring
    comprehensive_info = create_comprehensive_info(
        train_dataset, val_dataset, train_loader, val_loader, sampling_strategy
    )
    
    # Print summary
    print_dataloader_summary(comprehensive_info)
    
    return train_loader, val_loader, comprehensive_info


def create_class_balanced_sampler(dataset: MultiTaskFoodDataset) -> SubsetRandomSampler:
    """Create sampler that ensures equal representation of each class"""
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, food_label, _, _ = dataset[idx]
        class_indices[food_label].append(idx)
    
    # Calculate samples per class for balanced representation
    min_samples_per_class = min(len(indices) for indices in class_indices.values())
    samples_per_class = max(1, min_samples_per_class)  # At least 1 sample per class
    
    # Sample equally from each class
    balanced_indices = []
    for class_indices_list in class_indices.values():
        if len(class_indices_list) >= samples_per_class:
            # Random sample without replacement
            selected = np.random.choice(
                class_indices_list, 
                size=samples_per_class, 
                replace=False
            ).tolist()
        else:
            # Use all available samples
            selected = class_indices_list
        balanced_indices.extend(selected)
    
    np.random.shuffle(balanced_indices)
    return SubsetRandomSampler(balanced_indices)


def _worker_init_fn(worker_id: int):
    """Initialize worker with different random seed for reproducibility"""
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def create_comprehensive_info(
    train_dataset: MultiTaskFoodDataset,
    val_dataset: MultiTaskFoodDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sampling_strategy: str
) -> Dict[str, Any]:
    """Create comprehensive information about the data loading setup"""
    
    # Get dataset diagnostics
    train_diagnostics = train_dataset.diagnose_dataset()
    val_diagnostics = val_dataset.diagnose_dataset()
    
    # Calculate data loading efficiency metrics
    train_efficiency = {
        'samples_per_epoch': len(train_loader) * train_loader.batch_size,
        'batches_per_epoch': len(train_loader),
        'effective_dataset_size': len(train_dataset),
        'data_utilization': min(1.0, (len(train_loader) * train_loader.batch_size) / len(train_dataset))
    }
    
    return {
        # Dataset information
        'train_dataset_info': {
            'total_samples': len(train_dataset),
            'num_food_classes': train_diagnostics['class_alignment']['actual_classes'],
            'num_cuisine_classes': len(train_dataset.cuisine_classes),
            'class_aligned': train_diagnostics['class_alignment']['aligned'],
            'samples_per_class': train_diagnostics['samples_per_food_class']
        },
        'val_dataset_info': {
            'total_samples': len(val_dataset),
            'num_food_classes': val_diagnostics['class_alignment']['actual_classes'],
            'num_cuisine_classes': len(val_dataset.cuisine_classes)
        },
        
        # DataLoader configuration
        'dataloader_config': {
            'train_batch_size': train_loader.batch_size,
            'val_batch_size': val_loader.batch_size,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'num_workers': train_loader.num_workers,
            'sampling_strategy': sampling_strategy,
            'pin_memory': train_loader.pin_memory
        },
        
        # Training efficiency metrics
        'efficiency_metrics': train_efficiency,
        
        # Data quality indicators
        'data_quality': {
            'train_class_balance': train_diagnostics['data_quality'],
            'val_class_balance': val_diagnostics['data_quality'],
            'potential_issues': _identify_potential_issues(train_diagnostics, val_diagnostics)
        },
        
        # Raw diagnostics for debugging
        'raw_diagnostics': {
            'train': train_diagnostics,
            'val': val_diagnostics
        }
    }


def _identify_potential_issues(train_diag: Dict, val_diag: Dict) -> List[str]:
    """Identify potential training issues from diagnostics"""
    issues = []
    
    # Check for severe class imbalance
    train_stats = train_diag['samples_per_food_class']
    if train_stats['max'] > train_stats['min'] * 10:
        issues.append(
            f"Severe class imbalance: {train_stats['min']}-{train_stats['max']} samples per class"
        )
    
    # Check for insufficient training data
    if train_diag['total_samples'] < 1000:
        issues.append(f"Small training set: {train_diag['total_samples']} samples")
    
    # Check for classes with very few samples
    low_sample_classes = len(train_diag['data_quality']['low_sample_classes'])
    if low_sample_classes > 0:
        issues.append(f"{low_sample_classes} classes have insufficient samples")
    
    # Check class alignment
    if not train_diag['class_alignment']['aligned']:
        issues.append(
            f"Class count mismatch: {train_diag['class_alignment']['actual_classes']} "
            f"vs {train_diag['class_alignment']['target_classes']} target"
        )
    
    return issues


def print_dataloader_summary(info: Dict[str, Any]):
    """Print comprehensive dataloader summary"""
    print(f"\n{'='*60}")
    print(f"ENHANCED DATALOADER SUMMARY")
    print(f"{'='*60}")
    
    # Dataset info
    train_info = info['train_dataset_info']
    val_info = info['val_dataset_info']
    print(f"\nDataset Information:")
    print(f"  Train: {train_info['total_samples']} samples, {train_info['num_food_classes']} food classes")
    print(f"  Val:   {val_info['total_samples']} samples, {val_info['num_food_classes']} food classes")
    print(f"  Class alignment: {'Aligned!' if train_info['class_aligned'] else 'NOT Aligned!'}")
    
    # Class distribution
    stats = train_info['samples_per_class']
    print(f"  Samples per class: {stats['min']:.0f}-{stats['max']:.0f} (μ={stats['mean']:.1f}, σ={stats['std']:.1f})")
    
    # DataLoader config
    config = info['dataloader_config']
    print(f"\nDataLoader Configuration:")
    print(f"  Batch size: {config['train_batch_size']} (train), {config['val_batch_size']} (val)")
    print(f"  Batches per epoch: {config['train_batches']} (train), {config['val_batches']} (val)")
    print(f"  Workers: {config['num_workers']}, Pin memory: {config['pin_memory']}")
    print(f"  Sampling strategy: {config['sampling_strategy']}")
    
    # Efficiency metrics
    efficiency = info['efficiency_metrics']
    print(f"\nTraining Efficiency:")
    print(f"  Data utilization: {efficiency['data_utilization']:.1%}")
    print(f"  Samples per epoch: {efficiency['samples_per_epoch']}")
    
    # Potential issues
    issues = info['data_quality']['potential_issues']
    if issues:
        print(f"\nPotential Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print(f"\nNo significant data quality issues detected")
    
    print(f"\n{'='*60}")


def create_inference_dataloader(
    image_paths: List[str],
    batch_size: int = 1,
    num_workers: int = 0,
    transform: Optional[Any] = None
) -> DataLoader:
    """
    Create dataloader for inference on arbitrary images.
    
    Args:
        image_paths: List of paths to images
        batch_size: Batch size for inference
        num_workers: Number of workers
        transform: Transform to apply
        
    Returns:
        DataLoader for inference
    """
    from torch.utils.data import Dataset
    from PIL import Image
    
    class InferenceDataset(Dataset):
        def __init__(self, image_paths: List[str], transform=None):
            self.image_paths = image_paths
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, image_path
    
    if transform is None:
        transform = get_transforms('test')
    
    dataset = InferenceDataset(image_paths, transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )


def validate_dataloader_setup(train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, bool]:
    """
    Validate that dataloaders are properly configured.
    
    Returns:
        Dict with validation results
    """
    validation_results = {
        'train_loader_works': False,
        'val_loader_works': False,
        'batch_shapes_consistent': False,
        'labels_in_range': False
    }
    
    try:
        # Test train loader
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(train_loader):
            if batch_idx >= 2:  # Test a few batches
                break
            
            # Check shapes
            assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}D"
            assert images.size(1) == 3, f"Expected 3 channels, got {images.size(1)}"
            
            # Check label ranges
            assert food_labels.min() >= 0, f"Food label out of range: {food_labels.min()}"
            assert cuisine_labels.min() >= 0, f"Cuisine label out of range: {cuisine_labels.min()}"
            assert nutrition_targets.dim() == 2, f"Expected 2D nutrition tensor, got {nutrition_targets.dim()}D"
            assert nutrition_targets.size(1) == 4, f"Expected 4 nutrition values, got {nutrition_targets.size(1)}"
        
        validation_results['train_loader_works'] = True
        
        # Test val loader
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(val_loader):
            if batch_idx >= 1:  # Test one batch
                break
            
            # Same checks as train
            assert images.dim() == 4
            assert images.size(1) == 3
            assert food_labels.min() >= 0
            assert cuisine_labels.min() >= 0
            assert nutrition_targets.dim() == 2
            assert nutrition_targets.size(1) == 4
        
        validation_results['val_loader_works'] = True
        validation_results['batch_shapes_consistent'] = True
        validation_results['labels_in_range'] = True
        
    except Exception as e:
        logger.error(f"DataLoader validation failed: {e}")
    
    return validation_results


if __name__ == "__main__":
    print("Testing Enhanced DataLoader Configuration...")
    
    try:
        # Test with small subset for quick validation
        train_loader, val_loader, info = create_enhanced_dataloaders(
            batch_size=8,
            subset_size=50,
            num_workers=0,  # Avoid multiprocessing issues in testing
            sampling_strategy='smart_weighted',
            target_food_classes=101,
            augmentation_strength='medium'
        )
        
        print("\nTesting DataLoader Functionality...")
        
        # Validate setup
        validation_results = validate_dataloader_setup(train_loader, val_loader)
        
        print(f"\nValidation Results:")
        for test, passed in validation_results.items():
            status = "Passed!!" if passed else "NOT Passed!"
            print(f"  {test}: {status}")
        
        # Test a few batches
        print(f"\nTesting Sample Batches...")
        for batch_idx, (images, food_labels, cuisine_labels, nutrition) in enumerate(train_loader):
            print(f"  Batch {batch_idx + 1}: {images.shape}, food range: {food_labels.min()}-{food_labels.max()}")
            if batch_idx >= 2:
                break
        
        if all(validation_results.values()):
            print(f"\nEnhanced DataLoader test successful!")
        else:
            print(f"\nSome validation tests failed - check configuration")
        
    except Exception as e:
        print(f"\nDataLoader test failed: {e}")
        import traceback
        traceback.print_exc()