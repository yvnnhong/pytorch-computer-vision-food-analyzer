# Datasets Directory

PyTorch data handling components for multi-task food classification pipeline.

## Core Components

### dataset.py
Multi-task PyTorch Dataset class handling food images with food type, cuisine, and nutrition labels.

### data_loaders.py
Custom DataLoader configurations for training, validation, and testing with proper batching and sampling strategies.

### transforms.py
Image preprocessing pipeline including resizing, normalization, and basic augmentation transforms.

### preprocessing.py
Data cleaning and preparation utilities for converting raw datasets into training-ready format.

## Advanced Features

### augmentation.py
Data augmentation techniques including rotation, color jittering, and geometric transforms.

### synthetic_data.py
Synthetic data generation utilities for data balancing and augmentation.

### class_mapping.py
Mapping utilities between food categories, cuisine types, and nutrition databases.

## Usage

```python
from src.datasets.dataset import FoodDataset
from src.datasets.data_loaders import create_data_loaders

# Create dataset
dataset = FoodDataset(data_dir='data/processed', transform=transforms)

# Create data loaders
train_loader, val_loader = create_data_loaders(dataset, batch_size=32)
```

## Requirements

- PIL/Pillow for image loading
- torchvision.transforms for preprocessing
- albumentations for advanced augmentation