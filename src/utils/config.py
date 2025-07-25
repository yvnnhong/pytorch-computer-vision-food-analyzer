# Configuration file for Multi-Task Food Classifier

# Model Configuration
NUM_FOOD_CLASSES = 101
NUM_CUISINE_CLASSES = 10  # Will be updated based on actual data
NUTRITION_DIM = 4  # calories, protein, carbs, fat

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

# Multi-task Loss Weights
FOOD_WEIGHT = 1.0      # Primary task
CUISINE_WEIGHT = 0.5   # Secondary task
NUTRITION_WEIGHT = 1.0 # Important regression task

# Data Configuration
IMAGE_SIZE = 224
DATA_ROOT = './data/raw/food-101'
CUISINE_MAPPING_PATH = './data/cuisine_mappings.json'
NUTRITION_DB_PATH = './data/nutrition_db.json'

# Training Configuration
SAVE_DIR = './models'
LOG_DIR = './logs'
SAVE_EVERY = 5  # Save checkpoint every N epochs
EARLY_STOPPING_PATIENCE = 7

# Device Configuration
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader Configuration
NUM_WORKERS = 4
PIN_MEMORY = True

# Optimizer Configuration
OPTIMIZER = 'adam'  # 'adam', 'sgd', 'adamw'
SCHEDULER = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'

# Augmentation Configuration
AUGMENTATION_ENABLED = True
COLOR_JITTER_STRENGTH = 0.2
ROTATION_DEGREES = 10
HORIZONTAL_FLIP_PROB = 0.5

print(f"Configuration loaded - Device: {DEVICE}")