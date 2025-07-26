# Multi-task CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import json

class MultiTaskFoodModel(nn.Module):
    """
    Multi-task CNN for food classification, cuisine prediction, and nutrition regression.
    
    Architecture:
    - Shared ResNet50 backbone (pre-trained)
    - Three task-specific heads:
        1. Food classification (101 classes)
        2. Cuisine classification (10+ classes) 
        3. Nutrition regression (4 values: calories, protein, carbs, fat)
    """
    
    def __init__(self, num_food_classes=101, num_cuisine_classes=10, nutrition_dim=4, dropout_rate=0.3):
        super(MultiTaskFoodModel, self).__init__()
        
        # Load pre-trained ResNet50 backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer to get feature extractor
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features from ResNet50
        self.num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Task-specific heads
        
        # 1. Food Classification Head
        self.food_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_food_classes)
        )
        
        # 2. Cuisine Classification Head  
        self.cuisine_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_cuisine_classes)
        )
        
        # 3. Nutrition Regression Head
        self.nutrition_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, nutrition_dim)  # [calories, protein, carbs, fat]
        )
        
        # Store class mappings
        self.cuisine_to_idx = {}
        self.idx_to_cuisine = {}
        self.food_to_idx = {}
        self.idx_to_food = {}
        
    def forward(self, x):
        """
        Forward pass through the multi-task model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            tuple: (food_logits, cuisine_logits, nutrition_values)
        """
        # Extract features using shared backbone
        features = self.feature_extractor(x)  # (batch_size, 2048, 1, 1)
        features = torch.flatten(features, 1)  # (batch_size, 2048)
        
        # Shared processing
        shared_features = self.shared_fc(features)  # (batch_size, 512)
        
        # Task-specific predictions
        food_logits = self.food_classifier(shared_features)      # (batch_size, 101)
        cuisine_logits = self.cuisine_classifier(shared_features) # (batch_size, 10)
        nutrition_values = self.nutrition_regressor(shared_features) # (batch_size, 4)
        
        return food_logits, cuisine_logits, nutrition_values
    
    def predict_single_image(self, x):
        """
        Make predictions for a single image with nice formatting.
        
        Args:
            x: Input tensor (1, 3, 224, 224) or (3, 224, 224)
            
        Returns:
            dict: Formatted predictions with confidence scores
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension
                
            food_logits, cuisine_logits, nutrition_values = self.forward(x)
            
            # Get predictions
            food_probs = F.softmax(food_logits, dim=1)
            cuisine_probs = F.softmax(cuisine_logits, dim=1)
            
            food_pred = torch.argmax(food_probs, dim=1).item()
            cuisine_pred = torch.argmax(cuisine_probs, dim=1).item()
            
            food_confidence = food_probs[0, food_pred].item()
            cuisine_confidence = cuisine_probs[0, cuisine_pred].item()
            
            # Format nutrition values (ensure positive)
            nutrition = nutrition_values[0].cpu().numpy()
            nutrition = [max(0, float(val)) for val in nutrition]  # Ensure non-negative
            
            return {
                'food': {
                    'class_idx': food_pred,
                    'class_name': self.idx_to_food.get(food_pred, f'class_{food_pred}'),
                    'confidence': food_confidence
                },
                'cuisine': {
                    'class_idx': cuisine_pred,
                    'class_name': self.idx_to_cuisine.get(cuisine_pred, f'cuisine_{cuisine_pred}'),
                    'confidence': cuisine_confidence
                },
                'nutrition': {
                    'calories': float(round(nutrition[0], 1)),
                    'protein': float(round(nutrition[1], 1)),
                    'carbs': float(round(nutrition[2], 1)),
                    'fat': float(round(nutrition[3], 1))
                }
            }
    
    def load_class_mappings(self, cuisine_mapping_path='./data/cuisine_mappings.json'):
        """Load food and cuisine class mappings from JSON files."""
        try:
            with open(cuisine_mapping_path, 'r') as f:
                cuisine_mapping = json.load(f)
            
            # Create cuisine mappings
            unique_cuisines = list(set(cuisine_mapping.values()))
            self.cuisine_to_idx = {cuisine: idx for idx, cuisine in enumerate(unique_cuisines)}
            self.idx_to_cuisine = {idx: cuisine for cuisine, idx in self.cuisine_to_idx.items()}
            
            # Create food mappings (assuming Food-101 order)
            unique_foods = list(cuisine_mapping.keys())
            self.food_to_idx = {food: idx for idx, food in enumerate(unique_foods)}
            self.idx_to_food = {idx: food for food, idx in self.food_to_idx.items()}
            
            print(f"Loaded {len(unique_cuisines)} cuisine classes")
            print(f"Loaded {len(unique_foods)} food classes")
            
        except FileNotFoundError:
            print(f"Warning: Could not load class mappings from {cuisine_mapping_path}")
            print("Using default mappings...")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines:
    1. Food classification loss (CrossEntropy)
    2. Cuisine classification loss (CrossEntropy) 
    3. Nutrition regression loss (MSE)
    
    Uses learnable loss weights for optimal task balancing.
    """
    
    def __init__(self, food_weight=1.0, cuisine_weight=0.5, nutrition_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        
        # Loss functions for each task
        self.food_loss_fn = nn.CrossEntropyLoss()
        self.cuisine_loss_fn = nn.CrossEntropyLoss()
        self.nutrition_loss_fn = nn.MSELoss()
        
        # Task weights (can be learned or fixed)
        self.food_weight = food_weight
        self.cuisine_weight = cuisine_weight  
        self.nutrition_weight = nutrition_weight
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss.
        
        Args:
            predictions: tuple of (food_logits, cuisine_logits, nutrition_values)
            targets: tuple of (food_labels, cuisine_labels, nutrition_targets)
            
        Returns:
            tuple: (total_loss, individual_losses_dict)
        """
        food_logits, cuisine_logits, nutrition_values = predictions
        food_labels, cuisine_labels, nutrition_targets = targets
        
        # Calculate individual losses
        food_loss = self.food_loss_fn(food_logits, food_labels)
        cuisine_loss = self.cuisine_loss_fn(cuisine_logits, cuisine_labels)
        nutrition_loss = self.nutrition_loss_fn(nutrition_values, nutrition_targets)
        
        # Weighted combination
        total_loss = (
            self.food_weight * food_loss +
            self.cuisine_weight * cuisine_loss + 
            self.nutrition_weight * nutrition_loss
        )
        
        # Return detailed loss breakdown
        loss_dict = {
            'total_loss': total_loss.item(),
            'food_loss': food_loss.item(),
            'cuisine_loss': cuisine_loss.item(),
            'nutrition_loss': nutrition_loss.item()
        }
        
        return total_loss, loss_dict


def create_model(num_food_classes=101, num_cuisine_classes=10, device='cpu'):
    model = MultiTaskFoodModel(
        num_food_classes=num_food_classes,
        num_cuisine_classes=num_cuisine_classes,
        nutrition_dim=4,  # calories, protein, carbs, fat
        dropout_rate=0.3
    )
    
    # Load class mappings
    model.load_class_mappings()
    
    # Move to device
    model = model.to(device)
    
    print(f"Created multi-task model with:")
    print(f"  - Food classes: 101")
    print(f"  - Cuisine classes: {num_cuisine_classes}")
    print(f"  - Nutrition outputs: 4")
    print(f"  - Device: {device}")
    
    return model


if __name__ == "__main__":
    # Quick test of the model architecture
    print("Testing Multi-Task Food Model...")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(device=device)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nTesting with input shape: {test_input.shape}")
    
    # Forward pass
    food_logits, cuisine_logits, nutrition_values = model(test_input)
    
    print(f"Output shapes:")
    print(f"  - Food logits: {food_logits.shape}")
    print(f"  - Cuisine logits: {cuisine_logits.shape}")  
    print(f"  - Nutrition values: {nutrition_values.shape}")
    
    # Test single image prediction
    single_image = torch.randn(3, 224, 224).to(device)
    prediction = model.predict_single_image(single_image)
    
    print(f"\nSample prediction:")
    print(f"  - Food: {prediction['food']}")
    print(f"  - Cuisine: {prediction['cuisine']}")
    print(f"  - Nutrition: {prediction['nutrition']}")
    
    print("\nModel architecture test complete!")