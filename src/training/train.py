import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# Import our custom modules
import sys
sys.path.append('.')
from src.models.food_classifier import MultiTaskFoodModel, MultiTaskLoss, create_model
from src.datasets.data_loaders import create_dataloaders
from src.utils.config import *

class MultiTaskTrainer:
    """
    Trainer class for multi-task food classification model.
    Handles training, validation, checkpointing, and metrics tracking.
    """
    
    def __init__(self, model, train_loader, val_loader, device='cpu', 
                 learning_rate=0.001, save_dir='./models'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Loss function and optimizer
        self.criterion = MultiTaskLoss(
            food_weight=1.0,      # Food classification is primary task
            cuisine_weight=0.5,   # Cuisine is secondary  
            nutrition_weight=1.0  # Nutrition regression is important
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_food_acc': [],
            'val_food_acc': [],
            'train_cuisine_acc': [],
            'val_cuisine_acc': [],
            'learning_rates': []
        }
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Save directory: {save_dir}")
    
    def calculate_accuracy(self, logits, targets):
        """Calculate classification accuracy"""
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == targets).float()
            accuracy = correct.mean().item()
        return accuracy
    
    def calculate_nutrition_mae(self, predictions, targets):
        """Calculate Mean Absolute Error for nutrition regression"""
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets)).item()
        return mae
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_food_acc = 0.0
        running_cuisine_acc = 0.0
        running_nutrition_mae = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, food_labels, cuisine_labels, nutrition_targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            food_labels = food_labels.to(self.device)
            cuisine_labels = cuisine_labels.to(self.device)
            nutrition_targets = nutrition_targets.to(self.device)
            
            # Forward pass
            food_logits, cuisine_logits, nutrition_preds = self.model(images)
            
            # Calculate loss
            predictions = (food_logits, cuisine_logits, nutrition_preds)
            targets = (food_labels, cuisine_labels, nutrition_targets)
            
            total_loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            food_acc = self.calculate_accuracy(food_logits, food_labels)
            cuisine_acc = self.calculate_accuracy(cuisine_logits, cuisine_labels)
            nutrition_mae = self.calculate_nutrition_mae(nutrition_preds, nutrition_targets)
            
            # Update running averages
            running_loss += total_loss.item()
            running_food_acc += food_acc
            running_cuisine_acc += cuisine_acc
            running_nutrition_mae += nutrition_mae
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.3f}',
                'Food Acc': f'{food_acc:.3f}',
                'Cuisine Acc': f'{cuisine_acc:.3f}',
                'Nutrition MAE': f'{nutrition_mae:.1f}'
            })
        
        # Calculate epoch averages
        epoch_loss = running_loss / num_batches
        epoch_food_acc = running_food_acc / num_batches
        epoch_cuisine_acc = running_cuisine_acc / num_batches
        epoch_nutrition_mae = running_nutrition_mae / num_batches
        
        return {
            'loss': epoch_loss,
            'food_acc': epoch_food_acc,
            'cuisine_acc': epoch_cuisine_acc,
            'nutrition_mae': epoch_nutrition_mae
        }
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_food_acc = 0.0
        running_cuisine_acc = 0.0
        running_nutrition_mae = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for images, food_labels, cuisine_labels, nutrition_targets in pbar:
                # Move data to device
                images = images.to(self.device)
                food_labels = food_labels.to(self.device)
                cuisine_labels = cuisine_labels.to(self.device)
                nutrition_targets = nutrition_targets.to(self.device)
                
                # Forward pass
                food_logits, cuisine_logits, nutrition_preds = self.model(images)
                
                # Calculate loss
                predictions = (food_logits, cuisine_logits, nutrition_preds)
                targets = (food_labels, cuisine_labels, nutrition_targets)
                
                total_loss, loss_dict = self.criterion(predictions, targets)
                
                # Calculate metrics
                food_acc = self.calculate_accuracy(food_logits, food_labels)
                cuisine_acc = self.calculate_accuracy(cuisine_logits, cuisine_labels)
                nutrition_mae = self.calculate_nutrition_mae(nutrition_preds, nutrition_targets)
                
                # Update running averages
                running_loss += total_loss.item()
                running_food_acc += food_acc
                running_cuisine_acc += cuisine_acc
                running_nutrition_mae += nutrition_mae
                
                # Update progress bar
                pbar.set_postfix({
                    'Val Loss': f'{total_loss.item():.3f}',
                    'Food Acc': f'{food_acc:.3f}',
                    'Cuisine Acc': f'{cuisine_acc:.3f}',
                    'Nutrition MAE': f'{nutrition_mae:.1f}'
                })
        
        # Calculate epoch averages
        epoch_loss = running_loss / num_batches
        epoch_food_acc = running_food_acc / num_batches
        epoch_cuisine_acc = running_cuisine_acc / num_batches
        epoch_nutrition_mae = running_nutrition_mae / num_batches
        
        return {
            'loss': epoch_loss,
            'food_acc': epoch_food_acc,
            'cuisine_acc': epoch_cuisine_acc,
            'nutrition_mae': epoch_nutrition_mae
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch}")
        
        # Save history as JSON
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, num_epochs=10, save_every=5):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_food_acc'].append(train_metrics['food_acc'])
            self.history['val_food_acc'].append(val_metrics['food_acc'])
            self.history['train_cuisine_acc'].append(train_metrics['cuisine_acc'])
            self.history['val_cuisine_acc'].append(val_metrics['cuisine_acc'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Food Acc: {train_metrics['food_acc']:.3f} | Val Food Acc: {val_metrics['food_acc']:.3f}")
            print(f"  Train Cuisine Acc: {train_metrics['cuisine_acc']:.3f} | Val Cuisine Acc: {val_metrics['cuisine_acc']:.3f}")
            print(f"  Train Nutrition MAE: {train_metrics['nutrition_mae']:.1f} | Val Nutrition MAE: {val_metrics['nutrition_mae']:.1f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check if best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved in: {self.save_dir}")


def main():
    """Main training function"""
    print("Multi-Task Food Classification Training")
    print("=" * 50)
    
    # Configuration
    BATCH_SIZE = 16  # Smaller batch size for stability
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    SUBSET_SIZE = 200  # Use subset for quick training (remove for full training)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nCreating datasets...")
    train_loader, val_loader, class_info = create_dataloaders(
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,  # Remove this for full training
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_food_classes=class_info['num_food_classes'],
        num_cuisine_classes=class_info['num_cuisine_classes'],
        device=device
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    # Start training
    trainer.train(num_epochs=NUM_EPOCHS, save_every=3)
    
    print("\nTraining script complete!")


if __name__ == "__main__":
    main()