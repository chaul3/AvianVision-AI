"""
Attribute Model Training Module

Trains a ResNet-based multi-label classifier for 312 bird attributes.
Uses ImageNet-pretrained backbone with a linear classification head.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
from sklearn.metrics import average_precision_score, f1_score


class CUBAttributeDataset(Dataset):
    """Dataset class for CUB images with 312 attribute labels."""
    
    def __init__(self, 
                 manifest_df: pd.DataFrame, 
                 split: str,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            manifest_df: DataFrame with image paths and attributes
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.data = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
        # Extract attribute columns (1-312)
        self.attr_cols = [str(i) for i in range(1, 313)]
        
        print(f"Initialized {split} dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = row['full_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get attribute labels (312-dimensional binary vector)
        attributes = torch.tensor(
            row[self.attr_cols].values.astype(np.float32)
        )
        
        return {
            'image': image,
            'attributes': attributes,
            'image_id': row['image_id'],
            'class_id': row['class_id'],
            'species_name': row['species_name']
        }


class ResNetAttributeClassifier(nn.Module):
    """ResNet-based multi-label attribute classifier."""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 num_attributes: int = 312,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize the model.
        
        Args:
            backbone: ResNet variant ('resnet50', 'resnet101')
            num_attributes: Number of attributes to predict
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final layer
        """
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add attribute classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_attributes)
        )
        
        self.num_attributes = num_attributes
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class AttributeModelTrainer:
    """Trainer class for attribute model."""
    
    def __init__(self, 
                 config: Dict,
                 output_dir: str = "outputs"):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            output_dir: Directory to save model and logs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced GPU detection with debugging
        print("=== GPU Debug Information ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Force GPU usage if config specifies it or if CUDA is available
        force_gpu = config.get('training', {}).get('force_gpu', False)
        if force_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
        print("==============================")
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'cub-attributes'),
                config=config,
                name=config.get('experiment_name', 'attribute_model')
            )
    
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get training and validation transforms."""
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_transform
    
    def get_loss_weights(self, prevalence: Dict[str, float]) -> torch.Tensor:
        """Compute per-attribute loss weights based on prevalence."""
        weights = []
        
        for i in range(1, 313):  # attributes 1-312
            prev = prevalence[f'attr_{i}']
            # Weight inversely proportional to prevalence
            # Add small epsilon to avoid division by zero
            weight = 1.0 / (prev + 1e-6)
            weights.append(weight)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Normalize weights to have mean = 1
        weights = weights / weights.mean()
        
        return weights
    
    def train_epoch(self, 
                   model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['attributes'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss_step': loss.item(),
                    'epoch': epoch,
                    'step': epoch * num_batches + batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self,
                      model: nn.Module,
                      dataloader: DataLoader,
                      criterion: nn.Module,
                      epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0.0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['attributes'].to(self.device)
                
                logits = model(images)
                loss = criterion(logits, targets)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to numpy for sklearn metrics
        probs = torch.sigmoid(all_logits).numpy()
        targets_np = all_targets.numpy()
        
        print(f"Validation metrics - Probs shape: {probs.shape}, Targets shape: {targets_np.shape}")
        print(f"Targets range: {targets_np.min():.3f} - {targets_np.max():.3f}")
        
        # Ensure targets are binary (0/1)
        targets_np = (targets_np > 0.5).astype(int)
        
        # Compute mAP (mean Average Precision) for multi-label classification
        # Handle each attribute separately to avoid multiclass-multioutput error
        valid_aps = []
        for i in range(targets_np.shape[1]):  # For each attribute
            # Skip attributes with no positive samples
            if targets_np[:, i].sum() > 0 and targets_np[:, i].sum() < len(targets_np):
                try:
                    ap = average_precision_score(targets_np[:, i], probs[:, i])
                    valid_aps.append(ap)
                except Exception:
                    continue
        
        map_score = np.mean(valid_aps) if valid_aps else 0.0
        
        # Compute macro F1 with threshold 0.5
        preds = (probs > 0.5).astype(int)
        
        # Handle F1 score calculation more robustly
        try:
            macro_f1 = f1_score(targets_np, preds, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning: F1 score calculation failed: {e}")
            macro_f1 = 0.0
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'val_loss': avg_loss,
            'val_map': map_score,
            'val_macro_f1': macro_f1
        }
        
        return metrics
    
    def train(self, 
              manifest_path: str,
              prevalence_path: str) -> nn.Module:
        """Main training loop."""
        
        # Load data
        print("Loading dataset manifest...")
        manifest_df = pd.read_csv(manifest_path)
        
        with open(prevalence_path, 'r') as f:
            prevalence = json.load(f)
        
        # Get transforms
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = CUBAttributeDataset(manifest_df, 'train', train_transform)
        val_dataset = CUBAttributeDataset(manifest_df, 'val', val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Initialize model
        model = ResNetAttributeClassifier(
            backbone=self.config['backbone'],
            num_attributes=312,
            pretrained=self.config['pretrained'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Setup loss function with class weights
        if self.config.get('use_pos_weights', True):
            pos_weights = self.get_loss_weights(prevalence).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=self.config.get('lr_patience', 3),
            verbose=True
        )
        
        # Training loop
        best_metric = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Print metrics
            print(f"Train Loss: {metrics['train_loss']:.4f}")
            print(f"Val Loss: {metrics['val_loss']:.4f}")
            print(f"Val mAP: {metrics['val_map']:.4f}")
            print(f"Val Macro F1: {metrics['val_macro_f1']:.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log(metrics)
            
            # Learning rate scheduling
            scheduler.step(metrics['val_map'])
            
            # Early stopping and model saving
            current_metric = metrics['val_map']
            
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                
                # Save best model
                model_path = self.output_dir / "attr_model_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'metrics': metrics
                }, model_path)
                
                print(f"✓ New best model saved (mAP: {best_metric:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.get('early_stop_patience', 10):
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save final model
        final_model_path = self.output_dir / "attr_model_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'final_metrics': metrics
        }, final_model_path)
        
        print(f"✓ Training completed. Best mAP: {best_metric:.4f}")
        
        return model


def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        'backbone': 'resnet50',
        'pretrained': True,
        'dropout': 0.5,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'early_stop_patience': 10,
        'lr_patience': 3,
        'use_pos_weights': True,
        'num_workers': 4,
        'use_wandb': False,
        'wandb_project': 'cub-attributes',
        'experiment_name': 'resnet50_baseline'
    }


def main():
    """Example training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CUB attribute model")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to dataset manifest CSV")
    parser.add_argument("--prevalence", type=str, required=True,
                       help="Path to attribute prevalence JSON")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for model weights")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Train model
    trainer = AttributeModelTrainer(config, args.output_dir)
    model = trainer.train(args.manifest, args.prevalence)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
