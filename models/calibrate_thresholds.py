"""
Threshold Calibration Module

Calibrates per-attribute decision thresholds for optimal F1 score.
Also provides probability calibration using temperature scaling.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize_scalar
from tqdm import tqdm


class ThresholdCalibrator:
    """Calibrates per-attribute decision thresholds for optimal performance."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize calibrator.
        
        Args:
            output_dir: Directory to save calibration outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model_predictions(self, 
                              model_path: str,
                              manifest_path: str,
                              split: str = 'val') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load model and generate predictions on validation set.
        
        Args:
            model_path: Path to trained model
            manifest_path: Path to dataset manifest
            split: Dataset split to use ('val' or 'test')
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        from models.train_attribute_model import ResNetAttributeClassifier, CUBAttributeDataset
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['config']
        
        model = ResNetAttributeClassifier(
            backbone=model_config['backbone'],
            num_attributes=312,
            pretrained=False,  # We're loading trained weights
            dropout=model_config['dropout']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load dataset
        manifest_df = pd.read_csv(manifest_path)
        
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
        
        dataset = CUBAttributeDataset(manifest_df, split, val_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Generate predictions
        all_logits = []
        all_targets = []
        
        print(f"Generating predictions on {split} set...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images = batch['image'].to(self.device)
                targets = batch['attributes']
                
                logits = model(images)
                
                all_logits.append(logits.cpu())
                all_targets.append(targets)
        
        # Convert to numpy
        logits = torch.cat(all_logits, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        
        print(f"Generated predictions for {len(probs)} samples")
        
        return probs, targets
    
    def find_optimal_threshold(self, 
                              y_true: np.ndarray, 
                              y_probs: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal threshold for a single attribute.
        
        Args:
            y_true: Ground truth binary labels
            y_probs: Predicted probabilities
            metric: Metric to optimize ('f1', 'youden', 'precision_recall_balance')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        # Generate candidate thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic: sensitivity + specificity - 1
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tp = np.sum((y_true == 1) & (y_pred == 1))
                
                sensitivity = tp / (tp + fn + 1e-7)
                specificity = tn / (tn + fp + 1e-7)
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def calibrate_thresholds(self,
                           probs: np.ndarray,
                           targets: np.ndarray,
                           metric: str = 'f1') -> Dict[str, float]:
        """
        Calibrate thresholds for all 312 attributes.
        
        Args:
            probs: Predicted probabilities (N, 312)
            targets: Ground truth labels (N, 312)
            metric: Metric to optimize
            
        Returns:
            Dictionary mapping attribute indices to optimal thresholds
        """
        print(f"Calibrating thresholds using {metric} metric...")
        
        thresholds = {}
        scores = {}
        
        for attr_idx in tqdm(range(312), desc="Calibrating"):
            attr_probs = probs[:, attr_idx]
            attr_targets = targets[:, attr_idx]
            
            # Skip if all targets are the same (no positive/negative examples)
            if len(np.unique(attr_targets)) < 2:
                thresholds[f'attr_{attr_idx+1}'] = 0.5
                scores[f'attr_{attr_idx+1}'] = 0.0
                continue
            
            threshold, score = self.find_optimal_threshold(
                attr_targets, attr_probs, metric
            )
            
            thresholds[f'attr_{attr_idx+1}'] = float(threshold)
            scores[f'attr_{attr_idx+1}'] = float(score)
        
        # Print summary statistics
        threshold_values = list(thresholds.values())
        score_values = list(scores.values())
        
        print(f"\nThreshold calibration complete:")
        print(f"  Mean threshold: {np.mean(threshold_values):.3f}")
        print(f"  Std threshold: {np.std(threshold_values):.3f}")
        print(f"  Min threshold: {np.min(threshold_values):.3f}")
        print(f"  Max threshold: {np.max(threshold_values):.3f}")
        print(f"  Mean {metric} score: {np.mean(score_values):.3f}")
        
        return thresholds
    
    def evaluate_with_thresholds(self,
                               probs: np.ndarray,
                               targets: np.ndarray,
                               thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate model performance using calibrated thresholds.
        
        Args:
            probs: Predicted probabilities
            targets: Ground truth labels
            thresholds: Per-attribute thresholds
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Apply thresholds
        predictions = np.zeros_like(probs)
        
        for attr_idx in range(312):
            threshold = thresholds[f'attr_{attr_idx+1}']
            predictions[:, attr_idx] = (probs[:, attr_idx] >= threshold).astype(int)
        
        # Compute metrics
        # Per-attribute F1 scores
        f1_scores = []
        for attr_idx in range(312):
            f1 = f1_score(targets[:, attr_idx], predictions[:, attr_idx], zero_division=0)
            f1_scores.append(f1)
        
        # Overall metrics
        macro_f1 = np.mean(f1_scores)
        
        # Exact match accuracy (all attributes correct)
        exact_match = np.mean(np.all(predictions == targets, axis=1))
        
        # Hamming accuracy (average per-attribute accuracy)
        hamming_acc = np.mean(predictions == targets)
        
        metrics = {
            'macro_f1': macro_f1,
            'exact_match_accuracy': exact_match,
            'hamming_accuracy': hamming_acc,
            'mean_threshold': np.mean(list(thresholds.values())),
            'std_threshold': np.std(list(thresholds.values()))
        }
        
        return metrics
    
    def temperature_scaling(self,
                          logits: np.ndarray,
                          targets: np.ndarray) -> float:
        """
        Apply temperature scaling for probability calibration.
        
        Args:
            logits: Model logits before sigmoid
            targets: Ground truth binary labels
            
        Returns:
            Optimal temperature parameter
        """
        print("Performing temperature scaling...")
        
        def temperature_nll(T):
            """Negative log-likelihood with temperature scaling."""
            scaled_logits = logits / T
            probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            
            # Binary cross-entropy loss
            nll = -np.mean(
                targets * np.log(probs) + (1 - targets) * np.log(1 - probs)
            )
            return nll
        
        # Optimize temperature
        result = minimize_scalar(
            temperature_nll,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_temperature = result.x
        print(f"Optimal temperature: {optimal_temperature:.3f}")
        
        return optimal_temperature
    
    def plot_calibration_curves(self,
                               probs: np.ndarray,
                               targets: np.ndarray,
                               n_bins: int = 10,
                               save_path: Optional[str] = None):
        """
        Plot reliability diagrams for probability calibration assessment.
        
        Args:
            probs: Predicted probabilities
            targets: Ground truth labels
            n_bins: Number of bins for calibration curve
            save_path: Path to save plot
        """
        print("Generating calibration plots...")
        
        # Sample a subset of attributes for visualization
        attr_indices = [0, 50, 100, 150, 200, 250, 300, 311]  # Sample across range
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, attr_idx in enumerate(attr_indices):
            attr_probs = probs[:, attr_idx]
            attr_targets = targets[:, attr_idx]
            
            # Skip if all targets are the same
            if len(np.unique(attr_targets)) < 2:
                axes[i].text(0.5, 0.5, f'Attr {attr_idx+1}\n(No variance)',
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                attr_targets, attr_probs, n_bins=n_bins
            )
            
            # Plot
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            axes[i].plot(mean_predicted_value, fraction_of_positives, 'o-',
                        label='Model calibration')
            axes[i].set_xlabel('Mean Predicted Probability')
            axes[i].set_ylabel('Fraction of Positives')
            axes[i].set_title(f'Attribute {attr_idx+1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Calibration plot saved to {save_path}")
        
        return fig
    
    def calibrate_model(self,
                       model_path: str,
                       manifest_path: str,
                       split: str = 'val',
                       metric: str = 'f1') -> Tuple[Dict[str, float], float]:
        """
        Main calibration function.
        
        Args:
            model_path: Path to trained model
            manifest_path: Path to dataset manifest
            split: Dataset split to use for calibration
            metric: Metric to optimize for threshold calibration
            
        Returns:
            Tuple of (thresholds_dict, temperature)
        """
        print(f"Starting threshold calibration on {split} set...")
        
        # Generate predictions
        probs, targets = self.load_model_predictions(model_path, manifest_path, split)
        
        # Calibrate thresholds
        thresholds = self.calibrate_thresholds(probs, targets, metric)
        
        # Evaluate with thresholds
        print("\nEvaluating with calibrated thresholds...")
        metrics = self.evaluate_with_thresholds(probs, targets, thresholds)
        
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Temperature scaling (need logits for this, skip for now)
        # In practice, you'd want to save logits during prediction
        temperature = 1.0  # Default (no scaling)
        
        # Generate calibration plots
        plot_path = self.output_dir / "calibration_curves.png"
        self.plot_calibration_curves(probs, targets, save_path=plot_path)
        
        # Save results
        thresholds_path = self.output_dir / "attr_thresholds.json"
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        metrics_path = self.output_dir / "calibration_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'thresholds_metrics': metrics,
                'temperature': temperature,
                'calibration_metric': metric
            }, f, indent=2)
        
        print(f"✓ Thresholds saved to {thresholds_path}")
        print(f"✓ Metrics saved to {metrics_path}")
        
        return thresholds, temperature


def main():
    """Example calibration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate CUB attribute model thresholds")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.pt file)")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to dataset manifest CSV")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for calibration results")
    parser.add_argument("--split", type=str, default="val",
                       choices=['val', 'test'],
                       help="Dataset split to use for calibration")
    parser.add_argument("--metric", type=str, default="f1",
                       choices=['f1', 'youden'],
                       help="Metric to optimize for threshold selection")
    
    args = parser.parse_args()
    
    # Run calibration
    calibrator = ThresholdCalibrator(args.output_dir)
    thresholds, temperature = calibrator.calibrate_model(
        args.model, args.manifest, args.split, args.metric
    )
    
    print("Threshold calibration completed successfully!")


if __name__ == "__main__":
    main()
