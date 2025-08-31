"""
CUB-200-2011 Dataset Processing Module

This module handles the processing of the CUB-200-2011 dataset:
- Loads images, class labels, and 312 attribute annotations
- Creates train/val/test splits
- Builds dataset manifests
- Computes attribute prevalence statistics
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


class CUBDatasetProcessor:
    """Process CUB-200-2011 dataset with attributes for fine-grained bird classification."""
    
    def __init__(self, cub_root: str, output_dir: str = "outputs"):
        """
        Initialize the processor.
        
        Args:
            cub_root: Path to CUB-200-2011 dataset root directory
            output_dir: Directory to save processed outputs
        """
        self.cub_root = Path(cub_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Expected CUB dataset structure
        self.images_dir = self.cub_root / "images"
        self.attributes_dir = self.cub_root / "attributes"
        
        # Dataset files
        self.images_file = self.cub_root / "images.txt"
        self.classes_file = self.cub_root / "classes.txt"
        self.image_class_labels_file = self.cub_root / "image_class_labels.txt"
        self.train_test_split_file = self.cub_root / "train_test_split.txt"
        self.attributes_file = self.cub_root / "attributes" / "class_attribute_labels_continuous.txt"
        self.image_attributes_file = self.cub_root / "attributes" / "image_attribute_labels.txt"
        
    def verify_dataset_structure(self) -> bool:
        """Verify that all required dataset files exist."""
        required_files = [
            self.images_file,
            self.classes_file, 
            self.image_class_labels_file,
            self.train_test_split_file,
            self.image_attributes_file
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            return False
            
        print("✓ All required dataset files found")
        return True
    
    def load_images_metadata(self) -> pd.DataFrame:
        """Load image IDs and file paths."""
        images_df = pd.read_csv(
            self.images_file, 
            sep=' ', 
            header=None, 
            names=['image_id', 'filepath']
        )
        
        # Add full image paths
        images_df['full_path'] = images_df['filepath'].apply(
            lambda x: str(self.images_dir / x)
        )
        
        return images_df
    
    def load_classes(self) -> pd.DataFrame:
        """Load class IDs and names."""
        classes_df = pd.read_csv(
            self.classes_file,
            sep=' ',
            header=None,
            names=['class_id', 'class_name']
        )
        
        # Extract species name (remove numeric prefix)
        classes_df['species_name'] = classes_df['class_name'].str.split('.').str[1]
        
        return classes_df
    
    def load_image_labels(self) -> pd.DataFrame:
        """Load image-to-class mappings."""
        labels_df = pd.read_csv(
            self.image_class_labels_file,
            sep=' ',
            header=None,
            names=['image_id', 'class_id']
        )
        
        return labels_df
    
    def load_train_test_split(self) -> pd.DataFrame:
        """Load train/test split information."""
        split_df = pd.read_csv(
            self.train_test_split_file,
            sep=' ',
            header=None,
            names=['image_id', 'is_training_image']
        )
        
        # Convert to readable split names
        split_df['split'] = split_df['is_training_image'].map({1: 'train', 0: 'test'})
        
        return split_df
    
    def load_image_attributes(self) -> pd.DataFrame:
        """Load per-image 312-dimensional attribute annotations."""
        # CUB has image_attribute_labels.txt with format: image_id attribute_id is_present certainty
        attr_df = pd.read_csv(
            self.image_attributes_file,
            sep=' ',
            header=None,
            names=['image_id', 'attribute_id', 'is_present', 'certainty']
        )
        
        # Pivot to get 312 attributes per image
        # Use is_present as binary labels (1/0)
        attrs_pivot = attr_df.pivot(
            index='image_id', 
            columns='attribute_id', 
            values='is_present'
        ).fillna(0)
        
        # Ensure we have all 312 attributes
        expected_attrs = list(range(1, 313))  # 1-312
        missing_attrs = set(expected_attrs) - set(attrs_pivot.columns)
        
        for attr in missing_attrs:
            attrs_pivot[attr] = 0
            
        # Sort columns to ensure consistent ordering
        attrs_pivot = attrs_pivot[expected_attrs]
        
        # Reset index to get image_id as column
        attrs_pivot = attrs_pivot.reset_index()
        
        return attrs_pivot
    
    def create_validation_split(self, train_df: pd.DataFrame, val_ratio: float = 0.2) -> pd.DataFrame:
        """Create validation split from training data."""
        train_images = train_df[train_df['split'] == 'train'].copy()
        
        # Stratified split by class to maintain class balance
        val_samples = []
        
        for class_id in train_images['class_id'].unique():
            class_images = train_images[train_images['class_id'] == class_id]
            n_val = max(1, int(len(class_images) * val_ratio))
            
            # Randomly sample validation images for this class
            val_class_samples = class_images.sample(n=n_val, random_state=42)
            val_samples.append(val_class_samples)
        
        val_df = pd.concat(val_samples, ignore_index=True)
        val_image_ids = set(val_df['image_id'])
        
        # Update splits
        updated_df = train_df.copy()
        updated_df.loc[updated_df['image_id'].isin(val_image_ids), 'split'] = 'val'
        
        return updated_df
    
    def compute_attribute_prevalence(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """Compute per-attribute prevalence in training set for loss weighting."""
        attr_cols = [f'{i}' for i in range(1, 313)]  # attributes 1-312
        
        prevalence = {}
        train_attrs = train_df[train_df['split'] == 'train'][attr_cols]
        
        for attr in attr_cols:
            pos_count = train_attrs[attr].sum()
            total_count = len(train_attrs)
            prevalence[f'attr_{attr}'] = float(pos_count / total_count)
        
        return prevalence
    
    def build_dataset_manifest(self) -> pd.DataFrame:
        """Build complete dataset manifest with all metadata."""
        print("Loading dataset components...")
        
        # Load all components
        images_df = self.load_images_metadata()
        classes_df = self.load_classes()
        labels_df = self.load_image_labels()
        split_df = self.load_train_test_split()
        attributes_df = self.load_image_attributes()
        
        print(f"✓ Loaded {len(images_df)} images")
        print(f"✓ Loaded {len(classes_df)} classes") 
        print(f"✓ Loaded {len(attributes_df)} image attribute vectors")
        
        # Merge all data
        manifest = images_df.merge(labels_df, on='image_id')
        manifest = manifest.merge(classes_df, on='class_id')
        manifest = manifest.merge(split_df, on='image_id')
        manifest = manifest.merge(attributes_df, on='image_id')
        
        # Create validation split
        manifest = self.create_validation_split(manifest)
        
        print(f"✓ Created dataset manifest with {len(manifest)} samples")
        
        # Print split statistics
        split_counts = manifest['split'].value_counts()
        print(f"Split distribution: {dict(split_counts)}")
        
        class_counts = manifest.groupby(['split', 'class_id']).size().unstack(fill_value=0)
        print(f"Classes per split: train={len(class_counts.columns)}, val={len(class_counts.columns)}, test={len(class_counts.columns)}")
        
        return manifest
    
    def process_dataset(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Main processing function - creates manifest and computes statistics."""
        if not self.verify_dataset_structure():
            raise ValueError("Dataset structure verification failed")
        
        print("Building dataset manifest...")
        manifest = self.build_dataset_manifest()
        
        print("Computing attribute prevalence...")
        prevalence = self.compute_attribute_prevalence(manifest)
        
        # Save outputs
        manifest_path = self.output_dir / "dataset_manifest.csv"
        prevalence_path = self.output_dir / "attribute_prevalence.json"
        
        manifest.to_csv(manifest_path, index=False)
        with open(prevalence_path, 'w') as f:
            json.dump(prevalence, f, indent=2)
        
        print(f"✓ Saved dataset manifest to {manifest_path}")
        print(f"✓ Saved attribute prevalence to {prevalence_path}")
        
        # Print summary statistics
        self.print_summary_stats(manifest, prevalence)
        
        return manifest, prevalence
    
    def print_summary_stats(self, manifest: pd.DataFrame, prevalence: Dict[str, float]):
        """Print dataset summary statistics."""
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        # Basic counts
        print(f"Total images: {len(manifest)}")
        print(f"Total classes: {manifest['class_id'].nunique()}")
        print(f"Total attributes: 312")
        
        # Split distribution
        print(f"\nSplit distribution:")
        for split, count in manifest['split'].value_counts().items():
            print(f"  {split}: {count:,} ({count/len(manifest)*100:.1f}%)")
        
        # Class balance
        print(f"\nClass balance:")
        class_counts = manifest['class_id'].value_counts()
        print(f"  Min images per class: {class_counts.min()}")
        print(f"  Max images per class: {class_counts.max()}")
        print(f"  Mean images per class: {class_counts.mean():.1f}")
        
        # Attribute statistics
        attr_prevalences = list(prevalence.values())
        print(f"\nAttribute prevalence:")
        print(f"  Min prevalence: {min(attr_prevalences):.3f}")
        print(f"  Max prevalence: {max(attr_prevalences):.3f}")
        print(f"  Mean prevalence: {np.mean(attr_prevalences):.3f}")
        print(f"  Median prevalence: {np.median(attr_prevalences):.3f}")
        
        # Check for data leakage
        train_classes = set(manifest[manifest['split'] == 'train']['class_id'])
        val_classes = set(manifest[manifest['split'] == 'val']['class_id'])
        test_classes = set(manifest[manifest['split'] == 'test']['class_id'])
        
        print(f"\nLeakage check:")
        print(f"  Classes in train: {len(train_classes)}")
        print(f"  Classes in val: {len(val_classes)}")
        print(f"  Classes in test: {len(test_classes)}")
        
        if train_classes == val_classes == test_classes:
            print("  ✓ All splits contain same classes (no leakage)")
        else:
            print("  ⚠ Warning: Different classes across splits!")


def main():
    """Example usage of the CUB dataset processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process CUB-200-2011 dataset")
    parser.add_argument("--cub_root", type=str, required=True,
                       help="Path to CUB-200-2011 dataset root directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    processor = CUBDatasetProcessor(args.cub_root, args.output_dir)
    manifest, prevalence = processor.process_dataset()
    
    print(f"\nProcessing complete! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
