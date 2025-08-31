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
        
        print(f"Loading attributes from: {self.image_attributes_file}")
        
        # First, let's examine the file structure in detail
        with open(self.image_attributes_file, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        
        print("First 10 lines of attribute file:")
        for i, line in enumerate(first_lines):
            if line:
                parts = line.split()
                print(f"  Line {i+1}: {len(parts)} fields - {line}")
        
        # Count total lines and check file size
        with open(self.image_attributes_file, 'r') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total non-empty lines in file: {total_lines:,}")
        
        # Expected: 11,788 images × 312 attributes = 3,677,856 lines
        expected_lines = 11788 * 312
        print(f"Expected lines for CUB dataset: {expected_lines:,}")
        
        if total_lines != expected_lines:
            print(f"⚠️  WARNING: Line count mismatch! Got {total_lines:,}, expected {expected_lines:,}")
            print("This suggests the file format might be different than expected.")
            
            # Check if it might be a different format (e.g., one line per image)
            if total_lines < 20000:  # Much fewer lines than expected
                print("🔍 Checking if this is a compact format (one line per image)...")
                with open(self.image_attributes_file, 'r') as f:
                    sample_line = f.readline().strip()
                    sample_parts = sample_line.split()
                    print(f"Sample line has {len(sample_parts)} parts: {sample_line[:100]}...")
                    
                    if len(sample_parts) > 10:  # Probably compact format
                        print("⚠️  This appears to be a compact format, not the standard CUB format!")
                        print("Attempting to parse as compact format...")
                        return self._load_compact_attributes()
        
        # Continue with standard parsing
        try:
            attr_df = pd.read_csv(
                self.image_attributes_file,
                sep=r'\s+',  # Use regex for multiple whitespace
                header=None,
                names=['image_id', 'attribute_id', 'is_present', 'certainty'],
                on_bad_lines='skip',  # Skip malformed lines
                engine='python'       # Use python engine for better error handling
            )
        except Exception as e:
            print(f"Error reading attributes file with pandas, trying manual parsing: {e}")
            # Fallback to manual parsing
            attr_data = []
            skipped_lines = 0
            
            with open(self.image_attributes_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # Take only the first 4 fields, ignore extra
                            attr_data.append([
                                int(parts[0]),    # image_id
                                int(parts[1]),    # attribute_id
                                int(parts[2]),    # is_present
                                float(parts[3])   # certainty
                            ])
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line {line_num}: {line[:50]} - {e}")
                            skipped_lines += 1
                    else:
                        print(f"Skipping malformed line {line_num}: {line}")
                        skipped_lines += 1
            
            print(f"Manual parsing: {len(attr_data)} valid lines, {skipped_lines} skipped")
            attr_df = pd.DataFrame(attr_data, columns=['image_id', 'attribute_id', 'is_present', 'certainty'])
        
        return self._process_attribute_dataframe(attr_df)
    
    def _load_compact_attributes(self) -> pd.DataFrame:
        """Load attributes from compact format (one line per image with all attributes)."""
        print("Loading compact format attributes...")
        
        attr_data = []
        with open(self.image_attributes_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 313:  # image_id + 312 attributes
                    try:
                        image_id = int(parts[0])
                        attributes = [float(x) for x in parts[1:313]]  # 312 attributes
                        
                        # Convert to long format
                        for attr_id, attr_val in enumerate(attributes, 1):
                            attr_data.append([
                                image_id,
                                attr_id,
                                int(attr_val > 0.5),  # Convert to binary
                                float(attr_val)       # Keep original as certainty
                            ])
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing compact line {line_num}: {e}")
                        continue
                else:
                    print(f"Compact line {line_num} has {len(parts)} parts, expected 313+")
        
        print(f"Loaded {len(attr_data)} attribute entries from compact format")
        attr_df = pd.DataFrame(attr_data, columns=['image_id', 'attribute_id', 'is_present', 'certainty'])
        return self._process_attribute_dataframe(attr_df)
    
    def _process_attribute_dataframe(self, attr_df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate the attribute dataframe."""
        
        print(f"Loaded {len(attr_df)} attribute annotations")
        print(f"Image IDs range: {attr_df['image_id'].min()} - {attr_df['image_id'].max()}")
        print(f"Attribute IDs range: {attr_df['attribute_id'].min()} - {attr_df['attribute_id'].max()}")
        print(f"Unique images: {attr_df['image_id'].nunique()}")
        print(f"Unique attributes: {attr_df['attribute_id'].nunique()}")
        
        # Check for expected ranges
        expected_images = 11788  # CUB has 11,788 images
        expected_attributes = 312  # CUB has 312 attributes
        
        if attr_df['image_id'].max() != expected_images:
            print(f"⚠️  WARNING: Expected max image_id {expected_images}, got {attr_df['image_id'].max()}")
        
        if attr_df['attribute_id'].max() != expected_attributes:
            print(f"⚠️  WARNING: Expected max attribute_id {expected_attributes}, got {attr_df['attribute_id'].max()}")
        
        # Check for duplicates
        duplicates = attr_df.groupby(['image_id', 'attribute_id']).size()
        num_duplicates = (duplicates > 1).sum()
        if num_duplicates > 0:
            print(f"⚠️  Found {num_duplicates} duplicate image-attribute pairs")
            # Remove duplicates by taking the first occurrence
            attr_df = attr_df.drop_duplicates(subset=['image_id', 'attribute_id'], keep='first')
            print(f"After removing duplicates: {len(attr_df)} annotations")
        
        # Pivot to get 312 attributes per image
        # Use is_present as binary labels (1/0)
        try:
            attrs_pivot = attr_df.pivot(
                index='image_id', 
                columns='attribute_id', 
                values='is_present'
            ).fillna(0)
        except Exception as e:
            print(f"Error during pivot operation: {e}")
            print("Attempting to handle duplicate entries...")
            # Handle potential duplicates by taking the mean
            attrs_pivot = attr_df.groupby(['image_id', 'attribute_id'])['is_present'].mean().unstack().fillna(0)
        
        # Ensure we have all 312 attributes
        expected_attrs = list(range(1, 313))  # 1-312
        missing_attrs = set(expected_attrs) - set(attrs_pivot.columns)
        
        if missing_attrs:
            print(f"⚠️  Missing {len(missing_attrs)} attributes, adding as zeros")
            # Create a dataframe with missing attributes as zeros
            missing_df = pd.DataFrame(0, index=attrs_pivot.index, columns=list(missing_attrs))
            # Concatenate instead of inserting one by one to avoid performance warning
            attrs_pivot = pd.concat([attrs_pivot, missing_df], axis=1)
            
        # Sort columns to ensure consistent ordering
        attrs_pivot = attrs_pivot[expected_attrs]
        
        # Convert column names to strings for consistency
        attrs_pivot.columns = [str(col) for col in attrs_pivot.columns]
        
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
        attr_cols = [str(i) for i in range(1, 313)]  # attributes 1-312 as strings
        
        print(f"Computing prevalence for {len(attr_cols)} attributes")
        print(f"DataFrame columns: {list(train_df.columns)[:10]}...")  # Show first 10 columns
        print(f"Looking for attribute columns: {attr_cols[:5]}...")  # Show first 5 expected
        
        # Check which attribute columns exist
        available_attrs = [col for col in attr_cols if col in train_df.columns]
        missing_attrs = [col for col in attr_cols if col not in train_df.columns]
        
        print(f"Available attribute columns: {len(available_attrs)}")
        if missing_attrs:
            print(f"Missing attribute columns: {len(missing_attrs)} (first 5: {missing_attrs[:5]})")
        
        if not available_attrs:
            print("❌ No attribute columns found! Check column naming.")
            # Fallback: try to find numeric columns that might be attributes
            numeric_cols = [col for col in train_df.columns if str(col).isdigit()]
            print(f"Found numeric columns: {numeric_cols[:10]}...")
            if numeric_cols:
                available_attrs = [str(col) for col in numeric_cols if 1 <= int(col) <= 312]
                print(f"Using numeric columns as attributes: {len(available_attrs)}")
        
        prevalence = {}
        train_split = train_df[train_df['split'] == 'train']
        
        for attr in available_attrs:
            if attr in train_split.columns:
                pos_count = train_split[attr].sum()
                total_count = len(train_split)
                prevalence[f'attr_{attr}'] = float(pos_count / total_count) if total_count > 0 else 0.0
            else:
                prevalence[f'attr_{attr}'] = 0.0
        
        # Add zero prevalence for missing attributes
        for attr in missing_attrs:
            prevalence[f'attr_{attr}'] = 0.0
        
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
