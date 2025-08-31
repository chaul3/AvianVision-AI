"""
CUB-200-2011 Dataset Installer

Downloads and sets up the CUB-200-2011 dataset for bird identification.
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
import argparse


def download_file(url: str, filename: str) -> None:
    """Download file with progress bar."""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress


def extract_tar(tar_path: str, extract_to: str) -> None:
    """Extract tar file."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    print("Extraction complete!")


def setup_cub_dataset(data_dir: str = "data") -> str:
    """
    Download and setup CUB-200-2011 dataset.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the extracted CUB dataset
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset URL
    cub_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tar_filename = os.path.join(data_dir, "CUB_200_2011.tgz")
    
    # Check if dataset already exists
    cub_path = os.path.join(data_dir, "CUB_200_2011")
    if os.path.exists(cub_path):
        print(f"Dataset already exists at {cub_path}")
        return cub_path
    
    try:
        # Download dataset
        download_file(cub_url, tar_filename)
        
        # Extract dataset
        extract_tar(tar_filename, data_dir)
        
        # Clean up tar file
        os.remove(tar_filename)
        print(f"Cleaned up {tar_filename}")
        
        # Verify extraction
        if os.path.exists(cub_path):
            print(f"✓ Dataset successfully installed at {cub_path}")
            
            # Print dataset info
            images_dir = os.path.join(cub_path, "images")
            if os.path.exists(images_dir):
                num_species = len([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
                print(f"✓ Found {num_species} bird species")
            
            return cub_path
        else:
            raise Exception("Dataset extraction failed")
            
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        # Clean up on failure
        if os.path.exists(tar_filename):
            os.remove(tar_filename)
        if os.path.exists(cub_path):
            shutil.rmtree(cub_path)
        raise


def create_demo_config(cub_path: str, output_file: str = "config_demo.yaml") -> str:
    """
    Create a configuration file with the correct dataset path.
    
    Args:
        cub_path: Path to the CUB dataset
        output_file: Output configuration file name
        
    Returns:
        Path to the created config file
    """
    config_content = f"""# Demo Configuration for CUB-200-2011 Pipeline
# Auto-generated with dataset path

data:
  cub_root: "{cub_path}"
  output_dir: "outputs"

training:
  batch_size: 16  # Smaller for demo
  num_epochs: 5   # Fewer epochs for demo
  learning_rate: 0.001
  weight_decay: 0.0001
  num_workers: 2  # Fewer workers for demo

model:
  backbone: "resnet50"
  num_attributes: 312
  dropout_rate: 0.5

calibration:
  split: "val"
  metric: "f1"

shortlisting:
  k_candidates: 5
  min_similarity: 0.1
  use_ground_truth_centroids: false

llm:
  model_name: "llama3"
  temperature: 0.7
  max_tokens: 500

evaluation:
  test_split: "test"
  k_values: [1, 3, 5, 10]

outputs:
  dataset_manifest: "outputs/dataset_manifest.csv"
  attribute_prevalence: "outputs/attribute_prevalence.json"
  model_weights: "outputs/best_model.pth"
  thresholds: "outputs/calibrated_thresholds.json"
  centroids: "outputs/species_centroids.json"
  attribute_mapping: "outputs/attribute_mapping.json"
  results: "outputs/pipeline_results.json"

wandb:
  enabled: false
  project: "cub-bird-identification"
"""
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created configuration file: {output_file}")
    return output_file


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install CUB-200-2011 dataset and setup pipeline")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory to install dataset (default: data)")
    parser.add_argument("--config_output", type=str, default="config_demo.yaml",
                       help="Output configuration file (default: config_demo.yaml)")
    
    args = parser.parse_args()
    
    print("CUB-200-2011 Dataset Installer")
    print("=" * 40)
    
    try:
        # Install dataset
        cub_path = setup_cub_dataset(args.data_dir)
        
        # Create configuration
        config_path = create_demo_config(cub_path, args.config_output)
        
        print("\n" + "=" * 40)
        print("INSTALLATION COMPLETE!")
        print("=" * 40)
        print(f"Dataset location: {cub_path}")
        print(f"Configuration file: {config_path}")
        print(f"\nTo run the pipeline:")
        print(f"python evaluate_pipeline.py --config {config_path}")
        
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
