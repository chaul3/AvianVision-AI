"""
Complete Setup and Pipeline Execution Script

This script will:
1. Set up virtual environment (if needed)
2. Install dependencies
3. Download CUB-200-2011 dataset
4. Run the evaluation pipeline
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, description: str = None) -> None:
    """Run a shell command with error handling."""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        sys.exit(1)


def check_virtual_env() -> bool:
    """Check if we're in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def setup_environment():
    """Set up the Python environment and dependencies."""
    print("🐦 CUB-200-2011 Bird Identification Pipeline Setup")
    print("=" * 50)
    
    # Check virtual environment
    if not check_virtual_env():
        venv_path = Path("venv")
        if not venv_path.exists():
            print("📦 Creating virtual environment...")
            run_command("python3 -m venv venv")
        
        print("⚠️  Please activate the virtual environment and run this script again:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        print(f"   python {' '.join(sys.argv)}")
        sys.exit(0)
    else:
        print("✅ Virtual environment is active")
    
    # Install dependencies
    print("\n📚 Installing dependencies...")
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        run_command("pip install -r requirements.txt", "Installing requirements")
    else:
        # Install essential packages
        packages = [
            "torch>=1.12.0",
            "torchvision>=0.13.0", 
            "scikit-learn>=1.1.0",
            "pandas>=1.4.0",
            "numpy>=1.21.0",
            "Pillow>=9.0.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pyyaml>=6.0",
            "flask"
        ]
        for package in packages:
            run_command(f"pip install {package}", f"Installing {package}")


def setup_dataset():
    """Download and setup the CUB-200-2011 dataset."""
    print("\n🔍 Checking for CUB-200-2011 dataset...")
    
    dataset_path = Path("data/CUB_200_2011")
    if dataset_path.exists():
        print("✅ Dataset already exists, skipping download")
        return str(dataset_path)
    
    print("⬇️  Installing CUB-200-2011 dataset...")
    run_command("python install_dataset.py --data_dir data --config_output config_demo.yaml")
    
    return str(dataset_path)


def run_pipeline():
    """Execute the evaluation pipeline."""
    print("\n🚀 Running evaluation pipeline...")
    
    # Ensure config exists
    config_path = Path("config_demo.yaml")
    if not config_path.exists():
        print("⚠️  Configuration file not found, creating it...")
        run_command("python install_dataset.py --data_dir data --config_output config_demo.yaml")
    
    # Run pipeline
    print("Starting pipeline execution...")
    run_command("python evaluate_pipeline.py --config config_demo.yaml")


def main():
    """Main setup and execution function."""
    parser = argparse.ArgumentParser(description="Setup and run CUB-200-2011 pipeline")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset download (use existing dataset)")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--pipeline-only", action="store_true",
                       help="Only run pipeline (skip setup)")
    
    args = parser.parse_args()
    
    try:
        if not args.pipeline_only:
            if not args.skip_deps:
                setup_environment()
            
            if not args.skip_dataset:
                setup_dataset()
        
        run_pipeline()
        
        print("\n🎉 Pipeline execution complete!")
        print("\n📊 Check the following files for results:")
        print("   - outputs/pipeline_results.json (detailed results)")
        print("   - outputs/ (all pipeline outputs)")
        print("\n🌐 To run the web application:")
        print("   cd webapp && python app.py")
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
