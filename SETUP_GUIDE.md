# 🚀 CUB-200-2011 Pipeline Setup Guide

This guide provides multiple ways to install the CUB-200-2011 dataset and run the bird identification pipeline.

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Internet connection for dataset download
- ~1.2GB free disk space for dataset
- GPU recommended (but not required)

## 🔧 Setup Options

### Option 1: Automated Setup (Recommended)

The easiest way to get started:

```bash
# Complete automated setup
python setup_pipeline.py
```

This will:
1. ✅ Check virtual environment
2. 📦 Install all dependencies
3. ⬇️ Download CUB-200-2011 dataset (~1.2GB)
4. ⚙️ Create configuration file
5. 🚀 Run the complete pipeline

### Option 2: Bash Script (Unix/macOS)

```bash
# Make executable and run
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Option 3: Step-by-Step Manual Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset and create config
python install_dataset.py

# 4. Run pipeline
python evaluate_pipeline.py --config config_demo.yaml
```

### Option 4: Dataset Only

To just download and setup the dataset:

```bash
python install_dataset.py --data_dir data --config_output config_demo.yaml
```

## 🎛️ Advanced Options

### Skip Dataset Download
If you already have the dataset or want to use a different path:

```bash
python setup_pipeline.py --skip-dataset
```

### Skip Dependency Installation
If dependencies are already installed:

```bash
python setup_pipeline.py --skip-deps
```

### Pipeline Only
Skip all setup and just run the pipeline:

```bash
python setup_pipeline.py --pipeline-only
```

### Custom Dataset Location
```bash
python install_dataset.py --data_dir /path/to/your/data --config_output my_config.yaml
python evaluate_pipeline.py --config my_config.yaml
```

## 📊 Pipeline Stages

The complete pipeline includes:

1. **Data Processing** - Process CUB dataset and create manifest
2. **Model Training** - Train ResNet50 for attribute detection
3. **Threshold Calibration** - Calibrate per-attribute thresholds
4. **Centroid Computation** - Compute species attribute centroids
5. **Attribute Mapping** - Create verbalization mappings
6. **Pipeline Evaluation** - End-to-end evaluation

## 🎯 Pipeline Options

### Quick Evaluation (Skip Training)
```bash
python evaluate_pipeline.py --config config_demo.yaml --skip_training
```

### Specific Steps Only
```bash
python evaluate_pipeline.py --config config_demo.yaml --steps data mapping evaluate
```

### Evaluation Only
```bash
python evaluate_pipeline.py --config config_demo.yaml --eval_only
```

## 📁 Output Files

After running the pipeline, you'll find:

```
outputs/
├── dataset_manifest.csv          # Dataset metadata
├── attribute_prevalence.json     # Attribute statistics
├── best_model.pth                # Trained model weights
├── calibrated_thresholds.json    # Calibrated thresholds
├── species_centroids.json        # Species attribute centroids
├── attribute_mapping.json        # Attribute verbalizations
└── pipeline_results.json         # Final evaluation results
```

## 🌐 Web Application

After pipeline setup, run the web interface:

```bash
cd webapp
python app.py
```

Visit `http://localhost:5001` to use the bird identification interface.

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Issues
```bash
# Manual dataset download
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz -C data/
```

### Memory Issues
Reduce batch size in configuration:
```yaml
training:
  batch_size: 8  # Reduce from 16
  num_workers: 1 # Reduce from 2
```

### GPU Issues
For CPU-only training, the pipeline will automatically detect and use CPU.

## 📞 Support

If you encounter issues:
1. Check the error logs in the terminal output
2. Verify all prerequisites are installed
3. Try the manual setup steps
4. Check the GitHub repository for updates
