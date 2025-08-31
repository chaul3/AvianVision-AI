#!/bin/bash

# CUB-200-2011 Pipeline Setup and Execution Script
# This script will:
# 1. Set up virtual environment
# 2. Install dependencies
# 3. Download CUB dataset
# 4. Run the evaluation pipeline

set -e  # Exit on any error

echo "🐦 CUB-200-2011 Bird Identification Pipeline Setup"
echo "=================================================="

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Not in virtual environment. Creating one..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
fi

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install flask

echo ""
echo "🔍 Checking for existing dataset..."

# Check if dataset exists
if [ -d "data/CUB_200_2011" ]; then
    echo "✅ Dataset already exists, skipping download"
else
    echo "⬇️  Installing CUB-200-2011 dataset..."
    python install_dataset.py --data_dir data --config_output config_demo.yaml
fi

echo ""
echo "🚀 Running evaluation pipeline..."

# Check if config exists
if [ ! -f "config_demo.yaml" ]; then
    echo "⚠️  Configuration file not found, creating it..."
    python install_dataset.py --data_dir data --config_output config_demo.yaml
fi

# Run the pipeline
echo "Starting pipeline execution..."
python evaluate_pipeline.py --config config_demo.yaml

echo ""
echo "🎉 Pipeline execution complete!"
echo ""
echo "📊 Check the following files for results:"
echo "   - outputs/pipeline_results.json (detailed results)"
echo "   - outputs/ (all pipeline outputs)"
echo ""
echo "🌐 To run the web application:"
echo "   cd webapp && python app.py"
