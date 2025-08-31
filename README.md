# 🐦 AvianVision-AI

**Advanced Bird Species Identification using Vision→Attributes→LLM Pipeline**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive pipeline for bird species identification combining computer vision, attribute-based reasoning, and large language models. Built on the CUB-200-2011 dataset with 200 bird species and 312 visual attributes.

## 🌟 Features

- **🎯 Complete Pipeline**: Vision→Attributes→LLM reasoning for interpretable bird identification
- **🧠 ResNet50 Backbone**: Fine-tuned on 312 bird-specific visual attributes
- **📊 Real CUB Dataset**: Authentic CUB-200-2011 species with 200 bird classes
- **🤖 Multi-LLM Support**: Ollama Llama 3, HuggingFace API, and enhanced mock reasoning
- **🌐 Web Interface**: Beautiful Flask webapp with drag-and-drop image upload
- **⚡ Real-time Processing**: Fast inference with model fallbacks
- **🔍 Explainable AI**: Shows detected attributes and reasoning process
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/chaul3/AvianVision-AI.git
cd AvianVision-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
cd webapp
python app.py
```

Visit `http://localhost:5001` to start identifying birds! 🐦

## 📊 Pipeline Architecture

```
Image Input → ResNet50 → 312 Attributes → Verbalization → Candidate Shortlisting → LLM Reasoning → Species ID
```

### 🔬 Technical Components

1. **Vision Model**: ResNet50 fine-tuned for 312-dimensional attribute prediction
2. **Attribute Verbalization**: Converts predictions to natural language descriptions
3. **Candidate Shortlisting**: Cosine similarity with species centroids
4. **LLM Reasoning**: Expert ornithologist simulation for final classification

## 🛠️ Training Your Own Model

### Prerequisites
- CUB-200-2011 dataset
- GPU with CUDA support (recommended)
- 16GB+ RAM

### Training Pipeline
```bash
# Configure your paths in config_cluster.yaml
python evaluate_pipeline.py --config config_cluster.yaml
```

### Replace Webapp Model
After training, copy your model files:
```bash
./copy_trained_model.sh
```

Expected files:
- `webapp/models/bird_attributes_model.pth` - Trained ResNet50 weights
- `webapp/models/optimal_thresholds.json` - Calibrated thresholds
- `webapp/models/species_centroids.npy` - Species centroids (200, 312)

## 📁 Project Structure

```
AvianVision-AI/
├── 📂 webapp/                    # Web application
│   ├── app.py                   # Main Flask application
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JS, uploads
│   └── models/                  # Trained model weights
├── 📂 notebooks/                # Jupyter notebooks
│   └── CUB_Bird_Pipeline_Complete.ipynb
├── 📂 data/                     # Dataset processing
├── 📂 models/                   # Model definitions
├── 📂 utils/                    # Utility functions
├── 📂 evaluation/               # Evaluation scripts
├── evaluate_pipeline.py         # Complete training pipeline
├── config_cluster.yaml          # Training configuration
└── requirements.txt             # Dependencies
```

## 🎯 Model Performance

- **312 Visual Attributes**: Bill shape, wing color, size, patterns, etc.
- **200 Bird Species**: Complete CUB-200-2011 dataset coverage
- **Multi-tier LLM**: Ollama → HuggingFace → Enhanced Mock fallback
- **Real-time Inference**: < 2 seconds per image

## 🌐 Web Interface Features

- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Processing**: Live progress indicators
- **Detailed Results**: Species prediction with confidence scores
- **Attribute Visualization**: Shows detected bird features
- **LLM Prompt Display**: Transparency into reasoning process
- **Candidate Analysis**: Top-5 species with similarity scores

## 🔧 Configuration

### LLM Options
1. **Ollama Llama 3** (Recommended): Local inference with full privacy
2. **HuggingFace API**: Cloud-based models with API access
3. **Enhanced Mock**: Intelligent fallback with CUB species knowledge

### Model Modes
- **Trained Mode**: Uses your custom-trained weights
- **Demo Mode**: Pre-trained ResNet50 with mock attributes
- **Hybrid Mode**: Mix of real and simulated components

## 📈 Results & Examples

Upload a bird image to see:
- **Species Identification**: "Cardinal" with 87.3% confidence
- **Key Features**: "Brilliant red plumage, prominent crest, thick orange beak"
- **Reasoning**: "The observed features strongly indicate Cardinal based on the distinctive red coloration and crest shape..."
- **Attribute Analysis**: 45 active attributes across 6 feature groups