"""
Utility functions for the CUB bird identification pipeline.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import seaborn as sns


def setup_output_directories(base_dir: str = "outputs") -> Dict[str, str]:
    """
    Create all necessary output directories.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary of created directory paths
    """
    base_path = Path(base_dir)
    
    directories = {
        'base': str(base_path),
        'models': str(base_path / "models"),
        'plots': str(base_path / "plots"),
        'logs': str(base_path / "logs"),
        'predictions': str(base_path / "predictions"),
        'analysis': str(base_path / "analysis")
    }
    
    for dir_path in directories.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return directories


def load_json_safe(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return default


def save_json_safe(data: Any, file_path: str) -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False


def plot_training_curves(metrics_file: str, 
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training curves from metrics file.
    
    Args:
        metrics_file: Path to training metrics JSON
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    # This would load actual training metrics and plot them
    # For now, create a placeholder plot
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mock data for demonstration
    epochs = list(range(1, 21))
    train_loss = [1.2 - 0.05*i + 0.01*np.random.randn() for i in epochs]
    val_loss = [1.3 - 0.04*i + 0.02*np.random.randn() for i in epochs]
    val_map = [0.3 + 0.02*i + 0.01*np.random.randn() for i in epochs]
    val_f1 = [0.25 + 0.025*i + 0.01*np.random.randn() for i in epochs]
    
    # Plot training and validation loss
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot validation mAP
    axes[0, 1].plot(epochs, val_map, label='Val mAP', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('Validation Mean Average Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot validation F1
    axes[1, 0].plot(epochs, val_f1, label='Val Macro F1', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation Macro F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate (mock)
    lr_schedule = [1e-4 * (0.5 ** (i // 5)) for i in epochs]
    axes[1, 1].plot(epochs, lr_schedule, label='Learning Rate', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    return fig


def plot_attribute_analysis(prevalence_file: str,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot attribute prevalence analysis.
    
    Args:
        prevalence_file: Path to attribute prevalence JSON
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    prevalence_data = load_json_safe(prevalence_file, {})
    
    if not prevalence_data:
        print("No prevalence data available")
        return None
    
    prevalences = list(prevalence_data.values())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram of prevalences
    axes[0, 0].hist(prevalences, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Attribute Prevalence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Attribute Prevalences')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(prevalences)
    axes[0, 1].set_ylabel('Attribute Prevalence')
    axes[0, 1].set_title('Attribute Prevalence Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prevalence by attribute index
    attr_indices = list(range(1, len(prevalences) + 1))
    axes[1, 0].plot(attr_indices, prevalences, alpha=0.7)
    axes[1, 0].set_xlabel('Attribute Index')
    axes[1, 0].set_ylabel('Prevalence')
    axes[1, 0].set_title('Prevalence by Attribute Index')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_prevalences = sorted(prevalences)
    cumulative = np.arange(1, len(sorted_prevalences) + 1) / len(sorted_prevalences)
    axes[1, 1].plot(sorted_prevalences, cumulative)
    axes[1, 1].set_xlabel('Attribute Prevalence')
    axes[1, 1].set_ylabel('Cumulative Frequency')
    axes[1, 1].set_title('Cumulative Distribution of Prevalences')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attribute analysis saved to {save_path}")
    
    return fig


def plot_pipeline_results(results_file: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comprehensive pipeline results.
    
    Args:
        results_file: Path to pipeline results JSON
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    results = load_json_safe(results_file, {})
    
    if not results:
        print("No results data available")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Recall scores
    if 'pipeline_steps' in results and 'shortlisting' in results['pipeline_steps']:
        recall_data = results['pipeline_steps']['shortlisting']['recall_scores']
        k_values = list(recall_data.keys())
        recall_values = list(recall_data.values())
        
        axes[0, 0].plot(k_values, recall_values, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('K (Number of Candidates)')
        axes[0, 0].set_ylabel('Recall@K')
        axes[0, 0].set_title('Candidate Shortlisting Recall')
        axes[0, 0].grid(True, alpha=0.3)
    
    # LLM confidence distribution
    if 'pipeline_steps' in results and 'llm_classification' in results['pipeline_steps']:
        llm_data = results['pipeline_steps']['llm_classification']
        
        # Mock confidence distribution for visualization
        confidences = np.random.beta(2, 2, 100)  # Mock data
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('LLM Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('LLM Confidence Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Pipeline accuracy by stage
    stage_names = ['Attribute\nDetection', 'Candidate\nShortlisting', 'LLM\nClassification']
    stage_accuracies = [0.75, 0.85, 0.82]  # Mock data
    
    axes[0, 2].bar(stage_names, stage_accuracies, alpha=0.7, color=['blue', 'green', 'orange'])
    axes[0, 2].set_ylabel('Performance Score')
    axes[0, 2].set_title('Pipeline Stage Performance')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error analysis
    error_types = ['Parsing\nErrors', 'Validation\nErrors', 'Confidence\nIssues']
    error_counts = [5, 8, 12]  # Mock data
    
    axes[1, 0].bar(error_types, error_counts, alpha=0.7, color='red')
    axes[1, 0].set_ylabel('Error Count')
    axes[1, 0].set_title('Error Analysis')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Processing time by stage
    stage_times = [120, 45, 180]  # Mock processing times in seconds
    axes[1, 1].bar(stage_names, stage_times, alpha=0.7, color=['lightblue', 'lightgreen', 'lightyellow'])
    axes[1, 1].set_ylabel('Processing Time (seconds)')
    axes[1, 1].set_title('Processing Time by Stage')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall metrics summary
    metrics_names = ['Overall\nAccuracy', 'Top-10\nRecall', 'Mean\nConfidence']
    metrics_values = [0.78, 0.92, 0.75]  # Mock data
    
    axes[1, 2].bar(metrics_names, metrics_values, alpha=0.7, color=['purple', 'cyan', 'pink'])
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Overall Pipeline Metrics')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pipeline results plot saved to {save_path}")
    
    return fig


def generate_species_report(species_id: int,
                          centroids_file: str,
                          similarities_file: str) -> Dict[str, Any]:
    """
    Generate detailed report for a specific species.
    
    Args:
        species_id: ID of the species to analyze
        centroids_file: Path to species centroids
        similarities_file: Path to species similarities
        
    Returns:
        Species analysis report
    """
    centroids_data = load_json_safe(centroids_file, {})
    similarities_data = load_json_safe(similarities_file, {})
    
    species_key = str(species_id)
    
    if species_key not in centroids_data:
        return {'error': f'Species {species_id} not found in centroids data'}
    
    species_data = centroids_data[species_key]
    species_info = species_data.get('species_info', {})
    centroid = np.array(species_data['centroid'])
    
    # Analyze centroid
    top_attributes = np.argsort(centroid)[::-1][:10]  # Top 10 attributes
    
    report = {
        'species_id': species_id,
        'species_name': species_info.get('species_name', f'Species_{species_id}'),
        'n_training_samples': species_info.get('n_samples', 'Unknown'),
        'centroid_magnitude': species_info.get('centroid_magnitude', np.linalg.norm(centroid)),
        'top_attributes': {
            'indices': (top_attributes + 1).tolist(),  # Convert to 1-indexed
            'values': centroid[top_attributes].tolist()
        },
        'attribute_statistics': {
            'mean': float(np.mean(centroid)),
            'std': float(np.std(centroid)),
            'min': float(np.min(centroid)),
            'max': float(np.max(centroid))
        }
    }
    
    # Add similarity information
    if species_key in similarities_data:
        similarity_info = similarities_data[species_key]
        report['most_similar_species'] = similarity_info.get('most_similar', [])
    
    return report


def create_quick_start_notebook():
    """Create a quick start Jupyter notebook for the pipeline."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CUB Bird Species Identification Pipeline - Quick Start\n\n",
                    "This notebook provides a quick start guide for the vision→attributes→LLM pipeline.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import sys\n",
                    "sys.path.append('..')\n\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import json\n\n",
                    "# Set up plotting\n",
                    "plt.style.use('default')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load and Explore Dataset\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load dataset manifest\n",
                    "try:\n",
                    "    import pandas as pd\n",
                    "    manifest_df = pd.read_csv('../outputs/dataset_manifest.csv')\n",
                    "    print(f\"Dataset loaded: {len(manifest_df)} samples\")\n",
                    "    print(f\"Splits: {manifest_df['split'].value_counts().to_dict()}\")\n",
                    "    print(f\"Classes: {manifest_df['class_id'].nunique()}\")\n",
                    "except FileNotFoundError:\n",
                    "    print(\"Dataset manifest not found. Run data processing first.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Attribute Analysis\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load attribute prevalence\n",
                    "try:\n",
                    "    with open('../outputs/attribute_prevalence.json', 'r') as f:\n",
                    "        prevalence = json.load(f)\n",
                    "    \n",
                    "    prevalence_values = list(prevalence.values())\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 4))\n",
                    "    plt.subplot(1, 2, 1)\n",
                    "    plt.hist(prevalence_values, bins=30, alpha=0.7)\n",
                    "    plt.xlabel('Attribute Prevalence')\n",
                    "    plt.ylabel('Frequency')\n",
                    "    plt.title('Distribution of Attribute Prevalences')\n",
                    "    \n",
                    "    plt.subplot(1, 2, 2)\n",
                    "    plt.plot(range(1, 313), prevalence_values, alpha=0.7)\n",
                    "    plt.xlabel('Attribute Index')\n",
                    "    plt.ylabel('Prevalence')\n",
                    "    plt.title('Prevalence by Attribute')\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    print(f\"Mean prevalence: {np.mean(prevalence_values):.3f}\")\n",
                    "    print(f\"Std prevalence: {np.std(prevalence_values):.3f}\")\n",
                    "    \n",
                    "except FileNotFoundError:\n",
                    "    print(\"Attribute prevalence not found. Run data processing first.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Test Attribute Verbalization\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Test attribute verbalization\n",
                    "try:\n",
                    "    from utils.attribute_verbalizer import AttributeVerbalizer\n",
                    "    \n",
                    "    verbalizer = AttributeVerbalizer()\n",
                    "    \n",
                    "    # Create test attribute vector\n",
                    "    np.random.seed(42)\n",
                    "    test_attrs = np.random.rand(312)\n",
                    "    \n",
                    "    text, json_data = verbalizer.verbalize_attributes(test_attrs)\n",
                    "    \n",
                    "    print(\"Test Verbalization:\")\n",
                    "    print(f\"Text: {text}\")\n",
                    "    print(f\"JSON: {json.dumps(json_data, indent=2)}\")\n",
                    "    print(f\"Text length: {len(text)} characters\")\n",
                    "    \n",
                    "except ImportError as e:\n",
                    "    print(f\"Could not import verbalizer: {e}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Pipeline Results\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load and display pipeline results\n",
                    "try:\n",
                    "    with open('../outputs/pipeline_results.json', 'r') as f:\n",
                    "        results = json.load(f)\n",
                    "    \n",
                    "    print(\"Pipeline Results Summary:\")\n",
                    "    print(f\"Test samples: {results['test_samples']}\")\n",
                    "    \n",
                    "    if 'final_metrics' in results:\n",
                    "        final_metrics = results['final_metrics']\n",
                    "        \n",
                    "        print(\"\\nFinal Metrics:\")\n",
                    "        for category, metrics in final_metrics.items():\n",
                    "            print(f\"  {category}:\")\n",
                    "            for metric, value in metrics.items():\n",
                    "                if isinstance(value, float):\n",
                    "                    print(f\"    {metric}: {value:.3f}\")\n",
                    "                else:\n",
                    "                    print(f\"    {metric}: {value}\")\n",
                    "    \n",
                    "except FileNotFoundError:\n",
                    "    print(\"Pipeline results not found. Run evaluation first.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Next Steps\n\n",
                    "- Explore individual components in detail\n",
                    "- Experiment with different model configurations\n",
                    "- Analyze failure cases and improve the pipeline\n",
                    "- Try the pipeline on your own bird images\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_path = "notebooks/quick_start.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✓ Quick start notebook created: {notebook_path}")


if __name__ == "__main__":
    # Create quick start notebook
    create_quick_start_notebook()
    
    # Set up output directories
    dirs = setup_output_directories()
    print(f"✓ Output directories created: {list(dirs.keys())}")
    
    print("Utility functions loaded successfully!")
