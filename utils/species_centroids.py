"""
Class Attribute Centroids Module

Computes per-species attribute centroids for candidate shortlisting.
Uses ground truth or predicted attribute vectors to create species profiles.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class AttributeCentroidBuilder:
    """Builds and manages species-level attribute centroids."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize centroid builder.
        
        Args:
            output_dir: Directory to save centroid outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.centroids = {}
        self.species_info = {}
    
    def compute_ground_truth_centroids(self, manifest_path: str) -> Dict[int, np.ndarray]:
        """
        Compute centroids using ground truth attribute annotations.
        
        Args:
            manifest_path: Path to dataset manifest CSV
            
        Returns:
            Dictionary mapping species_id to 312-D centroid vector
        """
        print("Computing ground truth attribute centroids...")
        
        # Load manifest
        manifest_df = pd.read_csv(manifest_path)
        
        # Only use training data for centroids
        train_df = manifest_df[manifest_df['split'] == 'train'].copy()
        
        # Extract attribute columns (1-312)
        attr_cols = [str(i) for i in range(1, 313)]
        
        centroids = {}
        species_stats = {}
        
        for species_id in train_df['class_id'].unique():
            species_data = train_df[train_df['class_id'] == species_id]
            species_name = species_data['species_name'].iloc[0]
            
            # Get attribute vectors for this species
            attr_vectors = species_data[attr_cols].values.astype(np.float32)
            
            # Compute centroid (mean)
            centroid = np.mean(attr_vectors, axis=0)
            
            # L2 normalize
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            
            centroids[species_id] = centroid
            
            # Store species info
            species_stats[species_id] = {
                'species_name': species_name,
                'n_samples': len(species_data),
                'centroid_magnitude': float(centroid_norm)
            }
        
        print(f"✓ Computed centroids for {len(centroids)} species")
        
        # Store for later use
        self.centroids = centroids
        self.species_info = species_stats
        
        return centroids
    
    def compute_predicted_centroids(self,
                                  model_path: str,
                                  manifest_path: str) -> Dict[int, np.ndarray]:
        """
        Compute centroids using model predictions on training data.
        
        Args:
            model_path: Path to trained attribute model
            manifest_path: Path to dataset manifest
            
        Returns:
            Dictionary mapping species_id to 312-D centroid vector
        """
        print("Computing predicted attribute centroids...")
        
        # Import here to avoid dependency issues
        from models.train_attribute_model import ResNetAttributeClassifier, CUBAttributeDataset
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['config']
        
        model = ResNetAttributeClassifier(
            backbone=model_config['backbone'],
            num_attributes=312,
            pretrained=False,
            dropout=model_config['dropout']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load dataset
        manifest_df = pd.read_csv(manifest_path)
        
        # Inference transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Only use training data
        train_dataset = CUBAttributeDataset(manifest_df, 'train', transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Generate predictions
        species_predictions = defaultdict(list)
        species_names = {}
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Predicting"):
                images = batch['image'].to(device)
                class_ids = batch['class_id'].numpy()
                species_names_batch = batch['species_name']
                
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for i, class_id in enumerate(class_ids):
                    species_predictions[class_id].append(probs[i])
                    species_names[class_id] = species_names_batch[i]
        
        # Compute centroids
        centroids = {}
        species_stats = {}
        
        for species_id, predictions in species_predictions.items():
            # Stack predictions
            pred_array = np.stack(predictions)
            
            # Compute centroid
            centroid = np.mean(pred_array, axis=0)
            
            # L2 normalize
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            
            centroids[species_id] = centroid
            
            # Store species info
            species_stats[species_id] = {
                'species_name': species_names[species_id],
                'n_samples': len(predictions),
                'centroid_magnitude': float(centroid_norm)
            }
        
        print(f"✓ Computed predicted centroids for {len(centroids)} species")
        
        # Store for later use
        self.centroids = centroids
        self.species_info = species_stats
        
        return centroids
    
    def analyze_centroid_similarities(self, 
                                    centroids: Dict[int, np.ndarray],
                                    top_k: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """
        Analyze pairwise similarities between species centroids.
        
        Args:
            centroids: Species centroids
            top_k: Number of most similar species to return per species
            
        Returns:
            Dictionary mapping species_id to list of (similar_species_id, similarity_score)
        """
        print("Analyzing centroid similarities...")
        
        species_ids = list(centroids.keys())
        centroid_matrix = np.stack([centroids[sid] for sid in species_ids])
        
        # Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(centroid_matrix)
        
        # Find most similar species for each species
        similarities = {}
        
        for i, species_id in enumerate(species_ids):
            # Get similarities for this species (excluding self)
            sim_scores = similarity_matrix[i]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Get top-k most similar (excluding self)
            similar_species = []
            for j in sorted_indices[1:top_k+1]:  # Skip self (index 0)
                similar_id = species_ids[j]
                score = sim_scores[j]
                similar_species.append((similar_id, float(score)))
            
            similarities[species_id] = similar_species
        
        return similarities
    
    def visualize_centroid_distribution(self,
                                      centroids: Dict[int, np.ndarray],
                                      save_path: Optional[str] = None):
        """
        Visualize distribution of centroid magnitudes and attribute activations.
        
        Args:
            centroids: Species centroids
            save_path: Path to save visualization
        """
        print("Creating centroid visualizations...")
        
        # Extract centroid data
        species_ids = list(centroids.keys())
        centroid_matrix = np.stack([centroids[sid] for sid in species_ids])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Centroid magnitude distribution
        magnitudes = [np.linalg.norm(centroids[sid]) for sid in species_ids]
        axes[0, 0].hist(magnitudes, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Centroid Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Centroid Magnitudes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mean attribute activation per position
        mean_activations = np.mean(centroid_matrix, axis=0)
        axes[0, 1].plot(range(1, 313), mean_activations, alpha=0.7)
        axes[0, 1].set_xlabel('Attribute Index')
        axes[0, 1].set_ylabel('Mean Activation')
        axes[0, 1].set_title('Mean Attribute Activations Across Species')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Centroid similarity heatmap (sample)
        sample_size = min(20, len(species_ids))
        sample_indices = np.random.choice(len(species_ids), sample_size, replace=False)
        sample_centroids = centroid_matrix[sample_indices]
        sample_similarities = cosine_similarity(sample_centroids)
        
        sns.heatmap(sample_similarities, 
                   ax=axes[1, 0],
                   cmap='coolwarm',
                   center=0,
                   annot=False,
                   cbar_kws={'label': 'Cosine Similarity'})
        axes[1, 0].set_title(f'Pairwise Similarities (Sample of {sample_size} Species)')
        
        # 4. Attribute variance across species
        attr_variances = np.var(centroid_matrix, axis=0)
        axes[1, 1].hist(attr_variances, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Attribute Variance')
        axes[1, 1].set_ylabel('Frequency') 
        axes[1, 1].set_title('Distribution of Attribute Variances')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Centroid visualization saved to {save_path}")
        
        return fig
    
    def save_centroids(self, 
                      centroids: Dict[int, np.ndarray],
                      filename: str = "species_centroids.npz"):
        """
        Save centroids to disk.
        
        Args:
            centroids: Species centroids
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert to format suitable for numpy
        species_ids = list(centroids.keys())
        centroid_matrix = np.stack([centroids[sid] for sid in species_ids])
        
        # Convert species_info keys to strings for JSON serialization
        species_info_serializable = {str(k): v for k, v in self.species_info.items()}
        
        np.savez(
            output_path,
            centroids=centroid_matrix,
            species_ids=np.array(species_ids),
            species_info=json.dumps(species_info_serializable)
        )
        
        print(f"✓ Centroids saved to {output_path}")
        
        # Also save as JSON for easier loading
        json_output = self.output_dir / f"{filename.split('.')[0]}.json"
        centroid_dict = {}
        
        for species_id, centroid in centroids.items():
            centroid_dict[str(species_id)] = {
                'centroid': centroid.tolist(),
                'species_info': self.species_info.get(species_id, {})
            }
        
        with open(json_output, 'w') as f:
            json.dump(centroid_dict, f, indent=2)
        
        print(f"✓ Centroids also saved as JSON to {json_output}")
    
    def load_centroids(self, filepath: str) -> Dict[int, np.ndarray]:
        """
        Load centroids from disk.
        
        Args:
            filepath: Path to centroids file (.npz or .json)
            
        Returns:
            Dictionary of species centroids
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            data = np.load(filepath)
            species_ids = data['species_ids']
            centroid_matrix = data['centroids']
            
            centroids = {}
            for i, species_id in enumerate(species_ids):
                centroids[int(species_id)] = centroid_matrix[i]
            
            # Load species info if available
            if 'species_info' in data:
                self.species_info = json.loads(str(data['species_info']))
        
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            centroids = {}
            species_info = {}
            
            for species_id_str, species_data in data.items():
                species_id = int(species_id_str)
                centroids[species_id] = np.array(species_data['centroid'])
                species_info[species_id] = species_data.get('species_info', {})
            
            self.species_info = species_info
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.centroids = centroids
        print(f"✓ Loaded centroids for {len(centroids)} species")
        
        return centroids
    
    def build_centroids(self,
                       manifest_path: str,
                       model_path: Optional[str] = None,
                       use_ground_truth: bool = True) -> Dict[int, np.ndarray]:
        """
        Main function to build species centroids.
        
        Args:
            manifest_path: Path to dataset manifest
            model_path: Path to trained model (if using predictions)
            use_ground_truth: Whether to use ground truth or predicted attributes
            
        Returns:
            Species centroids dictionary
        """
        print("Building species attribute centroids...")
        
        if use_ground_truth:
            centroids = self.compute_ground_truth_centroids(manifest_path)
        else:
            if model_path is None:
                raise ValueError("model_path required when use_ground_truth=False")
            centroids = self.compute_predicted_centroids(model_path, manifest_path)
        
        # Analyze similarities
        similarities = self.analyze_centroid_similarities(centroids)
        
        # Save similarity analysis
        similarity_path = self.output_dir / "species_similarities.json"
        similarity_output = {}
        
        for species_id, similar_list in similarities.items():
            species_name = self.species_info[species_id]['species_name']
            similar_formatted = []
            
            for sim_id, sim_score in similar_list:
                sim_name = self.species_info[sim_id]['species_name']
                similar_formatted.append({
                    'species_id': int(sim_id),  # Convert numpy int64 to Python int
                    'species_name': sim_name,
                    'similarity': float(sim_score)  # Convert numpy float to Python float
                })
            
            similarity_output[str(species_id)] = {
                'species_name': species_name,
                'most_similar': similar_formatted
            }
        
        with open(similarity_path, 'w') as f:
            json.dump(similarity_output, f, indent=2)
        
        print(f"✓ Species similarities saved to {similarity_path}")
        
        # Create visualizations
        viz_path = self.output_dir / "centroid_analysis.png"
        self.visualize_centroid_distribution(centroids, viz_path)
        
        # Save centroids
        centroid_type = "ground_truth" if use_ground_truth else "predicted"
        self.save_centroids(centroids, f"species_centroids_{centroid_type}.npz")
        
        print("✓ Centroid building completed!")
        
        return centroids


def main():
    """Example usage of centroid builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CUB species centroids")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to dataset manifest CSV")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (for predicted centroids)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--use_predictions", action="store_true",
                       help="Use model predictions instead of ground truth")
    
    args = parser.parse_args()
    
    # Build centroids
    builder = AttributeCentroidBuilder(args.output_dir)
    centroids = builder.build_centroids(
        args.manifest,
        args.model,
        use_ground_truth=not args.use_predictions
    )
    
    print(f"Built centroids for {len(centroids)} species")


if __name__ == "__main__":
    main()
