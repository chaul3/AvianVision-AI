"""
Candidate Shortlisting Module

Generates top-K species candidates based on attribute similarity.
Uses cosine similarity between predicted attributes and species centroids.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class CandidateShortlister:
    """Generates species candidate shortlists based on attribute similarity."""
    
    def __init__(self, 
                 centroids_path: str,
                 species_info_path: Optional[str] = None):
        """
        Initialize shortlister with precomputed centroids.
        
        Args:
            centroids_path: Path to species centroids file
            species_info_path: Optional path to species information
        """
        self.centroids = self.load_centroids(centroids_path)
        self.species_ids = list(self.centroids.keys())
        self.centroid_matrix = np.stack([self.centroids[sid] for sid in self.species_ids])
        
        # Load species information if available
        if species_info_path and Path(species_info_path).exists():
            with open(species_info_path, 'r') as f:
                self.species_info = json.load(f)
        else:
            self.species_info = {}
        
        print(f"Initialized shortlister with {len(self.centroids)} species centroids")
    
    def load_centroids(self, centroids_path: str) -> Dict[int, np.ndarray]:
        """
        Load species centroids from file.
        
        Args:
            centroids_path: Path to centroids file (.npz or .json)
            
        Returns:
            Dictionary mapping species_id to centroid vector
        """
        centroids_path = Path(centroids_path)
        
        if centroids_path.suffix == '.npz':
            data = np.load(centroids_path)
            species_ids = data['species_ids']
            centroid_matrix = data['centroids']
            
            centroids = {}
            for i, species_id in enumerate(species_ids):
                centroids[int(species_id)] = centroid_matrix[i]
        
        elif centroids_path.suffix == '.json':
            with open(centroids_path, 'r') as f:
                data = json.load(f)
            
            centroids = {}
            for species_id_str, species_data in data.items():
                species_id = int(species_id_str)
                centroids[species_id] = np.array(species_data['centroid'])
        
        else:
            raise ValueError(f"Unsupported centroid file format: {centroids_path.suffix}")
        
        return centroids
    
    def normalize_prediction(self, prediction_vector: np.ndarray) -> np.ndarray:
        """
        Normalize prediction vector for similarity computation.
        
        Args:
            prediction_vector: Raw attribute prediction vector (312-D)
            
        Returns:
            L2-normalized prediction vector
        """
        norm = np.linalg.norm(prediction_vector)
        if norm > 0:
            return prediction_vector / norm
        else:
            return prediction_vector
    
    def compute_similarities(self, 
                           prediction_vector: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between prediction and all species centroids.
        
        Args:
            prediction_vector: Normalized attribute prediction vector
            
        Returns:
            Array of similarity scores for each species
        """
        # Reshape for sklearn
        pred_reshaped = prediction_vector.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(pred_reshaped, self.centroid_matrix)
        
        return similarities[0]  # Return 1D array
    
    def get_top_k_candidates(self,
                           prediction_vector: np.ndarray,
                           k: int = 10,
                           min_similarity: float = 0.0) -> List[Tuple[int, str, float]]:
        """
        Get top-K most similar species candidates.
        
        Args:
            prediction_vector: Attribute prediction vector (312-D)
            k: Number of candidates to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (species_id, species_name, similarity_score) tuples
        """
        # Normalize prediction
        norm_prediction = self.normalize_prediction(prediction_vector)
        
        # Compute similarities
        similarities = self.compute_similarities(norm_prediction)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter by minimum similarity if specified
        candidates = []
        
        for idx in top_k_indices:
            species_id = self.species_ids[idx]
            similarity = similarities[idx]
            
            if similarity >= min_similarity:
                # Get species name
                species_name = self.get_species_name(species_id)
                candidates.append((species_id, species_name, float(similarity)))
        
        return candidates
    
    def get_species_name(self, species_id: int) -> str:
        """
        Get species name for a given species ID.
        
        Args:
            species_id: Species ID
            
        Returns:
            Species name or default name
        """
        # Try to get from species_info
        if str(species_id) in self.species_info:
            species_data = self.species_info[str(species_id)]
            if isinstance(species_data, dict):
                return species_data.get('species_name', f'Species_{species_id}')
            else:
                return str(species_data)
        
        # Default fallback
        return f'Species_{species_id}'
    
    def get_candidate_descriptions(self,
                                 candidates: List[Tuple[int, str, float]],
                                 centroids_for_description: bool = True) -> List[Dict]:
        """
        Get detailed descriptions for candidate species.
        
        Args:
            candidates: List of candidate tuples
            centroids_for_description: Whether to use centroids for descriptions
            
        Returns:
            List of candidate dictionaries with descriptions
        """
        # Import verbalizer for centroid descriptions
        from utils.attribute_verbalizer import AttributeVerbalizer
        
        verbalizer = AttributeVerbalizer()
        candidate_descriptions = []
        
        for species_id, species_name, similarity in candidates:
            candidate_info = {
                'species_id': species_id,
                'species_name': species_name,
                'similarity_score': similarity
            }
            
            if centroids_for_description:
                # Generate description from centroid
                centroid = self.centroids[species_id]
                
                # Convert centroid to "attribute probabilities" for verbalization
                # Since centroids are normalized, we'll threshold at mean value
                threshold = np.mean(centroid)
                active_attrs = np.where(centroid > threshold)[0] + 1  # 1-indexed
                
                # Create a simple description
                if len(active_attrs) > 0:
                    # Just use the top attributes for a basic description
                    top_attrs = np.argsort(centroid)[::-1][:5] + 1  # Top 5 attributes
                    description = f"Characterized by attributes: {', '.join(map(str, top_attrs))}"
                else:
                    description = "No distinctive attributes"
                
                candidate_info['description'] = description
            
            candidate_descriptions.append(candidate_info)
        
        return candidate_descriptions
    
    def batch_shortlist(self,
                       predictions_batch: np.ndarray,
                       k: int = 10,
                       min_similarity: float = 0.0) -> List[List[Tuple[int, str, float]]]:
        """
        Generate shortlists for a batch of predictions.
        
        Args:
            predictions_batch: Batch of prediction vectors (N, 312)
            k: Number of candidates per prediction
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of candidate lists
        """
        batch_candidates = []
        
        for prediction in predictions_batch:
            candidates = self.get_top_k_candidates(prediction, k, min_similarity)
            batch_candidates.append(candidates)
        
        return batch_candidates
    
    def evaluate_recall(self,
                       predictions_batch: np.ndarray,
                       true_species_ids: List[int],
                       k_values: List[int] = [1, 3, 5, 10, 15, 20]) -> Dict[int, float]:
        """
        Evaluate top-K recall performance.
        
        Args:
            predictions_batch: Batch of prediction vectors
            true_species_ids: True species IDs for each prediction
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary mapping K to recall score
        """
        recall_scores = {}
        
        for k in k_values:
            correct = 0
            total = len(predictions_batch)
            
            for i, prediction in enumerate(predictions_batch):
                true_species = true_species_ids[i]
                candidates = self.get_top_k_candidates(prediction, k)
                
                # Check if true species is in top-K
                candidate_ids = [cand[0] for cand in candidates]
                if true_species in candidate_ids:
                    correct += 1
            
            recall_scores[k] = correct / total
        
        return recall_scores
    
    def analyze_shortlist_quality(self,
                                predictions_batch: np.ndarray,
                                true_species_ids: List[int],
                                k: int = 10) -> Dict:
        """
        Analyze the quality of generated shortlists.
        
        Args:
            predictions_batch: Batch of prediction vectors
            true_species_ids: True species IDs
            k: Number of candidates to analyze
            
        Returns:
            Dictionary with analysis results
        """
        total_samples = len(predictions_batch)
        
        # Track metrics
        recall_at_k = 0
        mean_rank = 0
        mean_similarity_correct = 0
        mean_similarity_top1 = 0
        
        rank_distribution = []
        similarity_distributions = {'correct': [], 'top1': [], 'all': []}
        
        for i, prediction in enumerate(predictions_batch):
            true_species = true_species_ids[i]
            candidates = self.get_top_k_candidates(prediction, k)
            
            # Extract candidate info
            candidate_ids = [cand[0] for cand in candidates]
            candidate_similarities = [cand[2] for cand in candidates]
            
            # Check if correct species is in top-K
            if true_species in candidate_ids:
                recall_at_k += 1
                rank = candidate_ids.index(true_species) + 1  # 1-indexed rank
                mean_rank += rank
                rank_distribution.append(rank)
                
                # Similarity of correct answer
                correct_similarity = candidate_similarities[candidate_ids.index(true_species)]
                mean_similarity_correct += correct_similarity
                similarity_distributions['correct'].append(correct_similarity)
            
            # Top-1 similarity
            if candidate_similarities:
                top1_sim = candidate_similarities[0]
                mean_similarity_top1 += top1_sim
                similarity_distributions['top1'].append(top1_sim)
                similarity_distributions['all'].extend(candidate_similarities)
        
        # Compute final metrics
        recall_at_k = recall_at_k / total_samples
        mean_rank = mean_rank / max(1, len(rank_distribution))  # Only count cases where found
        mean_similarity_correct = mean_similarity_correct / max(1, len(similarity_distributions['correct']))
        mean_similarity_top1 = mean_similarity_top1 / total_samples
        
        analysis = {
            f'recall_at_{k}': recall_at_k,
            'mean_rank_when_found': mean_rank,
            'mean_similarity_correct': mean_similarity_correct,
            'mean_similarity_top1': mean_similarity_top1,
            'rank_distribution': rank_distribution,
            'similarity_distributions': similarity_distributions,
            'total_samples': total_samples
        }
        
        return analysis


def main():
    """Example usage of candidate shortlister."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test candidate shortlisting")
    parser.add_argument("--centroids", type=str, required=True,
                       help="Path to species centroids file")
    parser.add_argument("--test_predictions", type=str, default=None,
                       help="Path to test predictions (optional)")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of candidates to return")
    
    args = parser.parse_args()
    
    # Initialize shortlister
    shortlister = CandidateShortlister(args.centroids)
    
    if args.test_predictions:
        # Load test predictions and evaluate
        test_data = np.load(args.test_predictions)
        predictions = test_data['predictions']
        true_species = test_data['true_species_ids']
        
        # Evaluate recall
        recall_scores = shortlister.evaluate_recall(predictions, true_species)
        
        print("Recall scores:")
        for k, recall in recall_scores.items():
            print(f"  Top-{k}: {recall:.3f}")
        
        # Analyze quality
        analysis = shortlister.analyze_shortlist_quality(predictions, true_species, args.k)
        
        print(f"\nShortlist analysis (K={args.k}):")
        print(f"  Recall@{args.k}: {analysis[f'recall_at_{args.k}']:.3f}")
        print(f"  Mean rank when found: {analysis['mean_rank_when_found']:.1f}")
        print(f"  Mean similarity (correct): {analysis['mean_similarity_correct']:.3f}")
        print(f"  Mean similarity (top-1): {analysis['mean_similarity_top1']:.3f}")
    
    else:
        # Test with random prediction
        np.random.seed(42)
        test_prediction = np.random.rand(312)
        
        candidates = shortlister.get_top_k_candidates(test_prediction, args.k)
        
        print(f"Top-{args.k} candidates for random prediction:")
        for i, (species_id, species_name, similarity) in enumerate(candidates):
            print(f"  {i+1}. {species_name} (ID: {species_id}, Sim: {similarity:.3f})")


if __name__ == "__main__":
    main()
