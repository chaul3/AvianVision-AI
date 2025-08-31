"""
LLM Reasoning Module

Uses language models to make final species predictions based on 
verbalized attributes and candidate shortlists.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class LLMSpeciesClassifier:
    """Uses LLM reasoning for final species classification."""
    
    def __init__(self, 
                 model_name: str = "local",
                 temperature: float = 0.2,
                 max_tokens: int = 500):
        """
        Initialize LLM classifier.
        
        Args:
            model_name: LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize model client based on type
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client - placeholder for actual implementation."""
        # This would be replaced with actual LLM client initialization
        # e.g., OpenAI, Anthropic, local model via transformers, etc.
        print(f"Initialized LLM client: {self.model_name}")
        return None
    
    def create_classification_prompt(self,
                                   attribute_text: str,
                                   attribute_json: Dict,
                                   candidates: List[Tuple[int, str, float]],
                                   include_descriptions: bool = True) -> str:
        """
        Create prompt for LLM species classification.
        
        Args:
            attribute_text: Compact text description of attributes
            attribute_json: Structured JSON of attributes
            candidates: List of candidate species
            include_descriptions: Whether to include species descriptions
            
        Returns:
            Formatted prompt string
        """
        # Format candidates list
        candidate_list = []
        for i, (species_id, species_name, similarity) in enumerate(candidates, 1):
            candidate_list.append(f"{i}. {species_name}")
        
        candidates_str = "\n".join(candidate_list)
        
        # Create the prompt
        prompt = f"""You are an expert ornithologist tasked with identifying a bird species based on observed attributes.

OBSERVED ATTRIBUTES:
{attribute_text}

STRUCTURED ATTRIBUTES:
{json.dumps(attribute_json, indent=2)}

CANDIDATE SPECIES:
{candidates_str}

INSTRUCTIONS:
1. Carefully analyze the observed attributes
2. Consider which candidate species best matches these attributes
3. Select EXACTLY ONE species from the candidate list
4. Provide your reasoning based on the specific attributes observed

RESPONSE FORMAT:
Respond with a JSON object containing:
{{
    "species": "exact_species_name_from_list",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of why this species matches the attributes"
}}

Your response must be valid JSON and the species name must exactly match one from the candidate list.

Response:"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured information.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed response dictionary
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['species', 'confidence', 'reasoning']
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")
                
                return parsed
            else:
                raise ValueError("No JSON found in response")
        
        except Exception as e:
            # Fallback parsing for malformed responses
            print(f"Failed to parse LLM response: {e}")
            print(f"Raw response: {response}")
            
            return {
                'species': 'parsing_error',
                'confidence': 0.0,
                'reasoning': f'Failed to parse response: {response[:200]}...',
                'error': str(e)
            }
    
    def classify_single(self,
                       attribute_text: str,
                       attribute_json: Dict,
                       candidates: List[Tuple[int, str, float]]) -> Dict[str, Any]:
        """
        Classify a single bird using LLM reasoning.
        
        Args:
            attribute_text: Compact attribute description
            attribute_json: Structured attributes
            candidates: Candidate species list
            
        Returns:
            Classification result dictionary
        """
        # Create prompt
        prompt = self.create_classification_prompt(
            attribute_text, attribute_json, candidates
        )
        
        # Get LLM response (placeholder - replace with actual LLM call)
        response = self._call_llm(prompt)
        
        # Parse response
        parsed_result = self.parse_llm_response(response)
        
        # Add metadata
        parsed_result['prompt'] = prompt
        parsed_result['raw_response'] = response
        parsed_result['candidates'] = candidates
        
        # Validate species selection
        candidate_names = [name for _, name, _ in candidates]
        if parsed_result['species'] not in candidate_names:
            parsed_result['validation_error'] = f"Selected species '{parsed_result['species']}' not in candidate list"
            # Default to top candidate
            if candidates:
                parsed_result['species'] = candidates[0][1]
                parsed_result['confidence'] = 0.1  # Low confidence due to error
        
        return parsed_result
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with the given prompt.
        
        This is a placeholder - replace with actual LLM API call.
        """
        # Placeholder implementation - returns a mock response
        # In practice, this would call OpenAI, Anthropic, local model, etc.
        
        mock_responses = [
            """{
    "species": "American Robin",
    "confidence": 0.85,
    "reasoning": "The observed attributes of medium size, red breast coloration, and perching-like behavior are characteristic of the American Robin. The bill shape and overall body structure also match this species well."
}""",
            """{
    "species": "Northern Cardinal",
    "confidence": 0.78,
    "reasoning": "The red coloration and conical bill shape strongly suggest a Northern Cardinal. The size and overall appearance are consistent with this species."
}""",
            """{
    "species": "Blue Jay",
    "confidence": 0.92,
    "reasoning": "The blue wing and upperparts coloration, combined with the medium size and crow-like behavior, clearly indicate a Blue Jay. The bill shape and overall pattern match perfectly."
}"""
        ]
        
        # Return a random mock response
        import random
        return random.choice(mock_responses)
    
    def classify_batch(self,
                      attribute_texts: List[str],
                      attribute_jsons: List[Dict],
                      candidates_batch: List[List[Tuple[int, str, float]]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of birds using LLM reasoning.
        
        Args:
            attribute_texts: List of attribute descriptions
            attribute_jsons: List of structured attributes
            candidates_batch: List of candidate lists
            
        Returns:
            List of classification results
        """
        results = []
        
        for i in range(len(attribute_texts)):
            result = self.classify_single(
                attribute_texts[i],
                attribute_jsons[i],
                candidates_batch[i]
            )
            results.append(result)
        
        return results
    
    def evaluate_accuracy(self,
                         predictions: List[Dict[str, Any]],
                         true_species_names: List[str]) -> Dict[str, float]:
        """
        Evaluate LLM classification accuracy.
        
        Args:
            predictions: List of LLM prediction dictionaries
            true_species_names: List of true species names
            
        Returns:
            Dictionary with accuracy metrics
        """
        total = len(predictions)
        correct = 0
        valid_predictions = 0
        
        confidence_scores = []
        error_count = 0
        
        for i, pred in enumerate(predictions):
            true_species = true_species_names[i]
            predicted_species = pred.get('species', '')
            confidence = pred.get('confidence', 0.0)
            
            confidence_scores.append(confidence)
            
            # Check for parsing errors
            if 'error' in pred or 'validation_error' in pred:
                error_count += 1
            else:
                valid_predictions += 1
                
                # Check if prediction is correct
                if predicted_species.lower() == true_species.lower():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        valid_rate = valid_predictions / total if total > 0 else 0.0
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'valid_prediction_rate': valid_rate,
            'mean_confidence': mean_confidence,
            'total_samples': total,
            'correct_predictions': correct,
            'valid_predictions': valid_predictions,
            'error_count': error_count
        }
    
    def analyze_confidence_calibration(self,
                                     predictions: List[Dict[str, Any]],
                                     true_species_names: List[str],
                                     n_bins: int = 10) -> Dict:
        """
        Analyze how well confidence scores correlate with accuracy.
        
        Args:
            predictions: List of LLM predictions
            true_species_names: List of true species names
            n_bins: Number of confidence bins
            
        Returns:
            Calibration analysis results
        """
        confidences = []
        accuracies = []
        
        for i, pred in enumerate(predictions):
            if 'error' not in pred and 'validation_error' not in pred:
                confidence = pred.get('confidence', 0.0)
                true_species = true_species_names[i]
                predicted_species = pred.get('species', '')
                
                is_correct = predicted_species.lower() == true_species.lower()
                
                confidences.append(confidence)
                accuracies.append(1.0 if is_correct else 0.0)
        
        if not confidences:
            return {'error': 'No valid predictions to analyze'}
        
        # Bin by confidence
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = [(c >= lower and c < upper) for c in confidences]
            if i == n_bins - 1:  # Include upper bound for last bin
                in_bin = [(c >= lower and c <= upper) for c in confidences]
            
            bin_conf = [confidences[j] for j, in_b in enumerate(in_bin) if in_b]
            bin_acc = [accuracies[j] for j, in_b in enumerate(in_bin) if in_b]
            
            if bin_acc:
                bin_accuracies.append(np.mean(bin_acc))
                bin_confidences.append(np.mean(bin_conf))
                bin_counts.append(len(bin_acc))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((lower + upper) / 2)
                bin_counts.append(0)
        
        return {
            'bin_edges': bin_edges.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'overall_accuracy': np.mean(accuracies),
            'mean_confidence': np.mean(confidences)
        }


# Integration class for the full pipeline
class FullPipelineClassifier:
    """Complete pipeline from image to species classification."""
    
    def __init__(self,
                 model_path: str,
                 thresholds_path: str,
                 centroids_path: str,
                 attribute_mapping_path: str):
        """
        Initialize the full pipeline.
        
        Args:
            model_path: Path to trained attribute model
            thresholds_path: Path to calibrated thresholds
            centroids_path: Path to species centroids
            attribute_mapping_path: Path to attribute mappings
        """
        self.model_path = model_path
        self.thresholds_path = thresholds_path
        self.centroids_path = centroids_path
        self.attribute_mapping_path = attribute_mapping_path
        
        # Initialize components
        self._load_components()
    
    def _load_components(self):
        """Load all pipeline components."""
        # Load thresholds
        with open(self.thresholds_path, 'r') as f:
            self.thresholds = json.load(f)
        
        # Initialize verbalizer
        from utils.attribute_verbalizer import AttributeVerbalizer
        self.verbalizer = AttributeVerbalizer(self.attribute_mapping_path)
        
        # Initialize shortlister
        from utils.candidate_shortlist import CandidateShortlister
        self.shortlister = CandidateShortlister(self.centroids_path)
        
        # Initialize LLM classifier
        self.llm_classifier = LLMSpeciesClassifier()
        
        print("✓ Full pipeline initialized")
    
    def classify_image(self,
                      image_path: str,
                      k_candidates: int = 10) -> Dict[str, Any]:
        """
        Classify a single image through the full pipeline.
        
        Args:
            image_path: Path to input image
            k_candidates: Number of candidate species
            
        Returns:
            Complete classification result
        """
        # This would implement the full pipeline:
        # 1. Load and preprocess image
        # 2. Run attribute model
        # 3. Apply thresholds
        # 4. Verbalize attributes
        # 5. Generate candidates
        # 6. LLM reasoning
        
        # Placeholder implementation
        return {
            'image_path': image_path,
            'predicted_species': 'American Robin',
            'confidence': 0.85,
            'reasoning': 'Pipeline classification result',
            'pipeline_steps': {
                'attributes_detected': 42,
                'candidates_generated': k_candidates,
                'llm_reasoning_applied': True
            }
        }


def main():
    """Example usage of LLM classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM species classification")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with mock data")
    
    args = parser.parse_args()
    
    if args.test_mode:
        # Test with mock data
        classifier = LLMSpeciesClassifier()
        
        # Mock attributes
        attribute_text = "size: medium; bill_shape: conical; upperparts_color: blue; underparts_color: white"
        attribute_json = {
            "size": ["medium"],
            "bill_shape": ["conical"],
            "upperparts_color": ["blue"],
            "underparts_color": ["white"]
        }
        
        # Mock candidates
        candidates = [
            (1, "Blue Jay", 0.92),
            (2, "Eastern Bluebird", 0.78),
            (3, "Indigo Bunting", 0.65),
            (4, "Belted Kingfisher", 0.58),
            (5, "American Robin", 0.45)
        ]
        
        # Test classification
        result = classifier.classify_single(attribute_text, attribute_json, candidates)
        
        print("LLM Classification Result:")
        print(f"Species: {result['species']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        
        if 'validation_error' in result:
            print(f"Validation Error: {result['validation_error']}")


if __name__ == "__main__":
    main()
