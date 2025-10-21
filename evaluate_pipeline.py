"""
Complete Pipeline Evaluation Script

Runs the full vision→attributes→LLM pipeline for bird species identification.
Evaluates performance at each stage and generates comprehensive results.
"""
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_data_processing(config: Dict) -> str:
    """
    Step 1: Process CUB dataset and create manifest.
    
    Returns:
        Path to dataset manifest
    """
    print("="*50)
    print("STEP 1: DATA PROCESSING")
    print("="*50)
    
    from data.process_cub_dataset import CUBDatasetProcessor
    
    processor = CUBDatasetProcessor(
        cub_root=config['data']['cub_root'],
        output_dir=config['data']['output_dir']
    )
    
    manifest, prevalence = processor.process_dataset()
    return config['outputs']['dataset_manifest']


def run_model_training(config: Dict, manifest_path: str) -> str:
    """
    Step 2: Train attribute detection model.
    
    Returns:
        Path to trained model
    """
    print("\n" + "="*50)
    print("STEP 2: ATTRIBUTE MODEL TRAINING")
    print("="*50)
    
    from models.train_attribute_model import AttributeModelTrainer, get_default_config
    
    # Merge configs
    training_config = get_default_config()
    training_config.update(config['training'])
    training_config.update(config['model'])
    
    if config.get('wandb', {}).get('enabled', False):
        training_config['use_wandb'] = True
        training_config['wandb_project'] = config['wandb']['project']
    
    trainer = AttributeModelTrainer(training_config, config['data']['output_dir'])
    
    model = trainer.train(
        manifest_path=manifest_path,
        prevalence_path=config['outputs']['attribute_prevalence']
    )
    
    return config['outputs']['model_weights']


def run_threshold_calibration(config: Dict, model_path: str, manifest_path: str) -> str:
    """
    Step 3: Calibrate per-attribute thresholds.
    
    Returns:
        Path to calibrated thresholds
    """
    print("\n" + "="*50)
    print("STEP 3: THRESHOLD CALIBRATION")
    print("="*50)
    
    from models.calibrate_thresholds import ThresholdCalibrator
    
    calibrator = ThresholdCalibrator(config['data']['output_dir'])
    
    thresholds, temperature = calibrator.calibrate_model(
        model_path=model_path,
        manifest_path=manifest_path,
        split=config['calibration']['split'],
        metric=config['calibration']['metric']
    )
    
    return config['outputs']['thresholds']


def run_centroid_computation(config: Dict, manifest_path: str, model_path: str = None) -> str:
    """
    Step 4: Compute species attribute centroids.
    
    Returns:
        Path to species centroids
    """
    print("\n" + "="*50)
    print("STEP 4: SPECIES CENTROID COMPUTATION")
    print("="*50)
    
    from utils.species_centroids import AttributeCentroidBuilder
    
    builder = AttributeCentroidBuilder(config['data']['output_dir'])
    
    centroids = builder.build_centroids(
        manifest_path=manifest_path,
        model_path=model_path,
        use_ground_truth=config['shortlisting']['use_ground_truth_centroids']
    )
    
    return config['outputs']['centroids']


def create_attribute_mapping(config: Dict) -> str:
    """
    Step 5: Create attribute verbalization mapping.
    
    Returns:
        Path to attribute mapping file
    """
    print("\n" + "="*50)
    print("STEP 5: ATTRIBUTE MAPPING CREATION")
    print("="*50)
    
    from utils.attribute_verbalizer import create_cub_attribute_mapping
    
    mapping_path = config['outputs']['attribute_mapping']
    create_cub_attribute_mapping(mapping_path)
    
    return mapping_path


def evaluate_pipeline(config: Dict, 
                     model_path: str,
                     thresholds_path: str,
                     centroids_path: str,
                     mapping_path: str,
                     manifest_path: str) -> Dict[str, Any]:
    """
    Step 6: Evaluate complete pipeline on test set.
    
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*50)
    print("STEP 6: PIPELINE EVALUATION")
    print("="*50)
    
    # Load components
    from utils.attribute_verbalizer import AttributeVerbalizer
    from utils.candidate_shortlist import CandidateShortlister
    from utils.llm_classifier import LLMSpeciesClassifier
    from models.calibrate_thresholds import ThresholdCalibrator
    
    # Initialize components
    print("Initializing pipeline components...")
    
    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)
    
    verbalizer = AttributeVerbalizer(mapping_path)
    shortlister = CandidateShortlister(centroids_path)
    llm_classifier = LLMSpeciesClassifier(
        model_name=config['llm']['model_name'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens']
    )
    
    # Generate predictions on test set
    print("Generating attribute predictions...")
    calibrator = ThresholdCalibrator(config['data']['output_dir'])
    
    test_probs, test_targets = calibrator.load_model_predictions(
        model_path, manifest_path, split=config['evaluation']['test_split']
    )
    
    # Load test metadata
    manifest_df = pd.read_csv(manifest_path)
    test_df = manifest_df[manifest_df['split'] == config['evaluation']['test_split']].reset_index(drop=True)
    
    print(f"Evaluating on {len(test_df)} test samples...")
    
    # Pipeline evaluation
    results = {
        'config': config,
        'test_samples': len(test_df),
        'pipeline_steps': {},
        'final_metrics': {}
    }
    
    # Step-by-step evaluation
    attribute_texts = []
    attribute_jsons = []
    candidates_batch = []
    
    print("Running pipeline steps...")
    
    for i in tqdm(range(len(test_probs)), desc="Processing samples"):
        # Verbalize attributes
        text, json_data = verbalizer.verbalize_attributes(test_probs[i], thresholds)
        attribute_texts.append(text)
        attribute_jsons.append(json_data)
        
        # Generate candidates
        candidates = shortlister.get_top_k_candidates(
            test_probs[i], 
            k=config['shortlisting']['k_candidates'],
            min_similarity=config['shortlisting']['min_similarity']
        )
        candidates_batch.append(candidates)
    
    # Evaluate shortlisting performance
    print("Evaluating candidate shortlisting...")
    true_species_ids = test_df['class_id'].tolist()
    
    recall_scores = shortlister.evaluate_recall(
        test_probs, true_species_ids, config['evaluation']['k_values']
    )
    
    shortlist_analysis = shortlister.analyze_shortlist_quality(
        test_probs, true_species_ids, config['shortlisting']['k_candidates']
    )
    
    results['pipeline_steps']['shortlisting'] = {
        'recall_scores': recall_scores,
        'analysis': shortlist_analysis
    }
    
    # LLM Classification (on subset for speed)
    print("Running LLM classification...")
    n_llm_samples = min(100, len(test_df))  # Limit for demo purposes
    
    llm_predictions = llm_classifier.classify_batch(
        attribute_texts[:n_llm_samples],
        attribute_jsons[:n_llm_samples],
        candidates_batch[:n_llm_samples]
    )
    
    # Evaluate LLM performance
    true_species_names = test_df['species_name'][:n_llm_samples].tolist()
    
    llm_accuracy = llm_classifier.evaluate_accuracy(llm_predictions, true_species_names)
    llm_calibration = llm_classifier.analyze_confidence_calibration(llm_predictions, true_species_names)
    
    results['pipeline_steps']['llm_classification'] = {
        'accuracy_metrics': llm_accuracy,
        'calibration_analysis': llm_calibration,
        'sample_predictions': llm_predictions[:5]  # Save first 5 for inspection
    }
    
    # Overall pipeline metrics
    results['final_metrics'] = {
        'attribute_detection': {
            'n_attributes': 312,
            'mean_threshold': np.mean(list(thresholds.values())),
            'std_threshold': np.std(list(thresholds.values()))
        },
        'candidate_shortlisting': {
            f'recall_at_{config["shortlisting"]["k_candidates"]}': shortlist_analysis[f'recall_at_{config["shortlisting"]["k_candidates"]}'],
            'mean_rank_when_found': shortlist_analysis['mean_rank_when_found'],
            'mean_similarity_top1': shortlist_analysis['mean_similarity_top1']
        },
        'llm_classification': {
            'accuracy': llm_accuracy['accuracy'],
            'mean_confidence': llm_accuracy['mean_confidence'],
            'valid_prediction_rate': llm_accuracy['valid_prediction_rate']
        }
    }
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print summary of pipeline results."""
    print("\n" + "="*50)
    print("PIPELINE EVALUATION SUMMARY")
    print("="*50)
    
    final_metrics = results['final_metrics']
    
    print(f"Test samples: {results['test_samples']}")
    
    print(f"\nAttribute Detection:")
    print(f"  Mean threshold: {final_metrics['attribute_detection']['mean_threshold']:.3f}")
    print(f"  Std threshold: {final_metrics['attribute_detection']['std_threshold']:.3f}")
    
    print(f"\nCandidate Shortlisting:")
    k = results['config']['shortlisting']['k_candidates']
    print(f"  Recall@{k}: {final_metrics['candidate_shortlisting'][f'recall_at_{k}']:.3f}")
    print(f"  Mean rank (when found): {final_metrics['candidate_shortlisting']['mean_rank_when_found']:.1f}")
    print(f"  Mean top-1 similarity: {final_metrics['candidate_shortlisting']['mean_similarity_top1']:.3f}")
    
    print(f"\nLLM Classification:")
    print(f"  Accuracy: {final_metrics['llm_classification']['accuracy']:.3f}")
    print(f"  Mean confidence: {final_metrics['llm_classification']['mean_confidence']:.3f}")
    print(f"  Valid prediction rate: {final_metrics['llm_classification']['valid_prediction_rate']:.3f}")


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description="Run CUB bird identification pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip model training (use existing model)")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation (skip all training steps)")
    parser.add_argument("--steps", nargs="+", 
                       choices=["data", "train", "calibrate", "centroids", "mapping", "evaluate"],
                       help="Specific steps to run")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("CUB-200-2011 Bird Species Identification Pipeline")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Output directory: {config['data']['output_dir']}")
    
    # Create output directory
    Path(config['data']['output_dir']).mkdir(exist_ok=True)
    Path("configs").mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # Determine which steps to run
    if args.eval_only:
        steps_to_run = ["evaluate"]
    elif args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = ["data", "train", "calibrate", "centroids", "mapping", "evaluate"]
        if args.skip_training:
            steps_to_run.remove("train")
    
    # Initialize paths
    manifest_path = config['outputs']['dataset_manifest']
    model_path = config['outputs']['model_weights']
    thresholds_path = config['outputs']['thresholds']
    centroids_path = config['outputs']['centroids']
    mapping_path = config['outputs']['attribute_mapping']
    
    # Run pipeline steps
    try:
        if "data" in steps_to_run:
            manifest_path = run_data_processing(config)
        
        if "train" in steps_to_run:
            model_path = run_model_training(config, manifest_path)
        
        if "calibrate" in steps_to_run:
            thresholds_path = run_threshold_calibration(config, model_path, manifest_path)
        
        if "centroids" in steps_to_run:
            centroids_path = run_centroid_computation(config, manifest_path, model_path)
        
        if "mapping" in steps_to_run:
            mapping_path = create_attribute_mapping(config)
        
        if "evaluate" in steps_to_run:
            results = evaluate_pipeline(
                config, model_path, thresholds_path, 
                centroids_path, mapping_path, manifest_path
            )
            
            # Save and display results
            save_results(results, config['outputs']['results'])
            print_summary(results)
    
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Report total time
    total_time = time.time() - start_time
    print(f"\n✓ Pipeline completed in {total_time:.1f} seconds")
    
    return 0


if __name__ == "__main__":
    exit(main())
