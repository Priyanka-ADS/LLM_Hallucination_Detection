#!/usr/bin/env python3
"""
Training Script for Hallucination Detector

Loads the prepared dataset, initializes the ensemble detector, 
trains it, and saves the trained model.

Usage: python train_detector.py
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/training_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_dir='data/processed'):
    """Load the prepared dataset splits."""
    logger.info("Loading dataset...")
    
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}")
        logger.error("Please run download_dataset.py first!")
        sys.exit(1)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    logger.info(f"Train: {len(train_df)} samples")
    if val_df is not None:
        logger.info(f"Val: {len(val_df)} samples")
    if test_df is not None:
        logger.info(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def train_model(train_df, val_df=None, corpus_dir='data/corpus',
                use_signals=None):
    """
    Train the hallucination detector.
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame (optional).
        corpus_dir: Path to knowledge corpus.
        use_signals: Which signals to use.
        
    Returns:
        Trained detector and training metrics.
    """
    from detector import HallucinationDetector
    
    logger.info("="*60)
    logger.info("Initializing Hallucination Detector")
    logger.info("="*60)
    
    # Initialize detector
    detector = HallucinationDetector(
        corpus_dir=corpus_dir,
        use_signals=use_signals
    )
    
    # Prepare training data
    questions = train_df['question'].tolist()
    answers = train_df['answer'].tolist()
    labels = train_df['is_hallucination'].tolist()
    
    # Prepare validation data
    val_questions = None
    val_answers = None
    val_labels = None
    if val_df is not None:
        val_questions = val_df['question'].tolist()
        val_answers = val_df['answer'].tolist()
        val_labels = val_df['is_hallucination'].tolist()
    
    # Train
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    
    metrics = detector.train(
        questions=questions,
        answers=answers,
        labels=labels,
        val_questions=val_questions,
        val_answers=val_answers,
        val_labels=val_labels
    )
    
    return detector, metrics


def print_training_results(metrics):
    """Print formatted training results."""
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Training Accuracy':<25} {metrics['train_accuracy']:>10.4f}")
    print(f"{'Training F1':<25} {metrics['train_f1']:>10.4f}")
    print(f"{'Samples Used':<25} {metrics['n_train_samples']:>10}")
    print(f"{'Features':<25} {metrics['n_features']:>10}")
    
    if 'val_accuracy' in metrics:
        print(f"{'Validation Accuracy':<25} {metrics['val_accuracy']:>10.4f}")
        print(f"{'Validation F1':<25} {metrics['val_f1']:>10.4f}")
    
    if 'feature_importance' in metrics:
        print(f"\n📈 Feature Importance (Coefficients):")
        for name, importance in sorted(metrics['feature_importance'].items(), 
                                        key=lambda x: abs(x[1]), reverse=True):
            bar = "█" * int(abs(importance) * 10)
            sign = "+" if importance > 0 else "-"
            print(f"  {name:<30} {sign}{abs(importance):>8.4f} {bar}")
    
    print("="*60)


def main():
    """Main training function."""
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 HALLUCINATION DETECTOR - TRAINING PIPELINE")
    logger.info(f"Started at: {start_time}")
    logger.info("="*80)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    
    # Step 1: Load data
    train_df, val_df, test_df = load_data()
    
    # Step 2: Train model
    # Start with confidence-only for speed, then try all signals
    signal_configs = [
        ['confidence'],  # Fast baseline
        ['self_consistency', 'confidence'],  # Without retrieval (faster)
        ['self_consistency', 'retrieval', 'confidence'],  # Full ensemble
    ]
    
    best_detector = None
    best_metrics = None
    best_f1 = 0
    
    for signals in signal_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with signals: {signals}")
        logger.info(f"{'='*60}")
        
        try:
            detector, metrics = train_model(
                train_df, val_df,
                use_signals=signals
            )
            
            # Track best model
            f1 = metrics.get('val_f1', metrics['train_f1'])
            if f1 > best_f1:
                best_f1 = f1
                best_detector = detector
                best_metrics = metrics
                best_metrics['signals_used'] = signals
            
            print_training_results(metrics)
            
        except Exception as e:
            logger.error(f"Training failed with signals {signals}: {e}")
            continue
    
    if best_detector is None:
        logger.error("No models were successfully trained!")
        return 1
    
    # Step 3: Save best model
    model_path = 'models/saved/hallucination_detector.pkl'
    best_detector.save(model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # Step 4: Save metrics
    # Convert non-serializable items
    save_metrics = {k: v for k, v in best_metrics.items() 
                    if isinstance(v, (int, float, str, list, dict, bool))}
    
    metrics_path = 'results/training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    print(f"Duration: {duration}")
    print(f"Best signals: {best_metrics.get('signals_used', 'unknown')}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
