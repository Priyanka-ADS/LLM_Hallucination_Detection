#!/usr/bin/env python3
"""
Evaluation Pipeline for Hallucination Detector

Comprehensive evaluation with:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion Matrix
- Results saved to JSON
- Formatted output tables
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(data_dir='data/processed'):
    """Load the test dataset."""
    test_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(test_path):
        logger.error(f"Test data not found at {test_path}")
        logger.error("Please run download_dataset.py first!")
        sys.exit(1)
    
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    return test_df


def evaluate_detector(detector, test_df):
    """
    Evaluate detector on test set with comprehensive metrics.
    
    Returns:
        Dictionary of evaluation metrics and predictions.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    y_true = test_df['is_hallucination'].values
    
    logger.info("Running predictions on test set...")
    
    # Get predictions
    y_pred = []
    y_prob = []
    
    from tqdm import tqdm
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Evaluating"):
        try:
            result = detector.predict(q, a)
            y_pred.append(int(result['is_hallucination']))
            y_prob.append(result.get('hallucination_probability', 
                         float(result['is_hallucination'])))
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            y_pred.append(0)
            y_prob.append(0.5)
    
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
    }
    
    # AUC-ROC (requires probability scores)
    try:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError as e:
        logger.warning(f"Could not compute AUC-ROC: {e}")
        metrics['auc_roc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=['Truthful', 'Hallucinated'],
                                   output_dict=True)
    metrics['classification_report'] = report
    
    return metrics, y_pred, y_prob


def print_results(metrics, method_name="Ensemble Detector"):
    """Print formatted evaluation results."""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION RESULTS: {method_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>10.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>10.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>10.4f}")
    print(f"{'F1 Score':<25} {metrics['f1']:>10.4f}")
    print(f"{'AUC-ROC':<25} {metrics['auc_roc']:>10.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                  Predicted")
    print(f"                  Truthful  Hallucinated")
    print(f"  Actual Truthful   {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Actual Halluc.    {cm[1][0]:>5}      {cm[1][1]:>5}")
    
    print(f"\n  True Negatives:  {metrics['true_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    print(f"\n  Total Samples: {metrics['n_samples']}")
    print(f"  Positive (Hallucinated): {metrics['n_positive']}")
    print(f"  Negative (Truthful): {metrics['n_negative']}")
    print(f"{'='*70}\n")


def save_results(metrics, output_dir='results/metrics', 
                 filename='evaluation_results.json'):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Add metadata
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    return output_path


def generate_confusion_matrix_plot(metrics, output_dir='figures/confusion'):
    """Generate and save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        cm = np.array(metrics['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Truthful', 'Hallucinated'],
                   yticklabels=['Truthful', 'Hallucinated'],
                   ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Hallucination Detection', fontsize=14)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")


def main():
    """Main evaluation function."""
    logger.info("="*80)
    logger.info("🔍 HALLUCINATION DETECTOR - EVALUATION PIPELINE")
    logger.info("="*80)
    
    # Load test data
    test_df = load_test_data()
    
    # Load trained detector
    model_path = 'models/saved/hallucination_detector.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        logger.error("Please run train_detector.py first!")
        sys.exit(1)
    
    from detector import HallucinationDetector
    detector = HallucinationDetector.load(model_path)
    
    # Evaluate
    metrics, y_pred, y_prob = evaluate_detector(detector, test_df)
    
    # Print results
    print_results(metrics)
    
    # Save results
    save_results(metrics)
    
    # Generate confusion matrix plot
    generate_confusion_matrix_plot(metrics)
    
    logger.info("Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
