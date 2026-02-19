#!/usr/bin/env python3
"""
Baseline Implementations for Hallucination Detection

Provides 4 baseline detectors for comparison:
1. RandomBaseline - random predictions
2. ConfidenceOnlyBaseline - uses only confidence signal
3. SelfConsistencyOnlyBaseline - uses only self-consistency
4. RetrievalOnlyBaseline - uses only retrieval verification
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomBaseline:
    """Random prediction baseline."""
    
    def __init__(self, seed=42):
        self.name = "Random Baseline"
        self.rng = np.random.RandomState(seed)
    
    def predict(self, question: str, answer: str) -> Dict:
        prob = self.rng.random()
        return {
            'is_hallucination': prob > 0.5,
            'confidence': abs(prob - 0.5) * 2,
            'hallucination_probability': prob
        }


class ConfidenceOnlyBaseline:
    """Uses only the confidence signal for detection."""
    
    def __init__(self, threshold=0.5):
        self.name = "Confidence Only"
        self.threshold = threshold
        self._detector = None
    
    @property
    def detector(self):
        if self._detector is None:
            from signals.confidence_signal import ConfidenceDetector
            self._detector = ConfidenceDetector()
        return self._detector
    
    def predict(self, question: str, answer: str) -> Dict:
        result = self.detector.detect(question, answer)
        return {
            'is_hallucination': result['score'] < self.threshold,
            'confidence': abs(result['score'] - self.threshold) * 2,
            'hallucination_probability': 1.0 - result['score']
        }


class SelfConsistencyOnlyBaseline:
    """Uses only the self-consistency signal for detection."""
    
    def __init__(self, threshold=0.5):
        self.name = "Self-Consistency Only"
        self.threshold = threshold
        self._detector = None
    
    @property
    def detector(self):
        if self._detector is None:
            from signals.self_consistency import SelfConsistencyDetector
            self._detector = SelfConsistencyDetector()
        return self._detector
    
    def predict(self, question: str, answer: str) -> Dict:
        result = self.detector.detect(question, answer)
        return {
            'is_hallucination': result['score'] < self.threshold,
            'confidence': abs(result['score'] - self.threshold) * 2,
            'hallucination_probability': 1.0 - result['score']
        }


class RetrievalOnlyBaseline:
    """Uses only the retrieval verification signal for detection."""
    
    def __init__(self, corpus_dir='data/corpus', threshold=0.5):
        self.name = "Retrieval Only"
        self.threshold = threshold
        self.corpus_dir = corpus_dir
        self._detector = None
    
    @property
    def detector(self):
        if self._detector is None:
            from signals.retrieval_verification import RetrievalVerifier
            self._detector = RetrievalVerifier(corpus_dir=self.corpus_dir)
        return self._detector
    
    def predict(self, question: str, answer: str) -> Dict:
        result = self.detector.detect(question, answer)
        return {
            'is_hallucination': result['score'] < self.threshold,
            'confidence': abs(result['score'] - self.threshold) * 2,
            'hallucination_probability': 1.0 - result['score']
        }


def evaluate_baseline(baseline, questions, answers, labels):
    """
    Evaluate a baseline detector on the test set.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    from tqdm import tqdm
    
    y_pred = []
    y_prob = []
    
    for q, a in tqdm(zip(questions, answers), total=len(questions), 
                     desc=f"Evaluating {baseline.name}"):
        try:
            result = baseline.predict(q, a)
            y_pred.append(int(result['is_hallucination']))
            y_prob.append(result.get('hallucination_probability', 
                         float(result['is_hallucination'])))
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            y_pred.append(0)
            y_prob.append(0.5)
    
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_true = np.array(labels)
    
    metrics = {
        'method': baseline.name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    try:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    return metrics


def evaluate_all_baselines(test_df, corpus_dir='data/corpus'):
    """
    Evaluate all baselines on the test set.
    
    Returns:
        DataFrame with comparison results.
    """
    baselines = [
        RandomBaseline(),
        ConfidenceOnlyBaseline(),
        SelfConsistencyOnlyBaseline(),
        RetrievalOnlyBaseline(corpus_dir=corpus_dir),
    ]
    
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    labels = test_df['is_hallucination'].tolist()
    
    all_results = []
    for baseline in baselines:
        try:
            logger.info(f"\nEvaluating: {baseline.name}")
            metrics = evaluate_baseline(baseline, questions, answers, labels)
            all_results.append(metrics)
            
            # Print results
            print(f"\n{baseline.name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {baseline.name}: {e}")
    
    return pd.DataFrame(all_results)


def main():
    """Main function to run baseline evaluations."""
    logger.info("="*60)
    logger.info("🔬 BASELINE EVALUATIONS")
    logger.info("="*60)
    
    # Load test data
    test_path = 'data/processed/test.csv'
    if not os.path.exists(test_path):
        logger.error("Test data not found. Run download_dataset.py first!")
        sys.exit(1)
    
    test_df = pd.read_csv(test_path)
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Evaluate all baselines
    results_df = evaluate_all_baselines(test_df)
    
    # Save results
    os.makedirs('results/comparisons', exist_ok=True)
    results_df.to_csv('results/comparisons/baseline_results.csv', index=False)
    
    # Print comparison table
    print("\n" + "="*80)
    print("📊 BASELINE COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
