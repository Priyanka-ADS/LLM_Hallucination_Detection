#!/usr/bin/env python3
"""
Ablation Study Script

Tests the ensemble detector with different signal combinations to measure
the contribution of each signal to overall performance.

Combinations tested:
- All three signals
- Any two signals (3 combinations)
- Each signal alone (3 combinations)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


ALL_SIGNALS = ['self_consistency', 'retrieval', 'confidence']


def load_data():
    """Load train, val, and test datasets."""
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    return train_df, val_df, test_df


def train_and_evaluate(signals, train_df, val_df, test_df):
    """
    Train a detector with specific signals and evaluate on test set.
    
    Returns:
        Dictionary with metrics.
    """
    from detector import HallucinationDetector
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Initialize and train
    detector = HallucinationDetector(
        corpus_dir='data/corpus',
        use_signals=list(signals)
    )
    
    # Train
    train_metrics = detector.train(
        questions=train_df['question'].tolist(),
        answers=train_df['answer'].tolist(),
        labels=train_df['is_hallucination'].tolist(),
        val_questions=val_df['question'].tolist(),
        val_answers=val_df['answer'].tolist(),
        val_labels=val_df['is_hallucination'].tolist()
    )
    
    # Evaluate on test set
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    y_true = test_df['is_hallucination'].values
    
    y_pred = []
    y_prob = []
    for q, a in zip(questions, answers):
        result = detector.predict(q, a)
        y_pred.append(int(result['is_hallucination']))
        y_prob.append(result.get('hallucination_probability', float(result['is_hallucination'])))
    
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    metrics = {
        'Signals': ' + '.join(signals),
        'N_Signals': len(signals),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    try:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['AUC-ROC'] = 0.0
    
    return metrics


def run_ablation_study():
    """
    Run the complete ablation study.
    
    Tests all possible signal combinations:
    - 3 single-signal configs 
    - 3 two-signal configs
    - 1 all-signal config
    """
    logger.info("="*60)
    logger.info("🔬 ABLATION STUDY")
    logger.info("="*60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    logger.info(f"Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Generate all signal combinations
    all_combos = []
    
    # Single signals
    for signal in ALL_SIGNALS:
        all_combos.append((signal,))
    
    # Two signals
    for combo in combinations(ALL_SIGNALS, 2):
        all_combos.append(combo)
    
    # All three signals
    all_combos.append(tuple(ALL_SIGNALS))
    
    # Run experiments
    all_results = []
    for combo in all_combos:
        logger.info(f"\nTesting: {' + '.join(combo)}")
        try:
            metrics = train_and_evaluate(combo, train_df, val_df, test_df)
            all_results.append(metrics)
            logger.info(f"  F1: {metrics['F1']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    return pd.DataFrame(all_results)


def generate_ablation_plots(results_df, output_dir='figures/ablation'):
    """Generate ablation study visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: F1 by number of signals
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Bar chart of all combinations
    sorted_df = results_df.sort_values('F1', ascending=True)
    colors = ['#FF6B6B' if n == 1 else '#FFEAA7' if n == 2 else '#4ECDC4' 
              for n in sorted_df['N_Signals']]
    
    axes[0].barh(sorted_df['Signals'], sorted_df['F1'], color=colors, 
                edgecolor='white', linewidth=1.5)
    axes[0].set_xlabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score by Signal Combination', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1.1)
    
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        axes[0].text(row['F1'] + 0.02, i, f"{row['F1']:.3f}", 
                    va='center', fontsize=10)
    
    # Performance by number of signals
    grouped = results_df.groupby('N_Signals').agg({
        'F1': ['mean', 'max', 'min'],
        'Accuracy': ['mean', 'max', 'min']
    }).reset_index()
    
    n_signals = grouped['N_Signals'].values
    f1_mean = grouped[('F1', 'mean')].values
    f1_max = grouped[('F1', 'max')].values
    f1_min = grouped[('F1', 'min')].values
    
    axes[1].fill_between(n_signals, f1_min, f1_max, alpha=0.3, color='#4ECDC4')
    axes[1].plot(n_signals, f1_mean, 'o-', color='#4ECDC4', markersize=10, 
                linewidth=2.5, label='F1 (mean)')
    axes[1].plot(n_signals, f1_max, 's--', color='#45B7D1', markersize=8,
                linewidth=1.5, alpha=0.7, label='F1 (best)')
    
    axes[1].set_xlabel('Number of Signals', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Performance vs. Number of Signals', fontsize=14, fontweight='bold')
    axes[1].set_xticks([1, 2, 3])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_study.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Ablation plot saved: {path}")


def main():
    """Main function for ablation study."""
    start_time = datetime.now()
    
    # Run ablation study
    results_df = run_ablation_study()
    
    if results_df.empty:
        logger.error("No results to analyze!")
        return 1
    
    # Print results
    print("\n" + "="*90)
    print("📊 ABLATION STUDY RESULTS")
    print("="*90)
    print(results_df.sort_values('F1', ascending=False).to_string(index=False, float_format='%.4f'))
    print("="*90)
    
    # Analyze signal contributions
    print("\n📌 Signal Contribution Analysis:")
    if len(results_df) >= 7:
        full_f1 = results_df[results_df['N_Signals'] == 3]['F1'].values[0]
        for signal in ALL_SIGNALS:
            without = results_df[
                (results_df['N_Signals'] == 2) & 
                (~results_df['Signals'].str.contains(signal))
            ]
            if not without.empty:
                without_f1 = without['F1'].values[0]
                contribution = full_f1 - without_f1
                print(f"  {signal}: {'+'if contribution>=0 else ''}{contribution:.4f} F1 impact")
    
    # Save results
    os.makedirs('results/ablation', exist_ok=True)
    results_df.to_csv('results/ablation/ablation_results.csv', index=False)
    
    # Generate plots
    generate_ablation_plots(results_df)
    
    duration = datetime.now() - start_time
    logger.info(f"\nAblation study complete! Duration: {duration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
