#!/usr/bin/env python3
"""
Figure Generation Script

Creates publication-quality figures:
1. Performance comparison bar chart
2. ROC curves for all methods
3. Confusion matrix heatmap
4. Feature importance plot
5. Error analysis examples
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib backend before import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results():
    """Load evaluation results from files."""
    results = {}
    
    # Main evaluation results
    eval_path = 'results/metrics/evaluation_results.json'
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            results['evaluation'] = json.load(f)
    
    # Comparison results
    comp_path = 'results/comparisons/full_comparison.csv'
    if os.path.exists(comp_path):
        results['comparison'] = pd.read_csv(comp_path)
    
    # Training metrics
    train_path = 'results/training_metrics.json'
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            results['training'] = json.load(f)
    
    # Ablation results
    ablation_path = 'results/ablation/ablation_results.csv'
    if os.path.exists(ablation_path):
        results['ablation'] = pd.read_csv(ablation_path)
    
    return results


def figure1_performance_comparison(results_df, output_dir='figures/performance'):
    """Create performance comparison bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    available_cols = [c for c in metrics_cols if c in results_df.columns]
    
    plot_data = results_df.melt(
        id_vars='Method', value_vars=available_cols,
        var_name='Metric', value_name='Score'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    n_methods = len(results_df)
    palette = colors[:n_methods]
    
    sns.barplot(data=plot_data, x='Metric', y='Score', hue='Method',
                palette=palette, ax=ax, edgecolor='white', linewidth=1.5)
    
    ax.set_title('Performance Comparison: Hallucination Detection Methods',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_xlabel('Metric', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=10, title_fontsize=11)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Figure 1 saved: {path}")


def figure2_roc_curves(test_df, output_dir='figures/roc'):
    """Create ROC curves for all methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    from sklearn.metrics import roc_curve, auc
    
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    y_true = test_df['is_hallucination'].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Try to evaluate each method
    methods = {}
    
    # Random baseline
    from baselines import RandomBaseline, ConfidenceOnlyBaseline
    
    try:
        random_bl = RandomBaseline(seed=42)
        probs = [random_bl.predict(q, a)['hallucination_probability'] 
                for q, a in zip(questions, answers)]
        methods['Random'] = np.array(probs)
    except:
        pass
    
    try:
        conf_bl = ConfidenceOnlyBaseline()
        probs = [conf_bl.predict(q, a)['hallucination_probability']
                for q, a in zip(questions, answers)]
        methods['Confidence Only'] = np.array(probs)
    except:
        pass
    
    # Ensemble
    model_path = 'models/saved/hallucination_detector.pkl'
    if os.path.exists(model_path):
        try:
            from detector import HallucinationDetector
            ensemble = HallucinationDetector.load(model_path)
            probs = [ensemble.predict(q, a)['hallucination_probability']
                    for q, a in zip(questions, answers)]
            methods['Ensemble'] = np.array(probs)
        except:
            pass
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for (name, probs), color in zip(methods.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
               label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - Hallucination Detection Methods',
                fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Figure 2 saved: {path}")


def figure3_confusion_matrix(metrics, output_dir='figures/confusion'):
    """Create confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = np.array(metrics.get('confusion_matrix', [[0,0],[0,0]]))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Truthful', 'Hallucinated'],
               yticklabels=['Truthful', 'Hallucinated'],
               ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=['Truthful', 'Hallucinated'],
               yticklabels=['Truthful', 'Hallucinated'],
               ax=axes[1], vmin=0, vmax=1,
               cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Ensemble Detector - Confusion Matrix Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix_detailed.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Figure 3 saved: {path}")


def figure4_feature_importance(metrics, output_dir='figures/performance'):
    """Create feature importance plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    importance = metrics.get('feature_importance', {})
    if not importance:
        logger.warning("No feature importance data available")
        return
    
    names = list(importance.keys())
    values = list(importance.values())
    
    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(values))
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Coefficient Value', fontsize=13)
    ax.set_title('Feature Importance (Logistic Regression Coefficients)',
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, 
               bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left' if val >= 0 else 'right',
               va='center', fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Figure 4 saved: {path}")


def figure5_error_analysis(test_df, output_dir='figures/performance'):
    """Create error analysis visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = 'models/saved/hallucination_detector.pkl'
    if not os.path.exists(model_path):
        logger.warning("No trained model found for error analysis")
        return
    
    from detector import HallucinationDetector
    detector = HallucinationDetector.load(model_path)
    
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    y_true = test_df['is_hallucination'].values
    
    # Get predictions
    y_pred = []
    confidences = []
    for q, a in zip(questions, answers):
        result = detector.predict(q, a)
        y_pred.append(int(result['is_hallucination']))
        confidences.append(result['confidence'])
    
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    
    # Error analysis
    correct = y_pred == y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confidence distribution for correct vs incorrect
    axes[0].hist(confidences[correct], bins=20, alpha=0.7, label='Correct', color='#4ECDC4')
    axes[0].hist(confidences[~correct], bins=20, alpha=0.7, label='Incorrect', color='#FF6B6B')
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    # Error types
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    categories = ['True Neg.', 'True Pos.', 'False Pos.', 'False Neg.']
    counts = [tn, tp, fp, fn]
    colors = ['#4ECDC4', '#45B7D1', '#FFEAA7', '#FF6B6B']
    
    axes[1].bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Prediction Breakdown', fontsize=14, fontweight='bold')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        axes[1].text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Error Analysis - Ensemble Detector', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_analysis.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Figure 5 saved: {path}")


def main():
    """Generate all figures."""
    logger.info("="*60)
    logger.info("📊 GENERATING PUBLICATION FIGURES")
    logger.info("="*60)
    
    results = load_results()
    
    # Figure 1: Performance comparison
    if 'comparison' in results:
        logger.info("Generating Figure 1: Performance Comparison...")
        figure1_performance_comparison(results['comparison'])
    else:
        logger.warning("Comparison results not found. Skipping Figure 1.")
    
    # Figure 2: ROC curves
    test_path = 'data/processed/test.csv'
    if os.path.exists(test_path):
        logger.info("Generating Figure 2: ROC Curves...")
        test_df = pd.read_csv(test_path)
        figure2_roc_curves(test_df)
    
    # Figure 3: Confusion matrix
    if 'evaluation' in results:
        logger.info("Generating Figure 3: Confusion Matrix...")
        figure3_confusion_matrix(results['evaluation'])
    
    # Figure 4: Feature importance
    if 'training' in results:
        logger.info("Generating Figure 4: Feature Importance...")
        figure4_feature_importance(results['training'])
    
    # Figure 5: Error analysis
    if os.path.exists(test_path):
        logger.info("Generating Figure 5: Error Analysis...")
        test_df = pd.read_csv(test_path)
        figure5_error_analysis(test_df)
    
    logger.info("\n✅ Figure generation complete!")
    logger.info("Check the figures/ directory for output files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
