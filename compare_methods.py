#!/usr/bin/env python3
"""
Methods Comparison Script

Compares all baselines and the ensemble detector on the same test set.
Generates comparison tables, bar charts, and statistical significance tests.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data():
    """Load test dataset."""
    test_path = 'data/processed/test.csv'
    if not os.path.exists(test_path):
        logger.error("Test data not found!")
        sys.exit(1)
    return pd.read_csv(test_path)


def evaluate_method(method, questions, answers, labels, method_name):
    """Evaluate a single method and return metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from tqdm import tqdm
    
    y_pred = []
    y_prob = []
    
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc=method_name):
        try:
            result = method.predict(q, a)
            y_pred.append(int(result['is_hallucination']))
            y_prob.append(result.get('hallucination_probability', float(result['is_hallucination'])))
        except Exception as e:
            y_pred.append(0)
            y_prob.append(0.5)
    
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    y_true = np.array(labels)
    
    metrics = {
        'Method': method_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    try:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['AUC-ROC'] = 0.0
    
    return metrics, y_pred, y_prob


def run_significance_tests(predictions_dict, labels):
    """
    Run McNemar's test for statistical significance between methods.
    
    Returns:
        DataFrame of pairwise p-values.
    """
    methods = list(predictions_dict.keys())
    n = len(methods)
    p_values = np.ones((n, n))
    
    y_true = np.array(labels)
    
    for i in range(n):
        for j in range(i+1, n):
            pred_i = predictions_dict[methods[i]]
            pred_j = predictions_dict[methods[j]]
            
            # Contingency table for McNemar's test
            correct_i = (pred_i == y_true)
            correct_j = (pred_j == y_true)
            
            # b: i correct, j wrong
            b = np.sum(correct_i & ~correct_j)
            # c: i wrong, j correct
            c = np.sum(~correct_i & correct_j)
            
            # McNemar's test
            if b + c > 0:
                # Use exact binomial test for small counts
                if b + c < 25:
                    p_value = stats.binom_test(b, b + c, 0.5)
                else:
                    # Chi-squared approximation
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    p_value = 1 - stats.chi2.cdf(chi2, df=1)
            else:
                p_value = 1.0
            
            p_values[i, j] = p_value
            p_values[j, i] = p_value
    
    return pd.DataFrame(p_values, index=methods, columns=methods)


def generate_comparison_chart(results_df, output_dir='figures/performance'):
    """Generate bar chart comparing all methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
        plot_data = results_df.melt(
            id_vars='Method', 
            value_vars=metrics_cols,
            var_name='Metric', 
            value_name='Score'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Color palette
        colors = sns.color_palette("husl", n_colors=len(results_df))
        
        # Bar chart
        bar_plot = sns.barplot(
            data=plot_data, 
            x='Metric', y='Score', hue='Method',
            palette=colors, ax=ax
        )
        
        ax.set_title('Performance Comparison: Hallucination Detection Methods',
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13)
        ax.set_xlabel('Metric', fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'method_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate comparison chart: {e}")


def main():
    """Main comparison function."""
    logger.info("="*80)
    logger.info("📊 METHOD COMPARISON PIPELINE")
    logger.info("="*80)
    
    # Load test data
    test_df = load_test_data()
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    labels = test_df['is_hallucination'].tolist()
    
    all_results = []
    all_predictions = {}
    
    # 1. Evaluate baselines
    from baselines import (
        RandomBaseline, ConfidenceOnlyBaseline, 
        SelfConsistencyOnlyBaseline, RetrievalOnlyBaseline
    )
    
    baselines = [
        (RandomBaseline(), "Random"),
        (ConfidenceOnlyBaseline(), "Confidence Only"),
        (SelfConsistencyOnlyBaseline(), "Self-Consistency Only"),
        (RetrievalOnlyBaseline(), "Retrieval Only"),
    ]
    
    for baseline, name in baselines:
        try:
            metrics, preds, probs = evaluate_method(baseline, questions, answers, labels, name)
            all_results.append(metrics)
            all_predictions[name] = preds
        except Exception as e:
            logger.error(f"Failed: {name}: {e}")
    
    # 2. Evaluate ensemble detector
    model_path = 'models/saved/hallucination_detector.pkl'
    if os.path.exists(model_path):
        try:
            from detector import HallucinationDetector
            ensemble = HallucinationDetector.load(model_path)
            metrics, preds, probs = evaluate_method(
                ensemble, questions, answers, labels, "Ensemble"
            )
            all_results.append(metrics)
            all_predictions["Ensemble"] = preds
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
    else:
        logger.warning("Ensemble model not found. Skipping.")
    
    # 3. Create comparison table
    results_df = pd.DataFrame(all_results)
    
    # Sort by F1 score
    results_df = results_df.sort_values('F1', ascending=False).reset_index(drop=True)
    
    # Print comparison table
    print("\n" + "="*90)
    print("📊 COMPLETE METHOD COMPARISON")
    print("="*90)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # 4. Highlight best results
    print("\n📌 Best Results:")
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']:
        best_idx = results_df[col].idxmax()
        best_method = results_df.loc[best_idx, 'Method']
        best_value = results_df.loc[best_idx, col]
        print(f"  Best {col}: {best_method} ({best_value:.4f})")
    
    # 5. Statistical significance tests
    if len(all_predictions) > 1:
        print("\n" + "="*90)
        print("📈 STATISTICAL SIGNIFICANCE (McNemar's Test p-values)")
        print("="*90)
        try:
            p_values_df = run_significance_tests(all_predictions, labels)
            print(p_values_df.to_string(float_format='%.4f'))
        except Exception as e:
            logger.warning(f"Significance tests failed: {e}")
    
    print("="*90)
    
    # 6. Save results
    os.makedirs('results/comparisons', exist_ok=True)
    results_df.to_csv('results/comparisons/full_comparison.csv', index=False)
    
    # 7. Generate charts
    generate_comparison_chart(results_df)
    
    logger.info("Comparison complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
