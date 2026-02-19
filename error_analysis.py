#!/usr/bin/env python3
"""
Error Analysis Script

Analyzes prediction errors to understand detector weaknesses:
1. Identifies false positives and false negatives
2. Categorizes errors by question type
3. Shows most egregious errors
4. Saves detailed error report
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_errors(test_df, detector):
    """
    Perform comprehensive error analysis.
    
    Returns:
        DataFrame of all errors with analysis.
    """
    from tqdm import tqdm
    
    questions = test_df['question'].tolist()
    answers = test_df['answer'].tolist()
    y_true = test_df['is_hallucination'].values
    
    errors = []
    all_predictions = []
    
    for i, (q, a) in tqdm(enumerate(zip(questions, answers)), 
                          total=len(questions), desc="Analyzing errors"):
        try:
            result = detector.predict(q, a)
            pred = int(result['is_hallucination'])
            prob = result.get('hallucination_probability', float(pred))
            confidence = result.get('confidence', 0.5)
            
            all_predictions.append({
                'index': i,
                'question': q,
                'answer': a,
                'true_label': int(y_true[i]),
                'predicted': pred,
                'probability': prob,
                'confidence': confidence,
                'correct': pred == y_true[i],
                'error_type': None if pred == y_true[i] else 
                             ('false_positive' if pred == 1 else 'false_negative'),
                'category': test_df.iloc[i].get('category', 'unknown') if 'category' in test_df.columns else 'unknown'
            })
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
    
    predictions_df = pd.DataFrame(all_predictions)
    errors_df = predictions_df[~predictions_df['correct']].copy()
    
    return predictions_df, errors_df


def categorize_errors(errors_df):
    """Categorize errors by question type and error type."""
    print("\n" + "="*70)
    print("📊 ERROR CATEGORIZATION")
    print("="*70)
    
    # By error type
    print("\n📌 By Error Type:")
    fp = errors_df[errors_df['error_type'] == 'false_positive']
    fn = errors_df[errors_df['error_type'] == 'false_negative']
    print(f"  False Positives (truthful flagged as hallucination): {len(fp)}")
    print(f"  False Negatives (hallucination missed): {len(fn)}")
    
    # By category (if available)
    if 'category' in errors_df.columns:
        print("\n📌 By Category:")
        cat_counts = errors_df.groupby(['category', 'error_type']).size().unstack(fill_value=0)
        print(cat_counts.to_string())
    
    # By confidence level
    print("\n📌 By Confidence Level:")
    bins = [0, 0.3, 0.6, 1.0]
    labels = ['Low', 'Medium', 'High']
    errors_df['confidence_level'] = pd.cut(errors_df['confidence'], bins=bins, labels=labels)
    conf_counts = errors_df.groupby(['confidence_level', 'error_type']).size().unstack(fill_value=0)
    print(conf_counts.to_string())
    
    return fp, fn


def show_worst_errors(errors_df, n=10):
    """Show the most egregious errors (highest confidence wrong predictions)."""
    print("\n" + "="*70)
    print(f"🔴 TOP {n} MOST EGREGIOUS ERRORS")
    print("="*70)
    
    # Sort by confidence (highest confidence errors are most egregious)
    worst = errors_df.sort_values('confidence', ascending=False).head(n)
    
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        error_type = "FALSE POSITIVE" if row['error_type'] == 'false_positive' else "FALSE NEGATIVE"
        
        print(f"\n{'─'*60}")
        print(f"Error #{i} [{error_type}] (Confidence: {row['confidence']:.4f})")
        print(f"{'─'*60}")
        print(f"  Question: {row['question']}")
        print(f"  Answer:   {row['answer'][:200]}{'...' if len(str(row['answer'])) > 200 else ''}")
        print(f"  True:     {'Hallucinated' if row['true_label'] == 1 else 'Truthful'}")
        print(f"  Predicted: {'Hallucinated' if row['predicted'] == 1 else 'Truthful'}")
        print(f"  Category: {row.get('category', 'unknown')}")


def main():
    """Main error analysis function."""
    logger.info("="*60)
    logger.info("🔍 ERROR ANALYSIS")
    logger.info("="*60)
    
    # Load test data
    test_path = 'data/processed/test.csv'
    if not os.path.exists(test_path):
        logger.error("Test data not found!")
        sys.exit(1)
    test_df = pd.read_csv(test_path)
    
    # Load model
    model_path = 'models/saved/hallucination_detector.pkl'
    if not os.path.exists(model_path):
        logger.error("Trained model not found!")
        sys.exit(1)
    
    from detector import HallucinationDetector
    detector = HallucinationDetector.load(model_path)
    
    # Analyze
    predictions_df, errors_df = analyze_errors(test_df, detector)
    
    # Summary stats
    total = len(predictions_df)
    correct = predictions_df['correct'].sum()
    print(f"\n📊 Overall: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"   Errors: {len(errors_df)}/{total} ({len(errors_df)/total*100:.1f}%)")
    
    # Categorize
    fp, fn = categorize_errors(errors_df)
    
    # Show worst errors
    show_worst_errors(errors_df, n=10)
    
    # Save error report
    os.makedirs('results', exist_ok=True)
    errors_df.to_csv('results/error_analysis.csv', index=False)
    predictions_df.to_csv('results/all_predictions.csv', index=False)
    logger.info("\nError reports saved to results/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
