#!/usr/bin/env python3
"""
Dataset Download and Preparation Script
Downloads TruthfulQA dataset and prepares it for hallucination detection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/download_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def download_truthfulqa():
    """Download TruthfulQA dataset from HuggingFace."""
    logger.info("Downloading TruthfulQA dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("truthful_qa", "generation")
        logger.info(f"Dataset downloaded successfully. Split keys: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def process_dataset(dataset):
    """
    Convert TruthfulQA dataset to a DataFrame with hallucination labels.
    
    TruthfulQA contains questions with:
    - best_answer: The truthful/correct answer
    - correct_answers: List of correct answers
    - incorrect_answers: List of incorrect/hallucinated answers
    
    We create balanced positive (hallucinated) and negative (truthful) examples.
    """
    logger.info("Processing dataset...")
    
    records = []
    
    # Access the validation split (TruthfulQA only has 'validation')
    data = dataset['validation']
    
    for idx, item in enumerate(data):
        question = item['question']
        category = item.get('category', 'unknown')
        source = item.get('source', 'unknown')
        
        # Add correct answers (not hallucinated, label=0)
        best_answer = item.get('best_answer', '')
        if best_answer:
            records.append({
                'question': question,
                'answer': best_answer,
                'is_hallucination': 0,
                'answer_type': 'best_answer',
                'category': category,
                'source': source,
                'original_idx': idx
            })
        
        # Add other correct answers
        correct_answers = item.get('correct_answers', [])
        for ans in correct_answers[:2]:  # Limit to 2 correct answers per question
            if ans and ans != best_answer:
                records.append({
                    'question': question,
                    'answer': ans,
                    'is_hallucination': 0,
                    'answer_type': 'correct_answer',
                    'category': category,
                    'source': source,
                    'original_idx': idx
                })
        
        # Add incorrect answers (hallucinated, label=1)
        incorrect_answers = item.get('incorrect_answers', [])
        for ans in incorrect_answers[:3]:  # Limit to 3 incorrect answers per question
            if ans:
                records.append({
                    'question': question,
                    'answer': ans,
                    'is_hallucination': 1,
                    'answer_type': 'incorrect_answer',
                    'category': category,
                    'source': source,
                    'original_idx': idx
                })
    
    df = pd.DataFrame(records)
    logger.info(f"Created DataFrame with {len(df)} records")
    return df


def balance_dataset(df, random_state=42):
    """
    Balance the dataset to have equal positive and negative examples.
    Uses undersampling of the majority class.
    """
    logger.info("Balancing dataset...")
    
    pos_count = df[df['is_hallucination'] == 1].shape[0]
    neg_count = df[df['is_hallucination'] == 0].shape[0]
    
    logger.info(f"Before balancing: Positive (hallucinated)={pos_count}, Negative (truthful)={neg_count}")
    
    min_count = min(pos_count, neg_count)
    
    # Undersample majority class
    pos_samples = df[df['is_hallucination'] == 1].sample(n=min_count, random_state=random_state)
    neg_samples = df[df['is_hallucination'] == 0].sample(n=min_count, random_state=random_state)
    
    balanced_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"After balancing: {len(balanced_df)} total records ({min_count} per class)")
    return balanced_df


def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split
    
    logger.info("Splitting dataset into train/val/test...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df['is_hallucination']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=random_state,
        stratify=train_val['is_hallucination']
    )
    
    logger.info(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    return train, val, test


def print_statistics(df, name="Dataset"):
    """Print comprehensive dataset statistics."""
    print(f"\n{'='*60}")
    print(f"📊 {name} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(f"  Truthful (0): {(df['is_hallucination'] == 0).sum()} ({(df['is_hallucination'] == 0).mean()*100:.1f}%)")
    print(f"  Hallucinated (1): {(df['is_hallucination'] == 1).sum()} ({(df['is_hallucination'] == 1).mean()*100:.1f}%)")
    
    print(f"\nAnswer type distribution:")
    for atype, count in df['answer_type'].value_counts().items():
        print(f"  {atype}: {count}")
    
    if 'category' in df.columns:
        print(f"\nTop 10 categories:")
        for cat, count in df['category'].value_counts().head(10).items():
            print(f"  {cat}: {count}")
    
    print(f"\nAverage answer length: {df['answer'].str.len().mean():.1f} characters")
    print(f"Average question length: {df['question'].str.len().mean():.1f} characters")
    print(f"{'='*60}\n")


def main():
    """Main function to download and prepare the dataset."""
    logger.info("="*60)
    logger.info("Starting dataset download and preparation")
    logger.info("="*60)
    
    # Create output directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Download dataset
    dataset = download_truthfulqa()
    
    # Step 2: Process into DataFrame
    df = process_dataset(dataset)
    
    # Save raw processed data
    raw_path = 'data/raw/truthfulqa_raw.csv'
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path}")
    
    # Step 3: Balance the dataset
    balanced_df = balance_dataset(df)
    
    # Step 4: Split into train/val/test
    train_df, val_df, test_df = split_dataset(balanced_df)
    
    # Step 5: Save processed data
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    balanced_df.to_csv('data/processed/full_balanced.csv', index=False)
    
    logger.info("All data files saved successfully!")
    
    # Step 6: Print statistics
    print_statistics(df, "Raw Dataset")
    print_statistics(balanced_df, "Balanced Dataset")
    print_statistics(train_df, "Training Set")
    print_statistics(val_df, "Validation Set")
    print_statistics(test_df, "Test Set")
    
    logger.info("Dataset preparation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
