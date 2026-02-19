# Lightweight Hallucination Detection in Large Language Models

**Multi-Signal Ensemble Approach for Detecting Hallucinations in LLM Outputs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📝 Overview

This project implements a lightweight, offline hallucination detection system for Large Language Models (LLMs) using a multi-signal ensemble approach. It combines three independent detection signals:

1. **Self-Consistency Signal** — Measures semantic consistency of answer variations using sentence embeddings
2. **Retrieval Verification Signal** — Verifies answers against a knowledge corpus using BM25 + NLI
3. **Confidence Signal** — Analyzes text-based confidence heuristics (hedging, repetition, specificity)

These signals are combined via a Logistic Regression meta-classifier to produce a final hallucination prediction.

## 🏗️ Project Structure

```
LLM_HALLUCINATION/
├── signals/                      # Detection signal implementations
│   ├── self_consistency.py       # Self-consistency using sentence-transformers
│   ├── retrieval_verification.py # BM25 + NLI verification
│   └── confidence_signal.py      # Text-based confidence heuristics
├── data/                         # Dataset storage
│   ├── raw/                      # Raw downloaded datasets
│   ├── processed/                # Train/val/test splits
│   └── corpus/                   # Knowledge corpus + BM25 index
├── models/saved/                 # Trained model files
├── results/                      # Experimental results
│   ├── metrics/                  # Evaluation metrics (JSON)
│   ├── comparisons/              # Method comparison tables
│   └── ablation/                 # Ablation study results
├── figures/                      # Generated visualizations
│   ├── performance/              # Bar charts, feature importance
│   ├── roc/                      # ROC curve plots
│   ├── confusion/                # Confusion matrix heatmaps
│   └── ablation/                 # Ablation study plots
├── paper/                        # Paper materials (LaTeX tables, sections)
├── detector.py                   # Main ensemble detector class
├── train_detector.py             # Training pipeline
├── evaluate.py                   # Evaluation pipeline
├── baselines.py                  # Baseline implementations (4 methods)
├── compare_methods.py            # Full method comparison
├── generate_figures.py           # Publication figure generation
├── ablation_study.py             # Ablation experiments
├── error_analysis.py             # Error analysis and reporting
├── demo.py                       # Interactive demo
├── download_dataset.py           # Dataset download script
├── prepare_knowledge_corpus.py   # Knowledge corpus preparation
├── setup_environment.py          # Environment setup script
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download TruthfulQA dataset
python download_dataset.py

# Build knowledge corpus
python prepare_knowledge_corpus.py
```

### 3. Train Detector

```bash
python train_detector.py
```

### 4. Evaluate

```bash
# Run evaluation
python evaluate.py

# Compare all methods
python compare_methods.py

# Run ablation study
python ablation_study.py
```

### 5. Generate Figures

```bash
python generate_figures.py
```

### 6. Interactive Demo

```bash
python demo.py
```

## 📊 Methods

### Signal 1: Self-Consistency
Uses `all-MiniLM-L6-v2` sentence-transformer to compute semantic similarity between answer variations. Consistent answers suggest factual grounding.

### Signal 2: Retrieval Verification
Retrieves relevant documents via BM25, then uses `facebook/bart-large-mnli` NLI model to check if evidence entails the answer.

### Signal 3: Confidence Heuristics
Analyzes text patterns without model access:
- **Hedging detection** — uncertainty markers ("maybe", "I think")
- **Repetition analysis** — excessive word/phrase repetition
- **Specificity scoring** — presence of dates, numbers, proper nouns
- **Length analysis** — very short or very long answers
- **Entropy measurement** — character-level information entropy

### Ensemble
Logistic Regression combines scaled signal features into a final hallucination probability.

## 📈 Baselines

| Method | Description |
|--------|-------------|
| Random | Random predictions (50/50) |
| Confidence Only | Single signal: text heuristics |
| Self-Consistency Only | Single signal: semantic similarity |
| Retrieval Only | Single signal: evidence verification |
| **Ensemble** | **All three signals combined** |

## 📋 Running the Full Pipeline

```bash
# Complete pipeline in order:
source venv/bin/activate

# Phase 1: Setup
python setup_environment.py

# Phase 2: Data
python download_dataset.py
python prepare_knowledge_corpus.py

# Phase 3-4: Train
python train_detector.py

# Phase 5: Evaluate
python evaluate.py
python baselines.py
python compare_methods.py

# Phase 6: Analyze
python generate_figures.py
python ablation_study.py
python error_analysis.py

# Demo
python demo.py
```

## 📝 Citation

```bibtex
@article{hallucination_detection_2025,
  title={Lightweight Hallucination Detection in Large Language Models via Multi-Signal Ensemble},
  year={2025}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
