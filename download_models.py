#!/usr/bin/env python3
"""
Download Pre-trained Models
Separate script to download models after packages are installed
"""

import sys

def download_models():
    """Download all required pre-trained models."""
    print("="*70)
    print("🤖 Downloading Pre-trained Models")
    print("="*70)
    
    success = True
    
    # Download sentence-transformers model
    print("\n📥 Downloading Sentence-Transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence-Transformers model: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"❌ Failed: {e}")
        success = False
    
    # Download NLI model
    print("\n📥 Downloading NLI model (BART)...")
    try:
        from transformers import pipeline
        nli = pipeline('zero-shot-classification', 
                      model='facebook/bart-large-mnli',
                      device=-1)
        print("✅ NLI model: facebook/bart-large-mnli")
    except Exception as e:
        print(f"❌ Failed: {e}")
        success = False
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data: punkt, stopwords")
    except Exception as e:
        print(f"❌ Failed: {e}")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("🎉 All models downloaded successfully!")
        print("="*70)
        return 0
    else:
        print("⚠️  Some models failed to download")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(download_models())
