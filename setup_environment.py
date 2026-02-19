#!/usr/bin/env python3
"""
Environment Setup Script for Hallucination Detection Project
This script installs all required dependencies and downloads necessary models.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def install_packages():
    """Install all required Python packages."""
    print("\n" + "="*60)
    print("📦 INSTALLING REQUIRED PACKAGES")
    print("="*60)
    
    packages = {
        "Core ML Libraries": [
            "torch==2.0.1",
            "transformers==4.30.2",
            "sentence-transformers==2.2.2",
        ],
        "Data Processing": [
            "datasets==2.12.0",
            "pandas==2.0.2",
            "numpy==1.24.3",
        ],
        "Machine Learning": [
            "scikit-learn==1.3.0",
            "rank-bm25==0.2.2",
        ],
        "Visualization": [
            "matplotlib==3.7.1",
            "seaborn==0.12.2",
        ],
        "Utilities": [
            "tqdm==4.65.0",
            "nltk==3.8.1",
        ]
    }
    
    all_success = True
    for category, pkgs in packages.items():
        print(f"\n📂 {category}")
        for pkg in pkgs:
            cmd = f"{sys.executable} -m pip install {pkg}"
            if not run_command(cmd, f"Installing {pkg}"):
                all_success = False
    
    return all_success

def download_models():
    """Download required pre-trained models."""
    print("\n" + "="*60)
    print("🤖 DOWNLOADING PRE-TRAINED MODELS")
    print("="*60)
    
    print("\n📥 Downloading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence-Transformers model downloaded")
    except Exception as e:
        print(f"❌ Failed to download sentence-transformers: {e}")
        return False
    
    print("\n📥 Downloading NLI model...")
    try:
        from transformers import pipeline
        nli = pipeline('zero-shot-classification', 
                      model='facebook/bart-large-mnli',
                      device=-1)  # CPU
        print("✅ NLI model downloaded")
    except Exception as e:
        print(f"❌ Failed to download NLI model: {e}")
        return False
    
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False
    
    return True

def verify_installation():
    """Verify that all packages are correctly installed."""
    print("\n" + "="*60)
    print("✔️  VERIFYING INSTALLATION")
    print("="*60)
    
    required_imports = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "sentence_transformers": "Sentence-Transformers",
        "datasets": "HuggingFace Datasets",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "rank_bm25": "BM25",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "tqdm": "TQDM",
        "nltk": "NLTK",
    }
    
    all_verified = True
    for module, name in required_imports.items():
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NOT FOUND")
            all_verified = False
    
    return all_verified

def create_project_structure():
    """Create the basic project directory structure."""
    print("\n" + "="*60)
    print("📁 CREATING PROJECT STRUCTURE")
    print("="*60)
    
    base_dir = Path(__file__).parent
    directories = [
        "signals",
        "data",
        "results",
        "models",
        "figures",
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created: {dir_path}")
        
        # Create __init__.py for Python packages
        if dir_name == "signals":
            (dir_path / "__init__.py").touch()
            print(f"   └── __init__.py")
    
    return True

def main():
    """Main setup function."""
    print("\n" + "="*80)
    print("🚀 HALLUCINATION DETECTION PROJECT - ENVIRONMENT SETUP")
    print("="*80)
    
    steps = [
        ("Installing packages", install_packages),
        ("Creating project structure", create_project_structure),
        ("Downloading models", download_models),
        ("Verifying installation", verify_installation),
    ]
    
    results = {}
    for step_name, step_func in steps:
        results[step_name] = step_func()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 SETUP SUMMARY")
    print("="*80)
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step_name}: {status}")
    
    if all(results.values()):
        print("\n" + "="*80)
        print("🎉 SETUP COMPLETE! You're ready to start the project.")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  SETUP INCOMPLETE - Please fix the errors above")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
