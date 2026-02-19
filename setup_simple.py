#!/usr/bin/env python3
"""
Simplified Environment Setup Script (No Sudo Required)
Installs packages using pip install --user
"""

import subprocess
import sys
import os

def run_pip_install(package):
    """Install a package using pip with --user flag."""
    print(f"Installing {package}...", end=" ")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", package],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ({e.stderr.strip()[:50]})")
        return False

def main():
    print("="*70)
    print("🚀 Installing Python Packages (User Mode)")
    print("="*70)
    
    # Essential packages only, using compatible versions
    packages = [
        "pip==24.0",  # Upgrade pip first
        "torch==2.0.1",
        "transformers==4.30.2",
        "sentence-transformers==2.2.2",
        "datasets==2.12.0",
        "pandas==2.0.2",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "rank-bm25==0.2.2",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "tqdm==4.65.0",
        "nltk==3.8.1",
    ]
    
    print(f"\n📦 Installing {len(packages)} packages...\n")
    
    success_count = 0
    for pkg in packages:
        if run_pip_install(pkg):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully installed {success_count}/{len(packages)} packages")
    print(f"{'='*70}")
    
    if success_count == len(packages):
        print("\n🎉 All packages installed successfully!")
        print("\nNext step: python3 download_models.py")
        return 0
    else:
        print(f"\n⚠️  {len(packages) - success_count} packages failed")
        print("You may need to install them manually")
        return 1

if __name__ == "__main__":
    sys.exit(main())
