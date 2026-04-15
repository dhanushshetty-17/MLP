#!/usr/bin/env python3
"""
Quick Start Guide - Run this to verify everything is working
"""
import sys
import numpy as np

print("=" * 70)
print("DEEP LEARNING PROJECT - VERIFICATION SCRIPT")
print("=" * 70)

# Check Python version
print(f"\n[OK] Python Version: {sys.version.split()[0]}")

# Check imports
modules_to_check = [
    ('numpy', 'NumPy'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'Scikit-learn'),
]

print("\n[OK] Checking dependencies:")
for module, name in modules_to_check:
    try:
        __import__(module)
        print(f"  [OK] {name:20s} - OK")
    except ImportError:
        print(f"  *** {name:20s} - MISSING")
        sys.exit(1)

# Check project modules
print("\n[OK] Checking project modules:")
project_modules = ['utils', 'data_loader', 'mlp', 'autoencoder', 'rbm']
for module in project_modules:
    try:
        __import__(module)
        print(f"  [OK] {module:20s} - OK")
    except ImportError as e:
        print(f"  *** {module:20s} - ERROR: {e}")
        sys.exit(1)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED!")
print("=" * 70)
print("\nYou're ready to train the models. Run:")
print("  python train.py           # Train all models")
print("  python evaluate.py        # Evaluate and analyze results")
print("\n" + "=" * 70)
