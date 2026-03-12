"""
Data Leakage Validation Script

Run this script to validate your data pipeline for data leakage before training.

Usage:
    python experiments/validate_leakage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils import load_mnist_local, load_fashion_mnist_local
from torchvision import transforms
from validate_data import (
    run_all_leak_checks,
    set_seed,
    check_train_test_overlap,
    check_augmentation_leakage
)

def validate_mnist():
    """Validate MNIST data pipeline."""
    print("\n" + "="*70)
    print("VALIDATING MNIST DATA PIPELINE")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load data
    mnist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'MNIST')
    (x_train, y_train), (x_test, y_test) = load_mnist_local(mnist_dir)
    
    print(f"\nLoaded MNIST:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Image shape: {x_train.shape[1:]}")
    print(f"  Data type: {x_train.dtype}")
    print(f"  Pixel range: {x_train.min()} - {x_train.max()}")
    
    # Define transforms (matching train_efta.py)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Run all checks
    all_passed = run_all_leak_checks(
        x_train, y_train, x_test, y_test,
        train_transform, test_transform,
        verbose=True
    )
    
    return all_passed


def validate_fashion_mnist():
    """Validate Fashion-MNIST data pipeline."""
    print("\n" + "="*70)
    print("VALIDATING FASHION-MNIST DATA PIPELINE")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load data
    fashion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'FashionMNIST')
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fashion_dir)
    
    print(f"\nLoaded Fashion-MNIST:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Image shape: {x_train.shape[1:]}")
    print(f"  Data type: {x_train.dtype}")
    print(f"  Pixel range: {x_train.min()} - {x_train.max()}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Run all checks
    all_passed = run_all_leak_checks(
        x_train, y_train, x_test, y_test,
        train_transform, test_transform,
        verbose=True
    )
    
    return all_passed


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("DATA LEAKAGE VALIDATION SUITE")
    print("="*70)
    print("This script checks for common forms of data leakage in your pipeline.")
    print("All checks should pass before training production models.\n")
    
    # Validate MNIST
    mnist_passed = validate_mnist()
    
    # Validate Fashion-MNIST
    fashion_passed = validate_fashion_mnist()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"MNIST:           [PASS] PASSED" if mnist_passed else "MNIST:           [FAIL] FAILED")
    print(f"Fashion-MNIST:   [PASS] PASSED" if fashion_passed else "Fashion-MNIST:   [FAIL] FAILED")
    print("="*70)
    
    if mnist_passed and fashion_passed:
        print("\n[PASS] ALL VALIDATIONS PASSED")
        print("Your data pipeline is clean and ready for training!")
        print("\nYou can now run:")
        print("  python experiments/train_efta.py")
        return 0
    else:
        print("\n[FAIL] SOME VALIDATIONS FAILED")
        print("Please review the warnings above and fix any issues before training.")
        return 1


if __name__ == '__main__':
    exit(main())
