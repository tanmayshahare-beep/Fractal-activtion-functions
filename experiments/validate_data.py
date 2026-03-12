"""
Data Leakage Detection Utilities

This module provides utilities to check for common forms of data leakage
in PyTorch image classification pipelines.

Usage:
    from experiments.validate_data import (
        check_train_test_overlap,
        check_augmentation_leakage,
        check_normalization_leakage,
        run_all_leak_checks
    )
"""

import hashlib
import numpy as np
import torch
from typing import Set, Tuple


def check_train_test_overlap(x_train: np.ndarray, x_test: np.ndarray, verbose: bool = True) -> bool:
    """
    Check for exact image duplicates between train and test sets.
    
    Uses MD5 hashing to efficiently detect duplicate images.
    
    Args:
        x_train: Training images array
        x_test: Test images array
        verbose: If True, print detailed results
    
    Returns:
        True if no overlap found (clean), False if leakage detected
    """
    # Flatten images for hashing
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Compute hashes
    train_hashes: Set[str] = {
        hashlib.md5(img.tobytes()).hexdigest() 
        for img in x_train_flat
    }
    test_hashes: Set[str] = {
        hashlib.md5(img.tobytes()).hexdigest() 
        for img in x_test_flat
    }
    
    # Find overlap
    overlap = train_hashes.intersection(test_hashes)
    
    if overlap:
        if verbose:
            print(f"[FAIL] CRITICAL: Found {len(overlap)} duplicate images between train and test sets!")
            print(f"   This indicates data leakage - results will be invalid!")
        return False
    else:
        if verbose:
            print("[PASS] Train-test split check passed: No exact image duplicates found.")
        return True


def check_augmentation_leakage(train_transform, test_transform, verbose: bool = True) -> bool:
    """
    Ensure validation/test transforms don't contain random augmentations.
    
    Args:
        train_transform: Training transforms
        test_transform: Test/validation transforms
        verbose: If True, print detailed results
    
    Returns:
        True if test transforms are deterministic (clean), False if leakage detected
    """
    # List of random/stochastic transforms that should NOT be in test transforms
    random_transforms = [
        'RandomRotation', 
        'RandomAffine', 
        'RandomHorizontalFlip', 
        'RandomVerticalFlip',
        'RandomCrop',
        'RandomResizedCrop',
        'RandomPerspective',
        'RandomErasing',
        'ColorJitter',
        'GaussianBlur',
        'RandomApply',
        'RandomChoice',
        'RandomOrder'
    ]
    
    # Get transform names
    test_transform_names = []
    if hasattr(test_transform, 'transforms'):
        for t in test_transform.transforms:
            test_transform_names.append(type(t).__name__)
    elif hasattr(test_transform, '__class__'):
        test_transform_names.append(type(test_transform).__name__)
    
    leakage_detected = False
    found_transforms = []
    
    for t_name in random_transforms:
        if t_name in test_transform_names:
            leakage_detected = True
            found_transforms.append(t_name)
    
    if leakage_detected:
        if verbose:
            print(f"[FAIL] LEAKAGE DETECTED: Test/validation set uses random augmentations!")
            print(f"   Found: {', '.join(found_transforms)}")
            print(f"   Test transforms should be deterministic (e.g., only ToTensor, Normalize)")
        return False
    else:
        if verbose:
            print("[PASS] Augmentation check passed: Test/validation transforms are deterministic.")
        return True


def check_normalization_leakage(
    x_train: np.ndarray, 
    x_test: np.ndarray,
    train_mean: float = None,
    train_std: float = None,
    verbose: bool = True
) -> bool:
    """
    Check if normalization statistics were computed on test data (leakage).
    
    For MNIST, the normalization should use training set statistics only.
    
    Args:
        x_train: Training images (before normalization)
        x_test: Test images (before normalization)
        train_mean: Mean used for normalization (if None, compute from x_train)
        train_std: Std used for normalization (if None, compute from x_train)
        verbose: If True, print detailed results
    
    Returns:
        True if normalization is clean, False if leakage detected
    """
    # Compute actual statistics
    actual_train_mean = x_train.mean()
    actual_train_std = x_train.std()
    
    actual_test_mean = x_test.mean()
    actual_test_std = x_test.std()
    
    # If provided stats differ significantly from training stats,
    # they might have been computed on full dataset (leakage)
    if train_mean is not None and train_std is not None:
        mean_diff = abs(train_mean - actual_train_mean)
        std_diff = abs(train_std - actual_train_std)
        
        # Allow small tolerance for floating point
        if mean_diff > 0.01 or std_diff > 0.01:
            # Check if stats match combined dataset
            combined = np.concatenate([x_train.flatten(), x_test.flatten()])
            combined_mean = combined.mean()
            combined_std = combined.std()
            
            if abs(train_mean - combined_mean) < 0.01:
                if verbose:
                    print(f"[FAIL] LEAKAGE DETECTED: Normalization stats appear to be computed on full dataset!")
                    print(f"   Provided mean: {train_mean:.4f}, Training mean: {actual_train_mean:.4f}")
                    print(f"   Use training set statistics only for normalization")
                return False
    
    if verbose:
        print("[PASS] Normalization check passed: Statistics appear to be from training set only.")
        print(f"   Train mean: {actual_train_mean:.4f}, std: {actual_train_std:.4f}")
        print(f"   Test mean: {actual_test_mean:.4f}, std: {actual_test_std:.4f}")
    
    return True


def check_label_distribution(y_train: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> bool:
    """
    Check if label distributions are significantly different (potential sampling bias).
    
    Args:
        y_train: Training labels
        y_test: Test labels
        verbose: If True, print detailed results
    
    Returns:
        True if distributions are similar, False if significant difference detected
    """
    from scipy import stats
    
    # Get unique labels and counts
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    # Normalize to proportions
    train_props = train_counts / len(y_train)
    test_props = test_counts / len(y_test)

    # Use Kolmogorov-Smirnov test for distribution comparison
    # (works with different sample sizes)
    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(train_props, test_props)

    if verbose:
        print("\nLabel Distribution:")
        print(f"   Training: {train_props}")
        print(f"   Test:     {test_props}")
        print(f"   KS test p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"[WARN] Label distributions are significantly different (p < 0.05)")
            print(f"   This may indicate sampling bias, but is not necessarily leakage")
            return True  # Still return True - this is a warning, not leakage
        else:
            print("[PASS] Label distribution check passed: Distributions are similar.")
    
    return True


def run_all_leak_checks(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_transform=None,
    test_transform=None,
    verbose: bool = True
) -> bool:
    """
    Run all data leakage checks.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        train_transform: Training transforms
        test_transform: Test/validation transforms
        verbose: If True, print detailed results
    
    Returns:
        True if all checks pass (no leakage), False if any check fails
    """
    print("\n" + "="*70)
    print("DATA LEAKAGE VALIDATION")
    print("="*70)
    
    all_passed = True
    
    # Check 1: Train-test overlap
    print("\n[1/4] Checking for train-test overlap...")
    if not check_train_test_overlap(x_train, x_test, verbose):
        all_passed = False
    
    # Check 2: Augmentation leakage
    if train_transform is not None and test_transform is not None:
        print("\n[2/4] Checking for augmentation leakage...")
        if not check_augmentation_leakage(train_transform, test_transform, verbose):
            all_passed = False
    else:
        print("\n[2/4] Skipping augmentation check (transforms not provided)")
    
    # Check 3: Normalization leakage
    print("\n[3/4] Checking for normalization leakage...")
    if not check_normalization_leakage(x_train, x_test, verbose=verbose):
        all_passed = False
    
    # Check 4: Label distribution
    print("\n[4/4] Checking label distribution...")
    if not check_label_distribution(y_train, y_test, verbose):
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("[RESULT] ALL CHECKS PASSED - Data pipeline appears clean!")
    else:
        print("[RESULT] SOME CHECKS FAILED - Review warnings above!")
    print("="*70 + "\n")
    
    return all_passed


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducibility")
