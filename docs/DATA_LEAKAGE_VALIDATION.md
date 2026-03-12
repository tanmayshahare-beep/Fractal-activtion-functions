# Data Leakage Validation Report

## Overview

This document describes the data leakage validation checks implemented for the Dynamic Activation Functions research project. These checks ensure that the reported results (e.g., EFTA achieving 99.55% on MNIST) are valid and not due to methodological flaws.

## Types of Data Leakage Checked

### 1. Train-Test Overlap (Contamination)

**What it checks:** Whether any exact duplicate images exist between training and test sets.

**Why it matters:** If the same image appears in both sets, the model can "memorize" it during training and then perform artificially well on it during testing.

**Detection method:** MD5 hashing of all images to find exact duplicates.

**Expected result:** ✅ No overlap (0 duplicate images)

---

### 2. Augmentation Leakage

**What it checks:** Whether random data augmentations are applied to validation/test data.

**Why it matters:** Random augmentations (rotation, affine transforms) should only be applied to training data. Applying them to test data would mean the model is evaluated on augmented versions of test images, not the original images.

**Detection method:** Inspect test transforms for stochastic operations.

**Expected result:** ✅ Test transforms are deterministic (only ToTensor, no Random*)

---

### 3. Normalization Leakage

**What it checks:** Whether normalization statistics (mean, std) were computed on the full dataset instead of training data only.

**Why it matters:** Computing normalization statistics on the full dataset (including test data) leaks information about the test distribution into the training process.

**Detection method:** Compare provided normalization stats with actual training set statistics.

**Expected result:** ✅ Normalization uses training set statistics only

---

### 4. Label Distribution Bias

**What it checks:** Whether training and test sets have significantly different label distributions.

**Why it matters:** While not strictly "leakage," significant differences in label distribution can indicate sampling bias that affects result generalizability.

**Detection method:** Chi-squared test for distribution difference.

**Expected result:** ✅ Similar distributions (p-value > 0.05)

---

## How to Run Validation

### Quick Validation

```bash
cd "C:\All projects\Dynamic activation functions\Handwritten numbers on mnist"
python experiments/validate_leakage.py
```

This runs all checks on both MNIST and Fashion-MNIST datasets.

### Programmatic Validation

```python
from experiments.validate_data import run_all_leak_checks, set_seed
from src.utils import load_mnist_local
from torchvision import transforms

# Set seed for reproducibility
set_seed(42)

# Load data
(x_train, y_train), (x_test, y_test) = load_mnist_local('datasets/MNIST')

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Run validation
all_passed = run_all_leak_checks(
    x_train, y_train, x_test, y_test,
    train_transform, test_transform
)

if all_passed:
    print("✅ Data pipeline is clean - ready for training!")
else:
    print("❌ Issues detected - review warnings above")
```

---

## Validation Results

### MNIST Dataset

| Check | Status | Details |
|-------|--------|---------|
| Train-Test Overlap | ✅ PASS | 0 duplicate images |
| Augmentation Leakage | ✅ PASS | Test transforms are deterministic |
| Normalization Leakage | ✅ PASS | Statistics from training set only |
| Label Distribution | ✅ PASS | Distributions similar (p > 0.05) |

### Fashion-MNIST Dataset

| Check | Status | Details |
|-------|--------|---------|
| Train-Test Overlap | ✅ PASS | 0 duplicate images |
| Augmentation Leakage | ✅ PASS | Test transforms are deterministic |
| Normalization Leakage | ✅ PASS | Statistics from training set only |
| Label Distribution | ✅ PASS | Distributions similar (p > 0.05) |

---

## Best Practices Followed

### 1. Data Preprocessing

```python
# ✅ CORRECT: Keep data as uint8, let ToTensor() handle scaling
x_train = x_train  # Keep as uint8 (0-255)
transform = transforms.Compose([
    transforms.ToTensor(),  # Scales 0-255 → 0-1
])

# ❌ WRONG: Converting to float32 before ToTensor() corrupts data
x_train = x_train.astype('float32')  # PIL doesn't handle float32 correctly
```

### 2. Train-Validation Split

```python
# ✅ CORRECT: Split before creating datasets
val_split = int(0.9 * len(x_train))
x_val = x_train[val_split:]
x_train_sub = x_train[:val_split]

# Apply transforms separately
train_dataset = NumpyDataset(x_train_sub, y_train_sub, transform=train_transform)
val_dataset = NumpyDataset(x_val, y_val, transform=test_transform)
```

### 3. Data Augmentation

```python
# ✅ CORRECT: Augmentation only on training data
train_transform = transforms.Compose([
    transforms.RandomRotation(10),           # Random - training only
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random - training only
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Deterministic - safe for test
])
```

### 4. Reproducibility

```python
# ✅ Set all random seeds
set_seed(42)  # Sets torch, numpy, python.random seeds
```

---

## Documentation for Paper

When reporting results in a paper, include this statement:

> **Data Leakage Prevention:** We implemented comprehensive checks to prevent data leakage:
> - No duplicate images between training and test sets (verified via MD5 hashing)
> - Data augmentation (rotation, affine transforms) applied exclusively to training data
> - Normalization statistics computed from training set only
> - Label distributions verified to be similar across splits (chi-squared test, p > 0.05)
> - Random seeds fixed for reproducibility (seed = 42)

---

## Troubleshooting

### If Train-Test Overlap is Detected

1. Check your data loading code
2. Ensure you're not accidentally concatenating train and test before splitting
3. Verify the dataset source files are correct

### If Augmentation Leakage is Detected

1. Review your test_transform definition
2. Remove any Random* transforms from test pipeline
3. Test transforms should only include: ToTensor, Normalize (with train stats)

### If Normalization Leakage is Detected

1. Compute mean/std from training set ONLY
2. Apply same stats to both train and test transforms
3. Never compute stats on test data

---

## Files

- `validate_data.py` - Core validation functions
- `validate_leakage.py` - Standalone validation script
- `DATA_LEAKAGE_VALIDATION.md` - This documentation

---

## References

1. Kaufman, S. et al. "Leakage in Data Mining: Formulation, Detection, and Avoidance" (2012)
2. Scikit-learn Documentation: "Common pitfalls and best practices"
3. PyTorch Documentation: "Data loading and processing"

---

**Last Updated:** March 6, 2026  
**Status:** ✅ All validations passing
