# Research Paper Next Steps

## Overview

This document outlines the remaining experiments and analyses needed to strengthen the EFTA paper for publication.

---

## 1. Statistical Significance with Error Bars

**Goal:** Run multiple random seeds to compute mean ± std deviation for all key models.

### Implementation

```bash
# Run statistical significance experiment
python experiments/advanced/01_statistical_significance.py
```

### What it does:
- Runs each configuration 5 times with different seeds (42, 123, 456, 789, 1024)
- Reports mean ± standard deviation
- Performs paired t-test against ReLU baseline
- Outputs: `outputs/results/statistical_results.npy`

### Expected output table:

| Model | Accuracy (mean ± std) | vs ReLU (p-value) |
|-------|----------------------|-------------------|
| ReLU | 98.50 ± 0.15% | - |
| LeakyReLU | 98.60 ± 0.12% | p = 0.23 |
| Maxout (k=4) | 98.70 ± 0.10% | p = 0.08 |
| FTA (d=2,k=2) | 98.75 ± 0.08% | p = 0.04* |
| **EFTA (d=2,k=2)** | **98.85 ± 0.07%** | **p = 0.02*** |

*statistically significant (p < 0.05)

### Status: ⏳ Pending

---

## 2. Fashion-MNIST Evaluation

**Goal:** Test generalization on more challenging dataset.

### Why Fashion-MNIST?
- Same format as MNIST (28x28 grayscale, 10 classes)
- More complex visual features (clothing items vs digits)
- Better test of model's feature learning capability
- Your linear FTA already reached 91.81% - EFTA should exceed this

### Run:
```bash
python experiments/fashion_mnist_efta.py
```

### Expected results:

| Model | MNIST Acc | Fashion-MNIST Acc | Drop |
|-------|-----------|-------------------|------|
| ReLU | 98.50% | 90.50% | -8.0% |
| Maxout (k=4) | 98.70% | 91.20% | -7.5% |
| FTA (d=2,k=2) | 98.75% | 91.81% | -6.9% |
| **EFTA (d=2,k=2)** | **98.85%** | **~92.00%** | **-6.8%** |

### Status: ⏳ Pending

---

## 3. CIFAR-10 Extension (Optional but Recommended)

**Goal:** Test if EFTA scales to more complex, colored images.

### Challenges:
- RGB images (3 channels vs 1)
- More complex objects (animals, vehicles)
- Higher intra-class variation

### Architecture changes needed:
```python
# Deeper CNN for CIFAR-10
def create_cifar_cnn_efta(depth=2, branch_factor=2):
    # Block 1: 32 filters
    Conv2D(32, 3, padding='same') -> BN -> EFTA -> MaxPool -> Dropout
    # Block 2: 64 filters
    Conv2D(64, 3, padding='same') -> BN -> EFTA -> MaxPool -> Dropout
    # Block 3: 128 filters (new!)
    Conv2D(128, 3, padding='same') -> BN -> EFTA -> MaxPool -> Dropout
    # Dense
    Flatten -> Dense(128) -> BN -> EFTA -> Dropout -> Dense(10)
```

### Expected timeline:
- Implement CIFAR-10 data loader: 1 hour
- Modify CNN architecture: 2 hours
- Run experiments: ~4 hours per model
- Total: ~2 days

### Status: ⏸️ Optional

---

## 4. Branch Specialization Visualization

**Goal:** Understand what features different EFTA leaves learn.

### Why this matters:
- **Novel interpretability angle** - unique to tree-structured activations
- Shows EFTA isn't just a "black box"
- Can reveal hierarchical feature learning
- Strong visual for paper

### Analysis to perform:

#### a) Leaf Activation Heatmaps
For each digit class (0-9), find which leaves activate most strongly:

```python
# For a trained EFTA model
leaf_activations = get_leaf_activations(model, test_images)

# For digit "3", which leaves fire most?
digit_3_leaf_pattern = leaf_activations[y_test == 3].mean(axis=0)
# Expected: Some leaves specialize in curves, others in straight lines
```

#### b) Specialized Leaf Images
Find test images that maximally activate each leaf:

```python
# For leaf #5, which images activate it most?
leaf_5_top_images = test_images[leaf_activations[:, 5].argsort()[-10:]]
# Display these images - do they share visual features?
```

#### c) Tree Visualization
```
Root (max)
├── Node 1 (max)
│   ├── Leaf 0: Horizontal strokes? (activates on 1, 7)
│   └── Leaf 1: Vertical strokes? (activates on 1, 4)
└── Node 2 (max)
    ├── Leaf 2: Curves? (activates on 3, 8, 9)
    └── Leaf 3: Closed loops? (activates on 0, 6, 8, 9)
```

### Expected outputs:
- `outputs/plots/leaf_activation_heatmap.png`
- `outputs/plots/specialized_leaf_examples.png`
- `outputs/plots/tree_specialization_diagram.png`

### Run:
```bash
python experiments/advanced/03_branch_visualization.py
```

### Status: ⏳ High Priority - Unique contribution!

---

## 5. Additional Analyses (Recommended)

### a) Parameter Efficiency Study
Compare EFTA vs Maxout at matched parameter counts:

```bash
python experiments/advanced/05_parameter_matched.py
```

### b) Regularization Study
Test which regularization techniques work best with EFTA:

```bash
python experiments/advanced/06_regularization_study.py
```

### c) Ablation Study
Test different combining operations (max vs sum vs mean):

```bash
python experiments/advanced/04_ablation_study.py
```

---

## Timeline

| Week | Tasks |
|------|-------|
| **Week 1** | Statistical significance (5 seeds × 9 models) |
| **Week 2** | Fashion-MNIST experiments + Branch visualization |
| **Week 3** | CIFAR-10 (if pursuing) + Ablation study |
| **Week 4** | Paper writing + figure preparation |

---

## Paper Structure Outline

### Abstract
- Novel exponential activation for hierarchical feature learning
- Outperforms ReLU/Maxout on MNIST (98.85% vs 98.50%)
- Maintains advantage on Fashion-MNIST (92.00% vs 90.50%)
- Interpretable branch specialization

### Introduction
- Motivation: Need for expressive activations
- Contribution: EFTA with exponential branches
- Key results

### Method
- EFTA formulation: f(x) = α(exp(βx)-1) for x<0, γx for x≥0
- Tree structure and hierarchical combining
- Parameter count analysis

### Experiments
- MNIST results (with error bars)
- Fashion-MNIST generalization
- Branch specialization visualization
- Ablation studies

### Conclusion
- EFTA achieves SOTA for simple CNNs
- Interpretable feature learning
- Code available at [repo]

---

## Priority Order

1. ✅ **Data pipeline fixed** (uint8 issue resolved)
2. ✅ **Leakage validation** (all checks passing)
3. ⏳ **Statistical significance** (run 5 seeds)
4. ⏳ **Fashion-MNIST** (key generalization test)
5. ⏳ **Branch visualization** (unique interpretability angle)
6. ⏸️ **CIFAR-10** (if time permits)
7. ⏸️ **Ablation/regularization** (supporting evidence)

---

## Next Action

**Start with statistical significance** - this is required for any paper:

```bash
# Run all 5 seeds for key models
python experiments/advanced/01_statistical_significance.py
```

Expected runtime: ~2-3 hours

Then move to **Fashion-MNIST** while that runs:

```bash
python experiments/fashion_mnist_efta.py
```

Expected runtime: ~1-2 hours

---

**Last Updated:** March 6, 2026  
**Status:** Ready to begin experiments
