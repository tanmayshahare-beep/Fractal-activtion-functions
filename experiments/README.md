# Advanced Experiments (PyTorch)

This folder contains advanced experiments for the Fractal Tree Activation (FTA) and Exponential Fractal Tree Activation (EFTA) research using **PyTorch**.

## 📁 Structure

```
experiments/
├── README.md                     # This file
├── mnist_efta.py                 # Main EFTA experiment on MNIST
├── fashion_mnist_efta.py         # Main EFTA experiment on Fashion-MNIST
└── advanced/                     # Advanced experiment scripts
    ├── 01_statistical_significance.py
    ├── 03_branch_visualization.py
    ├── 04_ablation_study.py
    ├── 05_parameter_matched.py
    ├── 06_regularization_study.py
    └── 07_cnn_fashion_mnist.py
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib seaborn scipy idx2numpy tqdm
```

### Run Main Experiments

```bash
# From project root directory
python experiments/mnist_efta.py
python experiments/fashion_mnist_efta.py
```

### Run Advanced Experiments

```bash
cd experiments

# Statistical significance testing
python advanced/01_statistical_significance.py

# Branch visualization
python advanced/03_branch_visualization.py

# Ablation study
python advanced/04_ablation_study.py

# Parameter-matched comparison
python advanced/05_parameter_matched.py

# Regularization study
python advanced/06_regularization_study.py

# CNN on Fashion-MNIST
python advanced/07_cnn_fashion_mnist.py
```

## 📊 Experiments Overview

| File | Description | Runtime (est.) |
|------|-------------|----------------|
| `01_statistical_significance.py` | Run 5 trials per activation, report mean ± std, perform t-tests | ~30 min |
| `03_branch_visualization.py` | Visualize which leaves activate for different digits | ~15 min |
| `04_ablation_study.py` | Compare max vs sum vs product vs other combining operations | ~40 min |
| `05_parameter_matched.py` | Fair comparison: FTA vs Maxout with matched parameters | ~45 min |
| `06_regularization_study.py` | Test dropout, L2, BN, early stopping, data augmentation | ~60 min |
| `07_cnn_fashion_mnist.py` | CNN with FTA activation on Fashion-MNIST | ~90 min |

## 📈 Experiment Details

### 1. Statistical Significance Testing

**Purpose:** Determine if performance differences are statistically significant.

**Method:**
- Run each configuration 5 times with different random seeds
- Report mean ± standard deviation
- Perform paired t-test against ReLU baseline

**Output:** `../outputs/results/statistical_results.npy`

---

### 2. Fashion-MNIST Generalization

**Purpose:** Test if FTA gains generalize to harder datasets.

**Method:**
- Same architecture as MNIST experiments
- Fashion-MNIST has same format but more complex visual features
- Compare performance drop from MNIST to Fashion-MNIST

**Output:** `../outputs/results/fashion_mnist_results.npy`, `../outputs/plots/fashion_mnist_comparison.png`

---

### 3. Branch Visualization

**Purpose:** Understand what features different leaves learn.

**Method:**
- Train FTA model on MNIST
- For each digit class, find which leaves activate most strongly
- Identify specialized leaves and visualize their preferred inputs
- Create tree visualization with activation heatmaps

**Output:** `../outputs/plots/leaf_activation_analysis.png`, `../outputs/plots/specialized_leaf_images.png`

**Key Questions:**
- Do different leaves specialize in different digit classes?
- Are there leaves that activate for similar digits (e.g., 4 and 9)?
- What visual features might each leaf detect?

---

### 4. Ablation Study

**Purpose:** Find optimal combining operation for FTA.

**Operations Tested:**

| Operation | Description |
|-----------|-------------|
| `max` | Standard maxout-style combining |
| `sum` | Sum all branch outputs |
| `mean` | Average of branches |
| `product` | Product (clipped for stability) |
| `weighted_sum` | Learnable weights per branch |
| `max_mean` | Hybrid: 0.7*max + 0.3*mean |

**Output:** `../outputs/results/ablation_results.npy`, `../outputs/plots/ablation_study_results.png`

---

### 5. Parameter-Matched Comparison

**Purpose:** Fair comparison between FTA and Maxout.

**Method:**
- Calculate parameter counts for all configurations
- Compare FTA vs Maxout at similar parameter budgets:
  - ~100k: Maxout k=2 vs FTA d=1,k=2
  - ~200k: Maxout k=4 vs FTA d=2,k=2
  - ~400k: Maxout k=8 vs FTA d=3,k=2
  - ~800k: Maxout k=16 vs FTA d=4,k=2

**Also tests:** FTA with weight sharing for parameter efficiency

**Output:** `../outputs/results/parameter_matched_results.npy`, `../outputs/plots/parameter_matched_comparison.png`

---

### 6. Regularization Study

**Purpose:** Address overfitting and improve generalization.

**Regularization Techniques Tested:**

| Technique | Configurations |
|-----------|---------------|
| Dropout | 0.3, 0.5 |
| L2 Weight Decay | 1e-4, 5e-4 |
| Batch Norm after FTA | Yes/No |
| Early Stopping | Patience=5 |
| Data Augmentation | Rotation ±10°, Shift ±10%, Zoom ±10% |
| Reduced Architecture | FTA (d=2,k=2) vs (d=3,k=2) |
| Combined | Full regularization + data aug |

**Metrics:**
- Best validation accuracy
- Test accuracy
- Generalization gap (train - val accuracy)
- Epochs trained (early stopping effect)

**Output:** `../outputs/results/regularization_results.npy`, `../outputs/plots/regularization_study_results.png`

**Expected Outcome:** Combined regularization should reduce overfitting and improve test accuracy.

---

### 7. CNN with FTA on Fashion-MNIST

**Purpose:** Evaluate FTA in a convolutional architecture on a challenging dataset.

**Architecture:**
```
Input (28x28x1)
↓
Conv2D(32) + BN + FTA + MaxPool + Dropout
↓
Conv2D(64) + BN + FTA + MaxPool + Dropout
↓
Flatten + Dense(128) + BN + FTA + Dropout
↓
Softmax(10)
```

**Models Compared:**

| Model | Activation |
|-------|-----------|
| CNN + ReLU | Standard baseline |
| CNN + LeakyReLU | Improved ReLU variant |
| CNN + Maxout (k=4) | Maxout baseline |
| CNN + FTA (d=2,k=2) | FTA 4 leaves |
| CNN + FTA (d=3,k=2) | FTA 8 leaves |
| CNN + FTA (d=2,k=3) | FTA 9 leaves |

**Features:**
- Data augmentation (rotation, shift, zoom)
- Early stopping with patience=5
- Learning rate scheduling

**Output:** `../outputs/results/cnn_fashion_mnist_results.npy`, `../outputs/plots/cnn_fashion_mnist_results.png`

---

## 📊 Parameter Count Reference

| Model | Configuration | Parameters |
|-------|--------------|------------|
| Maxout | k=2 | 100,874 |
| Maxout | k=4 | 201,482 |
| Maxout | k=8 | 402,698 |
| Maxout | k=16 | 805,130 |
| FTA | d=1, k=2 | 100,874 |
| FTA | d=2, k=2 | 201,482 |
| FTA | d=3, k=2 | 402,698 |
| FTA | d=4, k=2 | 805,130 |
| FTA (shared) | d=3, k=2 | 301,834 |
| FTA (shared) | d=4, k=2 | 402,322 |

## 🏆 Expected Results

Based on preliminary runs:

1. **Statistical Significance:** FTA (d=2,k=2) should show significant improvement over ReLU (p < 0.05)

2. **Fashion-MNIST:** Expect ~15-20% drop from MNIST; FTA should maintain relative advantage

3. **Branch Visualization:** Different leaves should specialize in different stroke patterns

4. **Ablation:** Max should outperform sum/product; weighted_sum may offer slight improvement

5. **Parameter-Matched:** FTA should outperform Maxout at equal parameter counts due to hierarchical feature learning

## 🛠️ Troubleshooting

### Out of Memory
- Reduce batch size to 32
- Use fewer epochs for initial testing

### Slow Training
- Enable GPU for faster execution:
  ```bash
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  ```

### Import Errors
```bash
pip install torch torchvision numpy matplotlib seaborn scipy idx2numpy tqdm
```

### No GPU Detected
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Using the Activation Layers

```python
import torch
from src.activations import FractalTreeActivation, ExponentialFTA, MaxoutLayer

# FTA layer
fta = FractalTreeActivation(num_units=128, depth=2, branch_factor=2, input_dim=784)
x = torch.randn(32, 784)
output = fta(x)

# EFTA layer
efta = ExponentialFTA(num_units=128, depth=2, branch_factor=2, input_dim=784)
output = efta(x)

# Maxout layer
maxout = MaxoutLayer(in_features=784, out_features=128, k=4)
output = maxout(x)
```

## 📚 Using the Model Builders

```python
from src.models import create_cnn_baseline, create_cnn_fta, create_cnn_efta, create_cnn_maxout

# Baseline CNN
model = create_cnn_baseline(activation='relu')

# FTA CNN
model = create_cnn_fta(depth=2, branch_factor=2)

# EFTA CNN
model = create_cnn_efta(depth=2, branch_factor=2)

# Maxout CNN
model = create_cnn_maxout(k=4)
```

## 📚 Loading Data

```python
from src.utils import load_mnist_local, load_fashion_mnist_local

# Load MNIST
mnist_dir = '../datasets/MNIST'
(x_train, y_train), (x_test, y_test) = load_mnist_local(mnist_dir)

# Load Fashion-MNIST
fm_dir = '../datasets/FashionMNIST'
(x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fm_dir)
```
