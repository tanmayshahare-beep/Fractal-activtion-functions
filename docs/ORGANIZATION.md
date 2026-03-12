# Project Organization Quick Reference

## 📁 New Directory Structure

```
Handwritten numbers on mnist/
│
├── 📄 README.md                  # Main project documentation
├── 📄 requirements.txt           # Python dependencies
│
├── 📂 src/                       # SOURCE CODE (importable package)
│   ├── activations/              # Custom activation layers
│   │   ├── FractalTreeActivation (FTA)
│   │   ├── ExponentialFTA (EFTA)
│   │   └── MaxoutLayer
│   ├── models/                   # Model architectures
│   │   ├── MLP models (create_mlp_*)
│   │   └── CNN models (create_cnn_*)
│   └── utils/                    # Utility functions
│       └── data_loader.py        # Dataset loading utilities
│
├── 📂 datasets/                  # DATASETS
│   ├── MNIST/                    # MNIST handwritten digits
│   └── FashionMNIST/             # Fashion-MNIST dataset
│
├── 📂 experiments/               # EXPERIMENT SCRIPTS
│   ├── mnist_efta.py             # Main EFTA experiment
│   ├── fashion_mnist_efta.py     # Fashion-MNIST experiment
│   ├── cnn_efta.py               # CNN comparison
│   ├── README.md                 # Experiments documentation
│   └── advanced/                 # Advanced experiments
│       ├── 01_statistical_significance.py
│       ├── 04_ablation_study.py
│       └── 06_regularization_study.py
│
├── 📂 scripts/                   # UTILITY SCRIPTS
│   ├── baseline_pytorch.py       # PyTorch baseline
│   ├── baseline_tensorflow.py    # TensorFlow baseline
│   ├── maxout_pytorch.py         # Maxout PyTorch
│   ├── maxout_tensorflow.py      # Maxout TensorFlow
│   └── check_gpu.py              # GPU check utility
│
├── 📂 outputs/                   # GENERATED OUTPUTS
│   ├── plots/                    # Generated plots (.png)
│   └── results/                  # Saved results (.npy)
│
└── 📂 docs/                      # DOCUMENTATION
```

## 🔑 Key Changes

### 1. Modular Source Code (`src/`)
- **Before**: Activation layers duplicated in every script
- **After**: Single source of truth in `src/activations/`

```python
# New import pattern
from src.activations import ExponentialFTA, FractalTreeActivation
from src.models import create_cnn_efta, create_cnn_fta
from src.utils import load_mnist_local
```

### 2. Centralized Datasets (`datasets/`)
- **Before**: `MNIST dataset/`, `fashionMNIST/` scattered
- **After**: `datasets/MNIST/`, `datasets/FashionMNIST/`

### 3. Organized Experiments (`experiments/`)
- **Before**: Experiment scripts mixed with everything
- **After**: Main experiments in `experiments/`, advanced in `experiments/advanced/`

### 4. Utility Scripts (`scripts/`)
- **Before**: Baseline scripts in root
- **After**: All utility scripts in `scripts/`

### 5. Generated Outputs (`outputs/`)
- **Before**: Plots and results scattered in root
- **After**: `outputs/plots/` for images, `outputs/results/` for .npy files

## 🚀 Usage Examples

### Run Main Experiments
```bash
python experiments/mnist_efta.py
python experiments/fashion_mnist_efta.py
```

### Use Activation Layers in Your Code
```python
from src.activations import ExponentialFTA

# Create model with EFTA
model = Sequential([
    Dense(128, use_bias=False),
    BatchNormalization(),
    ExponentialFTA(num_units=128, depth=2, branch_factor=2),
    Dense(10, activation='softmax')
])
```

### Load Datasets
```python
from src.utils import load_mnist_local

mnist_dir = 'datasets/MNIST'
(x_train, y_train), (x_test, y_test) = load_mnist_local(mnist_dir)
```

## 📝 File Path Updates

All experiment scripts now use relative paths:
- **Dataset paths**: `../datasets/MNIST`, `../datasets/FashionMNIST`
- **Output paths**: `../outputs/plots/`, `../outputs/results/`

## 🗑️ Old Files Cleaned Up

The following have been removed from root:
- ❌ `baseline_pytorch.py`, `baseline_tensorflow.py` → moved to `scripts/`
- ❌ `maxout_pytorch.py`, `maxout_tensorflow.py` → moved to `scripts/`
- ❌ `mnist_efta.py`, `fashion_mnist_efta.py` → moved to `experiments/`
- ❌ `fractal_tree_activation.py` → refactored to `src/activations/`
- ❌ `MNIST dataset/`, `fashionMNIST/` → moved to `datasets/`
- ❌ `*.png`, `*.npy` → moved to `outputs/`

## 📦 Package Structure

The `src/` directory is now a proper Python package:

```python
# Import activations
from src.activations import FractalTreeActivation, ExponentialFTA, MaxoutLayer

# Import model builders
from src.models import (
    create_mlp_baseline, create_mlp_fta, create_mlp_efta, create_mlp_maxout,
    create_cnn_baseline, create_cnn_fta, create_cnn_efta, create_cnn_maxout
)

# Import utilities
from src.utils import load_mnist_local, load_fashion_mnist_local, MNISTLocalDataset
```

## ✅ Next Steps

1. **Run a test experiment**:
   ```bash
   python experiments/mnist_efta.py
   ```

2. **Check GPU setup**:
   ```bash
   python scripts/check_gpu.py
   ```

3. **Read experiment documentation**:
   - `README.md` - Main project docs
   - `experiments/README.md` - Experiment details

4. **Start your own experiments**:
   ```python
   from src.activations import ExponentialFTA
   from src.models import create_cnn_efta
   
   model = create_cnn_efta(depth=2, branch_factor=2)
   ```
