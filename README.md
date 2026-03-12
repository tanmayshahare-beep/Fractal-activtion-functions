# Dynamic Activation Functions Research (PyTorch)

A comprehensive research project exploring novel activation functions for neural networks, including **Fractal Tree Activation (FTA)** and **Exponential Fractal Tree Activation (EFTA)**.

**Framework:** PyTorch

## 📁 Project Structure

```
Handwritten numbers on mnist/
├── src/                        # Source code package
│   ├── activations/            # Custom activation layers
│   │   ├── __init__.py
│   │   ├── fractal_tree_activation.py  # FTA layer
│   │   ├── exponential_fta.py          # EFTA layer
│   │   └── maxout.py                   # Maxout layer
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── mlp.py              # MLP models
│   │   └── cnn.py              # CNN models
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── data_loader.py      # Data loading utilities
│
├── datasets/                   # Dataset directory
│   ├── MNIST/                  # MNIST handwritten digits
│   └── FashionMNIST/           # Fashion-MNIST dataset
│
├── experiments/                # Experiment scripts
│   ├── README.md               # Experiments documentation
│   ├── mnist_efta.py           # EFTA on MNIST (CNN)
│   ├── fashion_mnist_efta.py   # EFTA on Fashion-MNIST (CNN)
│   └── advanced/               # Advanced experiments
│       ├── 01_statistical_significance.py
│       ├── 03_branch_visualization.py
│       ├── 04_ablation_study.py
│       ├── 05_parameter_matched.py
│       ├── 06_regularization_study.py
│       └── 07_cnn_fashion_mnist.py
│
├── scripts/                    # Utility scripts
│   ├── baseline_pytorch.py     # PyTorch baseline
│   ├── maxout_pytorch.py       # PyTorch Maxout
│   └── check_gpu.py            # GPU check utility
│
├── outputs/                    # Generated outputs
│   ├── plots/                  # Generated plots
│   └── results/                # Saved results (.npy files)
│
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify GPU Setup (Optional)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 3. Run Main Experiments

```bash
# EFTA on MNIST
python experiments/mnist_efta.py

# EFTA on Fashion-MNIST
python experiments/fashion_mnist_efta.py
```

## 📊 Activation Functions

### Fractal Tree Activation (FTA)

A hierarchical activation function where each neuron contains a tree of linear transformations combined via max operation.

```python
import torch
from src.activations import FractalTreeActivation

# Create FTA layer
fta = FractalTreeActivation(
    num_units=128,
    depth=2,
    branch_factor=2,
    input_dim=784
)

# Use in a model
x = torch.randn(32, 784)
output = fta(x)
print(output.shape)  # torch.Size([32, 128])
```

### Exponential Fractal Tree Activation (EFTA)

An enhanced FTA using learnable exponential functions in each leaf:

```
f(x) = α*(exp(β*x)-1)  if x < 0
f(x) = γ*x              if x >= 0
```

```python
from src.activations import ExponentialFTA

# Create EFTA layer
efta = ExponentialFTA(
    num_units=128,
    depth=2,
    branch_factor=2,
    input_dim=784
)
```

### Maxout

Standard maxout activation for comparison.

```python
from src.activations import MaxoutLayer

maxout = MaxoutLayer(
    in_features=784,
    out_features=128,
    k=4
)
```

## 🏗️ Model Architectures

### MLP Models

```python
from src.models import (
    create_mlp_baseline,
    create_mlp_fta,
    create_mlp_efta,
    create_mlp_maxout
)

# Baseline MLP
model = create_mlp_baseline(activation='relu')

# FTA MLP
model = create_mlp_fta(depth=2, branch_factor=2)

# EFTA MLP
model = create_mlp_efta(depth=2, branch_factor=2)

# Maxout MLP
model = create_mlp_maxout(k=4)
```

### CNN Models

```python
from src.models import (
    create_cnn_baseline,
    create_cnn_fta,
    create_cnn_efta,
    create_cnn_maxout
)

# Baseline CNN
model = create_cnn_baseline(activation='relu')

# FTA CNN
model = create_cnn_fta(depth=2, branch_factor=2)

# EFTA CNN
model = create_cnn_efta(depth=2, branch_factor=2)

# Maxout CNN
model = create_cnn_maxout(k=4)
```

## 📈 Experiments

### Main Experiments

| Script | Description | Dataset |
|--------|-------------|---------|
| `mnist_efta.py` | CNN with EFTA on MNIST | MNIST |
| `fashion_mnist_efta.py` | CNN with EFTA on Fashion-MNIST | Fashion-MNIST |

### Advanced Experiments

| Script | Description | Runtime |
|--------|-------------|---------|
| `01_statistical_significance.py` | Statistical significance testing | ~30 min |
| `03_branch_visualization.py` | Visualize leaf activations | ~15 min |
| `04_ablation_study.py` | Compare combining operations | ~40 min |
| `05_parameter_matched.py` | Parameter-matched comparison | ~45 min |
| `06_regularization_study.py` | Regularization techniques | ~60 min |
| `07_cnn_fashion_mnist.py` | CNN on Fashion-MNIST | ~90 min |

## 🔬 Key Research Questions

1. **Hierarchical Features**: Does the tree structure in FTA enable learning of multi-scale features?
2. **Exponential Branches**: Do exponential functions (EFTA) provide better expressivity than linear (FTA)?
3. **Parameter Efficiency**: How does FTA/EFTA compare to Maxout at equal parameter counts?
4. **Generalization**: Do gains transfer to more challenging datasets (Fashion-MNIST)?
5. **Specialization**: Do different leaves specialize in different features?

## 🛠️ GPU Setup (Windows)

### 1. Install NVIDIA Drivers

Download from: https://www.nvidia.com/Download/index.aspx

### 2. Install CUDA Toolkit 11.8

Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

### 3. Install cuDNN 8.7

Download from: https://developer.nvidia.com/cudnn-downloads

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 📦 Dependencies

- **PyTorch**: >= 2.0.0
- **torchvision**: >= 0.15.0
- **NumPy**: >= 1.23.5
- **Matplotlib**: >= 3.7.0
- **Seaborn**: >= 0.12.0
- **idx2numpy**: >= 1.0.0
- **scikit-learn**: >= 1.2.0
- **tqdm**: >= 4.65.0

See `requirements.txt` for full list.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dynamic_activations_2024,
  title={Dynamic Activation Functions: Fractal Tree and Exponential Variants},
  author={Your Name},
  year={2024}
}
```

## 📝 License

This project is provided for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaborations, please open an issue on the repository.
