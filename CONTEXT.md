# Project Context: Dynamic Activation Functions Research (FTA/EFTA)

**Generated:** 7 April 2026  
**Research Period:** 2024–2026  
**Target Publication:** NeurIPS 2026 (Deadline: May 6, 2026)  
**Framework:** PyTorch (CUDA 11.8)  
**Primary Hardware:** NVIDIA RTX 4060 Laptop GPU

---

## Overview

A comprehensive **academic research project** exploring novel hierarchical activation functions for neural networks. The core contributions are two new activation families:

1. **Fractal Tree Activation (FTA)** — each neuron contains a tree of linear transformations combined via hierarchical max-pooling
2. **Exponential Fractal Tree Activation (EFTA)** — same tree structure but each leaf computes a learnable exponential function

**Key Discovery — "Complexity Reversal Phenomenon":**  
| Dataset Complexity | Best Activation | Representative Result |
|---|---|---|
| Simple (MNIST) | FTA (99.53%) | Both near saturation |
| Moderate (Fashion-MNIST) | FTA (93.45%) | Linear branches excel |
| Complex (CIFAR-10) | EFTA (86.56%) | Exponential branches superior |
| Tabular (Adult) | Comparable (~86%) | Both competitive |

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│  Experiment Scripts (root-level .py files)              │
│  ├── cifar10_fta_vs_efta.py (single GPU, memory-optim.) │
│  ├── cifar10_fta_vs_efta_parallel.py (CUDA threading)   │
│  ├── cifar10_fta_vs_efta_3device.py (multi-device)      │
│  ├── imagenet_fta_vs_efta.py (ImageNet-scale)           │
│  ├── miniimagenet_fta_vs_efta.py                        │
│  ├── baselines_vs_fta_efta.py (comprehensive comparison)│
│  ├── adult_fta_efta_comparison.py                       │
│  ├── wine_fta_efta_comparison.py                        │
│  └── train_react_efta_comprehensive.py (REAct variant)  │
├─────────────────────────────────────────────────────────┤
│  src/ — Reusable Python Package                         │
│  ├── activations/  — Custom nn.Module layers            │
│  │   ├── fractal_tree_activation.py  (FTA)              │
│  │   ├── exponential_fta.py          (EFTA)             │
│  │   ├── maxout.py                   (baseline)         │
│  │   ├── react_efta.py               (REAct extension)  │
│  │   └── enhanced_react_efta.py                         │
│  ├── models/       — CNN and MLP builders               │
│  │   ├── cnn.py   (CNNBaseline, CNNFTA, CNNEFTA, ...)   │
│  │   └── mlp.py   (MLP variants)                        │
│  └── utils/        — Data loaders, training utilities   │
├─────────────────────────────────────────────────────────┤
│  experiments/ — Organized experiment suite              │
│  ├── mnist_efta.py / fashion_mnist_efta.py              │
│  └── advanced/  (statistical significance, ablation,    │
│                  branch viz, regularization, etc.)      │
├─────────────────────────────────────────────────────────┤
│  Visualization & Analysis                               │
│  ├── run_branch_viz.py        — Leaf specialization viz │
│  ├── enhanced_branch_viz.py   — Visualization module    │
│  ├── comgra_visualize_*.py    — Comgra graph analysis   │
│  └── outputs/                 — Generated plots/results │
└─────────────────────────────────────────────────────────┘
```

### Consistent CNN Architecture (used across all experiments)

```
Input → Conv1(32, 3×3) → BatchNorm → Activation → MaxPool(2×2) → Dropout(0.25)
     → Conv2(64, 3×3) → BatchNorm → Activation → MaxPool(2×2) → Dropout(0.25)
     → Flatten → Dense(128) → BatchNorm → Activation → Dropout(0.5)
     → Output(num_classes, softmax)
```

- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping:** patience=5 (varies by dataset)
- **Loss:** CrossEntropyLoss

---

## Directory Structure

```
Handwritten numbers on mnist/
├── src/                              # Reusable source package
│   ├── activations/                  # Custom activation layers (FTA, EFTA, Maxout, REAct)
│   ├── models/                       # CNN & MLP model factories
│   ├── utils/                        # Data loading & training utilities
│   └── __init__.py
├── experiments/                      # Organized experiment scripts
│   ├── mnist_efta.py                 # Main MNIST experiment
│   ├── fashion_mnist_efta.py         # Main Fashion-MNIST experiment
│   ├── advanced/                     # 6 advanced experiments (ablation, regularization, etc.)
│   └── README.md                     # Experiment documentation
├── datasets/                         # Local dataset storage (MNIST, FashionMNIST)
├── outputs/                          # Generated outputs
│   ├── plots/                        # PNG figures
│   ├── results/                      # JSON result files
│   ├── models/                       # Saved checkpoints (.pt)
│   ├── wine_branch_analysis/         # Leaf specialization visualizations
│   └── adult_branch_analysis/
├── comgra_data/                      # Comgra computation graph recordings
├── scripts/                          # Utility scripts (GPU check, baselines)
├── tools/                            # External tools (comgra)
├── docs/                             # Additional documentation
│
│── Root-level experiment scripts (standalone, self-contained):
├── cifar10_fta_vs_efta.py            # CIFAR-10 comparison (memory-optimized)
├── cifar10_fta_vs_efta_parallel.py   # CIFAR-10 with CUDA threading
├── cifar10_fta_vs_efta_3device.py    # CIFAR-10 multi-device (CUDA+XPU+NPU)
├── imagenet_fta_vs_efta.py           # ImageNet-scale experiment (AMP, gradient accum.)
├── miniimagenet_fta_vs_efta.py       # miniImageNet (84×84, 100 classes)
├── tinyimagenet_fta_vs_efta.py       # TinyImageNet variant
├── baselines_vs_fta_efta.py          # Comprehensive: MNIST + Fashion-MNIST, all activations
├── mnist_modern_baselines_comparison.py  # MNIST vs modern activations
├── adult_fta_efta_comparison.py      # Adult Census Income tabular data
├── wine_fta_efta_comparison.py       # Wine Recognition tabular data
├── train_react_efta_comprehensive.py # REAct-EFTA variant (4-param leaf function)
├── quick_train.py                    # Fast training with checkpoint saving
├── run_branch_viz.py                 # Branch visualization runner
├── enhanced_branch_viz.py            # Visualization module
├── comgra_visualize_parameters.py    # Comgra parameter recording
├── comgra_visualize_efta_fta.py      # Comgra full training recording
├── comgra_visualize_efta_fta_kpis.py # Comgra KPI tracking
│
│── Documentation:
├── README.md                         # Project overview, quick start
├── COMPREHENSIVE_RESEARCH_SUMMARY.md # Full research report (50+ pages)
├── ADULT_EXPERIMENT_SUMMARY.md       # Adult dataset experiment details
├── BRANCH_VIZ_ANALYSIS.md            # Branch viz analysis (random init)
├── BRANCH_VIZ_ANALYSIS_TRAINED.md    # Branch viz analysis (trained models)
├── COMGRA_VISUALIZATION_GUIDE.md     # Comgra tool usage guide
├── context.txt                       # Compact research context summary
│
│── Configuration:
├── requirements.txt                  # Python dependencies
├── .gitignore
│
└── 2402.09092v1.pdf                  # Reference paper (arXiv)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/activations/fractal_tree_activation.py` | **Core FTA implementation** — tree of linear transforms with hierarchical max |
| `src/activations/exponential_fta.py` | **Core EFTA implementation** — learnable exponential leaves (α, β, γ) |
| `src/models/cnn.py` | **CNN model factories** — consistent architecture across all activations |
| `src/models/mlp.py` | **MLP model factories** — for tabular data experiments |
| `baselines_vs_fta_efta.py` | **Primary comparison script** — MNIST + Fashion-MNIST, 13 model configs |
| `cifar10_fta_vs_efta.py` | **CIFAR-10 comparison** — memory-optimized, key evidence for thesis |
| `imagenet_fta_vs_efta.py` | **ImageNet-scale** — AMP, gradient accumulation, distributed-ready |
| `adult_fta_efta_comparison.py` | **Tabular data** — Adult Census Income, 5-seed statistical analysis |
| `train_react_efta_comprehensive.py` | **REAct-EFTA** — 4-parameter rational exponential leaf variant |
| `run_branch_viz.py` | **Leaf specialization analysis** — Gini coefficient, per-class heatmaps |
| `COMPREHENSIVE_RESEARCH_SUMMARY.md` | **Definitive research report** — all results, thesis, publication strategy |
| `experiments/advanced/01_statistical_significance.py` | Multi-seed statistical testing |
| `experiments/advanced/04_ablation_study.py` | Combining operations ablation (max, sum, product, etc.) |
| `experiments/advanced/06_regularization_study.py` | Dropout, L2, BN placement study |

---

## Mathematical Formulation

### FTA (Fractal Tree Activation)

**Leaf:** `f_leaf(x) = W·x + b`  
**Tree:** `k^d` leaves (depth `d`, branch factor `k`)  
**Combination:** Hierarchical max — internal nodes take element-wise max of children  

**Parameters:** `k^d × (n × m + n)` (weights + biases per leaf)

### EFTA (Exponential Fractal Tree Activation)

**Leaf:**
```
f(x) = α·(exp(β·x) − 1)   if x < 0
f(x) = γ·x                 if x ≥ 0
```

**Learnable per leaf per unit:** α (negative scale), β (exponential rate), γ (positive slope)  
**Parameters:** `k^d × 3n`

### REAct-EFTA (Extension)

**Leaf:** `REAct(x) = (exp(p1·x) − exp(−p2·x)) / (exp(p3·x) + exp(−p4·x))`  
**4 learnable parameters per leaf** — initialized to mimic tanh behavior

---

## Dependencies & Configuration

### Runtime Dependencies
- **PyTorch ≥ 2.0.0** (with CUDA 11.8)
- **torchvision ≥ 0.15.0**
- **NumPy, SciPy**
- **Matplotlib, Seaborn** (visualization)
- **scikit-learn ≥ 1.2.0** (metrics, preprocessing, datasets)
- **tqdm ≥ 4.65.0** (progress bars)
- **idx2numpy ≥ 1.0.0** (MNIST data loading)

### GPU Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Development Tools
- **Comgra** — computation graph visualization (in `tools/comgra/`)

---

## Coding Patterns & Conventions

### Experiment Script Structure
Most root-level experiment scripts follow this pattern:
1. **GPU setup** — device detection, memory limits (`torch.cuda.set_per_process_memory_fraction`)
2. **Model definitions** — inline activation classes + CNN/MLP builders (self-contained)
3. **Training loop** — standard PyTorch train/val/test with early stopping
4. **Results collection** — JSON serialization of metrics
5. **Visualization** — matplotlib/seaborn plots saved to `outputs/`

### Key Conventions
- **Self-contained scripts:** Root-level experiments inline their model definitions for portability
- **Modular package:** `src/` is importable (`from src.activations import FractalTreeActivation`)
- **Memory management:** `clear_memory()` helper calls `torch.cuda.empty_cache()` + `gc.collect()` between models
- **Result naming:** JSON files saved with timestamp and config metadata
- **Seeds:** Experiments use multiple random seeds (typically 5) for statistical significance

### Testing Approach
- **No formal test suite** — validation through multi-seed experiments and statistical testing
- **Ablation studies** in `experiments/advanced/` serve as systematic validation

---

## Development Workflow

### Quick Start
```bash
pip install -r requirements.txt

# Quick verification
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run main experiments
python experiments/mnist_efta.py
python baselines_vs_fta_efta.py
python cifar10_fta_vs_efta.py
```

### Run Specific Experiments
| Task | Command |
|------|---------|
| MNIST | `python experiments/mnist_efta.py` |
| Fashion-MNIST | `python experiments/fashion_mnist_efta.py` |
| CIFAR-10 | `python cifar10_fta_vs_efta.py` |
| CIFAR-10 (parallel) | `python cifar10_fta_vs_efta_parallel.py` |
| ImageNet | `python imagenet_fta_vs_efta.py` |
| Adult tabular | `python adult_fta_efta_comparison.py` |
| Wine tabular | `python wine_fta_efta_comparison.py` |
| REAct-EFTA | `python train_react_efta_comprehensive.py` |
| Branch visualization | `python run_branch_viz.py --dataset wine --model efta` |

### Outputs Location
- **Plots:** `outputs/plots/`
- **Results (JSON):** `outputs/results/`
- **Model checkpoints:** `outputs/models/` or dataset-specific subdirs
- **Branch analysis:** `outputs/wine_branch_analysis/`, `outputs/adult_branch_analysis/`

---

## Important Notes for Agents

### ⚠️ Critical Context
1. **Root scripts are self-contained** — they inline their own model definitions rather than importing from `src/`. This is intentional for portability but means changes to `src/activations/` won't automatically propagate to root scripts.
2. **`src/activations/` is the canonical source** — use these for any new experiments or modifications.
3. **The consistent CNN architecture** (2 conv blocks + 1 dense) is used across ALL dataset experiments for fair comparison.
4. **ImageNet experiments are preliminary** — miniImageNet had initial bugs (channel mismatch) that were fixed; full ImageNet runs are planned.

### 🔬 Research Thesis (Current State)
> "The optimal branch complexity in fractal tree activations is **dataset-dependent**: linear branches (FTA) dominate on moderately complex data, while exponential branches (EFTA) excel on complex natural images."

### 📊 Key Numbers to Know
| Dataset | Best Model | Test Accuracy |
|---------|-----------|---------------|
| MNIST | FTA (d=3, k=2) | 99.53% |
| Fashion-MNIST | FTA (d=2, k=3) | 93.45% |
| CIFAR-10 | EFTA (d=2, k=2) | 86.56% |
| Adult | FTA/EFTA (~tied) | ~85.9% |
| Wine | FTA/EFTA (~tied) | ~94.4% |

### 🚫 Files to NOT Modify Without Reason
- `COMPREHENSIVE_RESEARCH_SUMMARY.md` — serves as the definitive research report
- `2402.09092v1.pdf` — external reference paper
- `context.txt` — compact context summary for AI agents
- `training_log.txt` — historical training record

### 🔧 Areas Requiring Careful Attention
1. **Input tensor format:** FTA/EFTA CNN layers expect `(batch, H, W, channels)` and permute internally. Mismatches cause silent bugs.
2. **Memory management:** CIFAR-10+ experiments use `set_per_process_memory_fraction(0.7)` and explicit `clear_memory()` calls.
3. **EFTA parameter initialization:** α=1.0, β=0.1, γ=1.0 — changing defaults affects all experiments.
4. **Numerical stability:** EFTA clips `beta * x` to `[-10, 10]` to prevent `exp` overflow.

### 📝 NeurIPS 2026 Preparation
- **Abstract deadline:** May 4, 2026
- **Full paper deadline:** May 6, 2026
- **Remaining work items:** Multi-seed runs, modern baselines (Mish, Swish, GELU), leaf specialization on image datasets, theoretical analysis

---

## Glossary

| Term | Definition |
|------|-----------|
| **FTA** | Fractal Tree Activation — hierarchical linear branches with max-pooling |
| **EFTA** | Exponential Fractal Tree Activation — learnable exponential branch functions |
| **REAct-EFTA** | EFTA variant using Rational Exponential (4-param) leaf function |
| **Depth (d)** | Number of hierarchical levels in the activation tree |
| **Branch factor (k)** | Number of children per internal node |
| **Leaves** | `k^d` terminal nodes, each computing an activation function |
| **Gini coefficient** | Metric for leaf usage balance (0 = equal, 1 = one leaf dominates) |
| **Complexity Reversal** | Key finding: FTA→EFTA performance flip with increasing dataset complexity |
| **Comgra** | Computation graph analysis tool for visualizing network architectures |
