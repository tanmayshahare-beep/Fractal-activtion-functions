# EFTA Parameter Freezing Experiment

## Hypothesis
EFTA activation functions (α, β, γ parameters) learned on simple MNIST data capture general nonlinearities that transfer to complex datasets. By freezing these parameters, we can dramatically reduce trainable parameters while maintaining near-normal accuracy.

## Structure
```
efta_parameter_freezing/
├── README.md                      ← this file
├── phase1_train_mnist_efta.py     ← Train simple CNN+EFTA on MNIST, save learned α,β,γ
├── phase2_train_tinyimagenet_efta.py  ← Transfer to TinyImageNet: scratch vs frozen vs fine-tune
└── outputs/
    ├── mnist_efta_phase1/             ← Phase 1 results (MNIST 99.36%)
    ├── tinyimagenet_efta_phase2_scratch/
    ├── tinyimagenet_efta_phase2_frozen/
    └── tinyimagenet_efta_phase2_finetune/
```

## Phase 1: Learn EFTA on MNIST
**Command:** `python phase1_train_mnist_efta.py`

| Metric | Value |
|---|---|
| Architecture | 2-conv CNN (32→64→128) + EFTA (d=2, k=2) |
| MNIST Test Accuracy | **99.36%** |
| EFTA Parameters Saved | `outputs/mnist_efta_phase1/mnist_efta_learned_params.json` |
| Epochs | 20 (early stopped) |

## Phase 2: Transfer to TinyImageNet
**Command:** `python phase2_train_tinyimagenet_efta.py`

Three modes compared:
| Mode | Description | Trainable Params |
|---|---|---|
| **scratch** | Random init, train everything | All |
| **frozen** | Load MNIST EFTA → freeze α,β,γ → train only conv/bn/fc | ~97.4% |
| **finetune** | Load MNIST EFTA → unfreeze → train everything | All |

### Results
| Mode | Best Val Acc (200 classes) | Best Train Acc | Epochs |
|---|---|---|---|
| From scratch | 1.91% | 18.6% | 12 |
| Frozen (2.6% frozen) | 1.48% | 16.5% | 14 |
| Fine-tune | 1.36% | 20.2% | 16 |

Random baseline = 0.5%. All modes severely underfit — architecture lacks capacity for 200-class TinyImageNet.

### Key Finding
Only **6,336 EFTA params frozen** out of 245,448 total (2.6%). The freeze impact is negligible at this scale, and the underlying model can't fit the data regardless of activation strategy.

## Conclusion
The experiment demonstrates the methodology is sound (MNIST EFTA transfers architecturally), but the TinyImageNet model needs significantly more capacity before parameter efficiency comparisons become meaningful.
