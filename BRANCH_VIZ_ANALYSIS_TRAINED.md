# Branch Visualization Analysis - Wine & Adult Datasets (TRAINED MODELS)

## Execution Summary

**Date:** March 18, 2026  
**Datasets:** Wine (36 test samples), Adult (9,769 test samples)  
**Models:** FTA (d=2,k=2) and EFTA (d=2,k=2) with MLP architecture  
**Training:** 1 seed, trained to convergence

---

## ✅ Key Results with TRAINED Models

### EFTA (Exponential Fractal Tree Activation)

| Dataset | Leaf 0 | Leaf 1 | Leaf 2 | Leaf 3 | Gini |
|---------|--------|--------|--------|--------|------|
| **Wine** | 48.39% | 28.73% | 8.29% | 14.58% | **0.3362** |
| **Adult** | 49.44% | 23.71% | 12.54% | 14.31% | **0.3002** |

**Parameter Changes After Training:**

| Dataset | Alpha | Beta | Gamma |
|---------|-------|------|-------|
| **Wine** | 0.997 ± 0.013 (init: 1.0) | 0.503 ± 0.011 (init: 0.5) | 0.977 ± 0.027 (init: 1.0) |
| **Adult** | 0.310 ± 0.141 (init: 1.0) | 0.191 ± 0.098 (init: 0.5) | 0.273 ± 0.318 (init: 1.0) |

### FTA (Fractal Tree Activation)

| Dataset | Leaf 0 | Leaf 1 | Leaf 2 | Leaf 3 | Weight Norm Mean | Weight Norm Std |
|---------|--------|--------|--------|--------|------------------|-----------------|
| **Wine** | 22.05% | 25.30% | 26.56% | 26.09% | 11.29 | 0.024 |
| **Adult** | 26.11% | 24.05% | 25.25% | 24.58% | 10.07 | 0.055 |

---

## 🔬 Key Findings

### 1. EFTA Shows Leaf Specialization

**Gini Coefficient Interpretation:**
- **Wine: 0.3362** - Moderate specialization, one leaf dominates (~48%)
- **Adult: 0.3002** - Moderate specialization, one leaf dominates (~49%)

**Why one leaf dominates in EFTA:**
The max operation in the tree structure naturally leads to "winner-take-most" dynamics. One leaf learns slightly better parameters and wins more often, reinforcing its dominance.

**Parameter Divergence (Adult):**
- Alpha dropped from 1.0 → 0.31 (less negative scaling needed)
- Beta dropped from 0.5 → 0.19 (flatter exponential)
- Gamma dropped from 1.0 → 0.27 (smaller positive slope)

This suggests the Adult dataset benefits from **more conservative activation scaling** - the model learned to dampen the activation magnitude.

### 2. FTA Shows Balanced Leaf Usage

**Gini ≈ 0** (near-perfect balance):
- **Wine:** All leaves used ~25% each (22-27% range)
- **Adult:** All leaves used ~25% each (24-26% range)

**Why FTA is balanced:**
- Linear branches with random initialization start symmetric
- No inherent asymmetry in the activation function
- Weight norms are nearly identical across leaves (std < 0.06)
- The tree max operation distributes load evenly when leaves are similar

**Weight Norm Analysis:**
- Wine: Mean=11.29, Std=0.024 (very consistent)
- Adult: Mean=10.07, Std=0.055 (very consistent)

Small standard deviation indicates all leaves learned similar weight magnitudes - no leaf specialized to different input scales.

### 3. Dataset Complexity Affects Learning

**Wine (simple, 13 features):**
- EFTA parameters stayed close to initialization
- Small adjustments: α=-0.3%, β=+0.6%, γ=-2.3%
- Suggests default parameters were already near-optimal

**Adult (complex, 108 features):**
- EFTA parameters changed dramatically
- Large adjustments: α=-69%, β=-62%, γ=-73%
- Model learned fundamentally different activation shape
- Higher parameter variance (α std=0.14, γ std=0.32) indicates leaf specialization

---

## 📊 Generated Visualizations

### Wine Dataset (`./outputs/wine_branch_analysis/`)

1. **efta_leaf_specialization.png** - 4-panel analysis showing:
   - Leaf 0 dominates (48.39%)
   - Gini = 0.336 (moderate imbalance)
   - Parameters near initialization

2. **efta_activation_curves.png** - Function shapes for 4 leaves
   - Shows learned (α, β, γ) per leaf

3. **efta_parameter_heatmaps.png** - Parameter distributions
   - Uniform across leaves and units (low specialization)

4. **efta_parameter_distribution.png** - Violin plots
   - Tight distributions around mean values

5. **fta_leaf_specialization.png** - 4-panel FTA analysis
   - Balanced leaf usage (~25% each)
   - Consistent weight norms

6. **fta_weight_analysis.png** - Weight matrix heatmap
   - No clear specialization patterns

### Adult Dataset (`./outputs/adult_branch_analysis/`)

Same 6 visualizations, showing:
- Similar EFTA leaf dominance pattern
- Larger parameter changes from initialization
- FTA remains balanced

---

## 🧠 Interpretation for Research Thesis

### Thesis: "Linear branches (FTA) excel on moderate complexity; Exponential branches (EFTA) excel on high complexity"

**Evidence from Visualizations:**

1. **FTA Balance → Robustness**
   - Balanced leaf usage suggests no single point of failure
   - All leaves contribute equally to representation
   - May explain FTA's strong performance on Fashion-MNIST

2. **EFTA Specialization → Expressivity**
   - One dominant leaf + specialized supporting leaves
   - Parameters adapt to dataset statistics
   - Adult shows larger parameter changes → more adaptation needed
   - May explain EFTA's advantage on CIFAR-10 (complex images)

3. **Gini Coefficient as Specialization Metric**
   - FTA: Gini ≈ 0.01 (random) to 0.02 (trained) - minimal specialization
   - EFTA: Gini ≈ 0.30-0.34 (trained) - moderate specialization
   - Higher Gini in EFTA indicates functional differentiation

### Predictions for Image Datasets

Based on tabular results:

**Fashion-MNIST (moderate):**
- FTA should show balanced usage (Gini < 0.1)
- EFTA may show moderate specialization (Gini ~ 0.2-0.3)
- FTA's balance → better generalization

**CIFAR-10 (complex):**
- FTA may struggle (too much capacity, overfitting)
- EFTA's specialization → efficient feature extraction
- EFTA's parameter adaptation → better inductive bias

---

## 📈 Metrics Summary

| Model | Dataset | Test Acc | Gini | Param Change | Specialization |
|-------|---------|----------|------|--------------|----------------|
| **FTA** | Wine | 94.44% | ~0.02 | N/A | None |
| **FTA** | Adult | 85.94% | ~0.01 | N/A | None |
| **EFTA** | Wine | 94.44% | 0.336 | Minimal | Low |
| **EFTA** | Adult | 85.90% | 0.300 | Large | Moderate |

---

## 🔭 Next Steps

1. **Run on Image Datasets**
   - Apply same visualization to MNIST/CIFAR-10 trained models
   - Compare Gini coefficients across dataset complexities

2. **Correlate Gini with Accuracy**
   - Run multiple seeds
   - Plot Gini vs. test accuracy scatter

3. **Ablation: Force Leaf Balance**
   - Add loss term to encourage equal leaf usage
   - Test if balanced EFTA performs better

4. **Leaf Function Clustering**
   - Cluster leaves by (α, β, γ) similarity
   - Identify distinct "activation types" learned

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `quick_train.py` | Fast model training with checkpoint saving |
| `run_branch_viz.py` | Visualization runner (loads checkpoints) |
| `enhanced_branch_viz.py` | Visualization module |
| `./outputs/wine_branch_analysis/` | 6 Wine visualizations |
| `./outputs/adult_branch_analysis/` | 6 Adult visualizations |
| `./outputs/wine_results/model_*.pt` | Trained Wine checkpoints |
| `./outputs/adult_results/model_*.pt` | Trained Adult checkpoints |
| `BRANCH_VIZ_ANALYSIS_TRAINED.md` | This document |

---

## ✅ Conclusion

The trained model visualizations reveal:

1. **FTA maintains balanced leaf usage** across datasets - no specialization
2. **EFTA develops moderate specialization** - one dominant leaf emerges
3. **Parameter adaptation scales with complexity** - Adult shows larger changes than Wine
4. **Gini coefficient quantifies specialization** - useful metric for comparing activations

These findings support the thesis that **FTA and EFTA have fundamentally different specialization behaviors**, which may explain their dataset-dependent performance differences.
