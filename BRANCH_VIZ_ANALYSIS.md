# Branch Visualization Analysis - Wine & Adult Datasets

## Execution Summary

**Date:** March 18, 2026  
**Datasets:** Wine (36 test samples), Adult (9,769 test samples)  
**Models:** FTA (d=2,k=2) and EFTA (d=2,k=2) with MLP architecture

---

## ⚠️ Important Note

The visualizations were generated using **randomly initialized models** (not trained weights from experiments). This is because the training scripts don't save individual model checkpoints by default. 

**What this means:**
- FTA results show **expected behavior** with balanced leaf usage (~25% each for 4 leaves)
- EFTA results show **default behavior** with parameters at initialization (α=1.0, β=0.5, γ=1.0)
- To see **learned specialization**, run the visualization script with saved model checkpoints

---

## Key Findings from FTA (Random Initialization)

### Wine Dataset (13 features → 64 hidden units)

| Metric | Value |
|--------|-------|
| **Leaf Selection Balance** | Excellent (~25% each) |
| Leaf 0 | 25.39% |
| Leaf 1 | 23.52% |
| Leaf 2 | 24.70% |
| Leaf 3 | 26.39% |
| **Weight Norm Mean** | 11.41 |
| **Weight Norm Std** | 0.15 |

### Adult Dataset (108 features → 128 hidden units)

| Metric | Value |
|--------|-------|
| **Leaf Selection Balance** | Excellent (~25% each) |
| Leaf 0 | 26.31% |
| Leaf 1 | 24.19% |
| Leaf 2 | 25.63% |
| Leaf 3 | 23.87% |
| **Weight Norm Mean** | 16.03 |
| **Weight Norm Std** | 0.08 |

### Interpretation

With random initialization, FTA shows **perfectly balanced leaf usage** - each leaf wins approximately 25% of the time. This is expected because:
1. All leaf weights are drawn from the same distribution
2. No leaf has learned to specialize on specific input patterns
3. The max operation selects leaves essentially randomly

**After training**, we would expect:
- Some leaves to specialize on specific input feature combinations
- Selection frequencies to become imbalanced (some leaves win more often)
- Weight norms to diverge (different leaves learn different scales)

---

## EFTA Behavior (Random Initialization)

| Parameter | Initial Value | Observed Value |
|-----------|---------------|----------------|
| Alpha (α) | 1.0 | 1.0000 ± 0.0000 |
| Beta (β)  | 0.5 | 0.5000 ± 0.0000 |
| Gamma (γ) | 1.0 | 1.0000 ± 0.0000 |

**Leaf Selection:**
- Leaf 0: 100%
- Leaf 1-3: 0%

**Why one leaf dominates:** With identical parameters across all leaves, tiny numerical differences cause the same leaf to always win the max operation. After training, different leaves should learn different (α, β, γ) values and specialize.

---

## Generated Visualizations

### Wine Dataset (`./outputs/wine_branch_analysis/`)

1. **efta_leaf_specialization.png** - 4-panel EFTA analysis
   - Overall selection frequency
   - Per-class selection heatmap
   - Gini coefficient (usage balance)
   - Activation statistics

2. **efta_activation_curves.png** - Learned function shapes for each leaf
   - Shows f(x) curves with α, β, γ parameters

3. **efta_parameter_heatmaps.png** - α, β, γ distributions across leaves and units

4. **efta_parameter_distribution.png** - Violin and box plots of parameters

5. **fta_leaf_specialization.png** - 4-panel FTA analysis
   - Overall selection frequency
   - Per-class selection heatmap  
   - Weight norms
   - Activation statistics

6. **fta_weight_analysis.png** - Weight matrix heatmap and distributions

### Adult Dataset (`./outputs/adult_branch_analysis/`)

Same 6 visualizations as Wine, but with:
- Larger input dimension (108 vs 13)
- More hidden units (128 vs 64)
- Binary classification (2 classes vs 3)

---

## How to Use with Trained Models

To visualize **actual learned specialization**, modify the script to load trained checkpoints:

```python
# After training, save model:
torch.save(model.state_dict(), 'best_efta_wine.pt')

# Then in run_branch_viz.py:
model = MLPWithEFTA(...)
model.load_state_dict(torch.load('best_efta_wine.pt'))
model.eval()

# Then call visualization:
visualize_efta_leaf_specialization(model, X_test, y_test, save_dir)
```

---

## Expected Results with Trained Models

Based on the research thesis:

### FTA (Linear Branches)
- **Wine:** Some leaf specialization, moderate imbalance (Gini ~0.3-0.5)
- **Adult:** More pronounced specialization due to more features (Gini ~0.4-0.6)
- Weight norms should vary across leaves
- Different leaves should activate for different classes

### EFTA (Exponential Branches)
- **Wine:** Parameters should diverge from initialization
  - Some leaves learn larger α (strong negative response)
  - Some leaves learn larger γ (stronger linear response)
  - Beta typically stays small (stable exponential)
- **Adult:** More diverse parameter values due to complex feature interactions
- Leaf selection should be more balanced than random init (Gini < 0.5)

---

## Metrics to Analyze

### Gini Coefficient
- **0.0:** Perfect balance (all leaves used equally)
- **0.0-0.3:** Balanced usage (good diversity)
- **0.3-0.5:** Moderate specialization
- **0.5+:** High imbalance (few leaves dominate)

### Per-Class Selection Patterns
- **Uniform across classes:** Leaves don't specialize by class
- **Different patterns:** Leaves specialize for specific input types

### Parameter Statistics (EFTA only)
- **Alpha > 1:** Stronger negative exponential response
- **Beta > 0.5:** Steeper exponential curve
- **Gamma > 1:** Stronger positive linear response

---

## Next Steps

1. **Save model checkpoints** during training (add to training scripts)
2. **Re-run visualization** with trained models
3. **Compare FTA vs EFTA** specialization patterns
4. **Correlate with performance** - does specialization improve accuracy?
5. **Add to paper** - include best visualizations in NeurIPS submission

---

## Files Created

| File | Purpose |
|------|---------|
| `run_branch_viz.py` | Main visualization runner |
| `enhanced_branch_viz.py` | Visualization module |
| `./outputs/wine_branch_analysis/` | Wine visualizations |
| `./outputs/adult_branch_analysis/` | Adult visualizations |
| `BRANCH_VIZ_ANALYSIS.md` | This document |
