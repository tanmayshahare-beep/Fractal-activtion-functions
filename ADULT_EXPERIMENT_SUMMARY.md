# Adult Dataset Experiment Summary

## Files Created

### Main Script
- **`adult_fta_efta_comparison.py`** - Complete Adult (Census Income) dataset experiment

## Dataset Information

**Adult Income (UCI / OpenML Data ID: 1590)**
- **Samples:** 48,842 (after cleaning)
- **Original Features:** 14 (6 numeric, 8 categorical)
- **Features after preprocessing:** 108 (after OneHotEncoding)
- **Target:** Binary (income >50K vs <=50K)
- **Split:** 60% train / 20% val / 20% test
- **Class distribution:** 37,155 (<=50K) / 11,687 (>50K)

## Models Compared

### Standard Baselines
- ReLU
- LeakyReLU (0.01)
- GELU
- Swish/SiLU
- Mish
- EELU (custom exponential ELU)

### Maxout
- Maxout (k=4)

### FTA Variants
- FTA (d=2, k=2)
- FTA (d=3, k=2)
- FTA (d=2, k=3)

### EFTA Variants
- EFTA (d=2, k=2)
- EFTA (d=3, k=2)
- EFTA (d=2, k=3)

## Model Architecture

```
Input (108 features)
    ↓
Linear(108 → 128) + Activation
    ↓
Linear(128 → 64) + Activation
    ↓
Linear(64 → 32) + Activation
    ↓
Linear(32 → 2) [Output]
```

## Training Configuration

- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch size:** 128
- **Epochs:** 100 (with early stopping, patience=10)
- **Seeds:** 5 for statistical significance
- **Loss:** CrossEntropyLoss

## Expected Results (based on thesis)

Based on the research thesis that **linear branches (FTA) excel on moderately complex data**:

| Category | Expected Performance |
|----------|---------------------|
| Standard Baselines | ~85-86% |
| Maxout | ~85-86% |
| FTA (linear) | ~86-87% (best) |
| EFTA (exponential) | ~85-86% |

## Key Hypothesis

On the Adult dataset (moderate complexity tabular data):
- **FTA should outperform EFTA** because linear branches can capture feature interactions without the constraints of exponential parameterization
- This would mirror the Fashion-MNIST results and contrast with CIFAR-10 where EFTA excels
- Would support the thesis: "activation complexity should match dataset complexity"

## Runtime Estimates

- **Per seed:** ~2-5 minutes (depending on model complexity)
- **Total (5 seeds, 13 models):** ~2-4 hours
- **GPU:** NVIDIA RTX 4060 Laptop

## Output Files

After completion, results are saved to:
- `./outputs/adult_results/adult_comparison.png` - Bar chart comparison
- `./outputs/adult_results/adult_training_curves.png` - Training curves
- `./outputs/adult_results/adult_results_*.json` - Full numerical results
- `./outputs/adult_visualizations/` - FTA/EFTA branch visualizations

## Quick Run Version

For faster results, modify the `run_experiment()` call in `main()`:

```python
# Quick run (2 seeds, 50 epochs)
results, best_models, display_names, X_test = run_experiment(
    n_seeds=2, epochs=50, patience=10, batch_size=128
)
```

## Preprocessing Notes

1. **Numeric features:** Standardized (mean=0, std=1)
2. **Categorical features:** OneHotEncoded with unknown category handling
3. **Stratified splits:** Maintain class balance across train/val/test

## Next Steps After Adult

1. **If FTA > EFTA:** Strengthens thesis for tabular data
2. **If EFTA ≥ FTA:** Consider Covertype dataset (larger, more complex)
3. **Statistical testing:** Run paired t-test to confirm significance

## Related Experiments

| Dataset | Samples | Features | Complexity | Status |
|---------|---------|----------|------------|--------|
| Wine | 178 | 13 → 13 | Low | ✅ Complete |
| Adult | 48,842 | 14 → 108 | Moderate | 🔄 Running |
| MNIST | 70,000 | 784 | Low-Moderate | ✅ Complete |
| CIFAR-10 | 60,000 | 3072 | High | ✅ Complete |
| Covertype | 581,012 | 54 | High | ⏳ Future |
