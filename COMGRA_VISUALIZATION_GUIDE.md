# Comgra Visualization Guide for EFTA and FTA Models

This document explains how to use comgra to visualize the EFTA and FTA models.

## What is Comgra?

Comgra is a computation graph analysis tool that helps you:
- Visualize neural network architecture as dependency graphs
- Inspect tensor values and statistics at each layer
- Compare gradients across different layers
- Track parameter changes during training
- Identify anomalies like vanishing/exploding gradients

## Generated Visualizations

The script `comgra_visualize_parameters.py` has generated comgra recordings for:

| Model | Parameters | Description |
|-------|------------|-------------|
| ReLU Baseline | 422,090 | Standard CNN with ReLU activation |
| FTA (depth=1, k=2) | 465,546 | Fractal Tree Activation with 1 level, 2 branches |
| FTA (depth=2, k=2) | 509,002 | Fractal Tree Activation with 2 levels, 2 branches |
| EFTA (depth=1, k=2) | 423,434 | Exponential FTA with 1 level, 2 branches |
| EFTA (depth=2, k=2) | 424,778 | Exponential FTA with 2 levels, 2 branches |

## Launching the GUI

### Option 1: Using the comgra command
```bash
comgra --path "C:\All projects\Dynamic activation functions\Handwritten numbers on mnist\comgra_data\efta_fta_parameters"
```

### Option 2: Running the server directly
```bash
python tools\comgra\src\scripts\server.py --path "comgra_data\efta_fta_parameters"
```

Then open your browser to `http://127.0.0.1:8050/`

## Using the GUI

### Main Components

1. **Dependency Graph** (center)
   - Shows how tensors flow through the network
   - Green nodes = inputs
   - Blue nodes = parameters
   - Orange nodes = losses
   - Click nodes to see their values

2. **Selectors** (left panel)
   - **Trial**: Switch between ReLU, FTA, and EFTA models
   - **Training Step**: View different points in training
   - **Node**: Select which layer to inspect
   - **Batch or Sample**: View individual samples or batch statistics

3. **Tensor Statistics** (bottom panel)
   - Mean, std, min, max values
   - Per-neuron values for detailed inspection
   - Gradient statistics

### Key Things to Investigate

1. **Architecture Comparison**
   - Switch between trials to see how FTA/EFTA add parameters
   - Compare the tree structure of FTA vs exponential branches in EFTA

2. **Parameter Analysis**
   - Select parameter nodes to see weight distributions
   - Check if EFTA's exponential parameters (alpha, beta, gamma) are learning meaningful values

3. **Gradient Flow**
   - Look at gradient magnitudes across layers
   - Compare gradient flow in ReLU vs FTA vs EFTA

4. **Activation Values**
   - Check the output distributions of different activation functions
   - Look for dead neurons or saturation

## Creating Full Training Recordings

To record full training runs with per-batch statistics, use:

```bash
python comgra_visualize_efta_fta_kpis.py
```

This records loss/accuracy curves over time for comparison.

## Troubleshooting

### Recursion Error
If you encounter recursion errors, the comgra source has been patched. The fix converts recursive tensor indexing to an iterative approach.

### No Data Showing
Make sure to:
1. Select a trial from the dropdown
2. Select a training step
3. Click on a node in the dependency graph

### Slow Performance
- Reduce `max_num_batch_size_to_record` in the recorder
- Use `record_all_tensors_per_batch_index_by_default=False`

## File Locations

- **Comgra Data**: `comgra_data\efta_fta_parameters\`
- **Recording Script**: `comgra_visualize_parameters.py`
- **Comgra Source**: `tools\comgra\src\comgra\`

## Next Steps

1. Launch the GUI and explore the model architectures
2. Compare parameter initializations across activation types
3. Run full training recordings to see learning dynamics
4. Use the visualization to understand why EFTA outperforms FTA

## Additional Resources

- Comgra README: `tools\comgra\README.md`
- Comgra Tutorial: Run `comgra-test-run` and `comgra --use-path-for-test-run` for an interactive tutorial
