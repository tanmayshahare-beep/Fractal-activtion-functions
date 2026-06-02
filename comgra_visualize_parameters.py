"""
Comgra Visualization Script for EFTA and FTA Models - Parameters + KPIs Only

This script creates comgra recordings of EFTA and FTA models, recording:
- Model parameters (weights)
- KPIs (loss, accuracy curves)

It avoids the recursion bug by not recording per-neuron values.

Usage:
    python comgra_visualize_parameters.py
    
After running, launch the comgra GUI:
    comgra --path "<path_to_comgra_root>/efta_fta_parameters"
"""

import sys
import os
import shutil
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import comgra
from comgra.recorder import ComgraRecorder
from comgra.objects import DecisionMakerForRecordingsFrequencyPerType

# Import models
from src.models.cnn import create_cnn_efta, create_cnn_fta, create_cnn_baseline

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Configuration
COMGRA_ROOT_PATH = Path(project_root) / 'comgra_data'
COMGRA_GROUP = 'efta_fta_parameters'
BATCH_SIZE = 32
NUM_BATCHES = 3

# Ensure comgra root path exists and clean up previous data
COMGRA_ROOT_PATH.mkdir(exist_ok=True)
comgra_root = COMGRA_ROOT_PATH / COMGRA_GROUP
if comgra_root.exists():
    print(f"Cleaning up previous comgra data at {comgra_root}")
    shutil.rmtree(comgra_root)


def load_mnist_data():
    """Load MNIST dataset using torchvision."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=str(Path(project_root) / 'datasets'),
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_record(model, model_name, train_loader, trial_id, num_batches=3):
    """Train model briefly and record parameters with comgra."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize comgra recorder - record parameters only
    recorder = ComgraRecorder(
        comgra_root_path=str(COMGRA_ROOT_PATH),
        group=COMGRA_GROUP,
        trial_id=trial_id,
        decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(
            min_training_steps_difference=999999  # Never record batches
        ),
        prefixes_for_grouping_module_parameters_visually=['model.'],
        prefixes_for_grouping_module_parameters_in_nodes=['model.'],
        max_num_batch_size_to_record=0,  # Don't record any batch values
        comgra_is_active=True,
        calculate_svd_and_other_expensive_operations_of_parameters=False,
    )
    
    recorder.add_note(f"Model: {model_name}")
    recorder.add_note(f"Parameters: {count_parameters(model):,}")
    recorder.add_note(f"Batch size: {BATCH_SIZE}")
    recorder.track_module("model", model)
    
    # Run a forward-backward pass to record architecture with gradients
    model.train()
    
    images, labels = next(iter(train_loader))
    images = images.to(device).float()
    labels = labels.to(device).long()
    
    recorder.start_batch(
        training_step=0,
        current_batch_size=images.shape[0],
        type_of_execution='parameters_only',
        record_all_tensors_per_batch_index_by_default=False,
        override__recording_is_active=True,
    )
    
    recorder.start_iteration()
    
    # Register input
    recorder.register_tensor("input", images, is_input=True)
    
    # Forward pass
    outputs = model(images)
    recorder.register_tensor("outputs", outputs)
    
    loss = criterion(outputs, labels)
    recorder.register_tensor("loss", loss, is_loss=True)
    
    # Record KPIs
    recorder.record_kpi_in_graph("num_parameters", "", count_parameters(model))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    recorder.record_current_gradients("gradients")
    
    recorder.finish_iteration()
    recorder.finish_batch()
    
    recorder.finalize()
    print(f"  Recorded {model_name}: {count_parameters(model):,} parameters")


def main():
    print("="*70)
    print("Comgra Parameter Visualization for EFTA and FTA Models")
    print("="*70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader = load_mnist_data()
    
    # Models to record
    models_config = [
        ('trial_relu', 'ReLU Baseline', lambda: create_cnn_baseline(activation='relu')),
        ('trial_fta_d1k2', 'FTA (depth=1, k=2)', lambda: create_cnn_fta(depth=1, branch_factor=2)),
        ('trial_fta_d2k2', 'FTA (depth=2, k=2)', lambda: create_cnn_fta(depth=2, branch_factor=2)),
        ('trial_efta_d1k2', 'EFTA (depth=1, k=2)', lambda: create_cnn_efta(depth=1, branch_factor=2)),
        ('trial_efta_d2k2', 'EFTA (depth=2, k=2)', lambda: create_cnn_efta(depth=2, branch_factor=2)),
    ]
    
    print("\nRecording model parameters...")
    for trial_id, model_name, model_fn in models_config:
        print(f"\n{model_name}:")
        try:
            model = model_fn()
            train_and_record(model, model_name, train_loader, trial_id, num_batches=NUM_BATCHES)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Create a summary file
    summary_path = COMGRA_ROOT_PATH / COMGRA_GROUP / 'models_summary.json'
    summary = {
        'models': [
            {
                'trial_id': trial_id,
                'name': model_name,
                'parameters': count_parameters(model_fn()),
            }
            for trial_id, model_name, model_fn in models_config
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("Comgra recording complete!")
    print("="*70)
    print(f"\nData saved to: {COMGRA_ROOT_PATH / COMGRA_GROUP}")
    print("\nTo launch the comgra GUI, run:")
    print(f'  comgra --path "{COMGRA_ROOT_PATH / COMGRA_GROUP}"')
    print("\nThis will show:")
    print("  - Model architecture dependency graphs")
    print("  - Parameter values and statistics")
    print("  - Comparison between different activation functions")
    print("="*70)


if __name__ == '__main__':
    main()
