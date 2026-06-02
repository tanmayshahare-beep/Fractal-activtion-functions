"""
Comgra Visualization Script for EFTA and FTA Models on MNIST

This script trains EFTA and FTA models on MNIST while recording computation graphs
using comgra, allowing for detailed visualization and analysis of the models.

Usage:
    python comgra_visualize_efta_fta.py
    
After running, launch the comgra GUI:
    comgra --path "<path_to_comgra_root>/efta_fta_mnist_comparison"
"""

import sys
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

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
COMGRA_GROUP = 'efta_fta_mnist_comparison'
BATCH_SIZE = 8  # Very small batch for comgra compatibility
NUM_EPOCHS = 1  # Just one epoch for visualization
LEARNING_RATE = 0.001

# Ensure comgra root path exists and clean up previous data
COMGRA_ROOT_PATH.mkdir(exist_ok=True)
comgra_root = COMGRA_ROOT_PATH / COMGRA_GROUP
if comgra_root.exists():
    print(f"Cleaning up previous comgra data at {comgra_root}")
    shutil.rmtree(comgra_root)


def load_mnist_data():
    """Load MNIST dataset using torchvision."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load datasets
    train_dataset = datasets.MNIST(
        root=str(Path(project_root) / 'datasets'),
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=str(Path(project_root) / 'datasets'),
        train=False,
        download=True,
        transform=transform
    )
    
    # Split validation from training
    n_train = len(train_dataset)
    n_val = int(0.1 * n_train)
    n_train_sub = n_train - n_val
    
    train_sub_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [n_train_sub, n_val]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_with_comgra(model, model_name, train_loader, recorder, num_batches=3):
    """Train model for a few batches with comgra recording - minimal tensor registration."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} with comgra recording...")
    print(f"{'='*60}")
    
    batch_count = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_count >= num_batches:
                break
                
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            # Start batch recording
            recorder.start_batch(
                training_step=batch_count,
                current_batch_size=images.shape[0],
                type_of_execution=f'epoch_{epoch}_batch_{batch_idx}',
                record_all_tensors_per_batch_index_by_default=False,
                override__recording_is_active=True,
            )
            
            # Start iteration
            recorder.start_iteration()
            
            # Register input only
            recorder.register_tensor("input", images, is_input=True)
            
            # Forward pass with gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            
            # Register loss
            recorder.register_tensor("loss", loss, is_loss=True)
            
            # Record KPIs
            recorder.record_kpi_in_graph("loss", "", loss.item())
            recorder.record_kpi_in_graph("accuracy", "", accuracy.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Record gradients
            recorder.record_current_gradients("gradients")
            
            # Update parameters
            optimizer.step()
            
            # Finish iteration and batch
            recorder.finish_iteration()
            recorder.finish_batch()
            
            batch_count += 1
            
            print(f"  Batch {batch_count}/{num_batches}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}")
        
        if batch_count >= num_batches:
            break
    
    return model


def main():
    print("="*70)
    print("Comgra Visualization for EFTA and FTA Models on MNIST")
    print("="*70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist_data()
    print(f"Training batches: {len(train_loader)}")
    
    models_to_train = [
        ('trial_relu', 'ReLU Baseline', lambda: create_cnn_baseline(activation='relu')),
        # ('trial_fta_d1k2', 'FTA (depth=1, k=2)', lambda: create_cnn_fta(depth=1, branch_factor=2)),
        # ('trial_efta_d1k2', 'EFTA (depth=1, k=2)', lambda: create_cnn_efta(depth=1, branch_factor=2)),
    ]
    
    for trial_id, model_name, model_fn in models_to_train:
        print(f"\n\n{'#'*70}")
        print(f"# Training {model_name}")
        print(f"{'#'*70}")
        
        # Initialize comgra recorder for this trial
        recorder = ComgraRecorder(
            comgra_root_path=str(COMGRA_ROOT_PATH),
            group=COMGRA_GROUP,
            trial_id=trial_id,
            decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(
                min_training_steps_difference=1
            ),
            prefixes_for_grouping_module_parameters_visually=[
                'model.',
            ],
            prefixes_for_grouping_module_parameters_in_nodes=[
                'model.',
            ],
            max_num_batch_size_to_record=5,
            comgra_is_active=True,
            calculate_svd_and_other_expensive_operations_of_parameters=False,
        )
        
        recorder.add_note(f"Model: {model_name}")
        recorder.add_note(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
        
        # Create and train model
        model = model_fn()
        recorder.track_module("model", model)
        model = train_with_comgra(model, model_name, train_loader, recorder, num_batches=3)
        
        # Finalize
        print(f"\nFinalizing comgra recording for {model_name}...")
        recorder.finalize()
    
    print("\n" + "="*70)
    print("Comgra recording complete for all models!")
    print("="*70)
    print(f"\nData saved to: {COMGRA_ROOT_PATH / COMGRA_GROUP}")
    print("\nTo launch the comgra GUI, run:")
    print(f'  comgra --path "{COMGRA_ROOT_PATH / COMGRA_GROUP}"')
    print("\nOr if running from Python:")
    print(f'  python -m comgra --path "{COMGRA_ROOT_PATH / COMGRA_GROUP}"')
    print("="*70)


if __name__ == '__main__':
    main()
