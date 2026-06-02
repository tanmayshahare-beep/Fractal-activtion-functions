"""
Comgra Visualization Script for EFTA and FTA Models on MNIST - KPIs Only Version

This script trains EFTA and FTA models on MNIST while recording KPIs using comgra.
Due to a recursion issue with tensor value recording in the current comgra version,
this script only records KPIs (loss, accuracy curves) and model parameters.

Usage:
    python comgra_visualize_efta_fta_kpis.py
    
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
COMGRA_GROUP = 'efta_fta_mnist_kpis'
BATCH_SIZE = 32
NUM_BATCHES_TO_RECORD = 5

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
    
    train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model_with_comgra(model, model_name, train_loader, trial_id, num_batches=5):
    """Train model with comgra recording (KPIs and parameters only)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize comgra recorder
    recorder = ComgraRecorder(
        comgra_root_path=str(COMGRA_ROOT_PATH),
        group=COMGRA_GROUP,
        trial_id=trial_id,
        decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(
            min_training_steps_difference=1
        ),
        prefixes_for_grouping_module_parameters_visually=['model.'],
        prefixes_for_grouping_module_parameters_in_nodes=['model.'],
        max_num_batch_size_to_record=3,  # Record only first 3 samples
        comgra_is_active=True,
        calculate_svd_and_other_expensive_operations_of_parameters=False,
    )
    
    recorder.add_note(f"Model: {model_name}")
    recorder.add_note(f"Batch size: {BATCH_SIZE}")
    recorder.track_module("model", model)
    
    model.train()
    batch_count = 0
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(1):  # Single epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_count >= num_batches:
                break
            
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            # Start batch
            recorder.start_batch(
                training_step=batch_count,
                current_batch_size=images.shape[0],
                type_of_execution=f'batch_{batch_idx}',
                record_all_tensors_per_batch_index_by_default=True,
                override__recording_is_active=True,
            )
            
            # Start iteration
            recorder.start_iteration()
            
            # Register input
            recorder.register_tensor("input", images, is_input=True)
            
            # Forward
            outputs = model(images)
            recorder.register_tensor("outputs", outputs)
            
            loss = criterion(outputs, labels)
            recorder.register_tensor("loss", loss, is_loss=True)
            
            # Record KPIs
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            recorder.record_kpi_in_graph("loss", "", loss.item())
            recorder.record_kpi_in_graph("accuracy", "", accuracy.item())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            recorder.record_current_gradients("gradients")
            optimizer.step()
            
            # Finish iteration and batch
            recorder.finish_iteration()
            recorder.finish_batch()
            
            batch_count += 1
            print(f"  Batch {batch_count}/{num_batches}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}")
    
    # Finalize
    recorder.finalize()
    print(f"Recording finalized for {model_name}")
    
    return model


def main():
    print("="*70)
    print("Comgra Visualization for EFTA and FTA Models on MNIST (KPIs)")
    print("="*70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist_data()
    print(f"Training batches: {len(train_loader)}")
    
    # Models to train
    models_config = [
        ('trial_relu', 'ReLU Baseline', lambda: create_cnn_baseline(activation='relu')),
        ('trial_fta_d1k2', 'FTA (depth=1, k=2)', lambda: create_cnn_fta(depth=1, branch_factor=2)),
        ('trial_efta_d1k2', 'EFTA (depth=1, k=2)', lambda: create_cnn_efta(depth=1, branch_factor=2)),
    ]
    
    for trial_id, model_name, model_fn in models_config:
        print(f"\n\n{'#'*70}")
        print(f"# {model_name}")
        print(f"{'#'*70}")
        
        try:
            model = model_fn()
            train_model_with_comgra(
                model, model_name, train_loader, 
                trial_id=trial_id, num_batches=NUM_BATCHES_TO_RECORD
            )
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Comgra recording complete!")
    print("="*70)
    print(f"\nData saved to: {COMGRA_ROOT_PATH / COMGRA_GROUP}")
    print("\nTo launch the comgra GUI, run:")
    print(f'  comgra --path "{COMGRA_ROOT_PATH / COMGRA_GROUP}"')
    print("="*70)


if __name__ == '__main__':
    main()
