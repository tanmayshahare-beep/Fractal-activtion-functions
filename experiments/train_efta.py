"""
EFTA Training Script - PyTorch

This script trains ONLY the Exponential Fractal Tree Activation (EFTA) model on MNIST.
For full comparison with baselines, use mnist_efta.py instead.

Usage:
    python experiments/train_efta.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.activations import ExponentialFTA
from src.models import create_cnn_efta
from src.utils import load_mnist_local, NumpyDataset

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=30, model_name="EFTA"):
    """Train model and return metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {count_parameters(model):,}")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        'name': model_name,
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'params': count_parameters(model),
        'epochs_trained': len(history['train_loss'])
    }


def plot_training_history(result, project_root=None):
    """Plot training history for single model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    history = result['history']

    # Accuracy curves
    ax = axes[0]
    ax.plot(history['train_acc'], label='Train', linewidth=2)
    ax.plot(history['val_acc'], label='Validation', linewidth=2)
    ax.axhline(result['test_acc'], color='red', linestyle='--', label=f'Test ({result["test_acc"]:.2f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{result["name"]} - Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.axhline(result['test_loss'], color='red', linestyle='--', label=f'Test ({result["test_loss"]:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{result["name"]} - Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if project_root:
        outputs_dir = os.path.join(project_root, 'outputs', 'plots')
        os.makedirs(outputs_dir, exist_ok=True)
        plt.savefig(os.path.join(outputs_dir, f'efta_training_history.png'), dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {os.path.join(outputs_dir, 'efta_training_history.png')}")
    else:
        plt.savefig('efta_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_results(result):
    """Print results summary."""
    print("\n" + "="*70)
    print("EFTA TRAINING RESULTS".center(70))
    print("="*70)
    print(f"Model: {result['name']}")
    print(f"Parameters: {result['params']:,}")
    print(f"Epochs trained: {result['epochs_trained']}")
    print("-"*70)
    print(f"Best Validation Accuracy: {max(result['history']['val_acc']):.2f}%")
    print(f"Final Test Accuracy: {result['test_acc']:.2f}%")
    print(f"Final Test Loss: {result['test_loss']:.4f}")
    print("="*70)

    # Show improvement over epochs
    if len(result['history']['val_acc']) > 1:
        first_val_acc = result['history']['val_acc'][0]
        best_val_acc = max(result['history']['val_acc'])
        improvement = best_val_acc - first_val_acc
        print(f"\n📈 Validation accuracy improved from {first_val_acc:.2f}% to {best_val_acc:.2f}% (+{improvement:.2f}%)")

    print("="*70)


def main():
    print("="*70)
    print("Training EFTA on MNIST (PyTorch)")
    print("="*70)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mnist_dir = os.path.join(project_root, 'datasets', 'MNIST')

    (x_train, y_train), (x_test, y_test) = load_mnist_local(mnist_dir)
    print(f"\nDataset: MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Keep data as uint8 (0-255) - ToTensor() will scale to 0-1
    # DO NOT convert to float32 before NumpyDataset!
    # PIL.Image.fromarray() doesn't handle float32 correctly
    x_train = x_train  # Keep as uint8
    x_test = x_test    # Keep as uint8

    # Split validation from training
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = NumpyDataset(x_train_sub, y_train_sub, transform=train_transform)
    val_dataset = NumpyDataset(x_val, y_val, transform=test_transform)
    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Verify data range
    sample_images, sample_labels = next(iter(train_loader))
    print(f"\nData verification:")
    print(f"  Batch shape: {sample_images.shape}")
    print(f"  Image range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
    print(f"  (Should be [0.000, 1.000])")

    # EFTA configurations to test
    efta_configs = {
        'EFTA (d=2, k=2)': {'depth': 2, 'branch_factor': 2},
        'EFTA (d=3, k=2)': {'depth': 3, 'branch_factor': 2},
        'EFTA (d=2, k=3)': {'depth': 2, 'branch_factor': 3},
    }

    results = {}
    for name, config in efta_configs.items():
        try:
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            model = create_cnn_efta(depth=config['depth'], branch_factor=config['branch_factor'])
            
            # Verify data range
            sample_images, sample_labels = next(iter(train_loader))
            print(f"Data range check: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
            
            result = train_model(
                model, train_loader, val_loader, test_loader, device,
                epochs=30, model_name=name)
            results[name] = result
            
            print_results(result)
            
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Plot results
    if results:
        plot_training_history(list(results.values())[0], project_root)
        
        # Save results
        outputs_dir = os.path.join(project_root, 'outputs', 'results')
        os.makedirs(outputs_dir, exist_ok=True)
        np.save(os.path.join(outputs_dir, 'efta_only_results.npy'), results, allow_pickle=True)
        print(f"\nResults saved to: {os.path.join(outputs_dir, 'efta_only_results.npy')}")

    return results


if __name__ == '__main__':
    main()
