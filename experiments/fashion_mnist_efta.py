"""
CNN with Exponential Fractal Tree Activation (EFTA) on Fashion-MNIST - PyTorch

This script implements a Convolutional Neural Network using EFTA as the activation function
on the Fashion-MNIST dataset (more challenging than standard MNIST).

EFTA replaces linear branches in FTA with learnable exponential functions:
    f(x) = α * (exp(β*x) - 1)  if x < 0
    f(x) = γ * x                if x >= 0

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Usage:
    python experiments/fashion_mnist_efta.py
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

from src.activations import ExponentialFTA, FractalTreeActivation, MaxoutLayer
from src.models import create_cnn_baseline, create_cnn_fta, create_cnn_efta, create_cnn_maxout
from src.utils import load_fashion_mnist_local, NumpyDataset

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
                epochs=30, model_name="Model", use_augmentation=True):
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


def plot_results(results, project_root=None):
    """Plot training results comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    test_accs = [r['test_acc'] for r in results.values()]

    # Test accuracy bar chart
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), test_accs, color=colors)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Fashion-MNIST - Final Test Accuracy')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(test_accs) - 2, max(test_accs) + 2)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)

    # Validation accuracy curves
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['history']['val_acc'], color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Validation loss curves
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['history']['val_loss'], color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy vs Parameters
    ax = axes[1, 1]
    params = [r['params'] for r in results.values()]
    ax.scatter(params, test_accs, s=150, c=colors)
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], test_accs[i]), fontsize=7,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Parameter Efficiency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if project_root:
        outputs_dir = os.path.join(project_root, 'outputs', 'plots')
        os.makedirs(outputs_dir, exist_ok=True)
        plt.savefig(os.path.join(outputs_dir, 'fashion_mnist_efta_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {os.path.join(outputs_dir, 'fashion_mnist_efta_comparison.png')}")
    else:
        plt.savefig('fashion_mnist_efta_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_comparison_table(results):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("FASHION-MNIST CNN ACTIVATION COMPARISON RESULTS".center(90))
    print("="*90)
    print(f"{'Model':<25} | {'Test Acc':<12} | {'Test Loss':<12} | {'Params':<12} | {'Epochs':<8}")
    print("-"*90)
    for name, result in results.items():
        print(f"{name:<25} | {result['test_acc']:>10.2f}% | {result['test_loss']:>10.4f} | "
              f"{result['params']:>10,} | {result['epochs_trained']:>6}")
    print("="*90)

    best = max(results, key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 Best Model: {best} with {results[best]['test_acc']:.2f}% test accuracy")

    efta_models = [k for k in results.keys() if 'EFTA' in k]
    fta_models = [k for k in results.keys() if 'FTA' in k and 'EFTA' not in k]
    if efta_models and fta_models:
        print("\n📊 EFTA vs FTA Comparison:")
        for efta_name in efta_models:
            base_name = efta_name.replace('EFTA', 'FTA')
            if base_name in fta_models:
                diff = results[efta_name]['test_acc'] - results[base_name]['test_acc']
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {efta_name} vs {base_name}: {symbol} {abs(diff):.2f}%")
    print("="*90)


def main():
    print("="*70)
    print("CNN with EFTA on Fashion-MNIST (PyTorch)")
    print("="*70)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fashion_mnist_dir = os.path.join(project_root, 'datasets', 'FashionMNIST')

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fashion_mnist_dir)
    print(f"\nDataset: Fashion-MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(f"Classes: {', '.join(class_names)}")

    # Preprocess - Keep as uint8 (0-255), ToTensor() will scale to 0-1
    # DO NOT convert to float32 before NumpyDataset!
    # PIL.Image.fromarray() doesn't handle float32 correctly
    x_train = x_train  # Keep as uint8
    x_test = x_test    # Keep as uint8

    # Split validation from training
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]

    # Define transforms - ToTensor() will scale 0-255 to 0-1
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),  # Scales 0-255 -> 0-1
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Scales 0-255 -> 0-1
    ])

    # Create datasets
    train_dataset = NumpyDataset(x_train_sub, y_train_sub, transform=train_transform)
    val_dataset = NumpyDataset(x_val, y_val, transform=test_transform)
    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Models to compare
    models_config = {
        'CNN + ReLU': lambda: create_cnn_baseline(activation='relu'),
        'CNN + LeakyReLU': lambda: create_cnn_baseline(activation='leaky_relu'),
        'CNN + Maxout (k=4)': lambda: create_cnn_maxout(k=4),
        'CNN + FTA (d=2,k=2)': lambda: create_cnn_fta(depth=2, branch_factor=2),
        'CNN + FTA (d=3,k=2)': lambda: create_cnn_fta(depth=3, branch_factor=2),
        'CNN + FTA (d=2,k=3)': lambda: create_cnn_fta(depth=2, branch_factor=3),
        'CNN + EFTA (d=2,k=2)': lambda: create_cnn_efta(depth=2, branch_factor=2),
        'CNN + EFTA (d=3,k=2)': lambda: create_cnn_efta(depth=3, branch_factor=2),
        'CNN + EFTA (d=2,k=3)': lambda: create_cnn_efta(depth=2, branch_factor=3),
    }

    results = {}
    for name, model_fn in models_config.items():
        try:
            model = model_fn()
            result = train_model(
                model, train_loader, val_loader, test_loader, device,
                epochs=30, model_name=name, use_augmentation=True)
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print_comparison_table(results)
    plot_results(results, project_root)

    # Save results
    outputs_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(outputs_dir, exist_ok=True)
    np.save(os.path.join(outputs_dir, 'fashion_mnist_efta_results.npy'), results, allow_pickle=True)
    print(f"\nResults saved to: {os.path.join(outputs_dir, 'fashion_mnist_efta_results.npy')}")

    return results


if __name__ == '__main__':
    main()
