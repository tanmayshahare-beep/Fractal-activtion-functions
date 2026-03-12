"""
EFTA (d=2, k=6) on Fashion-MNIST - PyTorch

This script trains an EFTA model with:
- depth=2, branch_factor=6
- Total leaves: 6^2 = 36 leaves
- Tests on Fashion-MNIST dataset

This configuration tests a moderate depth model with higher branching
for better gradient flow and specialized feature learning.

Usage:
    python experiments/efta_fashion_mnist_d2k8.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from src.activations import ExponentialFTA
from src.utils import load_fashion_mnist_local, NumpyDataset

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNNEFTA(nn.Module):
    """
    CNN with Exponential Fractal Tree Activation (EFTA).

    Architecture:
        Input -> Conv2D(32) -> BN -> EFTA -> MaxPool -> Dropout
              -> Conv2D(64) -> BN -> EFTA -> MaxPool -> Dropout
              -> Flatten -> Dense(128) -> BN -> EFTA -> Dropout
              -> Dense(num_classes)
    """

    def __init__(self, depth=2, branch_factor=6, input_channels=1, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.efta1 = ExponentialFTA(num_units=32, depth=depth,
                                     branch_factor=branch_factor, input_dim=32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.efta2 = ExponentialFTA(num_units=64, depth=depth,
                                     branch_factor=branch_factor, input_dim=64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.efta3 = ExponentialFTA(num_units=128, depth=depth,
                                     branch_factor=branch_factor, input_dim=128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Ensure input is (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout2d(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout2d(x)
        
        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, loader, criterion, optimizer, device, use_amp=False, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
                epochs=50, model_name="EFTA (d=2, k=6)", use_amp=True):
    """Train model and return metrics with mixed precision support."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision scaler for CUDA
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Using Mixed Precision: {use_amp and device.type == 'cuda'}")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # Early stopping with patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best model! Val Acc: {val_acc:.2f}%")
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
        'epochs_trained': len(history['train_loss']),
        'best_val_acc': best_val_acc
    }


def plot_training_history(result, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    history = result['history']
    
    # Accuracy curves
    ax = axes[0]
    ax.plot(history['train_acc'], label='Train', linewidth=2)
    ax.plot(history['val_acc'], label='Validation', linewidth=2)
    ax.axhline(result['test_acc'], color='red', linestyle='--', 
               label=f'Test ({result["test_acc"]:.2f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{result["name"]} - Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss curves
    ax = axes[1]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.axhline(result['test_loss'], color='red', linestyle='--', 
               label=f'Test ({result["test_loss"]:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{result["name"]} - Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()


def print_results(result):
    """Print results summary."""
    print("\n" + "="*70)
    print("EFTA (d=2, k=6) FASHION-MNIST RESULTS".center(70))
    print("="*70)
    print(f"Model: {result['name']}")
    print(f"Depth: 2, Branch Factor: 6")
    print(f"Total Leaves: 6^2 = 36")
    print(f"Parameters: {result['params']:,}")
    print(f"Epochs trained: {result['epochs_trained']}")
    print("-"*70)
    print(f"Best Validation Accuracy: {result['best_val_acc']:.2f}%")
    print(f"Final Test Accuracy: {result['test_acc']:.2f}%")
    print(f"Final Test Loss: {result['test_loss']:.4f}")
    print("="*70)

    # Compare with baseline expectations
    print("\n📊 Comparison with Expected Results:")
    print("  ReLU Baseline: ~90.5%")
    print("  Maxout (k=4): ~91.2%")
    print("  FTA (d=2,k=2): ~91.8%")
    print(f"  EFTA (d=2,k=6): {result['test_acc']:.2f}% (this work)")

    if result['test_acc'] > 92.5:
        print("\n✅ EFTA (d=2,k=6) OUTPERFORMS all smaller configurations!")
    elif result['test_acc'] > 92.0:
        print("\n✅ EFTA (d=2,k=6) shows strong performance!")
    elif result['test_acc'] > 91.8:
        print("\n✅ EFTA (d=2,k=6) outperforms FTA (d=2,k=2)!")
    elif result['test_acc'] > 91.2:
        print("\n✅ EFTA (d=2,k=6) outperforms Maxout baseline!")
    elif result['test_acc'] > 90.5:
        print("\n✅ EFTA (d=2,k=6) outperforms ReLU baseline!")

    print("="*70)


def main():
    print("="*70)
    print("EFTA (d=2, k=6) on Fashion-MNIST (PyTorch)")
    print("="*70)
    print("\nConfiguration:")
    print("  Depth: 2")
    print("  Branch Factor: 6")
    print("  Total Leaves: 6^2 = 36")
    print("  Dataset: Fashion-MNIST")
    print("  Mixed Precision: Enabled (for CUDA)")
    print("  Optimizer: AdamW with Cosine Annealing LR")
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

    # Keep as uint8 - ToTensor() will scale
    x_train = x_train  # Keep as uint8
    x_test = x_test    # Keep as uint8

    # Split validation from training
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]

    # Define transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = NumpyDataset(x_train_sub, y_train_sub, transform=train_transform)
    val_dataset = NumpyDataset(x_val, y_val, transform=test_transform)
    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    # Create DataLoaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Verify data range
    sample_images, sample_labels = next(iter(train_loader))
    print(f"\nData verification:")
    print(f"  Batch shape: {sample_images.shape}")
    print(f"  Image range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
    print(f"  (Should be [0.000, 1.000])")

    # Create model
    print("\nCreating EFTA model (d=2, k=6)...")
    model = CNNEFTA(depth=2, branch_factor=6)

    params = count_parameters(model)
    print(f"Model parameters: {params:,}")

    # Enable cudnn benchmarking for faster training on GPU
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"Enabled cuDNN benchmarking for GPU optimization")

    # Train
    result = train_model(
        model, train_loader, val_loader, test_loader, device,
        epochs=50, model_name="EFTA (d=2, k=6)", use_amp=True
    )

    # Print results
    print_results(result)

    # Plot training history
    outputs_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(outputs_dir, exist_ok=True)
    plot_path = os.path.join(outputs_dir, 'efta_d2k6_fashion_mnist_training.png')
    plot_training_history(result, save_path=plot_path)

    # Save results
    results_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'efta_d2k6_fashion_mnist_results.npy')
    np.save(results_path, result, allow_pickle=True)
    print(f"\nResults saved to: {results_path}")

    return result


if __name__ == '__main__':
    main()
