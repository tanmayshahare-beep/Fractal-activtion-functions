"""
CNN with REAct-EFTA on Fashion-MNIST - PyTorch

This script implements a Convolutional Neural Network using REAct-EFTA 
(Rational Exponential Fractal Tree Activation) on Fashion-MNIST.

REAct (ICASSP 2025) branch function:
    REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))

This provides a tanh-like activation with 4 learnable shape parameters
per leaf, enabling asymmetric saturation and adaptive curvature.

Usage:
    python experiments/fashion_mnist_react_efta.py
    
    # With custom number of seeds:
    python experiments/fashion_mnist_react_efta.py --seeds 10
    
    # With specific seed:
    python experiments/fashion_mnist_react_efta.py --seed 42
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.activations import ReActEFTA, ExponentialFTA, FractalTreeActivation, MaxoutLayer
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


class CNNReActEFTA(nn.Module):
    """
    CNN with REAct-EFTA activation.

    Architecture:
        Input -> Conv2D(32) -> BN -> ReAct-EFTA -> MaxPool -> Dropout
              -> Conv2D(64) -> BN -> ReAct-EFTA -> MaxPool -> Dropout
              -> Flatten -> Dense(128) -> BN -> ReAct-EFTA -> Dropout
              -> Dense(num_classes)

    Args:
        depth: Depth of the ReAct-EFTA tree
        branch_factor: Branching factor of the ReAct-EFTA tree
        input_channels: Number of input channels
        num_classes: Number of output classes
    """

    def __init__(self, depth=2, branch_factor=2, input_channels=1, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.react_efta1 = ReActEFTA(num_units=32, depth=depth,
                                      branch_factor=branch_factor, input_dim=32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.react_efta2 = ReActEFTA(num_units=64, depth=depth,
                                      branch_factor=branch_factor, input_dim=64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.react_efta3 = ReActEFTA(num_units=128, depth=depth,
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
        x = self.react_efta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.react_efta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.react_efta3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_cnn_react_efta(depth=2, branch_factor=2, input_shape=(1, 28, 28), num_classes=10):
    """Factory function for CNN with ReAct-EFTA."""
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 1
    return CNNReActEFTA(depth=depth, branch_factor=branch_factor,
                        input_channels=input_channels, num_classes=num_classes)


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


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=40, model_name="Model", use_augmentation=True, seed=42):
    """Train model and return metrics."""
    # Set seed for this training run
    set_seed(seed)
    
    # Re-initialize model weights with new seed
    def reset_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    
    model.apply(reset_weights)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*70}")
    print(f"Training: {model_name} (Seed: {seed})")
    print(f"{'='*70}")
    print(f"Parameters: {count_parameters(model):,}")

    history = {
        'seed': seed,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc_per_epoch': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Also track test accuracy per epoch
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc_per_epoch'].append(test_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")

        scheduler.step()

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
    history['final_test_acc'] = test_acc
    history['final_test_loss'] = test_loss
    history['best_val_acc'] = best_val_acc

    return {
        'name': model_name,
        'seed': seed,
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'params': count_parameters(model),
        'epochs_trained': len(history['train_loss'])
    }


def plot_results(results, all_runs_results, project_root=None):
    """Plot training results comparison with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute statistics across seeds
    model_names = list(all_runs_results.keys())
    test_accs_mean = [np.mean(all_runs_results[name]['test_accs']) for name in model_names]
    test_accs_std = [np.std(all_runs_results[name]['test_accs']) for name in model_names]

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # Test accuracy bar chart with error bars
    ax = axes[0, 0]
    x_pos = range(len(model_names))
    bars = ax.bar(x_pos, test_accs_mean, yerr=test_accs_std, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Fashion-MNIST - Final Test Accuracy (Mean ± Std)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(max(0, min(test_accs_mean) - 5), min(100, max(test_accs_mean) + 5))
    
    # Add value labels
    for bar, mean, std in zip(bars, test_accs_mean, test_accs_std):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', fontsize=7)

    # Validation accuracy curves (mean with shaded std)
    ax = axes[0, 1]
    for i, (name, result) in enumerate(all_runs_results.items()):
        val_accs = np.array(result['val_accs'])  # Shape: (num_seeds, max_epochs)
        max_epochs = val_accs.shape[1]
        mean_acc = np.mean(val_accs, axis=0)
        std_acc = np.std(val_accs, axis=0)
        epochs = range(max_epochs)
        ax.plot(epochs, mean_acc, color=colors[i], label=name, linewidth=2)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training (Mean ± Std)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Validation loss curves
    ax = axes[1, 0]
    for i, (name, result) in enumerate(all_runs_results.items()):
        val_losses = np.array(result['val_losses'])
        max_epochs = val_losses.shape[1]
        mean_loss = np.mean(val_losses, axis=0)
        std_loss = np.std(val_losses, axis=0)
        epochs = range(max_epochs)
        ax.plot(epochs, mean_loss, color=colors[i], label=name, linewidth=2)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Training (Mean ± Std)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Accuracy vs Parameters
    ax = axes[1, 1]
    params = [all_runs_results[name]['params'] for name in model_names]
    ax.scatter(params, test_accs_mean, s=150, c=colors)
    for i, name in enumerate(model_names):
        ax.annotate(name, (params[i], test_accs_mean[i]), fontsize=6,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Parameter Efficiency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if project_root:
        outputs_dir = os.path.join(project_root, 'outputs', 'plots')
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(outputs_dir, f'fashion_mnist_react_efta_comparison_{timestamp}.png'), dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {os.path.join(outputs_dir, f'fashion_mnist_react_efta_comparison_{timestamp}.png')}")
    else:
        plt.savefig('fashion_mnist_react_efta_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_comparison_table(all_runs_results):
    """Print formatted comparison table with statistics."""
    print("\n" + "="*100)
    print("FASHION-MNIST CNN ACTIVATION COMPARISON RESULTS (10 SEEDS)".center(100))
    print("="*100)
    print(f"{'Model':<30} | {'Test Acc':<18} | {'Best Val':<12} | {'Params':<12} | {'Epochs':<8}")
    print(f"{'':<30} | {'(mean ± std)':<18} | {'Acc':<12} | {'':<12} | {'':<8}")
    print("-"*100)
    
    model_names = list(all_runs_results.keys())
    for name in model_names:
        result = all_runs_results[name]
        test_accs = result['test_accs']
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        best_val = np.mean(result['best_val_accs'])
        params = result['params']
        epochs = np.mean(result['epochs'])
        
        print(f"{name:<30} | {mean_acc:>10.2f}±{std_acc:<5.2f}% | {best_val:>10.2f}% | "
              f"{params:>10,} | {epochs:>6.1f}")
    print("="*100)

    best_idx = np.argmax([np.mean(all_runs_results[name]['test_accs']) for name in model_names])
    best_model = model_names[best_idx]
    best_acc = np.mean(all_runs_results[best_model]['test_accs'])
    best_std = np.std(all_runs_results[best_model]['test_accs'])
    print(f"\n🏆 Best Model: {best_model} with {best_acc:.2f}±{best_std:.2f}% test accuracy")

    # Compare ReAct-EFTA with other variants
    react_models = [k for k in model_names if 'ReAct' in k]
    efta_models = [k for k in model_names if 'EFTA' in k and 'ReAct' not in k]
    
    if react_models and efta_models:
        print("\n📊 ReAct-EFTA vs EFTA Comparison:")
        for react_name in react_models:
            base_name = react_name.replace('ReAct-EFTA', 'EFTA').replace('ReAct', '')
            if base_name in efta_models:
                diff = np.mean(all_runs_results[react_name]['test_accs']) - np.mean(all_runs_results[base_name]['test_accs'])
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {react_name} vs {base_name}: {symbol} {abs(diff):.2f}%")
    
    print("="*100)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CNN with REAct-EFTA on Fashion-MNIST')
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds to run (default: 10)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs (default: 40)')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed value (default: 42)')
    args = parser.parse_args()

    print("="*70)
    print("CNN with REAct-EFTA on Fashion-MNIST (PyTorch)")
    print("="*70)
    print(f"Number of seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Starting seed: {args.start_seed}")
    
    # Generate seeds
    seeds = [args.start_seed + i for i in range(args.seeds)]
    print(f"Seeds to use: {seeds}")

    # Load data (once, before all runs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fashion_mnist_dir = os.path.join(project_root, 'datasets', 'FashionMNIST')

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fashion_mnist_dir)
    print(f"\nDataset: Fashion-MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Keep data as uint8 (0-255)
    x_train = x_train
    x_test = x_test

    # Split validation from training
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]

    # Define transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
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
        'CNN + EFTA (d=2,k=2)': lambda: create_cnn_efta(depth=2, branch_factor=2),
        'CNN + ReAct (d=1,k=1)': lambda: create_cnn_react_efta(depth=1, branch_factor=1),
        'CNN + ReAct-EFTA (d=2,k=2)': lambda: create_cnn_react_efta(depth=2, branch_factor=2),
        'CNN + ReAct-EFTA (d=3,k=2)': lambda: create_cnn_react_efta(depth=3, branch_factor=2),
        'CNN + ReAct-EFTA (d=2,k=3)': lambda: create_cnn_react_efta(depth=2, branch_factor=3),
        'CNN + ReAct-EFTA (d=2,k=7)': lambda: create_cnn_react_efta(depth=2, branch_factor=7),
    }

    # Storage for all runs
    all_runs_results = {name: {
        'test_accs': [],
        'test_losses': [],
        'best_val_accs': [],
        'val_accs': [],  # List of val_acc curves for each seed
        'val_losses': [],  # List of val_loss curves for each seed
        'params': 0,
        'epochs': []
    } for name in models_config.keys()}

    # Run all seeds
    total_runs = len(models_config) * args.seeds
    run_count = 0
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n\n{'='*70}")
        print(f"SEED {seed_idx + 1}/{args.seeds} (seed={seed})")
        print(f"{'='*70}\n")
        
        for model_idx, (name, model_fn) in enumerate(models_config.items()):
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Training: {name}")
            
            try:
                model = model_fn()
                result = train_model(
                    model, train_loader, val_loader, test_loader, device,
                    epochs=args.epochs, model_name=name, use_augmentation=True, seed=seed)
                
                # Store results
                all_runs_results[name]['test_accs'].append(result['test_acc'])
                all_runs_results[name]['test_losses'].append(result['test_loss'])
                all_runs_results[name]['best_val_accs'].append(result['history']['best_val_acc'])
                all_runs_results[name]['params'] = result['params']
                all_runs_results[name]['epochs'].append(result['epochs_trained'])
                
                # Pad val_acc and val_loss curves to same length
                val_acc_curve = result['history']['val_acc']
                val_loss_curve = result['history']['val_loss']
                all_runs_results[name]['val_accs'].append(val_acc_curve)
                all_runs_results[name]['val_losses'].append(val_loss_curve)
                
            except Exception as e:
                print(f"\n[ERROR] Training {name} failed: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate results for printing
    print("\n\n" + "="*70)
    print("ALL TRAINING COMPLETED - AGGREGATING RESULTS")
    print("="*70)
    
    print_comparison_table(all_runs_results)
    plot_results({}, all_runs_results, project_root)

    # Save all results
    outputs_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as numpy file
    np.save(os.path.join(outputs_dir, f'fashion_mnist_react_efta_results_{timestamp}.npy'), 
            all_runs_results, allow_pickle=True)
    
    # Also save as JSON for easy viewing
    json_results = {}
    for name, data in all_runs_results.items():
        json_results[name] = {
            'test_accs': [float(x) for x in data['test_accs']],
            'test_losses': [float(x) for x in data['test_losses']],
            'best_val_accs': [float(x) for x in data['best_val_accs']],
            'params': data['params'],
            'epochs': [int(x) for x in data['epochs']],
            'stats': {
                'test_acc_mean': float(np.mean(data['test_accs'])),
                'test_acc_std': float(np.std(data['test_accs'])),
                'test_acc_min': float(np.min(data['test_accs'])),
                'test_acc_max': float(np.max(data['test_accs'])),
            }
        }
    
    with open(os.path.join(outputs_dir, f'fashion_mnist_react_efta_results_{timestamp}.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {os.path.join(outputs_dir, f'fashion_mnist_react_efta_results_{timestamp}.npy')}")
    print(f"  - {os.path.join(outputs_dir, f'fashion_mnist_react_efta_results_{timestamp}.json')}")

    return all_runs_results


if __name__ == '__main__':
    main()
