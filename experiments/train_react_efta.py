"""
Train REAct-EFTA on MNIST and Fashion-MNIST

Trains only ReAct-EFTA models:
- ReAct (d=1, k=1) - plain REAct activation
- ReAct-EFTA (d=2, k=7) - high capacity tree

Usage:
    python experiments/train_react_efta.py --dataset mnist --seeds 10
    python experiments/train_react_efta.py --dataset fashion --seeds 10
    python experiments/train_react_efta.py --dataset both --seeds 10
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

from src.activations import ReActEFTA
from src.utils import load_mnist_local, load_fashion_mnist_local, NumpyDataset

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CNNReActEFTA(nn.Module):
    """
    CNN with REAct-EFTA activation.
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

        x = self.conv1(x)  # (B, 32, H, W)
        x = self.bn1(x)
        # Convert to channel-last for ReActEFTA: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.react_efta1(x)
        # Convert back to channel-first for pooling: (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        x = self.conv2(x)  # (B, 64, H, W)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.react_efta2(x)
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.react_efta3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
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
                epochs=40, model_name="Model", seed=42):
    set_seed(seed)
    
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

    print(f"\n{'='*60}")
    print(f"Training: {model_name} (Seed: {seed})")
    print(f"{'='*60}")
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc_per_epoch'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train={train_acc:.2f}%, Val={val_acc:.2f}%, Test={test_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    history['final_test_acc'] = final_test_acc
    history['best_val_acc'] = best_val_acc

    return {
        'name': model_name,
        'seed': seed,
        'history': history,
        'test_acc': final_test_acc,
        'test_loss': test_loss,
        'params': count_parameters(model),
        'epochs_trained': len(history['train_loss'])
    }


def get_dataloaders(dataset_name, batch_size=64, val_split=0.1):
    """Get dataloaders for MNIST or Fashion-MNIST."""
    if dataset_name == 'mnist':
        data_dir = os.path.join(project_root, 'datasets', 'MNIST')
        (x_train, y_train), (x_test, y_test) = load_mnist_local(data_dir)
        mean, std = 0.1307, 0.3081
    else:  # fashion
        data_dir = os.path.join(project_root, 'datasets', 'FashionMNIST')
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(data_dir)
        mean, std = 0.2860, 0.3530

    val_split_idx = int((1 - val_split) * len(x_train))
    x_val, y_val = x_train[val_split_idx:], y_train[val_split_idx:]
    x_train_sub, y_train_sub = x_train[:val_split_idx], y_train[:val_split_idx]

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_dataset = NumpyDataset(x_train_sub, y_train_sub, transform=train_transform)
    val_dataset = NumpyDataset(x_val, y_val, transform=test_transform)
    test_dataset = NumpyDataset(x_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_results(results, dataset_name, project_root):
    """Plot results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e']

    # Test accuracy bar chart
    ax = axes[0]
    test_accs = [np.mean(results[name]['test_accs']) for name in model_names]
    test_stds = [np.std(results[name]['test_accs']) for name in model_names]
    
    bars = ax.bar(model_names, test_accs, yerr=test_stds, capsize=5, color=colors[:len(model_names)])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{dataset_name} - Test Accuracy (Mean ± Std)')
    ax.set_xticklabels(model_names, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, test_accs, test_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                f'{mean:.2f}±{std:.2f}%', ha='center', va='bottom', fontsize=9)

    # Training curves
    ax = axes[1]
    for i, (name, result) in enumerate(results.items()):
        val_accs = np.array(result['val_accs'])
        max_epochs = val_accs.shape[1]
        mean_acc = np.mean(val_accs, axis=0)
        std_acc = np.std(val_accs, axis=0)
        ax.plot(range(max_epochs), mean_acc, color=colors[i], label=name, linewidth=2)
        ax.fill_between(range(max_epochs), mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    outputs_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(outputs_dir, f'react_efta_{dataset_name}_{timestamp}.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(outputs_dir, f'react_efta_{dataset_name}_{timestamp}.png')}")
    plt.show()


def print_results(results, dataset_name):
    """Print results table."""
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} RESULTS".center(70))
    print("="*70)
    print(f"{'Model':<25} | {'Test Acc':<18} | {'Params':<12} | {'Epochs':<8}")
    print(f"{'':<25} | {'(mean ± std)':<18} | {'':<12} | {'':<8}")
    print("-"*70)
    
    for name, data in results.items():
        test_accs = data['test_accs']
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        params = data['params']
        epochs = np.mean(data['epochs'])
        
        print(f"{name:<25} | {mean_acc:>10.2f}±{std_acc:<5.2f}% | {params:>10,} | {epochs:>6.1f}")
    
    print("="*70)
    best = max(results.keys(), key=lambda k: np.mean(results[k]['test_accs']))
    print(f"🏆 Best: {best} with {np.mean(results[best]['test_accs']):.2f}±{np.std(results[best]['test_accs']):.2f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train REAct-EFTA models')
    parser.add_argument('--dataset', type=str, default='both', choices=['mnist', 'fashion', 'both'],
                        help='Dataset to train on')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed')
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]
    
    models_config = {
        'ReAct (d=1,k=1)': {'depth': 1, 'branch_factor': 1},
        'ReAct-EFTA (d=2,k=7)': {'depth': 2, 'branch_factor': 7},
    }

    datasets = ['mnist', 'fashion'] if args.dataset == 'both' else [args.dataset]

    for dataset_name in datasets:
        print(f"\n\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        train_loader, val_loader, test_loader = get_dataloaders(dataset_name)

        all_results = {name: {
            'test_accs': [],
            'test_losses': [],
            'best_val_accs': [],
            'val_accs': [],
            'val_losses': [],
            'params': 0,
            'epochs': []
        } for name in models_config.keys()}

        total_runs = len(models_config) * args.seeds
        run_count = 0
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"SEED {seed_idx + 1}/{args.seeds} (seed={seed})")
            print(f"{'='*60}")
            
            for name, config in models_config.items():
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {name}")
                
                model = CNNReActEFTA(
                    depth=config['depth'],
                    branch_factor=config['branch_factor'],
                    input_channels=1,
                    num_classes=10
                )
                
                result = train_model(
                    model, train_loader, val_loader, test_loader, device,
                    epochs=args.epochs, model_name=name, seed=seed
                )
                
                all_results[name]['test_accs'].append(result['test_acc'])
                all_results[name]['test_losses'].append(result['test_loss'])
                all_results[name]['best_val_accs'].append(result['history']['best_val_acc'])
                all_results[name]['params'] = result['params']
                all_results[name]['epochs'].append(result['epochs_trained'])
                all_results[name]['val_accs'].append(result['history']['val_acc'])
                all_results[name]['val_losses'].append(result['history']['val_loss'])

        print_results(all_results, dataset_name)
        plot_results(all_results, dataset_name, project_root)

        # Save results
        outputs_dir = os.path.join(project_root, 'outputs', 'results')
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_results = {}
        for name, data in all_results.items():
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
        
        np.save(os.path.join(outputs_dir, f'react_efta_{dataset_name}_results_{timestamp}.npy'), 
                all_results, allow_pickle=True)
        
        with open(os.path.join(outputs_dir, f'react_efta_{dataset_name}_results_{timestamp}.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - {os.path.join(outputs_dir, f'react_efta_{dataset_name}_results_{timestamp}.npy')}")
        print(f"  - {os.path.join(outputs_dir, f'react_efta_{dataset_name}_results_{timestamp}.json')}")


if __name__ == '__main__':
    main()
