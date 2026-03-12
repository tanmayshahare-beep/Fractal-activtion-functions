"""
Comprehensive REAct-EFTA Configuration Study

Tests different tree configurations and initialization strategies:
- Tree configs: (d=2,k=7), (d=3,k=3), (d=3,k=4), (d=4,k=2), (d=1,k=1)
- Init strategies: tanh, asymmetric, sharp, smooth, diverse
- Regularization: weight decay analysis

Usage:
    python experiments/react_efta_config_study.py --dataset mnist --seeds 5
    python experiments/react_efta_config_study.py --dataset fashion --seeds 5
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

from src.activations import EnhancedReActEFTA, ReActEFTA
from src.utils import load_mnist_local, load_fashion_mnist_local, NumpyDataset

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CNNReActEFTA(nn.Module):
    def __init__(self, depth=2, branch_factor=2, input_channels=1, num_classes=10,
                 init_strategy='tanh', weight_decay=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.react_efta1 = EnhancedReActEFTA(
            num_units=32, depth=depth, branch_factor=branch_factor, input_dim=32,
            init_strategy=init_strategy, weight_decay=weight_decay)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.react_efta2 = EnhancedReActEFTA(
            num_units=64, depth=depth, branch_factor=branch_factor, input_dim=64,
            init_strategy=init_strategy, weight_decay=weight_decay)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.react_efta3 = EnhancedReActEFTA(
            num_units=128, depth=depth, branch_factor=branch_factor, input_dim=128,
            init_strategy=init_strategy, weight_decay=weight_decay)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.react_efta1(x)
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        x = self.conv2(x)
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
        
        # Add regularization loss
        for module in model.modules():
            if isinstance(module, EnhancedReActEFTA):
                loss = module.regularize_loss(loss)
        
        loss.backward()
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
                epochs=40, model_name="Model", seed=42, weight_decay=0.0):
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
        'test_acc_per_epoch': [],
        'param_summary': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Get parameter summary every 5 epochs
        param_summary = None
        if epoch % 5 == 0 or epoch == epochs - 1:
            for module in model.modules():
                if isinstance(module, EnhancedReActEFTA):
                    param_summary = module.get_param_summary()
                    break

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc_per_epoch'].append(test_acc)
        history['param_summary'].append(param_summary)

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

    # Final parameter summary
    final_param_summary = None
    for module in model.modules():
        if isinstance(module, EnhancedReActEFTA):
            final_param_summary = module.get_param_summary()
            break
    history['final_param_summary'] = final_param_summary

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
    if dataset_name == 'mnist':
        data_dir = os.path.join(project_root, 'datasets', 'MNIST')
        (x_train, y_train), (x_test, y_test) = load_mnist_local(data_dir)
        mean, std = 0.1307, 0.3081
    else:
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


def print_results(all_results):
    print("\n" + "="*90)
    print("CONFIGURATION STUDY RESULTS".center(90))
    print("="*90)
    print(f"{'Model':<35} | {'Test Acc':<18} | {'Params':<12} | {'Epochs':<8}")
    print(f"{'':<35} | {'(mean ± std)':<18} | {'':<12} | {'':<8}")
    print("-"*90)
    
    for name, data in all_results.items():
        test_accs = data['test_accs']
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        params = data['params']
        epochs = np.mean(data['epochs'])
        
        print(f"{name:<35} | {mean_acc:>10.2f}±{std_acc:<5.2f}% | {params:>10,} | {epochs:>6.1f}")
    
    print("="*90)
    best = max(all_results.keys(), key=lambda k: np.mean(all_results[k]['test_accs']))
    print(f"🏆 Best: {best} with {np.mean(all_results[best]['test_accs']):.2f}±{np.std(all_results[best]['test_accs']):.2f}%")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description='REAct-EFTA Configuration Study')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--start-seed', type=int, default=42)
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]
    
    # Tree configurations to test
    tree_configs = [
        {'depth': 1, 'branch_factor': 1, 'name': 'ReAct (d=1,k=1)'},  # Plain REAct
        {'depth': 2, 'branch_factor': 7, 'name': 'ReAct-EFTA (d=2,k=7)'},  # 49 leaves
        {'depth': 3, 'branch_factor': 3, 'name': 'ReAct-EFTA (d=3,k=3)'},  # 27 leaves
        {'depth': 3, 'branch_factor': 4, 'name': 'ReAct-EFTA (d=3,k=4)'},  # 64 leaves
        {'depth': 4, 'branch_factor': 2, 'name': 'ReAct-EFTA (d=4,k=2)'},  # 16 leaves
    ]
    
    # Initialization strategies
    init_strategies = ['tanh', 'asymmetric', 'diverse']
    
    # Weight decay options
    weight_decays = [0.0, 1e-5]

    # Build all configurations
    all_configs = []
    for tree in tree_configs:
        for init_strat in init_strategies:
            for wd in weight_decays:
                if tree['branch_factor'] == 1 and init_strat != 'tanh':
                    continue  # Only use tanh init for plain REAct
                if wd > 0 and tree['depth'] == 1:
                    continue  # No need to test WD on plain REAct twice
                
                config_name = f"{tree['name']} / {init_strat}"
                if wd > 0:
                    config_name += f" / WD={wd}"
                
                all_configs.append({
                    'name': config_name,
                    'depth': tree['depth'],
                    'branch_factor': tree['branch_factor'],
                    'init_strategy': init_strat,
                    'weight_decay': wd
                })

    print(f"\n{'='*70}")
    print(f"REAct-EFTA CONFIGURATION STUDY")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds} (starting from {args.start_seed})")
    print(f"Epochs: {args.epochs}")
    print(f"Total configurations: {len(all_configs)}")
    print(f"Total runs: {len(all_configs) * args.seeds}")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    all_results = {cfg['name']: {
        'test_accs': [],
        'test_losses': [],
        'best_val_accs': [],
        'val_accs': [],
        'val_losses': [],
        'params': 0,
        'epochs': [],
        'config': cfg
    } for cfg in all_configs}

    total_runs = len(all_configs) * args.seeds
    run_count = 0
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx + 1}/{args.seeds} (seed={seed})")
        print(f"{'='*70}")
        
        for cfg in all_configs:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {cfg['name']}")
            
            model = CNNReActEFTA(
                depth=cfg['depth'],
                branch_factor=cfg['branch_factor'],
                input_channels=1,
                num_classes=10,
                init_strategy=cfg['init_strategy'],
                weight_decay=cfg['weight_decay']
            )
            
            result = train_model(
                model, train_loader, val_loader, test_loader, device,
                epochs=args.epochs, model_name=cfg['name'], seed=seed,
                weight_decay=cfg['weight_decay']
            )
            
            all_results[cfg['name']]['test_accs'].append(result['test_acc'])
            all_results[cfg['name']]['test_losses'].append(result['test_loss'])
            all_results[cfg['name']]['best_val_accs'].append(result['history']['best_val_acc'])
            all_results[cfg['name']]['params'] = result['params']
            all_results[cfg['name']]['epochs'].append(result['epochs_trained'])
            all_results[cfg['name']]['val_accs'].append(result['history']['val_acc'])
            all_results[cfg['name']]['val_losses'].append(result['history']['val_loss'])

    print_results(all_results)

    # Save results
    outputs_dir = os.path.join(project_root, 'outputs', 'results')
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as JSON
    json_results = {}
    for name, data in all_results.items():
        json_results[name] = {
            'config': data['config'],
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
    
    with open(os.path.join(outputs_dir, f'react_efta_config_study_{args.dataset}_{timestamp}.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    np.save(os.path.join(outputs_dir, f'react_efta_config_study_{args.dataset}_{timestamp}.npy'), 
            all_results, allow_pickle=True)
    
    print(f"\nResults saved to:")
    print(f"  - {os.path.join(outputs_dir, f'react_efta_config_study_{args.dataset}_{timestamp}.json')}")
    print(f"  - {os.path.join(outputs_dir, f'react_efta_config_study_{args.dataset}_{timestamp}.npy')}")


if __name__ == '__main__':
    main()
