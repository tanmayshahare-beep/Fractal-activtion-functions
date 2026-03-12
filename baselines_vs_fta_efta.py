"""
Baselines vs FTA vs EFTA Comprehensive Comparison

Compares standard activation baselines against FTA and EFTA variants
on both MNIST and Fashion-MNIST datasets using a consistent CNN architecture.

Configurations tested:
- Baselines: ReLU, LeakyReLU, Maxout (k=4)
- FTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)
- EFTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)

Datasets: MNIST, Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
from datetime import datetime
import json

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")


# =============================================================================
# Activation Function Implementations
# =============================================================================

class FractalTreeActivation(nn.Module):
    """
    Fractal Tree Activation (FTA) with linear branches.
    
    Each leaf computes: W·x + b
    Leaves are combined via hierarchical max operation.
    """
    def __init__(self, num_units, depth=2, branch_factor=2):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        
        # Leaf weights and biases
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_units, num_units))
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))
        
        # Initialize weights
        nn.init.kaiming_normal_(self.leaf_weights, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # x shape: (batch, channels, height, width) for conv output
        # or (batch, features) for dense
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            # Convolutional: (B, C, H, W) -> reshape for leaf computation
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)  # (B*H*W, C)
            
            # Compute all leaves
            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x_flat, self.leaf_weights[i], self.leaf_biases[i])
                leaf_outputs.append(leaf_val)
            
            # Stack: (num_leaves, B*H*W, num_units)
            stacked = torch.stack(leaf_outputs, dim=0)
            
            # Hierarchical max pooling through tree
            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]  # Max over branches
            
            # Reshape back: (B, H, W, C) -> (B, C, H, W)
            output = current.view(b, h, w, self.num_units).permute(0, 3, 1, 2)
            return output
        else:
            # Dense layer
            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)  # (num_leaves, B, num_units)
            
            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.squeeze(0)


class ExponentialFTA(nn.Module):
    """
    Exponential Fractal Tree Activation (EFTA).
    
    Each leaf computes:
        f(x) = α * (exp(β * x) - 1)  if x < 0
        f(x) = γ * x                  if x >= 0
    
    Leaves are combined via hierarchical max operation.
    """
    def __init__(self, num_units, depth=2, branch_factor=2):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        
        # Leaf parameters: (alpha, beta, gamma) per unit per leaf
        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))
        
        # Initialize: alpha=1, beta=1, gamma=1 (identity-like)
        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)  # alpha
        nn.init.constant_(self.leaf_params[:, :, 1], 1.0)  # beta
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)  # gamma
    
    def _leaf_function(self, x, alpha, beta, gamma):
        """Apply exponential branch function."""
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        
        # Negative: α * (exp(β * x) - 1), clip for stability
        neg_part = alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0)
        
        # Positive: γ * x
        pos_part = gamma * x
        
        return neg_mask * neg_part + pos_mask * pos_part
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
            
            leaf_outputs = []
            for i in range(self.num_leaves):
                alpha = self.leaf_params[i, :, 0]
                beta = self.leaf_params[i, :, 1]
                gamma = self.leaf_params[i, :, 2]
                leaf_val = self._leaf_function(x_flat, alpha, beta, gamma)
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)
            
            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]
            
            output = current.view(b, h, w, self.num_units).permute(0, 3, 1, 2)
            return output
        else:
            leaf_outputs = []
            for i in range(self.num_leaves):
                alpha = self.leaf_params[i, :, 0]
                beta = self.leaf_params[i, :, 1]
                gamma = self.leaf_params[i, :, 2]
                leaf_val = self._leaf_function(x, alpha, beta, gamma)
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)
            
            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.squeeze(0)


class MaxoutActivation(nn.Module):
    """
    Maxout activation: max over k linear transformations.
    """
    def __init__(self, num_units, k=4):
        super().__init__()
        self.num_units = num_units
        self.k = k
        
        self.weights = nn.Parameter(torch.randn(k, num_units, num_units))
        self.biases = nn.Parameter(torch.zeros(k, num_units))
        
        nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
            
            outputs = []
            for i in range(self.k):
                out = F.linear(x_flat, self.weights[i], self.biases[i])
                outputs.append(out)
            
            stacked = torch.stack(outputs, dim=0)
            output = stacked.max(dim=0)[0]
            return output.view(b, h, w, self.num_units).permute(0, 3, 1, 2)
        else:
            outputs = []
            for i in range(self.k):
                out = F.linear(x, self.weights[i], self.biases[i])
                outputs.append(out)
            
            stacked = torch.stack(outputs, dim=0)
            return stacked.max(dim=0)[0]


# =============================================================================
# CNN Architecture (Consistent across all models)
# =============================================================================

class CNNBaseline(nn.Module):
    """CNN with standard activation (ReLU, LeakyReLU)."""
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation_name = activation
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        return x


class CNNMaxout(nn.Module):
    """CNN with Maxout activation."""
    def __init__(self, k=4):
        super().__init__()
        self.k = k
        self.maxout1 = MaxoutActivation(32, k)
        self.maxout2 = MaxoutActivation(64, k)
        self.maxout3 = MaxoutActivation(128, k)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxout1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxout2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.maxout3(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        return x


class CNNFTA(nn.Module):
    """CNN with Fractal Tree Activation."""
    def __init__(self, depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        self.fta1 = FractalTreeActivation(32, depth, branch_factor)
        self.fta2 = FractalTreeActivation(64, depth, branch_factor)
        self.fta3 = FractalTreeActivation(128, depth, branch_factor)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fta1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fta2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fta3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        return x


class CNNEFTA(nn.Module):
    """CNN with Exponential Fractal Tree Activation."""
    def __init__(self, depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        self.efta1 = ExponentialFTA(32, depth, branch_factor)
        self.efta2 = ExponentialFTA(64, depth, branch_factor)
        self.efta3 = ExponentialFTA(128, depth, branch_factor)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        return x


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_name(model_type, depth=None, branch_factor=None, k=None):
    """Generate descriptive model name."""
    if model_type == 'relu':
        return 'ReLU'
    elif model_type == 'leaky_relu':
        return 'LeakyReLU'
    elif model_type == 'maxout':
        return f'Maxout (k={k})'
    elif model_type == 'fta':
        return f'FTA (d={depth}, k={branch_factor})'
    elif model_type == 'efta':
        return f'EFTA (d={depth}, k={branch_factor})'
    else:
        return model_type


def create_model(model_type, depth=None, branch_factor=None, k=None):
    """Create model based on type."""
    if model_type == 'relu':
        return CNNBaseline('relu')
    elif model_type == 'leaky_relu':
        return CNNBaseline('leaky_relu')
    elif model_type == 'maxout':
        return CNNMaxout(k=k)
    elif model_type == 'fta':
        return CNNFTA(depth=depth, branch_factor=branch_factor)
    elif model_type == 'efta':
        return CNNEFTA(depth=depth, branch_factor=branch_factor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, test_loader, device, 
                epochs=30, model_name="Model", verbose=True):
    """Train and evaluate model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 5
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    total_time = time.time() - start_time
    params = count_parameters(model)
    
    return {
        'name': model_name,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_val_acc': best_val_acc,
        'params': params,
        'epochs_trained': len(history['train_loss']),
        'training_time': total_time,
        'history': history
    }


# =============================================================================
# Data Loading
# =============================================================================

def get_data_loaders(dataset_name, batch_size=128, val_split=0.1):
    """Get train/val/test data loaders."""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./datasets/MNIST', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./datasets/MNIST', train=False, download=True, transform=transform
        )
        
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root='./datasets/FashionMNIST', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./datasets/FashionMNIST', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split training into train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(dataset_name, epochs=30, verbose=True):
    """Run comprehensive comparison experiment."""
    print(f"\n{'='*70}")
    print(f"Running experiment on: {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(dataset_name)
    
    # Model configurations
    configurations = [
        # Baselines
        ('relu', None, None, None),
        ('leaky_relu', None, None, None),
        ('maxout', None, None, 4),
        
        # FTA variants
        ('fta', 2, 2, None),
        ('fta', 3, 2, None),
        ('fta', 2, 3, None),
        ('fta', 3, 3, None),
        
        # EFTA variants
        ('efta', 2, 2, None),
        ('efta', 3, 2, None),
        ('efta', 2, 3, None),
        ('efta', 3, 3, None),
    ]
    
    results = {}
    
    for model_type, depth, branch_factor, k in configurations:
        model_name = get_model_name(model_type, depth, branch_factor, k)
        print(f"\nTraining: {model_name}")
        print(f"  Type: {model_type.upper()}, Depth: {depth}, Branch: {branch_factor}, k: {k}")
        
        try:
            model = create_model(model_type, depth, branch_factor, k)
            model = model.to(device)
            
            result = train_model(
                model, train_loader, val_loader, test_loader,
                device, epochs=epochs, model_name=model_name, verbose=verbose
            )
            results[model_name] = result
            
            print(f"  ✓ Completed: Test Acc={result['test_acc']:.2f}%, "
                  f"Params={result['params']:,}, Time={result['training_time']:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_results_table(results, dataset_name):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*90}")
    print(f"{'Model':<25} | {'Test Acc':<10} | {'Val Acc':<10} | {'Params':<12} | {'Time (s)':<10}")
    print("-"*90)
    
    for name, result in results.items():
        print(f"{name:<25} | {result['test_acc']:>8.2f}% | {result['best_val_acc']:>8.2f}% | "
              f"{result['params']:>10,} | {result['training_time']:>8.1f}")
    
    print("="*90)
    
    # Find best models
    best_overall = max(results, key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 Best Model: {best_overall} ({results[best_overall]['test_acc']:.2f}%)")
    
    # Best by category
    baselines = {k: v for k, v in results.items() if k in ['ReLU', 'LeakyReLU', 'Maxout (k=4)']}
    fta_models = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
    efta_models = {k: v for k, v in results.items() if 'EFTA' in k}
    
    if baselines:
        best_baseline = max(baselines, key=lambda x: baselines[x]['test_acc'])
        print(f"🥇 Best Baseline: {best_baseline} ({baselines[best_baseline]['test_acc']:.2f}%)")
    
    if fta_models:
        best_fta = max(fta_models, key=lambda x: fta_models[x]['test_acc'])
        print(f"🥇 Best FTA: {best_fta} ({fta_models[best_fta]['test_acc']:.2f}%)")
    
    if efta_models:
        best_efta = max(efta_models, key=lambda x: efta_models[x]['test_acc'])
        print(f"🥇 Best EFTA: {best_efta} ({efta_models[best_efta]['test_acc']:.2f}%)")
    
    print("="*90)


def save_results(all_results, output_dir='./outputs/results'):
    """Save results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'baselines_vs_fta_efta_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Convert results to JSON-serializable format
    save_data = {}
    for dataset, results in all_results.items():
        save_data[dataset] = {}
        for name, result in results.items():
            save_data[dataset][name] = {
                'test_acc': result['test_acc'],
                'test_loss': result['test_loss'],
                'best_val_acc': result['best_val_acc'],
                'params': result['params'],
                'epochs_trained': result['epochs_trained'],
                'training_time': result['training_time']
            }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {filepath}")
    return filepath


def main():
    """Main entry point."""
    print(f"\n{'='*70}")
    print("BASELINES VS FTA VS EFTA COMPREHENSIVE COMPARISON")
    print(f"{'='*70}")
    print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfigurations:")
    print(f"  - Baselines: ReLU, LeakyReLU, Maxout (k=4)")
    print(f"  - FTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"  - EFTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"\nDatasets: MNIST, Fashion-MNIST")
    print(f"Architecture: Consistent CNN across all models")
    print(f"{'='*70}")
    
    all_results = {}
    
    # Run on MNIST
    mnist_results = run_experiment('mnist', epochs=30, verbose=True)
    all_results['mnist'] = mnist_results
    print_results_table(mnist_results, 'mnist')
    
    # Run on Fashion-MNIST
    fashion_results = run_experiment('fashion_mnist', epochs=30, verbose=True)
    all_results['fashion_mnist'] = fashion_results
    print_results_table(fashion_results, 'fashion_mnist')
    
    # Save results
    save_results(all_results)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*70}")
    print(f"\nFinal Summary:")
    print(f"  MNIST Best:        {max(all_results['mnist'], key=lambda x: all_results['mnist'][x]['test_acc'])}")
    print(f"  Fashion-MNIST Best: {max(all_results['fashion_mnist'], key=lambda x: all_results['fashion_mnist'][x]['test_acc'])}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
