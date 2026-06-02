"""
Modern Baselines vs FTA vs EFTA on MNIST with Branch Visualization

This script compares state-of-the-art activation functions against FTA and EFTA
on the MNIST dataset, with comprehensive visualization of learned branches.

Modern Baselines:
- ReLU (baseline)
- LeakyReLU (baseline)
- GELU (Google ViT, GPT models)
- Swish/SiLU (Google Brain, EfficientNet)
- Mish (ICASSP 2020, smooth non-monotonic)
- EELU (Exponential ELU, recent exponential activation)

FTA/EFTA Variants:
- FTA (d=2,k=2), FTA (d=3,k=2), FTA (d=2,k=3), FTA (d=3,k=3)
- EFTA (d=2,k=2), EFTA (d=3,k=2), EFTA (d=2,k=3), EFTA (d=3,k=3)

Includes:
- Training curves comparison
- Accuracy vs parameter count scatter plot
- Branch parameter visualization for FTA/EFTA
- Leaf activation heatmaps
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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")


# =============================================================================
# Modern Activation Function Implementations
# =============================================================================

class GELUActivation(nn.Module):
    """
    Gaussian Error Linear Unit (GELU).
    
    f(x) = x * Φ(x) where Φ is the standard Gaussian CDF
    
    Widely used in transformers (BERT, GPT, ViT).
    Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)


class SwishActivation(nn.Module):
    """
    Swish/SiLU (Sigmoid Linear Unit).
    
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Used in EfficientNet, MobileNetV3.
    Smooth, non-monotonic, bounded below.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.silu(x)  # PyTorch's SiLU is equivalent to Swish


class MishActivation(nn.Module):
    """
    Mish (ICASSP 2020).
    
    f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    
    Smooth, non-monotonic, unbounded above, bounded below.
    Often outperforms ReLU on image classification tasks.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EELUActivation(nn.Module):
    """
    EELU (Exponential ELU) - Recent exponential activation.
    
    f(x) = α * (exp(x) - 1)  if x < 0
    f(x) = x                  if x >= 0
    
    Similar to ELU but with learnable scaling parameter.
    Provides smooth negative saturation.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
    
    def forward(self, x):
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        
        neg_part = self.alpha * (torch.exp(torch.clamp(x, -10, 10)) - 1.0)
        pos_part = x
        
        return neg_mask * neg_part + pos_mask * pos_part


# =============================================================================
# FTA/EFTA Implementations (from existing code)
# =============================================================================

class FractalTreeActivation(nn.Module):
    """
    Fractal Tree Activation (FTA) with linear branches.
    Each leaf computes: W·x + b
    Leaves are combined via hierarchical max operation.
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim
        
        if input_dim is None:
            raise ValueError("input_dim must be specified for FractalTreeActivation")
        
        # Leaf weights and biases
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, input_dim, num_units) * (2.0 / input_dim) ** 0.5)
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            # Convolutional: (B, H, W, C) - note: expecting channels last
            b, h, w, c = x.shape
            x_flat = x.reshape(b * h * w, c)
            
            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x_flat, self.leaf_weights[i].T, self.leaf_biases[i])
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)
            current = stacked.reshape(self.num_leaves, b, h, w, self.num_units)
            
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.reshape(num_nodes, self.branch_factor, b, h, w, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.reshape(b, h, w, self.num_units)
        else:
            # Dense layer
            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x, self.leaf_weights[i].T, self.leaf_biases[i])
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)
            current = stacked
            
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.reshape(num_nodes, self.branch_factor, batch_size, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.squeeze(0)
    
    def get_leaf_values(self, x):
        """Get individual leaf outputs for visualization."""
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            b, h, w, c = x.shape
            x_flat = x.reshape(b * h * w, c)
        else:
            x_flat = x
        
        leaf_values = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x_flat, self.leaf_weights[i].T, self.leaf_biases[i])
            leaf_values.append(leaf_val)
        
        return torch.stack(leaf_values, dim=0)


class ExponentialFTA(nn.Module):
    """
    Exponential Fractal Tree Activation (EFTA).
    Each leaf computes:
        f(x) = α * (exp(β * x) - 1)  if x < 0
        f(x) = γ * x                  if x >= 0
    Leaves are combined via hierarchical max operation.
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim
        
        if input_dim is None:
            raise ValueError("input_dim must be specified for ExponentialFTA")
        
        # Leaf parameters: (alpha, beta, gamma) per unit per leaf
        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))
        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)  # alpha
        nn.init.constant_(self.leaf_params[:, :, 1], 0.5)  # beta
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)  # gamma

    def _leaf_function(self, x, alpha, beta, gamma):
        """Apply exponential branch function."""
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        
        neg_part = alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0)
        pos_part = gamma * x
        
        return neg_mask * neg_part + pos_mask * pos_part

    def forward(self, x):
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            b, h, w, c = x.shape
            x_flat = x.reshape(b * h * w, c)
            
            leaf_outputs = []
            for i in range(self.num_leaves):
                alpha = self.leaf_params[i, :, 0]
                beta = self.leaf_params[i, :, 1]
                gamma = self.leaf_params[i, :, 2]
                leaf_val = self._leaf_function(x_flat, alpha, beta, gamma)
                leaf_outputs.append(leaf_val)
            
            stacked = torch.stack(leaf_outputs, dim=0)
            current = stacked.reshape(self.num_leaves, b, h, w, self.num_units)
            
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.reshape(num_nodes, self.branch_factor, b, h, w, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.reshape(b, h, w, self.num_units)
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
                current = current.reshape(num_nodes, self.branch_factor, batch_size, self.num_units)
                current = current.max(dim=1)[0]
            
            return current.squeeze(0)
    
    def get_leaf_values(self, x):
        """Get individual leaf outputs for visualization."""
        batch_size = x.shape[0]
        
        if x.dim() == 4:
            b, h, w, c = x.shape
            x_flat = x.reshape(b * h * w, c)
        else:
            x_flat = x
        
        leaf_values = []
        for i in range(self.num_leaves):
            alpha = self.leaf_params[i, :, 0]
            beta = self.leaf_params[i, :, 1]
            gamma = self.leaf_params[i, :, 2]
            leaf_val = self._leaf_function(x_flat, alpha, beta, gamma)
            leaf_values.append(leaf_val)
        
        return torch.stack(leaf_values, dim=0)


# =============================================================================
# CNN Architecture (Consistent across all models)
# =============================================================================

class CNNBaseline(nn.Module):
    """CNN with standard activation functions."""
    def __init__(self, activation_module):
        super().__init__()
        self.activation = activation_module()
        
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
        self.num_units_conv = 32
        self.num_units_conv2 = 64
        self.num_units_fc = 128
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxout1 = MaxoutLayer(32, k)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxout2 = MaxoutLayer(64, k)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.maxout3 = MaxoutLayer(128, k)
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
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.maxout3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class MaxoutLayer(nn.Module):
    """Maxout layer for CNN."""
    def __init__(self, num_units, k=4):
        super().__init__()
        self.k = k
        self.weights = nn.Parameter(torch.randn(k, num_units, num_units))
        self.biases = nn.Parameter(torch.zeros(k, num_units))
        nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # x: (B, H, W, C) - channels last format expected
        b, h, w, c = x.shape
        x_flat = x.reshape(b * h * w, c)
        
        outputs = []
        for i in range(self.k):
            out = F.linear(x_flat, self.weights[i].T, self.biases[i])
            outputs.append(out)
        
        stacked = torch.stack(outputs, dim=0)
        output = stacked.max(dim=0)[0]
        return output.reshape(b, h, w, self.weights.shape[1])


class CNNFTA(nn.Module):
    """CNN with Fractal Tree Activation."""
    def __init__(self, depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.fta1 = FractalTreeActivation(32, depth, branch_factor, input_dim=32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fta2 = FractalTreeActivation(64, depth, branch_factor, input_dim=64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fta3 = FractalTreeActivation(128, depth, branch_factor, input_dim=128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fta1(x.permute(0, 2, 3, 1))  # Convert to channels-last
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fta2(x.permute(0, 2, 3, 1))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.efta1 = ExponentialFTA(32, depth, branch_factor, input_dim=32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.efta2 = ExponentialFTA(64, depth, branch_factor, input_dim=64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.efta3 = ExponentialFTA(128, depth, branch_factor, input_dim=128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x.permute(0, 2, 3, 1))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x.permute(0, 2, 3, 1))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Utility Functions
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
    elif model_type == 'gelu':
        return 'GELU'
    elif model_type == 'swish':
        return 'Swish'
    elif model_type == 'mish':
        return 'Mish'
    elif model_type == 'eelu':
        return 'EELU'
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
        return CNNBaseline(GELUActivation)  # Use wrapper for consistency
    elif model_type == 'leaky_relu':
        return CNNBaseline(lambda: nn.LeakyReLU(0.01))
    elif model_type == 'gelu':
        return CNNBaseline(GELUActivation)
    elif model_type == 'swish':
        return CNNBaseline(SwishActivation)
    elif model_type == 'mish':
        return CNNBaseline(MishActivation)
    elif model_type == 'eelu':
        return CNNBaseline(EELUActivation)
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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


def get_data_loaders(dataset_name='mnist', batch_size=128, val_split=0.1):
    """Get train/val/test data loaders."""
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
# Visualization Functions
# =============================================================================

def plot_training_curves(results, save_path='./outputs/plots/training_curves_comparison.png'):
    """Plot training curves for all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color map for different model categories
    colors = {
        'ReLU': '#1f77b4',
        'LeakyReLU': '#ff7f0e',
        'GELU': '#2ca02c',
        'Swish': '#d62728',
        'Mish': '#9467bd',
        'EELU': '#8c564b',
        'Maxout': '#e377c2',
    }
    
    fta_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
    efta_colors = plt.cm.Greens(np.linspace(0.4, 0.9, 4))
    
    # Training accuracy curves
    ax = axes[0]
    for name, result in results.items():
        epochs = list(range(1, len(result['history']['train_acc']) + 1))
        
        if 'FTA' in name and 'EFTA' not in name:
            idx = int(name.split('d=')[1].split(',')[0]) - 2
            ax.plot(epochs, result['history']['train_acc'], '--', color=fta_colors[idx], 
                   alpha=0.7, linewidth=1.5, label=name)
        elif 'EFTA' in name:
            idx = int(name.split('d=')[1].split(',')[0]) - 2
            ax.plot(epochs, result['history']['train_acc'], '-.', color=efta_colors[idx],
                   alpha=0.7, linewidth=1.5, label=name)
        else:
            color = colors.get(name, '#333333')
            ax.plot(epochs, result['history']['train_acc'], '-', color=color,
                   linewidth=2, alpha=0.8, label=name)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    # Validation accuracy curves
    ax = axes[1]
    for name, result in results.items():
        epochs = list(range(1, len(result['history']['val_acc']) + 1))
        
        if 'FTA' in name and 'EFTA' not in name:
            idx = int(name.split('d=')[1].split(',')[0]) - 2
            ax.plot(epochs, result['history']['val_acc'], '--', color=fta_colors[idx],
                   alpha=0.7, linewidth=1.5, label=name)
        elif 'EFTA' in name:
            idx = int(name.split('d=')[1].split(',')[0]) - 2
            ax.plot(epochs, result['history']['val_acc'], '-.', color=efta_colors[idx],
                   alpha=0.7, linewidth=1.5, label=name)
        else:
            color = colors.get(name, '#333333')
            ax.plot(epochs, result['history']['val_acc'], '-', color=color,
                   linewidth=2, alpha=0.8, label=name)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_accuracy_vs_params(results, save_path='./outputs/plots/accuracy_vs_params.png'):
    """Plot test accuracy vs parameter count."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate by category
    baselines = {k: v for k, v in results.items() 
                 if k in ['ReLU', 'LeakyReLU', 'GELU', 'Swish', 'Mish', 'EELU', 'Maxout (k=4)']}
    fta_models = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
    efta_models = {k: v for k, v in results.items() if 'EFTA' in k}
    
    # Plot baselines
    for name, result in baselines.items():
        ax.scatter(result['params'], result['test_acc'], s=150, marker='o', 
                  label=name, edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Plot FTA
    for name, result in fta_models.items():
        ax.scatter(result['params'], result['test_acc'], s=200, marker='^',
                  label=name, edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Plot EFTA
    for name, result in efta_models.items():
        ax.scatter(result['params'], result['test_acc'], s=200, marker='s',
                  label=name, edgecolors='black', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Parameter Count', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('MNIST: Accuracy vs Parameter Count', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs params plot to: {save_path}")


def plot_bar_comparison(results, save_path='./outputs/plots/bar_comparison.png'):
    """Plot bar chart comparing all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    names = list(results.keys())
    accuracies = [results[n]['test_acc'] for n in names]
    params = [results[n]['params'] for n in names]
    
    # Color by category
    colors = []
    for name in names:
        if 'EFTA' in name:
            colors.append('#2ecc71')  # Green
        elif 'FTA' in name:
            colors.append('#3498db')  # Blue
        elif name == 'Maxout (k=4)':
            colors.append('#9b59b6')  # Purple
        else:
            colors.append('#95a5a6')  # Gray for baselines
    
    bars = ax.bar(range(len(names)), accuracies, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for i, (bar, acc, param) in enumerate(zip(bars, accuracies, params)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('MNIST Test Accuracy: Modern Baselines vs FTA vs EFTA', fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', edgecolor='black', label='Standard Baselines'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Maxout'),
        Patch(facecolor='#3498db', edgecolor='black', label='FTA (Linear)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='EFTA (Exponential)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar comparison to: {save_path}")


def visualize_fta_branches(model, test_loader, device, save_dir='./outputs/visualizations/fta'):
    """Visualize FTA branch weights and activations."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # Get a batch of test data
    data, target = next(iter(test_loader))
    data = data[:16].to(device)  # 16 samples
    
    # Get FTA layer (fta3 is the final dense layer)
    fta_layer = model.fta3
    
    # Get leaf values for this batch
    with torch.no_grad():
        # Forward through fc1 first to get input to FTA
        x = model.conv1(data)
        x = model.bn1(x)
        x = model.fta1(x.permute(0, 2, 3, 1))
        x = model.pool1(x)
        x = model.dropout1(x)
        x = model.conv2(x)
        x = model.bn2(x)
        x = model.fta2(x.permute(0, 2, 3, 1))
        x = model.pool2(x)
        x = model.dropout2(x)
        x = x.view(x.size(0), -1)
        x = model.fc1(x)
        x = model.bn3(x)
        
        # Now x is input to fta3
        fta_input = x
        leaf_values = fta_layer.get_leaf_values(x)  # (num_leaves, batch, num_units)
    
    num_leaves = fta_layer.num_leaves
    depth = fta_layer.depth
    branch_factor = fta_layer.branch_factor
    
    # Plot 1: Leaf weight norms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Weight matrix norms
    ax = axes[0, 0]
    weight_norms = []
    for i in range(num_leaves):
        norm = fta_layer.leaf_weights[i].data.norm(p=2).item()
        weight_norms.append(norm)
    
    ax.bar(range(num_leaves), weight_norms, color='steelblue', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Weight Norm (L2)')
    ax.set_title(f'FTA Leaf Weight Norms (d={depth}, k={branch_factor})')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Leaf activation statistics
    ax = axes[0, 1]
    leaf_means = leaf_values.mean(dim=[1, 2]).cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    ax.bar(x_pos - 0.2, leaf_means, width=0.4, label='Mean', color='coral', edgecolor='black')
    ax.bar(x_pos + 0.2, leaf_stds, width=0.4, label='Std', color='teal', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Activation Statistics')
    ax.set_title('Leaf Activation Mean & Std (16 samples, 128 units)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Leaf selection frequency (which leaf wins the max)
    ax = axes[1, 0]
    # For each unit, which leaf had the max value?
    winning_leaf = leaf_values.argmax(dim=0)  # (batch, num_units)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    
    ax.bar(range(num_leaves), win_counts.cpu().numpy(), color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Times Selected (as max)')
    ax.set_title('Leaf Selection Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tree structure diagram
    ax = axes[1, 1]
    ax.axis('off')
    
    # Draw simple tree diagram
    def draw_tree(ax, depth, branch_factor, num_leaves):
        levels = depth + 1
        y_positions = np.linspace(0.8, 0.2, levels)
        
        # Root
        ax.plot([0.5], [y_positions[0]], 'o', color='red', markersize=20)
        ax.text(0.5, y_positions[0] + 0.05, 'Root', ha='center', fontsize=10, fontweight='bold')
        
        # Draw levels
        nodes_at_level = 1
        for level in range(1, levels):
            nodes_at_level *= branch_factor
            x_positions = np.linspace(0.1, 0.9, nodes_at_level)
            
            for i, x_pos in enumerate(x_positions):
                if level == depth:
                    # Leaf node
                    ax.plot([x_pos], [y_positions[level]], 's', color='green', markersize=15)
                    ax.text(x_pos, y_positions[level] - 0.08, f'L{i}', ha='center', fontsize=8)
                else:
                    # Internal node
                    ax.plot([x_pos], [y_positions[level]], 'o', color='blue', markersize=15)
                
                # Draw connection to parent
                parent_idx = i // branch_factor
                parent_x = np.linspace(0.1, 0.9, nodes_at_level // branch_factor)[parent_idx]
                parent_y = y_positions[level - 1]
                ax.plot([parent_x, x_pos], [parent_y, y_positions[level]], 'k-', linewidth=1, alpha=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Tree Structure (depth={depth}, branch_factor={branch_factor})')
    
    draw_tree(ax, depth, branch_factor, num_leaves)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/branch_analysis_d{depth}_k{branch_factor}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved FTA visualization to: {save_dir}/branch_analysis_d{depth}_k{branch_factor}.png")


def visualize_efta_branches(model, test_loader, device, save_dir='./outputs/visualizations/efta'):
    """Visualize EFTA branch parameters and activations."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # Get a batch of test data
    data, target = next(iter(test_loader))
    data = data[:16].to(device)
    
    # Get EFTA layer (efta3 is the final dense layer)
    efta_layer = model.efta3
    
    # Get leaf values for this batch
    with torch.no_grad():
        x = model.conv1(data)
        x = model.bn1(x)
        x = model.efta1(x.permute(0, 2, 3, 1))
        x = model.pool1(x)
        x = model.dropout1(x)
        x = model.conv2(x)
        x = model.bn2(x)
        x = model.efta2(x.permute(0, 2, 3, 1))
        x = model.pool2(x)
        x = model.dropout2(x)
        x = x.view(x.size(0), -1)
        x = model.fc1(x)
        x = model.bn3(x)
        
        fta_input = x
        leaf_values = efta_layer.get_leaf_values(x)
    
    num_leaves = efta_layer.num_leaves
    depth = efta_layer.depth
    branch_factor = efta_layer.branch_factor
    
    # Extract parameters
    alphas = efta_layer.leaf_params[:, :, 0].mean(dim=1).cpu().numpy()
    betas = efta_layer.leaf_params[:, :, 1].mean(dim=1).cpu().numpy()
    gammas = efta_layer.leaf_params[:, :, 2].mean(dim=1).cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Alpha parameters
    ax = axes[0, 0]
    ax.bar(range(num_leaves), alphas, color='coral', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Alpha (mean over units)')
    ax.set_title('EFTA Alpha Parameters (negative scaling)')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Beta parameters
    ax = axes[0, 1]
    ax.bar(range(num_leaves), betas, color='teal', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Beta (mean over units)')
    ax.set_title('EFTA Beta Parameters (exponential rate)')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Initial value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gamma parameters
    ax = axes[0, 2]
    ax.bar(range(num_leaves), gammas, color='steelblue', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Gamma (mean over units)')
    ax.set_title('EFTA Gamma Parameters (positive slope)')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Leaf activation statistics
    ax = axes[1, 0]
    leaf_means = leaf_values.mean(dim=[1, 2]).cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    ax.bar(x_pos - 0.2, leaf_means, width=0.4, label='Mean', color='coral', edgecolor='black')
    ax.bar(x_pos + 0.2, leaf_stds, width=0.4, label='Std', color='teal', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Activation Statistics')
    ax.set_title('Leaf Activation Mean & Std')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Leaf selection frequency
    ax = axes[1, 1]
    winning_leaf = leaf_values.argmax(dim=0)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    
    ax.bar(range(num_leaves), win_counts.cpu().numpy(), color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Times Selected (as max)')
    ax.set_title('Leaf Selection Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Parameter distribution
    ax = axes[1, 2]
    all_params = efta_layer.leaf_params.cpu().numpy()
    ax.hist(all_params[:, :, 0].flatten(), bins=30, alpha=0.5, label='Alpha', color='coral')
    ax.hist(all_params[:, :, 1].flatten(), bins=30, alpha=0.5, label='Beta', color='teal')
    ax.hist(all_params[:, :, 2].flatten(), bins=30, alpha=0.5, label='Gamma', color='steelblue')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Learned Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/branch_analysis_d{depth}_k{branch_factor}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved EFTA visualization to: {save_dir}/branch_analysis_d{depth}_k{branch_factor}.png")


def plot_activation_functions(save_path='./outputs/plots/activation_functions_comparison.png'):
    """Plot the shape of different activation functions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x = np.linspace(-3, 3, 500)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # ReLU
    y = np.maximum(0, x)
    axes[0].plot(x, y, linewidth=2, color='blue')
    axes[0].set_title('ReLU: max(0, x)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)
    
    # LeakyReLU
    y = np.where(x > 0, x, 0.01 * x)
    axes[1].plot(x, y, linewidth=2, color='green')
    axes[1].set_title('LeakyReLU: max(0.01x, x)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)
    
    # GELU (approximation)
    y = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    axes[2].plot(x, y, linewidth=2, color='red')
    axes[2].set_title('GELU: x * Φ(x)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linewidth=0.5)
    axes[2].axvline(x=0, color='k', linewidth=0.5)
    
    # Swish
    y = x / (1 + np.exp(-x))
    axes[3].plot(x, y, linewidth=2, color='purple')
    axes[3].set_title('Swish/SiLU: x * sigmoid(x)')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='k', linewidth=0.5)
    axes[3].axvline(x=0, color='k', linewidth=0.5)
    
    # Mish
    y = x * np.tanh(np.log(1 + np.exp(x)))
    axes[4].plot(x, y, linewidth=2, color='orange')
    axes[4].set_title('Mish: x * tanh(softplus(x))')
    axes[4].grid(True, alpha=0.3)
    axes[4].axhline(y=0, color='k', linewidth=0.5)
    axes[4].axvline(x=0, color='k', linewidth=0.5)
    
    # EELU (alpha=1)
    y = np.where(x > 0, x, np.exp(x) - 1)
    axes[5].plot(x, y, linewidth=2, color='brown')
    axes[5].set_title('EELU: x if x>0 else exp(x)-1')
    axes[5].grid(True, alpha=0.3)
    axes[5].axhline(y=0, color='k', linewidth=0.5)
    axes[5].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved activation functions plot to: {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(epochs=30, verbose=True):
    """Run comprehensive comparison experiment on MNIST."""
    print(f"\n{'='*70}")
    print(f"Running MNIST Experiment: Modern Baselines vs FTA vs EFTA")
    print(f"{'='*70}\n")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders('mnist')
    
    # Model configurations
    configurations = [
        # Modern baselines
        ('relu', None, None, None),
        ('leaky_relu', None, None, None),
        ('gelu', None, None, None),
        ('swish', None, None, None),
        ('mish', None, None, None),
        ('eelu', None, None, None),
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
    trained_models = {}  # Store trained models for visualization
    
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
            trained_models[model_name] = (model, model_type, depth, branch_factor)
            
            print(f"  ✓ Completed: Test Acc={result['test_acc']:.2f}%, "
                  f"Params={result['params']:,}, Time={result['training_time']:.1f}s")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results, trained_models


def print_results_table(results):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"RESULTS FOR MNIST")
    print(f"{'='*100}")
    print(f"{'Model':<25} | {'Test Acc':<10} | {'Val Acc':<10} | {'Params':<12} | {'Time (s)':<10} | {'Epochs':<8}")
    print("-"*100)
    
    for name, result in results.items():
        print(f"{name:<25} | {result['test_acc']:>8.2f}% | {result['best_val_acc']:>8.2f}% | "
              f"{result['params']:>10,} | {result['training_time']:>8.1f} | {result['epochs_trained']:>6}")
    
    print("="*100)
    
    # Find best models
    best_overall = max(results, key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 Best Model: {best_overall} ({results[best_overall]['test_acc']:.2f}%)")
    
    # Best by category
    standard_baselines = {k: v for k, v in results.items() 
                         if k in ['ReLU', 'LeakyReLU', 'GELU', 'Swish', 'Mish', 'EELU']}
    maxout = {k: v for k, v in results.items() if 'Maxout' in k}
    fta_models = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
    efta_models = {k: v for k, v in results.items() if 'EFTA' in k}
    
    if standard_baselines:
        best_baseline = max(standard_baselines, key=lambda x: standard_baselines[x]['test_acc'])
        print(f"🥇 Best Standard Baseline: {best_baseline} ({standard_baselines[best_baseline]['test_acc']:.2f}%)")
    
    if maxout:
        best_maxout = max(maxout, key=lambda x: maxout[x]['test_acc'])
        print(f"🥇 Best Maxout: {best_maxout} ({maxout[best_maxout]['test_acc']:.2f}%)")
    
    if fta_models:
        best_fta = max(fta_models, key=lambda x: fta_models[x]['test_acc'])
        print(f"🥇 Best FTA: {best_fta} ({fta_models[best_fta]['test_acc']:.2f}%)")
    
    if efta_models:
        best_efta = max(efta_models, key=lambda x: efta_models[x]['test_acc'])
        print(f"🥇 Best EFTA: {best_efta} ({efta_models[best_efta]['test_acc']:.2f}%)")
    
    print("="*100)


def save_results(results, output_dir='./outputs/results'):
    """Save results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'mnist_baselines_fta_efta_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Convert results to JSON-serializable format
    save_data = {}
    for name, result in results.items():
        save_data[name] = {
            'test_acc': result['test_acc'],
            'test_loss': result['test_loss'],
            'best_val_acc': result['best_val_acc'],
            'params': result['params'],
            'epochs_trained': result['epochs_trained'],
            'training_time': result['training_time'],
            'final_train_acc': result['history']['train_acc'][-1],
            'final_val_acc': result['history']['val_acc'][-1]
        }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """Main function to run the full experiment."""
    # Create output directories
    os.makedirs('./outputs/plots', exist_ok=True)
    os.makedirs('./outputs/results', exist_ok=True)
    os.makedirs('./outputs/visualizations/fta', exist_ok=True)
    os.makedirs('./outputs/visualizations/efta', exist_ok=True)
    
    # Plot activation function shapes
    print("\nGenerating activation function comparison plot...")
    plot_activation_functions()
    
    # Run experiment
    results, trained_models = run_experiment(epochs=30, verbose=True)
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results(results)
    
    # Generate visualizations
    print("\nGenerating comparison plots...")
    plot_training_curves(results)
    plot_accuracy_vs_params(results)
    plot_bar_comparison(results)
    
    # Visualize branches for best FTA and EFTA models
    print("\nGenerating branch visualizations...")
    
    # Find best FTA and EFTA
    fta_models = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
    efta_models = {k: v for k, v in results.items() if 'EFTA' in k}
    
    if fta_models:
        best_fta = max(fta_models, key=lambda x: fta_models[x]['test_acc'])
        model, model_type, depth, branch_factor = trained_models[best_fta]
        test_loader = get_data_loaders('mnist')[2]
        visualize_fta_branches(model, test_loader, device)
    
    if efta_models:
        best_efta = max(efta_models, key=lambda x: efta_models[x]['test_acc'])
        model, model_type, depth, branch_factor = trained_models[best_efta]
        test_loader = get_data_loaders('mnist')[2]
        visualize_efta_branches(model, test_loader, device)
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Results: ./outputs/results/")
    print(f"  - Plots: ./outputs/plots/")
    print(f"  - Visualizations: ./outputs/visualizations/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
