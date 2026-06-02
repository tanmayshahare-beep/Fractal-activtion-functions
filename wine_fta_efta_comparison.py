"""
Wine Dataset: FTA vs EFTA vs Modern Baselines Comparison

Tests fractal tree activations on tabular data to validate the thesis:
- FTA (linear branches) should excel on moderately complex data
- EFTA (exponential branches) may be competitive but not necessarily superior

Dataset: Wine Recognition (UCI)
- Samples: 178
- Features: 13 (chemical properties)
- Classes: 3 (cultivar types)
- Complexity: Low-medium (small dataset, non-linear boundaries)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import os
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")


# =============================================================================
# Activation Function Implementations
# =============================================================================

class GELUActivation(nn.Module):
    """Gaussian Error Linear Unit."""
    def forward(self, x):
        return F.gelu(x)


class SwishActivation(nn.Module):
    """Swish/SiLU activation."""
    def forward(self, x):
        return F.silu(x)


class MishActivation(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EELUActivation(nn.Module):
    """Exponential ELU."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
    
    def forward(self, x):
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        neg_part = self.alpha * (torch.exp(torch.clamp(x, -10, 10)) - 1.0)
        pos_part = x
        return neg_mask * neg_part + pos_mask * pos_part


class FractalTreeActivation(nn.Module):
    """
    Fractal Tree Activation (FTA) for MLP.

    Each leaf computes: W·x + b
    Leaves are combined via hierarchical max operation.

    Args:
        num_units: number of output neurons
        depth: tree depth
        branch_factor: branches per internal node
        input_dim: input feature dimension
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units

        # Leaf weights and biases
        # F.linear expects weight shape [out_features, in_features]
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_units, self.input_dim) * (2.0 / self.input_dim) ** 0.5)
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        """
        x shape: (batch, input_dim)
        output shape: (batch, num_units)
        """
        batch_size = x.shape[0]

        # Compute all leaf outputs
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)

        # Stack: (num_leaves, batch, num_units)
        stacked = torch.stack(leaf_outputs, dim=0)

        # Hierarchical max pooling through tree
        current = stacked
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = current.view(num_nodes, self.branch_factor, batch_size, self.num_units)
            current = current.max(dim=1)[0]

        return current.squeeze(0)

    def get_leaf_values(self, x):
        """Get individual leaf outputs for visualization."""
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class ExponentialFTA(nn.Module):
    """
    Exponential Fractal Tree Activation (EFTA) for MLP.
    
    Each leaf computes:
        f(x) = α * (exp(β * x) - 1)  if x < 0
        f(x) = γ * x                  if x >= 0
    
    Args:
        num_units: number of output neurons
        depth: tree depth
        branch_factor: branches per internal node
        input_dim: input feature dimension
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
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
        """
        x shape: (batch, input_dim)
        output shape: (batch, num_units)
        """
        batch_size = x.shape[0]
        
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
            current = current.view(num_nodes, self.branch_factor, batch_size, self.num_units)
            current = current.max(dim=1)[0]
        
        return current.squeeze(0)
    
    def get_leaf_values(self, x):
        """Get individual leaf outputs for visualization."""
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            alpha = self.leaf_params[i, :, 0]
            beta = self.leaf_params[i, :, 1]
            gamma = self.leaf_params[i, :, 2]
            leaf_val = self._leaf_function(x, alpha, beta, gamma)
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class MaxoutActivation(nn.Module):
    """
    Maxout activation for MLP.

    f(x) = max(W_1·x, W_2·x, ..., W_k·x)
    """
    def __init__(self, in_features, out_features, k=4):
        super().__init__()
        self.k = k
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features * k)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.out_features, self.k)
        return x.max(dim=2)[0]


# =============================================================================
# MLP Models
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with pluggable activation.
    
    Args:
        input_dim: input feature dimension
        hidden_dims: list of hidden layer dimensions
        output_dim: output dimension (num classes)
        activation_fn: callable that returns activation module
    """
    def __init__(self, input_dim=13, hidden_dims=[64, 32], output_dim=3, activation_fn=None):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPWithFTA(nn.Module):
    """MLP with FTA activation."""
    def __init__(self, input_dim=13, hidden_dims=[64, 32], output_dim=3,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            # FTA receives h_dim-dimensional input from the Linear layer
            # Each unit in the h_dim output has its own tree
            layers.append(FractalTreeActivation(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPWithEFTA(nn.Module):
    """MLP with EFTA activation."""
    def __init__(self, input_dim=13, hidden_dims=[64, 32], output_dim=3,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(ExponentialFTA(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPWithMaxout(nn.Module):
    """MLP with Maxout activation."""
    def __init__(self, input_dim=13, hidden_dims=[64, 32], output_dim=3, k=4):
        super().__init__()
        self.k = k

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(MaxoutActivation(prev_dim, h_dim, k))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Data Loading
# =============================================================================

def load_wine_data(val_split=0.2, test_split=0.2, random_state=42):
    """Load and preprocess Wine dataset."""
    # Load data
    data = load_wine()
    X, y = data.data, data.target
    
    print(f"Dataset info:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_split + val_split, random_state=random_state, stratify=y
    )
    
    # Adjust val/test ratio
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16):
    """Create DataLoaders."""
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Training Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=200, lr=0.001, patience=15, model_name="Model"):
    """Train and evaluate model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
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
        
        if patience_counter >= patience:
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            pred = out.argmax(dim=1)
            test_correct += (pred == yb).sum().item()
            test_total += yb.size(0)
    
    test_acc = 100.0 * test_correct / test_total
    total_time = time.time() - start_time
    params = count_parameters(model)
    
    return {
        'name': model_name,
        'test_acc': test_acc,
        'val_acc': best_val_acc,
        'params': params,
        'training_time': total_time,
        'history': history
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_results(results, save_dir='./outputs/wine_results'):
    """Plot comparison results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(results.keys())
    mean_accs = [np.mean(results[n]['accs']) for n in names]
    std_accs = [np.std(results[n]['accs']) for n in names]
    
    colors = []
    for name in names:
        if 'EFTA' in name:
            colors.append('#2ecc71')
        elif 'FTA' in name:
            colors.append('#3498db')
        elif 'Maxout' in name:
            colors.append('#9b59b6')
        else:
            colors.append('#95a5a6')
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, mean_accs, yerr=std_accs, color=colors, 
                  edgecolor='black', capsize=5, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Wine Dataset: Multi-Seed Comparison (10 seeds)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_accs, std_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{mean:.2f}±{std:.2f}%', ha='center', va='bottom', fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', edgecolor='black', label='Standard Baselines'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Maxout'),
        Patch(facecolor='#3498db', edgecolor='black', label='FTA (Linear)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='EFTA (Exponential)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/wine_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {save_dir}/wine_comparison.png")
    
    # Training curves for best run of each category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = {
        'ReLU': None,
        'FTA (d=2,k=2)': None,
        'EFTA (d=2,k=2)': None
    }
    
    for name, data in results.items():
        if name in categories and data['best_history'] is not None:
            epochs = list(range(1, len(data['best_history']['val_acc']) + 1))
            ax = axes[0] if 'val_acc' in data['best_history'] else axes[1]
            
            if 'FTA' in name and 'EFTA' not in name:
                color = '#3498db'
                style = '--'
            elif 'EFTA' in name:
                color = '#2ecc71'
                style = '-.'
            else:
                color = '#95a5a6'
                style = '-'
            
            axes[0].plot(epochs, data['best_history']['val_acc'], style, 
                        color=color, linewidth=2, label=name, alpha=0.8)
            axes[1].plot(epochs, data['best_history']['val_loss'], style,
                        color=color, linewidth=2, label=name, alpha=0.8)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Validation Accuracy (Best Run)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss (Best Run)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/wine_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_dir}/wine_training_curves.png")


def visualize_fta_efta_branches(model, X_test, y_test, model_type, save_dir='./outputs/wine_visualizations'):
    """Visualize FTA/EFTA branch parameters and activations."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get a sample of test data
    X_sample = torch.tensor(X_test[:32], dtype=torch.float32).to(device)
    
    # Find the FTA/EFTA layer
    fta_layer = None
    for module in model.modules():
        if isinstance(module, (FractalTreeActivation, ExponentialFTA)):
            fta_layer = module
            break
    
    if fta_layer is None:
        print("No FTA/EFTA layer found in model")
        return
    
    # Get leaf values
    with torch.no_grad():
        # Forward through first linear layer to get input to activation
        x = model.net[0](X_sample)  # First linear layer
        leaf_values = fta_layer.get_leaf_values(x)  # (num_leaves, batch, num_units)
    
    num_leaves = fta_layer.num_leaves
    depth = fta_layer.depth
    branch_factor = fta_layer.branch_factor
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Leaf activation statistics
    ax = axes[0, 0]
    leaf_means = leaf_values.mean(dim=[1, 2]).cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    ax.bar(x_pos - 0.2, leaf_means, width=0.4, label='Mean', color='coral', edgecolor='black')
    ax.bar(x_pos + 0.2, leaf_stds, width=0.4, label='Std', color='teal', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Activation Statistics')
    ax.set_title(f'{model_type} Leaf Activation Mean & Std (32 samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Leaf selection frequency
    ax = axes[0, 1]
    winning_leaf = leaf_values.argmax(dim=0)  # (batch, num_units)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    
    ax.bar(range(num_leaves), win_counts.cpu().numpy(), color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Times Selected (as max)')
    ax.set_title('Leaf Selection Frequency')
    ax.grid(True, alpha=0.3)

    # Plot 3: Parameter visualization (EFTA only)
    if isinstance(fta_layer, ExponentialFTA):
        ax = axes[1, 0]
        alphas = fta_layer.leaf_params[:, :, 0].mean(dim=1).detach().cpu().numpy()
        betas = fta_layer.leaf_params[:, :, 1].mean(dim=1).detach().cpu().numpy()
        gammas = fta_layer.leaf_params[:, :, 2].mean(dim=1).detach().cpu().numpy()

        x_pos = np.arange(num_leaves)
        width = 0.25
        ax.bar(x_pos - width, alphas, width, label='Alpha', color='coral', edgecolor='black')
        ax.bar(x_pos, betas, width, label='Beta', color='teal', edgecolor='black')
        ax.bar(x_pos + width, gammas, width, label='Gamma', color='steelblue', edgecolor='black')
        ax.set_xlabel('Leaf Index')
        ax.set_ylabel('Parameter Value (mean)')
        ax.set_title('EFTA Learned Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Parameter distribution
        ax = axes[1, 1]
        all_params = fta_layer.leaf_params.detach().cpu().numpy()
        ax.hist(all_params[:, :, 0].flatten(), bins=20, alpha=0.5, label='Alpha', color='coral')
        ax.hist(all_params[:, :, 1].flatten(), bins=20, alpha=0.5, label='Beta', color='teal')
        ax.hist(all_params[:, :, 2].flatten(), bins=20, alpha=0.5, label='Gamma', color='steelblue')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Learned Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # FTA: Plot weight norms
        ax = axes[1, 0]
        weight_norms = [fta_layer.leaf_weights[i].data.norm(p=2).item() for i in range(num_leaves)]
        ax.bar(range(num_leaves), weight_norms, color='steelblue', edgecolor='black')
        ax.set_xlabel('Leaf Index')
        ax.set_ylabel('Weight Norm (L2)')
        ax.set_title('FTA Leaf Weight Norms')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Tree structure
        ax = axes[1, 1]
        ax.axis('off')
        
        levels = depth + 1
        y_positions = np.linspace(0.8, 0.2, levels)
        
        ax.plot([0.5], [y_positions[0]], 'o', color='red', markersize=20)
        ax.text(0.5, y_positions[0] + 0.05, 'Root', ha='center', fontsize=10, fontweight='bold')
        
        nodes_at_level = 1
        for level in range(1, levels):
            nodes_at_level *= branch_factor
            x_positions = np.linspace(0.1, 0.9, nodes_at_level)
            
            for i, x_pos in enumerate(x_positions):
                if level == depth:
                    ax.plot([x_pos], [y_positions[level]], 's', color='green', markersize=15)
                    ax.text(x_pos, y_positions[level] - 0.08, f'L{i}', ha='center', fontsize=8)
                else:
                    ax.plot([x_pos], [y_positions[level]], 'o', color='blue', markersize=15)
                
                parent_idx = i // branch_factor
                parent_x = np.linspace(0.1, 0.9, nodes_at_level // branch_factor)[parent_idx]
                parent_y = y_positions[level - 1]
                ax.plot([parent_x, x_pos], [parent_y, y_positions[level]], 'k-', linewidth=1, alpha=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Tree Structure (depth={depth}, branch_factor={branch_factor})')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_type.lower().replace(" ", "_")}_branches.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved branch visualization to: {save_dir}/{model_type.lower().replace(' ', '_')}_branches.png")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(n_seeds=10, epochs=200, patience=15, batch_size=16):
    """Run full experiment with multiple seeds."""
    print(f"\n{'='*70}")
    print(f"Wine Dataset Experiment: FTA vs EFTA vs Baselines")
    print(f"Seeds: {n_seeds}, Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*70}\n")
    
    # Load data (same split for all seeds)
    X_train, y_train, X_val, y_val, X_test, y_test = load_wine_data()
    train_loader, val_loader, test_loader = get_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    
    # Model configurations
    configurations = [
        # Standard baselines
        ('relu', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3, 
                            activation_fn=lambda: nn.ReLU())),
        ('leaky_relu', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                   activation_fn=lambda: nn.LeakyReLU(0.01))),
        ('gelu', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                            activation_fn=GELUActivation)),
        ('swish', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                             activation_fn=SwishActivation)),
        ('mish', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                            activation_fn=MishActivation)),
        ('eelu', lambda: MLP(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                            activation_fn=EELUActivation)),
        
        # Maxout
        ('maxout', lambda: MLPWithMaxout(input_dim=13, hidden_dims=[64, 32], output_dim=3, k=4)),
        
        # FTA variants
        ('fta_d2k2', lambda: MLPWithFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                        depth=2, branch_factor=2)),
        ('fta_d3k2', lambda: MLPWithFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                        depth=3, branch_factor=2)),
        ('fta_d2k3', lambda: MLPWithFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                        depth=2, branch_factor=3)),
        
        # EFTA variants
        ('efta_d2k2', lambda: MLPWithEFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                          depth=2, branch_factor=2)),
        ('efta_d3k2', lambda: MLPWithEFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                          depth=3, branch_factor=2)),
        ('efta_d2k3', lambda: MLPWithEFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                                          depth=2, branch_factor=3)),
    ]
    
    display_names = {
        'relu': 'ReLU',
        'leaky_relu': 'LeakyReLU',
        'gelu': 'GELU',
        'swish': 'Swish',
        'mish': 'Mish',
        'eelu': 'EELU',
        'maxout': 'Maxout (k=4)',
        'fta_d2k2': 'FTA (d=2,k=2)',
        'fta_d3k2': 'FTA (d=3,k=2)',
        'fta_d2k3': 'FTA (d=2,k=3)',
        'efta_d2k2': 'EFTA (d=2,k=2)',
        'efta_d3k2': 'EFTA (d=3,k=2)',
        'efta_d2k3': 'EFTA (d=2,k=3)',
    }
    
    results = {}
    best_models = {}
    
    for config_key, model_fn in configurations:
        name = display_names[config_key]
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        accs = []
        best_history = None
        best_run_result = None
        
        for seed in range(n_seeds):
            set_seed(seed)
            
            model = model_fn()
            model = model.to(device)
            
            result = train_model(
                model, train_loader, val_loader, test_loader, device,
                epochs=epochs, patience=patience, model_name=name
            )
            
            accs.append(result['test_acc'])
            print(f"  Seed {seed:2d}: Test Acc={result['test_acc']:.2f}%, "
                  f"Params={result['params']:,}, Time={result['training_time']:.1f}s")
            
            # Keep best run for visualization
            if best_run_result is None or result['test_acc'] > best_run_result['test_acc']:
                best_run_result = result
                best_history = result['history']
                best_models[config_key] = model
        
        results[name] = {
            'accs': accs,
            'mean': np.mean(accs),
            'std': np.std(accs),
            'best_history': best_history,
            'best_result': best_run_result
        }
        
        print(f"\n  {name}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    
    return results, best_models, display_names


def print_results_table(results):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"WINE DATASET RESULTS (Multi-Seed Average)")
    print(f"{'='*80}")
    print(f"{'Model':<20} | {'Mean Acc':<12} | {'Std':<10} | {'Best':<10} | {'Worst':<10}")
    print("-"*80)
    
    for name, data in results.items():
        best = max(data['accs'])
        worst = min(data['accs'])
        print(f"{name:<20} | {data['mean']:>10.2f}% | {data['std']:>8.2f}% | "
              f"{best:>8.2f}% | {worst:>8.2f}%")
    
    print("="*80)
    
    # Find best by category
    standard = {k: v for k, v in results.items() 
                if k in ['ReLU', 'LeakyReLU', 'GELU', 'Swish', 'Mish', 'EELU']}
    fta = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
    efta = {k: v for k, v in results.items() if 'EFTA' in k}
    
    if standard:
        best_std = max(standard, key=lambda x: results[x]['mean'])
        print(f"\n🥇 Best Standard: {best_std} ({results[best_std]['mean']:.2f}% ± {results[best_std]['std']:.2f}%)")
    
    if fta:
        best_fta = max(fta, key=lambda x: results[x]['mean'])
        print(f"🥇 Best FTA: {best_fta} ({results[best_fta]['mean']:.2f}% ± {results[best_fta]['std']:.2f}%)")
    
    if efta:
        best_efta = max(efta, key=lambda x: results[x]['mean'])
        print(f"🥇 Best EFTA: {best_efta} ({results[best_efta]['mean']:.2f}% ± {results[best_efta]['std']:.2f}%)")
    
    overall_best = max(results, key=lambda x: results[x]['mean'])
    print(f"\n🏆 Overall Best: {overall_best} ({results[overall_best]['mean']:.2f}% ± {results[overall_best]['std']:.2f}%)")
    print("="*80)


def save_results(results, best_models, save_dir='./outputs/wine_results'):
    """Save results to JSON and model checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'{save_dir}/wine_results_{timestamp}.json'
    
    save_data = {}
    for name, data in results.items():
        save_data[name] = {
            'mean': float(data['mean']),
            'std': float(data['std']),
            'all_accs': [float(a) for a in data['accs']],
            'best_test_acc': float(data['best_result']['test_acc']),
            'params': data['best_result']['params'],
            'training_time': data['best_result']['training_time']
        }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    # Save model checkpoints
    print("Saving model checkpoints...")
    for config_key, model in best_models.items():
        checkpoint_path = f'{save_dir}/model_{config_key}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config_key
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
    
    return filepath


def main():
    """Main function."""
    # Create output directories
    os.makedirs('./outputs/wine_results', exist_ok=True)
    os.makedirs('./outputs/wine_visualizations', exist_ok=True)
    
    # Run experiment
    results, best_models, display_names = run_experiment(
        n_seeds=10, epochs=200, patience=15, batch_size=16
    )
    
    # Print results
    print_results_table(results)
    
    # Save results and model checkpoints
    save_results(results, best_models)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results)
    
    # Visualize branches for best FTA and EFTA
    print("\nGenerating branch visualizations...")
    
    # Load data for visualization
    X_train, y_train, X_val, y_val, X_test, y_test = load_wine_data()
    
    if 'fta_d2k2' in best_models:
        model = best_models['fta_d2k2']
        visualize_fta_efta_branches(model, X_test, y_test, 'FTA (d=2,k=2)')
    
    if 'efta_d2k2' in best_models:
        model = best_models['efta_d2k2']
        visualize_fta_efta_branches(model, X_test, y_test, 'EFTA (d=2,k=2)')
    
    print("\n" + "="*70)
    print("Wine Dataset Experiment Complete!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Results: ./outputs/wine_results/")
    print(f"  - Plots: ./outputs/wine_results/")
    print(f"  - Visualizations: ./outputs/wine_visualizations/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
