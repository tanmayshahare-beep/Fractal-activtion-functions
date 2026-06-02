"""
Adult (Census Income) Dataset: FTA vs EFTA vs Modern Baselines

Tests fractal tree activations on real-world tabular data with mixed feature types.

Dataset: Adult Income (UCI / OpenML)
- Samples: 48,842 (after cleaning)
- Features: 14 (mix of numeric and categorical → ~108 after one-hot encoding)
- Target: Binary (income >50K vs <=50K)
- Complexity: Moderate (non-linear relationships, feature interactions)

This script includes:
- Proper preprocessing (StandardScaler for numeric, OneHotEncoder for categorical)
- MLP models with various activations
- Multi-seed evaluation (5 seeds by default)
- Comprehensive results and visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
        # Leaf weights and biases (F.linear expects [out, in])
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_units, self.input_dim) * (2.0 / self.input_dim) ** 0.5)
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        batch_size = x.shape[0]
        
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
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
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class ExponentialFTA(nn.Module):
    """
    Exponential Fractal Tree Activation (EFTA) for MLP.
    
    Each leaf computes:
        f(x) = α * (exp(β * x) - 1)  if x < 0
        f(x) = γ * x                  if x >= 0
    """
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))
        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)  # alpha
        nn.init.constant_(self.leaf_params[:, :, 1], 0.5)  # beta
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)  # gamma

    def _leaf_function(self, x, alpha, beta, gamma):
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        neg_part = alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0)
        pos_part = gamma * x
        return neg_mask * neg_part + pos_mask * pos_part

    def forward(self, x):
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
    """Maxout activation for MLP."""
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
    """MLP with standard activation."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2, activation_fn=None):
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
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(FractalTreeActivation(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPWithEFTA(nn.Module):
    """MLP with EFTA activation."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(ExponentialFTA(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPWithMaxout(nn.Module):
    """MLP with Maxout activation."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=2, k=4):
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
# Data Loading and Preprocessing
# =============================================================================

def load_adult_data(val_split=0.2, test_split=0.2, random_state=42):
    """Load and preprocess Adult dataset."""
    print("Loading Adult dataset from OpenML...")
    
    # Load data
    adult = fetch_openml(data_id=1590, as_frame=True)
    X, y = adult.data, adult.target
    
    # Convert target to binary
    y = (y == '>50K').astype(int)
    
    print(f"Dataset info:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split/(1-test_split), random_state=random_state, stratify=y_train
    )
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    # Fit and transform
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"  Features after encoding: {X_train_processed.shape[1]}")
    print(f"  Train samples: {len(X_train_processed):,}")
    print(f"  Val samples: {len(X_val_processed):,}")
    print(f"  Test samples: {len(X_test_processed):,}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)
    
    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t


def get_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    """Create DataLoaders."""
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Training Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=100, lr=0.001, patience=10, model_name="Model"):
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
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item()
                pred = out.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step(100.0 - val_acc)
        
        history['train_loss'].append(train_loss)
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

def plot_results(results, save_dir='./outputs/adult_results'):
    """Plot comparison results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
    ax.set_title('Adult Dataset: Multi-Seed Comparison (5 seeds)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(80, 87)
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_accs, std_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{mean:.2f}±{std:.2f}%', ha='center', va='bottom', fontsize=8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', edgecolor='black', label='Standard Baselines'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Maxout'),
        Patch(facecolor='#3498db', edgecolor='black', label='FTA (Linear)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='EFTA (Exponential)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/adult_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {save_dir}/adult_comparison.png")
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in results.items():
        if data['best_history'] is not None:
            epochs = list(range(1, len(data['best_history']['val_acc']) + 1))
            
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
                        color=color, linewidth=2, label=name, alpha=0.7)
            axes[1].plot(epochs, data['best_history']['val_loss'], style,
                        color=color, linewidth=2, label=name, alpha=0.7)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Validation Accuracy (Best Run)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='lower right', ncol=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss (Best Run)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/adult_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_dir}/adult_training_curves.png")


def visualize_fta_efta_branches(model, X_test, model_type, save_dir='./outputs/adult_visualizations'):
    """Visualize FTA/EFTA branch parameters and activations."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Sample test data
    X_sample = torch.tensor(X_test[:64], dtype=torch.float32).to(device)
    
    # Find FTA/EFTA layer
    fta_layer = None
    for module in model.modules():
        if isinstance(module, (FractalTreeActivation, ExponentialFTA)):
            fta_layer = module
            break
    
    if fta_layer is None:
        print("No FTA/EFTA layer found")
        return
    
    # Get leaf values
    with torch.no_grad():
        x = model.net[0](X_sample)  # First linear layer
        leaf_values = fta_layer.get_leaf_values(x)
    
    num_leaves = fta_layer.num_leaves
    depth = fta_layer.depth
    branch_factor = fta_layer.branch_factor
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Leaf activation statistics
    ax = axes[0, 0]
    leaf_means = leaf_values.mean(dim=[1, 2]).detach().cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).detach().cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    ax.bar(x_pos - 0.2, leaf_means, width=0.4, label='Mean', color='coral', edgecolor='black')
    ax.bar(x_pos + 0.2, leaf_stds, width=0.4, label='Std', color='teal', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Activation Statistics')
    ax.set_title(f'{model_type} Leaf Activation (64 samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Leaf selection frequency
    ax = axes[0, 1]
    winning_leaf = leaf_values.argmax(dim=0)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    
    ax.bar(range(num_leaves), win_counts.cpu().numpy(), color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Times Selected (as max)')
    ax.set_title('Leaf Selection Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameters
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
        
        ax = axes[1, 1]
        all_params = fta_layer.leaf_params.detach().cpu().numpy()
        ax.hist(all_params[:, :, 0].flatten(), bins=30, alpha=0.5, label='Alpha', color='coral')
        ax.hist(all_params[:, :, 1].flatten(), bins=30, alpha=0.5, label='Beta', color='teal')
        ax.hist(all_params[:, :, 2].flatten(), bins=30, alpha=0.5, label='Gamma', color='steelblue')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Parameter Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax = axes[1, 0]
        weight_norms = [fta_layer.leaf_weights[i].data.norm(p=2).item() for i in range(num_leaves)]
        ax.bar(range(num_leaves), weight_norms, color='steelblue', edgecolor='black')
        ax.set_xlabel('Leaf Index')
        ax.set_ylabel('Weight Norm (L2)')
        ax.set_title('FTA Leaf Weight Norms')
        ax.grid(True, alpha=0.3)
        
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
        ax.set_title(f'Tree Structure (depth={depth}, k={branch_factor})')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_type.lower().replace(" ", "_").replace("(", "").replace(")", "")}_branches.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved branch visualization: {save_dir}/{model_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}_branches.png")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(n_seeds=5, epochs=100, patience=10, batch_size=128):
    """Run full experiment."""
    print(f"\n{'='*70}")
    print(f"Adult Dataset Experiment: FTA vs EFTA vs Baselines")
    print(f"Seeds: {n_seeds}, Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*70}\n")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_adult_data()
    train_loader, val_loader, test_loader = get_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    
    input_dim = X_train.shape[1]
    print(f"\nInput dimension after preprocessing: {input_dim}")
    
    # Model configurations
    configurations = [
        # Standard baselines
        ('relu', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                            activation_fn=lambda: nn.ReLU())),
        ('leaky_relu', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                   activation_fn=lambda: nn.LeakyReLU(0.01))),
        ('gelu', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                            activation_fn=GELUActivation)),
        ('swish', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                             activation_fn=SwishActivation)),
        ('mish', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                            activation_fn=MishActivation)),
        ('eelu', lambda: MLP(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                            activation_fn=EELUActivation)),
        
        # Maxout
        ('maxout', lambda: MLPWithMaxout(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2, k=4)),
        
        # FTA variants
        ('fta_d2k2', lambda: MLPWithFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                        depth=2, branch_factor=2)),
        ('fta_d3k2', lambda: MLPWithFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                        depth=3, branch_factor=2)),
        ('fta_d2k3', lambda: MLPWithFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                        depth=2, branch_factor=3)),
        
        # EFTA variants
        ('efta_d2k2', lambda: MLPWithEFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                          depth=2, branch_factor=2)),
        ('efta_d3k2', lambda: MLPWithEFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
                                          depth=3, branch_factor=2)),
        ('efta_d2k3', lambda: MLPWithEFTA(input_dim=input_dim, hidden_dims=[128, 64, 32], output_dim=2,
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
            print(f"  Seed {seed}: Test Acc={result['test_acc']:.2f}%, "
                  f"Params={result['params']:,}, Time={result['training_time']:.1f}s")
            
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
    
    return results, best_models, display_names, X_test.numpy()


def print_results_table(results):
    """Print results table."""
    print(f"\n{'='*85}")
    print(f"ADULT DATASET RESULTS (Multi-Seed Average)")
    print(f"{'='*85}")
    print(f"{'Model':<20} | {'Mean Acc':<12} | {'Std':<10} | {'Best':<10} | {'Worst':<10}")
    print("-"*85)
    
    for name, data in results.items():
        best = max(data['accs'])
        worst = min(data['accs'])
        print(f"{name:<20} | {data['mean']:>10.2f}% | {data['std']:>8.2f}% | "
              f"{best:>8.2f}% | {worst:>8.2f}%")
    
    print("="*85)
    
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
    print("="*85)


def save_results(results, best_models, save_dir='./outputs/adult_results'):
    """Save results to JSON and model checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'{save_dir}/adult_results_{timestamp}.json'
    
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
    os.makedirs('./outputs/adult_results', exist_ok=True)
    os.makedirs('./outputs/adult_visualizations', exist_ok=True)
    
    # Run experiment
    results, best_models, display_names, X_test = run_experiment(
        n_seeds=5, epochs=100, patience=10, batch_size=128
    )
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results)
    
    # Visualize branches
    print("\nGenerating branch visualizations...")
    
    if 'fta_d2k2' in best_models:
        model = best_models['fta_d2k2']
        visualize_fta_efta_branches(model, X_test, 'FTA (d=2,k=2)')
    
    if 'efta_d2k2' in best_models:
        model = best_models['efta_d2k2']
        visualize_fta_efta_branches(model, X_test, 'EFTA (d=2,k=2)')
    
    print("\n" + "="*70)
    print("Adult Dataset Experiment Complete!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Results: ./outputs/adult_results/")
    print(f"  - Plots: ./outputs/adult_results/")
    print(f"  - Visualizations: ./outputs/adult_visualizations/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
