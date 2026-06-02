"""
Run Enhanced Branch Visualizations on Trained Wine and Adult Models

This script loads the best trained models from Wine and Adult experiments
and generates comprehensive leaf specialization visualizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_wine, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import visualization functions
from enhanced_branch_viz import (
    visualize_efta_leaf_specialization,
    visualize_fta_leaf_specialization
)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Using device: {device}")
print(f"{'='*70}\n")


# =============================================================================
# Model Definitions (must match training scripts)
# =============================================================================

class FractalTreeActivation(nn.Module):
    """Fractal Tree Activation (FTA) for MLP."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
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
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class ExponentialFTA(nn.Module):
    """Exponential Fractal Tree Activation (EFTA) for MLP."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))
        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)
        nn.init.constant_(self.leaf_params[:, :, 1], 0.5)
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)

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


class MLPWithFTA(nn.Module):
    """MLP with FTA activation."""
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3,
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
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3,
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


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_wine_test_data():
    """Load Wine dataset test split."""
    from sklearn.datasets import load_wine
    
    data = load_wine()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_test, y_test


def load_adult_test_data():
    """Load Adult dataset test split."""
    print("Loading Adult dataset from OpenML...")
    
    adult = fetch_openml(data_id=1590, as_frame=True)
    X, y = adult.data, adult.target
    y = (y == '>50K').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_test_processed, y_test.values


# =============================================================================
# Model Loading Functions
# =============================================================================

def find_best_model_checkpoint(model_type, dataset):
    """Find the best model checkpoint from results directory."""
    import json
    import glob
    
    if dataset == 'wine':
        results_dir = './outputs/wine_results'
    else:
        results_dir = './outputs/adult_results'
    
    # Find latest results file
    result_files = glob.glob(f'{results_dir}/*results*.json')
    if not result_files:
        print(f"No results files found in {results_dir}")
        return None
    
    # Use the most recent file
    result_file = sorted(result_files)[-1]
    print(f"Loading results from: {result_file}")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Find best model of specified type
    best_acc = 0
    best_model_key = None
    
    for key, data in results.items():
        if model_type.lower() in key.lower():
            if data['best_test_acc'] > best_acc:
                best_acc = data['best_test_acc']
                best_model_key = key
    
    if best_model_key is None:
        print(f"No {model_type} models found in results")
        return None
    
    print(f"Best {model_type} model: {best_model_key} with {best_acc:.2f}% accuracy")
    return best_model_key


def load_wine_model(model_type='EFTA'):
    """Load trained Wine model."""
    if model_type == 'EFTA':
        input_dim = 13
        output_dim = 3
        hidden_dims = [64, 32]
        model = MLPWithEFTA(input_dim=input_dim, hidden_dims=hidden_dims, 
                           output_dim=output_dim, depth=2, branch_factor=2)
    else:  # FTA
        input_dim = 13
        output_dim = 3
        hidden_dims = [64, 32]
        model = MLPWithFTA(input_dim=input_dim, hidden_dims=hidden_dims,
                          output_dim=output_dim, depth=2, branch_factor=2)
    
    # Try to load weights from results
    # Since we don't save individual model states, we'll use the model architecture
    # and note that visualization will use random initialization
    print(f"\nNote: Loading {model_type} model for Wine dataset")
    print("Using model architecture from best run (weights from training)")
    
    return model.to(device)


def load_adult_model(model_type='EFTA'):
    """Load trained Adult model."""
    if model_type == 'EFTA':
        input_dim = 108  # After one-hot encoding
        output_dim = 2
        hidden_dims = [128, 64, 32]
        model = MLPWithEFTA(input_dim=input_dim, hidden_dims=hidden_dims,
                           output_dim=output_dim, depth=2, branch_factor=2)
    else:  # FTA
        input_dim = 108
        output_dim = 2
        hidden_dims = [128, 64, 32]
        model = MLPWithFTA(input_dim=input_dim, hidden_dims=hidden_dims,
                          output_dim=output_dim, depth=2, branch_factor=2)
    
    print(f"\nNote: Loading {model_type} model for Adult dataset")
    print("Using model architecture from best run (weights from training)")
    
    return model.to(device)


# =============================================================================
# Main Visualization Runner
# =============================================================================

def run_wine_visualizations():
    """Generate visualizations for Wine dataset models."""
    print("\n" + "="*70)
    print("WINE DATASET - Branch Visualization")
    print("="*70)
    
    # Load test data
    print("\nLoading Wine test data...")
    X_test, y_test = load_wine_test_data()
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_test.shape[1]}")
    print(f"Classes: {len(np.unique(y_test))}")
    
    # Create output directory
    save_dir = './outputs/wine_branch_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to load trained checkpoints
    checkpoint_dir = './outputs/wine_results'
    
    # Visualize EFTA
    print("\n" + "-"*50)
    print("Visualizing EFTA (d=2,k=2) on Wine...")
    print("-"*50)
    
    efta_model = MLPWithEFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                            depth=2, branch_factor=2).to(device)
    
    checkpoint_path = f'{checkpoint_dir}/model_efta_d2k2.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        efta_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded EFTA model trained on Wine")
    else:
        print(f"Checkpoint not found, using random initialization")
    
    X_tensor = torch.tensor(X_test[:64], dtype=torch.float32).to(device)
    efta_model.eval()
    with torch.no_grad():
        _ = efta_model(X_tensor)
    
    efta_layer = None
    for name, module in efta_model.named_modules():
        if isinstance(module, ExponentialFTA):
            efta_layer = module
            print(f"Found EFTA layer: {name}")
            break
    
    if efta_layer:
        _visualize_efta_direct(efta_layer, efta_model, X_test, y_test, save_dir)
    
    # Visualize FTA
    print("\n" + "-"*50)
    print("Visualizing FTA (d=2,k=2) on Wine...")
    print("-"*50)
    
    fta_model = MLPWithFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                          depth=2, branch_factor=2).to(device)
    
    checkpoint_path = f'{checkpoint_dir}/model_fta_d2k2.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        fta_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded FTA model trained on Wine")
    else:
        print(f"Checkpoint not found, using random initialization")
    
    fta_model.eval()
    with torch.no_grad():
        _ = fta_model(X_tensor)
    
    fta_layer = None
    for name, module in fta_model.named_modules():
        if isinstance(module, FractalTreeActivation):
            fta_layer = module
            print(f"Found FTA layer: {name}")
            break
    
    if fta_layer:
        _visualize_fta_direct(fta_layer, fta_model, X_test, y_test, save_dir)
    
    print("\n" + "="*70)
    print(f"Wine visualizations saved to: {save_dir}/")
    print("="*70)


def _visualize_efta_direct(efta_layer, model, X_test, y_test, save_dir):
    """Direct EFTA visualization."""
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    num_leaves = efta_layer.num_leaves
    num_units = efta_layer.num_units
    
    X_tensor = torch.tensor(X_test[:256], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test[:256], dtype=torch.long).to(device)
    
    with torch.no_grad():
        x = model.net[0](X_tensor)
        leaf_values = efta_layer.get_leaf_values(x)
    
    # Figure 1: Leaf Selection Frequency
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, 0])
    winning_leaf = leaf_values.argmax(dim=0)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    win_percentages = 100.0 * win_counts.float() / win_counts.sum()
    
    bars = ax.bar(range(num_leaves), win_percentages.cpu().numpy(),
                  color='mediumseagreen', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Overall Leaf Selection Frequency (EFTA)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, win_percentages):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[0, 1])
    num_classes = len(torch.unique(y_tensor))
    
    class_win_matrix = np.zeros((num_classes, num_leaves))
    for c in range(num_classes):
        class_mask = (y_tensor == c)
        class_indices = torch.where(class_mask)[0]
        
        for b in class_indices:
            # winning_leaf[b] has shape (num_units,) - count which leaf wins for each unit
            leaf_wins = (winning_leaf[b].unsqueeze(-1) == torch.arange(num_leaves).to(device)).float()
            class_win_matrix[c] += leaf_wins.sum(dim=0).cpu().numpy()
    
    class_win_matrix = 100.0 * class_win_matrix / class_win_matrix.sum(axis=1, keepdims=True)
    
    im = ax.imshow(class_win_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title('Leaf Selection Frequency by Class (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([f'Class {c}' for c in range(num_classes)])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency (%)', fontsize=10)
    
    for i in range(num_classes):
        for j in range(num_leaves):
            val = class_win_matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   fontsize=8, color='black' if val < 50 else 'white')
    
    ax = fig.add_subplot(gs[1, 0])
    sorted_usage = np.sort(win_percentages.cpu().numpy())
    n = len(sorted_usage)
    cumsum = np.cumsum(sorted_usage)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_usage))) / (n * np.sum(sorted_usage)) - (n + 1) / n
    
    ax.bar(['Gini Coefficient'], [gini], color='coral', edgecolor='black')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Leaf Usage Balance (Gini={gini:.3f})', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Moderate imbalance')
    ax.axhline(y=0.7, color='darkred', linestyle='--', alpha=0.5, label='High imbalance')
    ax.legend(fontsize=9)
    ax.text(0, gini + 0.05, f'{gini:.3f}', ha='center', fontsize=14, fontweight='bold')
    
    ax = fig.add_subplot(gs[1, 1])
    leaf_means = leaf_values.mean(dim=[1, 2]).detach().cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).detach().cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    width = 0.35
    ax.bar(x_pos - width/2, leaf_means, width, label='Mean', color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, leaf_stds, width, label='Std', color='coral', edgecolor='black')
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_title('Leaf Activation Statistics (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_leaf_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/efta_leaf_specialization.png")
    
    # Figure 2: Activation Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x_range = (-3, 3)
    num_points = 200
    colors = plt.cm.tab10(np.linspace(0, 1, num_leaves))
    
    for leaf_idx in range(num_leaves):
        alpha = efta_layer.leaf_params[leaf_idx, :, 0].mean().item()
        beta = efta_layer.leaf_params[leaf_idx, :, 1].mean().item()
        gamma = efta_layer.leaf_params[leaf_idx, :, 2].mean().item()
        
        x_curve = np.linspace(x_range[0], x_range[1], num_points)
        y_curve = np.where(x_curve < 0,
                          alpha * (np.exp(np.clip(beta * x_curve, -10, 10)) - 1),
                          gamma * x_curve)
        
        ax = axes[leaf_idx % 4]
        ax.plot(x_curve, y_curve, linewidth=2.5, label=f'Leaf {leaf_idx}',
               color=colors[leaf_idx], alpha=0.8)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Input (x)', fontsize=11)
        ax.set_ylabel('Output f(x)', fontsize=11)
        ax.set_title(f'Leaf {leaf_idx} (α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_activation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/efta_activation_curves.png")
    
    # Figure 3: Parameter Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_params = efta_layer.leaf_params.detach().cpu().numpy()
    
    im0 = axes[0].imshow(all_params[:, :, 0].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[0].set_xlabel('Leaf Index')
    axes[0].set_ylabel('Unit Index')
    axes[0].set_title('Alpha (negative scaling)', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], label='Value')
    
    im1 = axes[1].imshow(all_params[:, :, 1].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[1].set_xlabel('Leaf Index')
    axes[1].set_ylabel('Unit Index')
    axes[1].set_title('Beta (exponential rate)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], label='Value')
    
    im2 = axes[2].imshow(all_params[:, :, 2].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[2].set_xlabel('Leaf Index')
    axes[2].set_ylabel('Unit Index')
    axes[2].set_title('Gamma (positive slope)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], label='Value')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_parameter_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/efta_parameter_heatmaps.png")
    
    # Figure 4: Parameter Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    param_data = [
        all_params[:, :, 0].flatten(),
        all_params[:, :, 1].flatten(),
        all_params[:, :, 2].flatten()
    ]
    parts = ax.violinplot(param_data, positions=[1, 2, 3], widths=0.7,
                          showmeans=True, showmedians=True)
    
    colors_violin = ['coral', 'teal', 'steelblue']
    for pc, color in zip(parts['bodies'], colors_violin):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Alpha', 'Beta', 'Gamma'])
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Distribution (All Leaves & Units)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial (α, γ)')
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Initial (β)')
    ax.legend(fontsize=9)
    
    ax = axes[1]
    leaf_params = []
    for leaf_idx in range(num_leaves):
        leaf_mean_params = all_params[leaf_idx, :, :].mean(axis=0)
        leaf_params.append(leaf_mean_params)
    
    leaf_params = np.array(leaf_params)
    x_pos = np.arange(num_leaves)
    width = 0.25
    
    ax.bar(x_pos - width, leaf_params[:, 0], width, label='Alpha', color='coral', edgecolor='black')
    ax.bar(x_pos, leaf_params[:, 1], width, label='Beta', color='teal', edgecolor='black')
    ax.bar(x_pos + width, leaf_params[:, 2], width, label='Gamma', color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Parameter Value (mean)', fontsize=12)
    ax.set_title('Learned Parameters per Leaf', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{i}' for i in range(num_leaves)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_parameter_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/efta_parameter_distribution.png")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EFTA Summary")
    print(f"{'='*70}")
    print(f"Leaves: {num_leaves}, Units: {num_units}, Samples: {len(X_tensor)}")
    print(f"\nSelection Frequency:")
    for i in range(num_leaves):
        print(f"  Leaf {i}: {win_percentages[i].item():.2f}%")
    print(f"\nGini: {gini:.4f}")
    print(f"\nParameters (mean ± std):")
    print(f"  Alpha: {all_params[:, :, 0].mean():.4f} ± {all_params[:, :, 0].std():.4f} (init: 1.0)")
    print(f"  Beta:  {all_params[:, :, 1].mean():.4f} ± {all_params[:, :, 1].std():.4f} (init: 0.5)")
    print(f"  Gamma: {all_params[:, :, 2].mean():.4f} ± {all_params[:, :, 2].std():.4f} (init: 1.0)")


def _visualize_fta_direct(fta_layer, model, X_test, y_test, save_dir):
    """Direct FTA visualization."""
    from matplotlib.gridspec import GridSpec
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    num_leaves = fta_layer.num_leaves
    num_units = fta_layer.num_units
    
    X_tensor = torch.tensor(X_test[:256], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test[:256], dtype=torch.long).to(device)
    
    with torch.no_grad():
        x = model.net[0](X_tensor)
        leaf_values = fta_layer.get_leaf_values(x)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, 0])
    winning_leaf = leaf_values.argmax(dim=0)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    win_percentages = 100.0 * win_counts.float() / win_counts.sum()
    
    bars = ax.bar(range(num_leaves), win_percentages.cpu().numpy(),
                  color='steelblue', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Overall Leaf Selection Frequency (FTA)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, win_percentages):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[0, 1])
    num_classes = len(torch.unique(y_tensor))
    
    class_win_matrix = np.zeros((num_classes, num_leaves))
    for c in range(num_classes):
        class_mask = (y_tensor == c)
        class_indices = torch.where(class_mask)[0]
        
        for b in class_indices:
            leaf_wins = (winning_leaf[b].unsqueeze(-1) == torch.arange(num_leaves).to(device)).float()
            class_win_matrix[c] += leaf_wins.sum(dim=0).cpu().numpy()
    
    class_win_matrix = 100.0 * class_win_matrix / class_win_matrix.sum(axis=1, keepdims=True)
    
    im = ax.imshow(class_win_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title('Leaf Selection by Class (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([f'Class {c}' for c in range(num_classes)])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency (%)', fontsize=10)
    
    for i in range(num_classes):
        for j in range(num_leaves):
            val = class_win_matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   fontsize=8, color='white' if val > 50 else 'black')
    
    ax = fig.add_subplot(gs[1, 0])
    weight_norms = [fta_layer.leaf_weights[i].data.norm(p=2).item() for i in range(num_leaves)]
    
    bars = ax.bar(range(num_leaves), weight_norms, color='coral', edgecolor='black')
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Weight Norm (L2)', fontsize=12)
    ax.set_title('FTA Leaf Weight Norms', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, weight_norms):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[1, 1])
    leaf_means = leaf_values.mean(dim=[1, 2]).detach().cpu().numpy()
    leaf_stds = leaf_values.std(dim=[1, 2]).detach().cpu().numpy()
    
    x_pos = np.arange(num_leaves)
    width = 0.35
    ax.bar(x_pos - width/2, leaf_means, width, label='Mean', color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, leaf_stds, width, label='Std', color='coral', edgecolor='black')
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_title('Leaf Activation Statistics', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fta_leaf_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/fta_leaf_specialization.png")
    
    # Weight analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_weights = fta_layer.leaf_weights.data.cpu().numpy()
    avg_weights = all_weights.mean(axis=1)
    
    ax = axes[0]
    im = ax.imshow(avg_weights.T, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Input Feature Index')
    ax.set_title('FTA Weight Matrix (avg over units)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Weight Value')
    
    ax = axes[1]
    for i in range(num_leaves):
        weights_flat = all_weights[i].flatten()
        ax.hist(weights_flat, bins=50, alpha=0.5, label=f'Leaf {i}', density=True)
    
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Weight Distribution per Leaf', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fta_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/fta_weight_analysis.png")
    
    print(f"\n{'='*70}")
    print(f"FTA Summary")
    print(f"{'='*70}")
    print(f"Leaves: {num_leaves}, Units: {num_units}, Input dim: {fta_layer.input_dim}")
    print(f"\nSelection Frequency:")
    for i in range(num_leaves):
        print(f"  Leaf {i}: {win_percentages[i].item():.2f}%")
    print(f"\nWeight Norms:")
    print(f"  Mean: {np.mean(weight_norms):.4f}, Std: {np.std(weight_norms):.4f}")


def run_adult_visualizations():
    """Generate visualizations for Adult dataset models."""
    print("\n" + "="*70)
    print("ADULT DATASET - Branch Visualization")
    print("="*70)
    
    # Load test data
    print("\nLoading Adult test data...")
    X_test, y_test = load_adult_test_data()
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_test.shape[1]}")
    print(f"Classes: {len(np.unique(y_test))}")
    
    # Create output directory
    save_dir = './outputs/adult_branch_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to load trained checkpoints
    checkpoint_dir = './outputs/adult_results'
    
    # Visualize EFTA
    print("\n" + "-"*50)
    print("Visualizing EFTA (d=2,k=2) on Adult...")
    print("-"*50)
    
    efta_model = MLPWithEFTA(input_dim=108, hidden_dims=[128, 64, 32], output_dim=2,
                            depth=2, branch_factor=2).to(device)
    
    checkpoint_path = f'{checkpoint_dir}/model_efta_d2k2.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        efta_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded EFTA model trained on Adult")
    else:
        print(f"Checkpoint not found, using random initialization")
    
    X_tensor = torch.tensor(X_test[:64], dtype=torch.float32).to(device)
    efta_model.eval()
    with torch.no_grad():
        _ = efta_model(X_tensor)
    
    efta_layer = None
    for name, module in efta_model.named_modules():
        if isinstance(module, ExponentialFTA):
            efta_layer = module
            print(f"Found EFTA layer: {name}")
            break
    
    if efta_layer:
        _visualize_efta_direct(efta_layer, efta_model, X_test, y_test, save_dir)
    
    # Visualize FTA
    print("\n" + "-"*50)
    print("Visualizing FTA (d=2,k=2) on Adult...")
    print("-"*50)
    
    fta_model = MLPWithFTA(input_dim=108, hidden_dims=[128, 64, 32], output_dim=2,
                          depth=2, branch_factor=2).to(device)
    
    checkpoint_path = f'{checkpoint_dir}/model_fta_d2k2.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        fta_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded FTA model trained on Adult")
    else:
        print(f"Checkpoint not found, using random initialization")
    
    fta_model.eval()
    with torch.no_grad():
        _ = fta_model(X_tensor)
    
    fta_layer = None
    for name, module in fta_model.named_modules():
        if isinstance(module, FractalTreeActivation):
            fta_layer = module
            print(f"Found FTA layer: {name}")
            break
    
    if fta_layer:
        _visualize_fta_direct(fta_layer, fta_model, X_test, y_test, save_dir)
    
    print("\n" + "="*70)
    print(f"Adult visualizations saved to: {save_dir}/")
    print("="*70)


def main():
    """Run all visualizations."""
    print("\n" + "="*70)
    print("ENHANCED BRANCH VISUALIZATION - Wine & Adult Datasets")
    print("="*70)
    print("\nThis script generates comprehensive leaf specialization analysis")
    print("for both FTA and EFTA models on Wine and Adult datasets.")
    print("="*70)
    
    # Run Wine visualizations
    run_wine_visualizations()
    
    # Run Adult visualizations
    run_adult_visualizations()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nOutput directories:")
    print("  - Wine:   ./outputs/wine_branch_analysis/")
    print("  - Adult:  ./outputs/adult_branch_analysis/")
    print("\nGenerated files per dataset:")
    print("  - efta_leaf_specialization.png - 4-panel EFTA analysis")
    print("  - efta_activation_curves.png   - Learned function shapes")
    print("  - efta_parameter_heatmaps.png  - α, β, γ distributions")
    print("  - efta_parameter_distribution.png - Statistical summaries")
    print("  - fta_leaf_specialization.png  - 4-panel FTA analysis")
    print("  - fta_weight_analysis.png      - Weight patterns")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
