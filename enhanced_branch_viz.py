"""
Enhanced FTA/EFTA Branch Visualization

Provides detailed analysis of leaf specialization including:
1. Leaf selection frequency per class
2. Learned activation function curves
3. Input-output correlation analysis
4. Parameter distribution heatmaps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os


# =============================================================================
# FTA/EFTA Implementations (for reference)
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
    
    def get_leaf_function_curve(self, leaf_idx, x_range=(-3, 3), num_points=100):
        """Get the activation curve for a specific leaf."""
        alpha = self.leaf_params[leaf_idx, :, 0].mean().item()
        beta = self.leaf_params[leaf_idx, :, 1].mean().item()
        gamma = self.leaf_params[leaf_idx, :, 2].mean().item()
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.where(x < 0, 
                     alpha * (np.exp(np.clip(beta * x, -10, 10)) - 1),
                     gamma * x)
        return x, y, (alpha, beta, gamma)


# =============================================================================
# Enhanced Visualization Functions
# =============================================================================

def visualize_efta_leaf_specialization(model, X_test, y_test, save_dir='./outputs/efta_analysis'):
    """
    Comprehensive EFTA leaf specialization analysis.
    
    Args:
        model: Trained MLPWithEFTA model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Find EFTA layer
    efta_layer = None
    for module in model.modules():
        if isinstance(module, ExponentialFTA):
            efta_layer = module
            break
    
    if efta_layer is None:
        print("No EFTA layer found in model")
        return
    
    num_leaves = efta_layer.num_leaves
    num_units = efta_layer.num_units
    
    # Convert test data to tensor
    X_tensor = torch.tensor(X_test[:256], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test[:256], dtype=torch.long).to(device)
    
    # Get leaf values through the network
    with torch.no_grad():
        # Forward through first linear layer
        x = model.net[0](X_tensor)  # (batch, hidden_dim)
        leaf_values = efta_layer.get_leaf_values(x)  # (num_leaves, batch, num_units)
    
    # =========================================================
    # Figure 1: Leaf Selection Frequency (Overall and Per Class)
    # =========================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1a: Overall leaf selection frequency
    ax = fig.add_subplot(gs[0, 0])
    winning_leaf = leaf_values.argmax(dim=0)  # (batch, num_units)
    win_counts = torch.bincount(winning_leaf.flatten(), minlength=num_leaves)
    win_percentages = 100.0 * win_counts.float() / win_counts.sum()
    
    bars = ax.bar(range(num_leaves), win_percentages.cpu().numpy(), 
                  color='mediumseagreen', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Overall Leaf Selection Frequency', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, win_percentages):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 1b: Leaf selection per class
    ax = fig.add_subplot(gs[0, 1])
    num_classes = len(torch.unique(y_tensor))
    
    class_win_matrix = np.zeros((num_classes, num_leaves))
    for c in range(num_classes):
        class_mask = (y_tensor == c)
        for b in range(len(class_mask)):
            if class_mask[b]:
                leaf_wins = (winning_leaf[b] == torch.arange(num_leaves).to(device)).float()
                class_win_matrix[c] += leaf_wins.cpu().numpy()
    
    # Normalize to percentages
    class_win_matrix = 100.0 * class_win_matrix / class_win_matrix.sum(axis=1, keepdims=True)
    
    im = ax.imshow(class_win_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title('Leaf Selection Frequency by Class (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(num_leaves))
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([f'Class {c}' for c in range(num_classes)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency (%)', fontsize=10)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_leaves):
            val = class_win_matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   fontsize=8, color='black' if val < 50 else 'white')
    
    # Plot 1c: Leaf usage balance (Gini coefficient)
    ax = fig.add_subplot(gs[1, 0])
    
    # Calculate Gini coefficient for leaf usage
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
    
    # Plot 1d: Activation statistics per leaf
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
    
    plt.savefig(f'{save_dir}/efta_leaf_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved leaf specialization analysis to: {save_dir}/efta_leaf_specialization.png")
    
    # =========================================================
    # Figure 2: Learned Activation Function Curves
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x_range = (-3, 3)
    num_points = 200
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_leaves))
    
    for leaf_idx in range(num_leaves):
        x_curve, y_curve, params = efta_layer.get_leaf_function_curve(leaf_idx, x_range, num_points)
        alpha, beta, gamma = params
        ax = axes[leaf_idx % 4]
        ax.plot(x_curve, y_curve, linewidth=2.5, label=f'Leaf {leaf_idx}', 
               color=colors[leaf_idx], alpha=0.8)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Input (x)', fontsize=11)
        ax.set_ylabel('Output f(x)', fontsize=11)
        ax.set_title(f'Leaf {leaf_idx} Activation Function\n(α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f})', 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_activation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved activation curves to: {save_dir}/efta_activation_curves.png")
    
    # =========================================================
    # Figure 3: Parameter Distribution Heatmaps
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_params = efta_layer.leaf_params.detach().cpu().numpy()  # (num_leaves, num_units, 3)
    
    # Alpha heatmap
    im0 = axes[0].imshow(all_params[:, :, 0].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[0].set_xlabel('Leaf Index')
    axes[0].set_ylabel('Unit Index')
    axes[0].set_title('Alpha Parameters (negative scaling)', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], label='Value')
    
    # Beta heatmap
    im1 = axes[1].imshow(all_params[:, :, 1].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[1].set_xlabel('Leaf Index')
    axes[1].set_ylabel('Unit Index')
    axes[1].set_title('Beta Parameters (exponential rate)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], label='Value')
    
    # Gamma heatmap
    im2 = axes[2].imshow(all_params[:, :, 2].T, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    axes[2].set_xlabel('Leaf Index')
    axes[2].set_ylabel('Unit Index')
    axes[2].set_title('Gamma Parameters (positive slope)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], label='Value')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_parameter_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter heatmaps to: {save_dir}/efta_parameter_heatmaps.png")
    
    # =========================================================
    # Figure 4: Parameter Distribution (Violin + Box)
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Violin plot
    ax = axes[0]
    param_data = [
        all_params[:, :, 0].flatten(),  # Alpha
        all_params[:, :, 1].flatten(),  # Beta
        all_params[:, :, 2].flatten()   # Gamma
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
    
    # Add initial value lines
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial (α, γ)')
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Initial (β)')
    ax.legend(fontsize=9)
    
    # Box plot per leaf
    ax = axes[1]
    leaf_params = []
    leaf_labels = []
    for leaf_idx in range(num_leaves):
        leaf_mean_params = all_params[leaf_idx, :, :].mean(axis=0)  # (3,)
        leaf_params.append(leaf_mean_params)
        leaf_labels.append(f'L{leaf_idx}')
    
    leaf_params = np.array(leaf_params)  # (num_leaves, 3)
    x_pos = np.arange(num_leaves)
    width = 0.25
    
    ax.bar(x_pos - width, leaf_params[:, 0], width, label='Alpha', color='coral', edgecolor='black')
    ax.bar(x_pos, leaf_params[:, 1], width, label='Beta', color='teal', edgecolor='black')
    ax.bar(x_pos + width, leaf_params[:, 2], width, label='Gamma', color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Leaf Index', fontsize=12)
    ax.set_ylabel('Parameter Value (mean ± std)', fontsize=12)
    ax.set_title('Learned Parameters per Leaf (mean over units)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(leaf_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efta_parameter_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter distribution to: {save_dir}/efta_parameter_distribution.png")
    
    # =========================================================
    # Print Summary Statistics
    # =========================================================
    print(f"\n{'='*70}")
    print(f"EFTA Leaf Specialization Summary")
    print(f"{'='*70}")
    print(f"Number of leaves: {num_leaves}")
    print(f"Number of units: {num_units}")
    print(f"Samples analyzed: {len(X_tensor)}")
    print(f"\nLeaf Selection Frequency:")
    for i in range(num_leaves):
        print(f"  Leaf {i}: {win_percentages[i].item():.2f}%")
    print(f"\nGini Coefficient: {gini:.4f} (0=perfect balance, 1=complete imbalance)")
    print(f"\nParameter Statistics (mean ± std):")
    print(f"  Alpha: {all_params[:, :, 0].mean():.4f} ± {all_params[:, :, 0].std():.4f} (init: 1.0)")
    print(f"  Beta:  {all_params[:, :, 1].mean():.4f} ± {all_params[:, :, 1].std():.4f} (init: 0.5)")
    print(f"  Gamma: {all_params[:, :, 2].mean():.4f} ± {all_params[:, :, 2].std():.4f} (init: 1.0)")
    print(f"{'='*70}\n")


def visualize_fta_leaf_specialization(model, X_test, y_test, save_dir='./outputs/fta_analysis'):
    """
    Comprehensive FTA leaf specialization analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Find FTA layer
    fta_layer = None
    for module in model.modules():
        if isinstance(module, FractalTreeActivation):
            fta_layer = module
            break
    
    if fta_layer is None:
        print("No FTA layer found in model")
        return
    
    num_leaves = fta_layer.num_leaves
    num_units = fta_layer.num_units
    
    X_tensor = torch.tensor(X_test[:256], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test[:256], dtype=torch.long).to(device)
    
    with torch.no_grad():
        x = model.net[0](X_tensor)
        leaf_values = fta_layer.get_leaf_values(x)
    
    # =========================================================
    # Figure 1: Leaf Selection and Weight Analysis
    # =========================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1a: Overall leaf selection frequency
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
    
    # Plot 1b: Leaf selection per class
    ax = fig.add_subplot(gs[0, 1])
    num_classes = len(torch.unique(y_tensor))
    
    class_win_matrix = np.zeros((num_classes, num_leaves))
    for c in range(num_classes):
        class_mask = (y_tensor == c)
        for b in range(len(class_mask)):
            if class_mask[b]:
                leaf_wins = (winning_leaf[b] == torch.arange(num_leaves).to(device)).float()
                class_win_matrix[c] += leaf_wins.cpu().numpy()
    
    class_win_matrix = 100.0 * class_win_matrix / class_win_matrix.sum(axis=1, keepdims=True)
    
    im = ax.imshow(class_win_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
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
                   fontsize=8, color='white' if val > 50 else 'black')
    
    # Plot 1c: Weight norms per leaf
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
    
    # Plot 1d: Activation statistics
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
    plt.savefig(f'{save_dir}/fta_leaf_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved FTA leaf specialization to: {save_dir}/fta_leaf_specialization.png")
    
    # =========================================================
    # Figure 2: Weight Matrix Heatmap
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full weight matrix
    ax = axes[0]
    all_weights = fta_layer.leaf_weights.data.cpu().numpy()  # (num_leaves, num_units, input_dim)
    # Reshape for visualization: average over units
    avg_weights = all_weights.mean(axis=1)  # (num_leaves, input_dim)
    
    im = ax.imshow(avg_weights.T, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Input Feature Index')
    ax.set_title('FTA Weight Matrix (averaged over units)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Weight Value')
    
    # Weight distribution per leaf
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
    print(f"Saved FTA weight analysis to: {save_dir}/fta_weight_analysis.png")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"FTA Leaf Specialization Summary")
    print(f"{'='*70}")
    print(f"Number of leaves: {num_leaves}")
    print(f"Number of units: {num_units}")
    print(f"Input dimension: {fta_layer.input_dim}")
    print(f"\nLeaf Selection Frequency:")
    for i in range(num_leaves):
        print(f"  Leaf {i}: {win_percentages[i].item():.2f}%")
    print(f"\nWeight Norm Statistics:")
    print(f"  Mean: {np.mean(weight_norms):.4f}")
    print(f"  Std:  {np.std(weight_norms):.4f}")
    print(f"{'='*70}\n")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    print("Enhanced FTA/EFTA Branch Visualization Module")
    print("="*70)
    print("\nThis module provides detailed leaf specialization analysis.")
    print("\nUsage example:")
    print("""
    # After training your model:
    from enhanced_branch_viz import visualize_efta_leaf_specialization
    
    visualize_efta_leaf_specialization(
        model=model,
        X_test=X_test,
        y_test=y_test,
        save_dir='./outputs/efta_analysis'
    )
    """)
    print("="*70)
