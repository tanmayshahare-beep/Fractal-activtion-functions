"""
Branch Visualization - What do FTA/EFTA leaves learn? (PyTorch)

Feed digit images and trace which leaves activate most strongly.
Analyze if different branches specialize in different features.

Usage:
    python experiments/advanced/03_branch_visualization.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.activations import FractalTreeActivation
from src.utils import load_mnist_local

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


class FTAWithLeafAccess(nn.Module):
    """FTA wrapper that allows accessing leaf values."""
    def __init__(self, fta_layer):
        super().__init__()
        self.fta = fta_layer

    def forward(self, x):
        return self.fta(x)

    def get_leaf_values(self, x):
        """
        Get leaf values before max pooling for analysis.
        Returns: (num_leaves, batch, num_units)
        """
        return self.fta.get_leaf_values(x)


def create_fta_model(depth=2, branch_factor=2, input_dim=784, num_classes=10):
    """Create FTA model with leaf access."""
    
    class FTAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128, bias=False)
            self.bn1 = nn.BatchNorm1d(128)
            self.fta = FractalTreeActivation(
                num_units=128, depth=depth, branch_factor=branch_factor, input_dim=128)
            self.fc2 = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.fta(x)
            return self.fc2(x)
    
    model = FTAModel()
    return model, model.fta


def analyze_leaf_activation(model, fta_layer, images, labels, num_samples_per_digit=10):
    """
    Analyze which leaves activate most strongly for each digit class.
    
    Args:
        model: Full FTA model
        fta_layer: FTA layer with get_leaf_values method
        images: Numpy array of images (N, 784)
        labels: Numpy array of labels (N,)
        num_samples_per_digit: Number of samples per digit to analyze
    
    Returns:
        digit_leaf_activations: Dict mapping digit -> mean leaf activation
        num_leaves: Number of leaves in FTA
    """
    print("\nAnalyzing leaf activations...")
    
    # Convert to tensor
    images_tensor = torch.FloatTensor(images).to(device)
    
    # First pass through initial layers to get to FTA input
    with torch.no_grad():
        x = model.fc1(images_tensor)
        x = model.bn1(x)
        # Now x is (batch, 128) - the input to FTA
        
        # Get leaf values: (num_leaves, batch, num_units)
        leaf_values = fta_layer.get_leaf_values(x)
        leaf_values = leaf_values.cpu().numpy()
    
    num_leaves = leaf_values.shape[0]
    num_digits = 10
    
    # For each digit, find which leaves activate most strongly
    digit_leaf_activations = {d: [] for d in range(num_digits)}
    
    unique_digits = np.unique(labels)
    for digit in unique_digits:
        digit_mask = labels == digit
        digit_indices = np.where(digit_mask)[0][:num_samples_per_digit]
        
        for idx in digit_indices:
            # Get leaf values for this sample: (num_leaves, num_units)
            sample_leaf_values = leaf_values[:, idx, :]
            # Mean activation across neurons for each leaf
            mean_leaf_activation = np.mean(sample_leaf_values, axis=1)
            digit_leaf_activations[digit].append(mean_leaf_activation)
        
        digit_leaf_activations[digit] = np.mean(digit_leaf_activations[digit], axis=0)
    
    return digit_leaf_activations, num_leaves


def plot_leaf_analysis(digit_leaf_activations, num_leaves, depth, branch_factor):
    """Visualize leaf activation patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    digits = list(digit_leaf_activations.keys())
    
    # 1. Heatmap: digits x leaves
    ax = axes[0, 0]
    heatmap_data = np.array([digit_leaf_activations[d] for d in digits])
    sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Mean Activation'})
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Digit')
    ax.set_title(f'Leaf Activation Patterns by Digit\n(FTA depth={depth}, branch={branch_factor}, {num_leaves} leaves)')
    
    # 2. Top activating leaves per digit
    ax = axes[0, 1]
    top_k = min(5, num_leaves)
    for i, digit in enumerate(digits):
        activations = digit_leaf_activations[digit]
        top_indices = np.argsort(activations)[-top_k:][::-1]
        y_pos = np.arange(len(top_indices))
        ax.scatter(activations[top_indices], y_pos, label=f'Digit {digit}', alpha=0.7, s=50)
    
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Top-K Rank')
    ax.set_title('Top Activating Leaves per Digit')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    
    # 3. Leaf specialization score (variance across digits)
    ax = axes[1, 0]
    all_activations = np.array([digit_leaf_activations[d] for d in digits])
    specialization_score = np.std(all_activations, axis=0)  # High std = specialized leaf
    
    # Find most specialized leaves
    top_specialized = np.argsort(specialization_score)[-10:][::-1]
    ax.bar(range(len(top_specialized)), specialization_score[top_specialized])
    ax.set_xlabel('Specialized Leaf Rank')
    ax.set_ylabel('Specialization Score (std across digits)')
    ax.set_title('Most Specialized Leaves')
    ax.set_xticks(range(len(top_specialized)))
    ax.set_xticklabels([f'Leaf {i}' for i in top_specialized], rotation=45, ha='right')
    
    # 4. Dendrogram-like visualization of tree structure
    ax = axes[1, 1]
    
    # Show tree structure with activation intensity
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_leaves))
    
    # Draw tree levels
    for level in range(depth + 1):
        nodes_at_level = branch_factor ** level
        y = depth - level
        x_positions = np.linspace(0, 1, nodes_at_level + 1)[1:]
        
        for i, x in enumerate(x_positions):
            if level == depth:
                # Leaf node - color by average activation
                leaf_idx = i
                avg_activation = np.mean([digit_leaf_activations[d][leaf_idx] for d in digits])
                color = plt.cm.Reds(0.3 + 0.7 * (avg_activation - np.min([digit_leaf_activations[d] for d in digits])) /
                           (np.max([digit_leaf_activations[d] for d in digits]) - np.min([digit_leaf_activations[d] for d in digits]) + 1e-6))
                ax.plot(x, y, 'o', color=color, markersize=15, markeredgecolor='black')
            else:
                ax.plot(x, y, 'o', color='lightgray', markersize=10, markeredgecolor='black')
            
            # Draw connections to children
            if level < depth:
                children_start = i * branch_factor
                for c in range(branch_factor):
                    child_x = (children_start + c + 1) / (branch_factor ** (level + 1))
                    ax.plot([x, child_x], [y, y - 0.9], 'k-', alpha=0.3)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, depth + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Tree Structure with Leaf Activations\n(Darker = Higher Activation)')
    
    plt.tight_layout()
    
    # Save to outputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(outputs_dir, exist_ok=True)
    
    save_path = os.path.join(outputs_dir, 'leaf_activation_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {save_path}")
    plt.show()


def visualize_specialized_digits(model, fta_layer, test_images, test_labels, num_leaves, top_k=5):
    """Find and visualize images that maximally activate specific leaves."""
    print("\nFinding images that maximally activate specific leaves...")
    
    # Convert to tensor
    images_tensor = torch.FloatTensor(test_images).to(device)
    
    # First pass through initial layers
    with torch.no_grad():
        x = model.fc1(images_tensor)
        x = model.bn1(x)
        # Get all leaf values
        leaf_values = fta_layer.get_leaf_values(x)
        leaf_values = leaf_values.cpu().numpy()  # (num_leaves, batch, num_units)
    
    # For each leaf, find top activating images
    fig, axes = plt.subplots(top_k, min(6, num_leaves), figsize=(15, 3 * top_k))
    
    for leaf_idx in range(min(6, num_leaves)):
        # Mean activation across neurons for this leaf
        leaf_activations = np.mean(leaf_values[leaf_idx, :, :], axis=1)
        top_indices = np.argsort(leaf_activations)[-top_k:][::-1]
        
        for i, img_idx in enumerate(top_indices):
            if top_k > 1:
                ax = axes[i, leaf_idx]
            else:
                ax = axes[leaf_idx]
            
            ax.imshow(test_images[img_idx].reshape(28, 28), cmap='gray')
            digit = test_labels[img_idx]
            activation = leaf_activations[img_idx]
            ax.set_title(f'Digit: {digit}\nActivation: {activation:.2f}', fontsize=9)
            ax.axis('off')
    
    plt.suptitle(f'Top Activating Images for Leaves 0-{min(5, num_leaves-1)}', y=1.02)
    plt.tight_layout()
    
    # Save to outputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(outputs_dir, exist_ok=True)
    
    save_path = os.path.join(outputs_dir, 'specialized_leaf_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Specialized images saved to: {save_path}")
    plt.show()


def run_branch_visualization(dataset_dir):
    """Run branch visualization analysis."""
    print("="*70)
    print("BRANCH VISUALIZATION - What do FTA leaves learn? (PyTorch)")
    print("="*70)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    # Keep as uint8 for now, will convert to tensor later
    print(f"\nLoaded MNIST:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    
    # Normalize for training
    x_train_flat = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test_flat = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Create model
    print("\nCreating FTA model...")
    model, fta_layer = create_fta_model(depth=2, branch_factor=2)
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Split validation
    val_split = int(0.9 * len(x_train_flat))
    x_val, y_val = x_train_flat[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train_flat[:val_split], y_train[:val_split]
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train_sub).to(device)
    y_train_tensor = torch.LongTensor(y_train_sub).to(device)
    x_val_tensor = torch.FloatTensor(x_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Train
    print("\nTraining for 10 epochs...")
    batch_size = 64
    n_batches = len(x_train_sub) // batch_size
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        indices = torch.randperm(len(x_train_sub))
        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            batch_x = x_train_tensor[batch_idx]
            batch_y = y_train_tensor[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            _, val_predicted = val_outputs.max(1)
            val_acc = 100.0 * val_predicted.eq(y_val_tensor).sum().item() / len(y_val)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Analyze leaf activations (use subset for speed)
    print("\nAnalyzing leaf activations on test set...")
    digit_activations, num_leaves = analyze_leaf_activation(
        model, fta_layer, x_test_flat[:500], y_test[:500], num_samples_per_digit=50
    )
    
    print(f"\nNumber of leaves: {num_leaves}")
    print(f"Digits analyzed: {list(digit_activations.keys())}")
    
    # Plot analysis
    plot_leaf_analysis(digit_activations, num_leaves, depth=2, branch_factor=2)
    
    # Visualize specialized images
    visualize_specialized_digits(model, fta_layer, x_test_flat[:500], y_test[:500], num_leaves, top_k=5)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey questions to consider:")
    print("1. Do different leaves specialize in different digit classes?")
    print("2. Are there leaves that activate strongly for similar digits (e.g., 4 and 9)?")
    print("3. What visual features might each leaf be detecting?")
    print("="*70)


if __name__ == '__main__':
    # Get paths - go up TWO levels from experiments/advanced/ to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
    dataset_dir = os.path.join(project_dir, 'datasets', 'MNIST')
    
    print(f"Script directory: {script_dir}")
    print(f"Project directory: {project_dir}")
    print(f"Dataset directory: {dataset_dir}")
    
    run_branch_visualization(dataset_dir)
