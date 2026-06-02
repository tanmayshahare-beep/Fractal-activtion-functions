"""
FTA vs EFTA Comparison on Tiny ImageNet

Compares Fractal Tree Activation (FTA) against Exponential Fractal Tree Activation (EFTA)
on the Tiny ImageNet dataset using a consistent CNN architecture.

Tiny ImageNet:
- 200 classes
- 64x64 images
- 100,000 training images
- 10,000 validation images

Much faster to train than full ImageNet while maintaining similar characteristics.
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
import gc

# =============================================================================
# GPU Memory Management
# =============================================================================

def setup_gpu():
    """Setup GPU with memory limits."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        
        print(f"\n{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memory Limit: 85% ({torch.cuda.get_device_properties(0).total_memory * 0.85 / 1e9:.2f} GB)")
        print(f"{'='*70}\n")
        return device
    else:
        print("⚠ CUDA not available, using CPU")
        return torch.device('cpu')


def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# =============================================================================
# Activation Function Implementations
# =============================================================================

class FractalTreeActivation(nn.Module):
    """Fractal Tree Activation (FTA) with linear branches."""
    def __init__(self, num_units, depth=2, branch_factor=2):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth

        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_units, num_units))
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

        nn.init.kaiming_normal_(self.leaf_weights, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        batch_size = x.shape[0]

        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)

            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x_flat, self.leaf_weights[i], self.leaf_biases[i])
                leaf_outputs.append(leaf_val)

            stacked = torch.stack(leaf_outputs, dim=0)

            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]

            output = current.view(b, h, w, self.num_units).permute(0, 3, 1, 2).contiguous()
            return output
        else:
            leaf_outputs = []
            for i in range(self.num_leaves):
                leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
                leaf_outputs.append(leaf_val)

            stacked = torch.stack(leaf_outputs, dim=0)

            current = stacked
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = current.view(num_nodes, self.branch_factor, -1, self.num_units)
                current = current.max(dim=1)[0]

            return current.squeeze(0)


class ExponentialFTA(nn.Module):
    """Exponential Fractal Tree Activation (EFTA)."""
    def __init__(self, num_units, depth=2, branch_factor=2):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth

        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))

        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)
        nn.init.constant_(self.leaf_params[:, :, 1], 1.0)
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)

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

            output = current.view(b, h, w, self.num_units).permute(0, 3, 1, 2).contiguous()
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


# =============================================================================
# CNN Architecture for Tiny ImageNet (Constant across FTA/EFTA)
# =============================================================================

class TinyImageNetCNNFTA(nn.Module):
    """4-Block CNN with Fractal Tree Activation for Tiny ImageNet."""
    def __init__(self, depth=2, branch_factor=2, num_classes=200):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        self.fta1 = FractalTreeActivation(64, depth, branch_factor)
        self.fta2 = FractalTreeActivation(128, depth, branch_factor)
        self.fta3 = FractalTreeActivation(256, depth, branch_factor)
        self.fta4 = FractalTreeActivation(512, depth, branch_factor)

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout4 = nn.Dropout(0.25)

        # Classifier
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.fta3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.fta4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class TinyImageNetCNNEFTA(nn.Module):
    """4-Block CNN with Exponential Fractal Tree Activation for Tiny ImageNet."""
    def __init__(self, depth=2, branch_factor=2, num_classes=200):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        self.efta1 = ExponentialFTA(64, depth, branch_factor)
        self.efta2 = ExponentialFTA(128, depth, branch_factor)
        self.efta3 = ExponentialFTA(256, depth, branch_factor)
        self.efta4 = ExponentialFTA(512, depth, branch_factor)

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout4 = nn.Dropout(0.25)

        # Classifier
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.efta4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_name(model_type, depth, branch_factor):
    """Generate descriptive model name."""
    if model_type == 'fta':
        return f'FTA (d={depth}, k={branch_factor})'
    elif model_type == 'efta':
        return f'EFTA (d={depth}, k={branch_factor})'
    else:
        return model_type


def create_model(model_type, depth, branch_factor, device, num_classes=200):
    """Create model based on type."""
    if model_type == 'fta':
        model = TinyImageNetCNNFTA(depth=depth, branch_factor=branch_factor, num_classes=num_classes)
    elif model_type == 'efta':
        model = TinyImageNetCNNEFTA(depth=depth, branch_factor=branch_factor, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

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
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, device,
                epochs=30, model_name="Model"):
    """Train and evaluate model with memory monitoring."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()

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

        # Monitor GPU memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e6
            mem_reserved = torch.cuda.memory_reserved() / 1e6
        else:
            mem_allocated = mem_reserved = 0

        print(f"  Epoch {epoch+1:2d}/{epochs}: "
              f"Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
              f"GPU Mem={mem_allocated:.0f}MB")

        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final validation evaluation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    total_time = time.time() - start_time
    params = count_parameters(model)

    return {
        'name': model_name,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'best_val_acc': best_val_acc,
        'params': params,
        'epochs_trained': len(history['train_loss']),
        'training_time': total_time,
        'history': history
    }


# =============================================================================
# Data Loading
# =============================================================================

def get_data_loaders(data_dir, batch_size=128, num_workers=0):
    """Get train/val data loaders for Tiny ImageNet."""
    
    print(f"\nLoading Tiny ImageNet from: {data_dir}")

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Tiny ImageNet train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Tiny ImageNet val directory not found: {val_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(data_dir, epochs=30, batch_size=128):
    """Run FTA vs EFTA comparison experiment on Tiny ImageNet."""
    print(f"\n{'='*70}")
    print("FTA VS EFTA COMPARISON ON TINY IMAGENET")
    print(f"{'='*70}\n")

    # Setup GPU
    device = setup_gpu()
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)

    # Model configurations
    configurations = [
        # FTA variants
        ('fta', 2, 2),
        ('fta', 3, 2),
        ('fta', 2, 3),
        ('fta', 3, 3),

        # EFTA variants
        ('efta', 2, 2),
        ('efta', 3, 2),
        ('efta', 2, 3),
        ('efta', 3, 3),
    ]

    results = {}

    for idx, (model_type, depth, branch_factor) in enumerate(configurations):
        model_name = get_model_name(model_type, depth, branch_factor)
        print(f"\n{'='*60}")
        print(f"Training: {model_name} ({idx+1}/{len(configurations)})")
        print(f"{'='*60}")

        # Clear memory before each model
        clear_memory()
        time.sleep(2)

        try:
            model = create_model(model_type, depth, branch_factor, device, num_classes=200)
            print(f"  Type: {model_type.upper()}, Depth: {depth}, Branch: {branch_factor}")
            print(f"  Parameters: {count_parameters(model):,}")

            result = train_model(
                model, train_loader, val_loader, device,
                epochs=epochs, model_name=model_name
            )
            results[model_name] = result

            print(f"\n  ✓ Completed:")
            print(f"    Best Val Acc:  {result['best_val_acc']:.2f}%")
            print(f"    Final Val Acc: {result['val_acc']:.2f}%")
            print(f"    Parameters:    {result['params']:,}")
            print(f"    Training Time: {result['training_time']/60:.1f} min")

            # Clear memory after each model
            del model
            clear_memory()
            time.sleep(2)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            clear_memory()
            
            results[model_name] = {
                'name': model_name,
                'error': str(e)
            }

    return results


def print_results_table(results):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print("TINY IMAGENET RESULTS: FTA VS EFTA")
    print(f"{'='*90}")
    print(f"{'Model':<25} | {'Val Acc':<10} | {'Best Val':<10} | {'Params':<12} | {'Time (min)':<10}")
    print("-"*90)

    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<25} | {'FAILED':<10} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10}")
        else:
            time_min = result['training_time'] / 60
            print(f"{name:<25} | {result['val_acc']:>8.2f}% | {result['best_val_acc']:>8.2f}% | "
                  f"{result['params']:>10,} | {time_min:>8.1f}")

    print("="*90)

    # Filter successful results
    successful = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful:
        best_overall = max(successful, key=lambda x: successful[x]['best_val_acc'])
        print(f"\n🏆 Best Model: {best_overall} ({successful[best_overall]['best_val_acc']:.2f}%)")

        fta_models = {k: v for k, v in successful.items() if 'FTA' in k and 'EFTA' not in k}
        efta_models = {k: v for k, v in successful.items() if 'EFTA' in k}

        if fta_models:
            best_fta = max(fta_models, key=lambda x: fta_models[x]['best_val_acc'])
            print(f"🥇 Best FTA:  {best_fta} ({fta_models[best_fta]['best_val_acc']:.2f}%)")

        if efta_models:
            best_efta = max(efta_models, key=lambda x: efta_models[x]['best_val_acc'])
            print(f"🥇 Best EFTA: {best_efta} ({efta_models[best_efta]['best_val_acc']:.2f}%)")

        print(f"\n{'='*50}")
        print("FTA VS EFTA HEAD-TO-HEAD:")
        print(f"{'='*50}")

        configs = [(2,2), (3,2), (2,3), (3,3)]
        for d, k in configs:
            fta_name = f'FTA (d={d}, k={k})'
            efta_name = f'EFTA (d={d}, k={k})'

            if fta_name in successful and efta_name in successful:
                fta_acc = successful[fta_name]['best_val_acc']
                efta_acc = successful[efta_name]['best_val_acc']
                diff = efta_acc - fta_acc
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  (d={d}, k={k}): FTA={fta_acc:.2f}% vs EFTA={efta_acc:.2f}% {symbol} {abs(diff):.2f}%")

    print("="*90)


def save_results(results, output_dir='./outputs/results'):
    """Save results to file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'tinyimagenet_fta_vs_efta_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)

    save_data = {}
    
    for name, result in results.items():
        if 'error' not in result:
            save_data[name] = {
                'val_acc': result['val_acc'],
                'val_loss': result['val_loss'],
                'best_val_acc': result['best_val_acc'],
                'params': result['params'],
                'epochs_trained': result['epochs_trained'],
                'training_time': result['training_time']
            }
        else:
            save_data[name] = {'error': result['error']}

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n📁 Results saved to: {filepath}")
    return filepath


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FTA vs EFTA on Tiny ImageNet')
    parser.add_argument('--data_dir', type=str, default='./datasets/tiny-imagenet-200',
                        help='Path to Tiny ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("FTA VS EFTA ON TINY IMAGENET")
    print(f"{'='*70}")
    print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nDataset: Tiny ImageNet (200 classes, 64x64 images, 100k train)")
    print(f"\nConfiguration:")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"\nModel Configurations:")
    print(f"  - FTA:  (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"  - EFTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"\nArchitecture: 4-block CNN (64→128→256→512) + BatchNorm + Dropout")
    print(f"{'='*70}")
    
    # Verify data directory
    if not os.path.exists(args.data_dir):
        print(f"\n⚠ Tiny ImageNet directory not found: {args.data_dir}")
        print("\n📥 Download Tiny ImageNet:")
        print("   https://tiny-imagenet.herokuapp.com/")
        print("   Or: https://cs.stanford.edu/~karpathy/convnetjs/demo/tinyimagenet.html")
        print("\nExpected structure:")
        print("   tiny-imagenet-200/")
        print("   ├── train/")
        print("   │   ├── n01443537/")
        print("   │   └── ...")
        print("   ├── val/")
        print("   │   ├── images/")
        print("   │   └── val_annotations.txt")
        print("   └── wnids.txt")
        print("\nTo use a different directory:")
        print("  python tinyimagenet_fta_vs_efta.py --data_dir /path/to/tiny-imagenet")
        return
    
    # Run experiment
    results = run_experiment(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Print results
    print_results_table(results)

    # Save results
    save_results(results)

    # Final memory cleanup
    clear_memory()

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*70}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
