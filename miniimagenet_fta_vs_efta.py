"""
FTA vs EFTA Comparison on miniImageNet - DEBUGGED VERSION

Fixes:
- Proper 3-channel input handling
- Deeper architecture for 84x84 images
- Lower learning rate
- Data verification
- Gradient monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        
        print(f"\n{'='*70}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memory Limit: 90% ({torch.cuda.get_device_properties(0).total_memory * 0.9 / 1e9:.2f} GB)")
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


def verify_data_loader(loader, num_classes=100):
    """Verify data loader is working correctly."""
    print("\n" + "="*50)
    print("DATA VERIFICATION")
    print("="*50)
    
    images, labels = next(iter(loader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Image mean: {images.mean():.3f}, std: {images.std():.3f}")
    print(f"Labels range: [{labels.min()}, {labels.max()}]")
    print(f"Unique labels in batch: {labels.unique().sort()[0].tolist()}")
    
    # Check for NaN/Inf
    if torch.isnan(images).any():
        print("⚠ WARNING: NaN values detected in images!")
    if torch.isinf(images).any():
        print("⚠ WARNING: Inf values detected in images!")
    
    print("="*50 + "\n")
    
    return images.shape[1] == 3  # Return True if RGB


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
# CNN Architecture for miniImageNet (4-Block for 84x84 images)
# =============================================================================

class MiniImageNetCNNFTA(nn.Module):
    """4-Block CNN with Fractal Tree Activation for miniImageNet (84x84 RGB)."""
    def __init__(self, depth=2, branch_factor=2, num_classes=100):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        # 4 FTA layers for 4 conv blocks
        self.fta1 = FractalTreeActivation(64, depth, branch_factor)
        self.fta2 = FractalTreeActivation(128, depth, branch_factor)
        self.fta3 = FractalTreeActivation(256, depth, branch_factor)
        self.fta4 = FractalTreeActivation(512, depth, branch_factor)

        # Block 1: 84x84 -> 42x42 (no stride, just pool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 input channels for RGB
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 84 -> 42
        self.dropout1 = nn.Dropout(0.25)

        # Block 2: 42x42 -> 21x21
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 42 -> 21
        self.dropout2 = nn.Dropout(0.25)

        # Block 3: 21x21 -> 10x10
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 21 -> 10
        self.dropout3 = nn.Dropout(0.25)

        # Block 4: 10x10 -> 5x5
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((5, 5))  # -> 5x5
        self.dropout4 = nn.Dropout(0.25)

        # Classifier: 512 * 5 * 5 = 12800
        self.fc1 = nn.Linear(512 * 5 * 5, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1: 84x84 -> 42x42
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fta1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: 42x42 -> 21x21
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fta2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3: 21x21 -> 10x10
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.fta3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4: 10x10 -> 5x5
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.fta4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Classifier
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class MiniImageNetCNNEFTA(nn.Module):
    """4-Block CNN with Exponential Fractal Tree Activation for miniImageNet."""
    def __init__(self, depth=2, branch_factor=2, num_classes=100):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        self.efta1 = ExponentialFTA(64, depth, branch_factor)
        self.efta2 = ExponentialFTA(128, depth, branch_factor)
        self.efta3 = ExponentialFTA(256, depth, branch_factor)
        self.efta4 = ExponentialFTA(512, depth, branch_factor)

        # Block 1: 84x84 -> 42x42
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 84 -> 42
        self.dropout1 = nn.Dropout(0.25)

        # Block 2: 42x42 -> 21x21
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 42 -> 21
        self.dropout2 = nn.Dropout(0.25)

        # Block 3: 21x21 -> 10x10
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 21 -> 10
        self.dropout3 = nn.Dropout(0.25)

        # Block 4: 10x10 -> 5x5
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((5, 5))  # -> 5x5
        self.dropout4 = nn.Dropout(0.25)

        # Classifier: 512 * 5 * 5 = 12800
        self.fc1 = nn.Linear(512 * 5 * 5, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1: 84x84 -> 42x42
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: 42x42 -> 21x21
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3: 21x21 -> 10x10
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4: 10x10 -> 5x5
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.efta4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        # Classifier
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


def create_model(model_type, depth, branch_factor, device, num_classes=100):
    """Create model based on type."""
    if model_type == 'fta':
        model = MiniImageNetCNNFTA(depth=depth, branch_factor=branch_factor, num_classes=num_classes)
    elif model_type == 'efta':
        model = MiniImageNetCNNEFTA(depth=depth, branch_factor=branch_factor, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch with gradient monitoring."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Monitor gradients
        if batch_idx == 0:
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item()
            grad_norms.append(total_grad_norm)
        
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
    
    return total_loss / len(loader), 100.0 * correct / total, avg_grad_norm


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
                epochs=50, model_name="Model"):
    """Train and evaluate model with memory monitoring."""
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 8  # Increased patience

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'grad_norms': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['grad_norms'].append(grad_norm)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Monitor GPU memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e6
        else:
            mem_allocated = 0

        print(f"  Epoch {epoch+1:2d}/{epochs}: "
              f"Train={train_acc:.2f}%, Val={val_acc:.2f}%, "
              f"Grad={grad_norm:.2f}, GPU Mem={mem_allocated:.0f}MB")

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

def get_data_loaders(data_dir, batch_size=64, val_split=0.1, num_workers=0):
    """Get train/val data loaders for miniImageNet."""
    
    print(f"\nLoading miniImageNet from: {data_dir}")

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(84, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    print(f"Total samples: {len(full_dataset):,}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    
    # Split into train/val
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply training transforms to train subset
    train_dataset.dataset.transform = train_transform

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

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

def run_experiment(data_dir, epochs=50, batch_size=64):
    """Run FTA vs EFTA comparison experiment on miniImageNet."""
    print(f"\n{'='*70}")
    print("FTA VS EFTA COMPARISON ON MINI IMAGENET (DEBUGGED)")
    print(f"{'='*70}\n")

    # Setup GPU
    device = setup_gpu()
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    # Verify data
    is_rgb = verify_data_loader(train_loader, num_classes=100)
    if not is_rgb:
        print("⚠ WARNING: Data may not be RGB! Check your dataset.")

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
            model = create_model(model_type, depth, branch_factor, device, num_classes=100)
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
    print("MINI IMAGENET RESULTS: FTA VS EFTA")
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
    filename = f'miniimagenet_fta_vs_efta_{timestamp}.json'
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
    
    parser = argparse.ArgumentParser(description='FTA vs EFTA on miniImageNet')
    parser.add_argument('--data_dir', type=str, default='./datasets/miniImageNet',
                        help='Path to miniImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split (default: 0.1)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("FTA VS EFTA ON MINI IMAGENET (DEBUGGED)")
    print(f"{'='*70}")
    print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nDataset: miniImageNet (100 classes, 84x84 RGB images, ~60k samples)")
    print(f"\nConfiguration:")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Validation split: {args.val_split*100:.0f}%")
    print(f"  - Learning rate: 0.0005 (reduced for stability)")
    print(f"\nModel Configurations:")
    print(f"  - FTA:  (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"  - EFTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"\nArchitecture: 4-block CNN (64→128→256→512) + BatchNorm + Dropout")
    print(f"Input: 84x84 RGB images (3 channels)")
    print(f"{'='*70}")
    
    # Verify data directory
    if not os.path.exists(args.data_dir):
        print(f"\n⚠ miniImageNet directory not found: {args.data_dir}")
        print("\nExpected structure:")
        print("   miniImageNet/")
        print("   ├── n01440764/")
        print("   ├── n01443537/")
        print("   └── ... (100 class folders)")
        print("\nTo use a different directory:")
        print("  python miniimagenet_fta_vs_efta.py --data_dir /path/to/miniImageNet")
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
