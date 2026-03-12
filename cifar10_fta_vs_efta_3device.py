"""
FTA vs EFTA Comparison on CIFAR-10 - Multi-Device Parallel Training

Uses three devices in parallel:
- NVIDIA RTX 4060 (CUDA)
- Intel Arc GPU (XPU)
- Intel NPU (XPU)

Different model configurations are trained in parallel across devices.
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
import threading
from queue import Queue

# =============================================================================
# Device Setup
# =============================================================================

def setup_devices():
    """Detect and setup available devices."""
    devices = []
    
    # Check for CUDA (NVIDIA RTX 4060)
    if torch.cuda.is_available():
        devices.append(('cuda', 0, f"NVIDIA RTX 4060 (CUDA)"))
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Check for XPU via Intel Extension for PyTorch (IPEX)
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            xpu_device_count = ipex.xpu.device_count()
            print(f"✓ XPU available (via IPEX): {xpu_device_count} device(s)")
            
            for i in range(min(xpu_device_count, 2)):
                device_name = ipex.xpu.get_device_name(i)
                devices.append(('xpu', i, f"Intel Device {i} ({device_name})"))
                print(f"  - XPU Device {i}: {device_name}")
    except ImportError:
        print("✗ intel_extension_for_pytorch not installed (Intel devices unavailable)")
    except Exception as e:
        print(f"✗ XPU error: {e}")
    
    # Fallback to CPU if no accelerators
    if not devices:
        devices.append(('cpu', None, "CPU"))
        print("⚠ No accelerators found, using CPU")
    
    print(f"\nTotal devices available: {len(devices)}")
    return devices


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
# CNN Architecture for CIFAR-10
# =============================================================================

class CNNFTA(nn.Module):
    """CNN with Fractal Tree Activation for CIFAR-10."""
    def __init__(self, depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        self.fta1 = FractalTreeActivation(64, depth, branch_factor)
        self.fta2 = FractalTreeActivation(128, depth, branch_factor)
        self.fta3 = FractalTreeActivation(256, depth, branch_factor)
        self.fta4 = FractalTreeActivation(512, depth, branch_factor)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

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
        x = self.fta4(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class CNNEFTA(nn.Module):
    """CNN with Exponential Fractal Tree Activation for CIFAR-10."""
    def __init__(self, depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        self.efta1 = ExponentialFTA(64, depth, branch_factor)
        self.efta2 = ExponentialFTA(128, depth, branch_factor)
        self.efta3 = ExponentialFTA(256, depth, branch_factor)
        self.efta4 = ExponentialFTA(512, depth, branch_factor)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

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
        x = self.efta4(x)
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


def create_model(model_type, depth, branch_factor):
    """Create model based on type."""
    if model_type == 'fta':
        return CNNFTA(depth=depth, branch_factor=branch_factor)
    elif model_type == 'efta':
        return CNNEFTA(depth=depth, branch_factor=branch_factor)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, loader, criterion, optimizer, device, device_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        if device_type == 'xpu':
            data, target = data.to(device), target.to(device)
        else:
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


def evaluate(model, loader, criterion, device, device_type):
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


def train_model_on_device(device_info, model_type, depth, branch_factor,
                          train_loader, val_loader, test_loader,
                          epochs=50, result_queue=None):
    """Train a model on a specific device."""
    
    device_type, device_idx, device_name = device_info
    
    if device_type == 'cuda':
        device = torch.device(f'cuda:{device_idx}')
    elif device_type == 'xpu':
        import intel_extension_for_pytorch as ipex
        device = ipex.xpu.device(device_idx)
    else:
        device = torch.device('cpu')
    
    model_name = get_model_name(model_type, depth, branch_factor)
    
    print(f"\n[{device_name}] Starting: {model_name}")
    
    try:
        model = create_model(model_type, depth, branch_factor)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 10

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, device_type)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, device_type)

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

            if (epoch + 1) % 10 == 0:
                print(f"[{device_name}] {model_name} - Epoch {epoch+1:2d}/{epochs}: "
                      f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

            if patience_counter >= early_stop_patience:
                print(f"[{device_name}] {model_name} - Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device, device_type)

        total_time = time.time() - start_time
        params = count_parameters(model)

        result = {
            'name': model_name,
            'device': device_name,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'best_val_acc': best_val_acc,
            'params': params,
            'epochs_trained': len(history['train_loss']),
            'training_time': total_time,
            'history': history
        }

        print(f"[{device_name}] ✓ {model_name} completed: Test Acc={test_acc:.2f}%, Time={total_time/60:.1f}min")
        
        if result_queue:
            result_queue.put(result)
            
        return result

    except Exception as e:
        print(f"[{device_name}] ✗ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {
            'name': model_name,
            'device': device_name,
            'error': str(e)
        }
        
        if result_queue:
            result_queue.put(error_result)
        
        return error_result


# =============================================================================
# Data Loading
# =============================================================================

def get_data_loaders(batch_size=128, val_split=0.1):
    """Get train/val/test data loaders for CIFAR-10 with augmentation."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root='./datasets/CIFAR10', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./datasets/CIFAR10', train=False, download=True, transform=test_transform
    )

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    full_train_dataset = datasets.CIFAR10(
        root='./datasets/CIFAR10', train=True, download=True, transform=test_transform
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# =============================================================================
# Multi-Device Parallel Training
# =============================================================================

def run_parallel_training(devices, epochs=50):
    """Run training in parallel across multiple devices."""
    
    print(f"\n{'='*70}")
    print("PARALLEL TRAINING CONFIGURATION")
    print(f"{'='*70}")
    
    # Model configurations to test
    configurations = [
        ('fta', 2, 2), ('fta', 3, 2), ('fta', 2, 3), ('fta', 3, 3),
        ('efta', 2, 2), ('efta', 3, 2), ('efta', 2, 3), ('efta', 3, 3),
    ]
    
    # Distribute configurations across devices (round-robin)
    device_assignments = {i: [] for i in range(len(devices))}
    for idx, config in enumerate(configurations):
        device_idx = idx % len(devices)
        device_assignments[device_idx].append(config)
    
    print(f"\nDevice assignments:")
    for i, (device_info, configs) in enumerate(zip(devices, device_assignments.values())):
        device_name = device_info[2]
        config_names = [get_model_name(c[0], c[1], c[2]) for c in configs]
        print(f"  {device_name}: {', '.join(config_names)}")
    
    print(f"\n{'='*70}\n")
    
    # Get shared data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print("Dataset loaded.\n")
    
    # Create result queue
    result_queue = Queue()
    
    # Launch training threads
    threads = []
    start_time = time.time()
    
    for device_idx, device_info in enumerate(devices):
        device_name = device_info[2]
        configs = device_assignments[device_idx]
        
        for config in configs:
            model_type, depth, branch_factor = config
            
            thread = threading.Thread(
                target=train_model_on_device,
                args=(
                    device_info, model_type, depth, branch_factor,
                    train_loader, val_loader, test_loader,
                    epochs, result_queue
                ),
                daemon=True
            )
            threads.append(thread)
            thread.start()
    
    print(f"Launched {len(threads)} parallel training jobs on {len(devices)} devices\n")
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        if 'error' not in result:
            results[result['name']] = result
    
    return results, total_time


# =============================================================================
# Results Display
# =============================================================================

def print_results_table(results, total_time):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print("CIFAR-10 RESULTS: FTA VS EFTA (Multi-Device Parallel Training)")
    print(f"{'='*100}")
    print(f"{'Model':<25} | {'Device':<25} | {'Test Acc':<10} | {'Val Acc':<10} | {'Params':<12} | {'Time (min)':<10}")
    print("-"*100)

    for name, result in results.items():
        time_min = result['training_time'] / 60
        device_name = result.get('device', 'N/A')[:25]
        print(f"{name:<25} | {device_name:<25} | {result['test_acc']:>8.2f}% | {result['best_val_acc']:>8.2f}% | "
              f"{result['params']:>10,} | {time_min:>8.1f}")

    print("="*100)
    print(f"Total parallel training time: {total_time/60:.1f} minutes")

    # Find best models
    if results:
        best_overall = max(results, key=lambda x: results[x]['test_acc'])
        print(f"\n🏆 Best Model: {best_overall} ({results[best_overall]['test_acc']:.2f}%) on {results[best_overall].get('device', 'N/A')}")

        # Best by category
        fta_models = {k: v for k, v in results.items() if 'FTA' in k and 'EFTA' not in k}
        efta_models = {k: v for k, v in results.items() if 'EFTA' in k}

        if fta_models:
            best_fta = max(fta_models, key=lambda x: fta_models[x]['test_acc'])
            print(f"🥇 Best FTA:  {best_fta} ({fta_models[best_fta]['test_acc']:.2f}%)")

        if efta_models:
            best_efta = max(efta_models, key=lambda x: efta_models[x]['test_acc'])
            print(f"🥇 Best EFTA: {best_efta} ({efta_models[best_efta]['test_acc']:.2f}%)")

        # Compare matched configurations
        print(f"\n{'='*50}")
        print("FTA VS EFTA HEAD-TO-HEAD:")
        print(f"{'='*50}")

        configs = [(2,2), (3,2), (2,3), (3,3)]
        for d, k in configs:
            fta_name = f'FTA (d={d}, k={k})'
            efta_name = f'EFTA (d={d}, k={k})'

            if fta_name in results and efta_name in results:
                fta_acc = results[fta_name]['test_acc']
                efta_acc = results[efta_name]['test_acc']
                diff = efta_acc - fta_acc
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  (d={d}, k={k}): FTA={fta_acc:.2f}% vs EFTA={efta_acc:.2f}% {symbol} {abs(diff):.2f}%")

    print("="*100)


def save_results(results, total_time, output_dir='./outputs/results'):
    """Save results to file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'cifar10_fta_vs_efta_3device_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)

    save_data = {
        'total_time': total_time,
        'devices_used': list(set(r.get('device', 'N/A') for r in results.values())),
        'results': {}
    }
    
    for name, result in results.items():
        save_data['results'][name] = {
            'device': result.get('device', 'N/A'),
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


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    """Main entry point."""
    print(f"\n{'='*70}")
    print("FTA VS EFTA ON CIFAR-10 - MULTI-DEVICE PARALLEL TRAINING")
    print(f"{'='*70}")
    print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfigurations:")
    print(f"  - FTA:  (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"  - EFTA: (d=2,k=2), (d=3,k=2), (d=2,k=3), (d=3,k=3)")
    print(f"\nDataset: CIFAR-10 (32x32 RGB, 10 classes, 50k train / 10k test)")
    print(f"Architecture: 4-block CNN with BatchNorm + Dropout")
    print(f"\n💡 For Intel GPU/NPU support, install:")
    print(f"   pip install intel_extension_for_pytorch")
    print(f"{'='*70}")

    # Setup devices
    devices = setup_devices()
    
    if len(devices) < 2:
        print(f"\n⚠ Warning: Only {len(devices)} device(s) detected. Multi-device benefits limited.")
        print("Expected: NVIDIA RTX 4060 (CUDA) + Intel Arc GPU (XPU) + Intel NPU (XPU)")
    
    # Run parallel training
    results, total_time = run_parallel_training(devices, epochs=50)
    
    # Print results
    print_results_table(results, total_time)
    
    # Save results
    save_results(results, total_time)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*70}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
