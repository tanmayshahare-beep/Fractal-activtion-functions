"""
Standalone Reviewer Experiments: ResNet-18 Scaling, Parameter Efficiency, and Inference Latency.

This script is a self-contained version of the reviewer experiments, including
all custom activation layers (FTA, EFTA, REAct-EFTA) inlined.

Usage:
    python review_experiments_standalone.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import os
import time
import json
import gc
from datetime import datetime

# Try to import fvcore, but provide a fallback if not installed
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not found. FLOP counting will be skipped. Install with 'pip install fvcore'.")

# =============================================================================
# ACTIVATION LAYERS (Inlined)
# =============================================================================

class FractalTreeActivation(nn.Module):
    """Fractal Tree Activation (FTA). Optimized with iterative max."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim

        if input_dim is None:
            raise ValueError("input_dim must be specified")

        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, input_dim, num_units) * (2.0 / input_dim) ** 0.5)
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            x_flat = x.reshape(-1, channels)
        else:
            x_flat = x

        current_max = None
        for i in range(self.num_leaves):
            leaf_val = F.linear(x_flat, self.leaf_weights[i].t(), self.leaf_biases[i])
            if current_max is None:
                current_max = leaf_val
            else:
                current_max = torch.max(current_max, leaf_val)

        return current_max.reshape(original_shape)

class ExponentialFTA(nn.Module):
    """Exponential Fractal Tree Activation (EFTA). Optimized with iterative max."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        
        if input_dim is None:
            raise ValueError("input_dim must be specified")

        self.leaf_params = nn.Parameter(torch.randn(self.num_leaves, num_units, 3) * 0.1)
        with torch.no_grad():
            self.leaf_params[:, :, 0].fill_(1.0)  # alpha
            self.leaf_params[:, :, 1].fill_(0.1)  # beta
            self.leaf_params[:, :, 2].fill_(1.0)  # gamma

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            x_flat = x.reshape(-1, channels)
        else:
            x_flat = x

        current_max = None
        for i in range(self.num_leaves):
            alpha, beta, gamma = self.leaf_params[i, :, 0], self.leaf_params[i, :, 1], self.leaf_params[i, :, 2]
            leaf_val = torch.where(x_flat < 0, alpha * (torch.exp(torch.clamp(beta * x_flat, -10, 10)) - 1.0), gamma * x_flat)
            if current_max is None:
                current_max = leaf_val
            else:
                current_max = torch.max(current_max, leaf_val)

        return current_max.reshape(original_shape)

class ReActEFTA(nn.Module):
    """Rational Exponential Fractal Tree Activation (REAct-EFTA). Optimized with iterative max."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None, clamp_value=5.0):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.clamp_value = clamp_value

        if input_dim is None:
            raise ValueError("input_dim must be specified")

        self.leaf_params = nn.Parameter(torch.randn(self.num_leaves, num_units, 4) * 0.05)
        with torch.no_grad():
            self.leaf_params.fill_(1.0)

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            x_flat = x.reshape(-1, channels)
        else:
            x_flat = x

        current_max = None
        for i in range(self.num_leaves):
            p = torch.clamp(self.leaf_params[i], 0.01, self.clamp_value)
            p1, p2, p3, p4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
            
            num = torch.exp(torch.clamp(p1 * x_flat, -self.clamp_value, self.clamp_value)) - \
                  torch.exp(torch.clamp(-p2 * x_flat, -self.clamp_value, self.clamp_value))
            den = torch.exp(torch.clamp(p3 * x_flat, -self.clamp_value, self.clamp_value)) + \
                  torch.exp(torch.clamp(-p4 * x_flat, -self.clamp_value, self.clamp_value))
            
            leaf_val = num / (den + 1e-8)
            
            if current_max is None:
                current_max = leaf_val
            else:
                current_max = torch.max(current_max, leaf_val)

        return current_max.reshape(original_shape)

# =============================================================================
# ResNet-18 with Custom Activation
# =============================================================================

class ActivationWrapper(nn.Module):
    def __init__(self, activation_layer):
        super().__init__()
        self.activation = activation_layer
        self.is_fta_family = isinstance(activation_layer, (FractalTreeActivation, ExponentialFTA, ReActEFTA))

    def forward(self, x):
        if self.is_fta_family and len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.activation(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            return x
        else:
            return self.activation(x)

def get_activation(act_type, num_channels):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'fta':
        return ActivationWrapper(FractalTreeActivation(num_units=num_channels, depth=2, branch_factor=2, input_dim=num_channels))
    elif act_type == 'efta':
        return ActivationWrapper(ExponentialFTA(num_units=num_channels, depth=2, branch_factor=2, input_dim=num_channels))
    elif act_type == 'react_efta':
        return ActivationWrapper(ReActEFTA(num_units=num_channels, depth=2, branch_factor=2, input_dim=num_channels))
    elif act_type == 'maxout':
        return ActivationWrapper(FractalTreeActivation(num_units=num_channels, depth=1, branch_factor=2, input_dim=num_channels))
    else:
        return nn.ReLU(inplace=True)

class BasicBlockCustom(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='relu'):
        super(BasicBlockCustom, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = get_activation(act_type, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = get_activation(act_type, planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act2(out)
        return out

class ResNetCustom(nn.Module):
    def __init__(self, block, layers, num_classes=10, act_type='relu'):
        super(ResNetCustom, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = get_activation(act_type, 64)
        self.layer1 = self._make_layer(block, 64, layers[0], act_type=act_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, act_type=act_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, act_type=act_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act_type=act_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, act_type='relu'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, act_type=act_type)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act_type=act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18_custom(act_type='relu', num_classes=10):
    return ResNetCustom(BasicBlockCustom, [2, 2, 2, 2], num_classes=num_classes, act_type=act_type)

# =============================================================================
# Utilities
# =============================================================================

def setup_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, device):
    if not HAS_FVCORE: return 0
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)
    flops = FlopCountAnalysis(model, x)
    return flops.total()

def measure_latency(model, device, input_size=(128, 3, 32, 32), repetitions=100):
    model.eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(10): _ = model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.time() - start) / repetitions * 1000

# =============================================================================
# Training logic
# =============================================================================

def get_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root='./datasets/CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testset = datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader

def train_model(model, trainloader, testloader, device, epochs=50, act_name='relu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        grad_norms = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if act_name == 'react_efta' and epoch < 5:
                norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
                grad_norms.append(norm)
            optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        if act_name == 'react_efta' and epoch < 5 and grad_norms:
            print(f"    [Epoch {epoch+1}] Mean Grad Norm: {np.mean(grad_norms):.6f}")
        scheduler.step()
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                t_total += targets.size(0)
                t_correct += pred.eq(targets).sum().item()
        acc = 100. * t_correct / t_total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(trainloader):.3f} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {acc:.2f}%")
    return best_acc

def run():
    device = setup_device()
    print(f"Using device: {device}")
    os.makedirs('outputs/results', exist_ok=True)
    # Increased batch size to 1024 to utilize high-end GPU memory
    batch_size = 1024
    trainloader, testloader = get_loaders(batch_size=batch_size)
    act_types = ['relu', 'maxout', 'fta', 'efta', 'react_efta']
    seeds = [42, 123, 456, 789, 1011]
    results = {}
    for act in act_types:
        print(f"\n{'='*60}\nExperiment: ResNet-18 with {act.upper()}\n{'='*60}")
        temp = resnet18_custom(act).to(device)
        params, flops = count_parameters(temp), count_flops(temp, device)
        print(f"Parameters: {params:,} | FLOPs: {flops:,}")
        del temp
        clear_memory()
        seed_accs, latencies = [], []
        for i, seed in enumerate(seeds):
            print(f"\n--- Run {i+1}/5 (Seed: {seed}) ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
            clear_memory()
            model = resnet18_custom(act).to(device)
            
            # Speed Boost: PyTorch 2.0 Compilation for custom math
            if hasattr(torch, 'compile'):
                try:
                    print("  Compiling model for speed optimization...")
                    model = torch.compile(model)
                except Exception as e:
                    print(f"  Compilation skipped: {e}")
                    
            best_acc = train_model(model, trainloader, testloader, device, epochs=50, act_name=act)
            seed_accs.append(best_acc)
            latencies.append(measure_latency(model, device))
            print(f"Seed {seed} Result: {best_acc:.2f}%")
            del model
            clear_memory()
        results[act] = {'params': params, 'flops': flops, 'mean_acc': np.mean(seed_accs), 'std_acc': np.std(seed_accs), 
                        'mean_latency': np.mean(latencies), 'std_latency': np.std(latencies), 'all_accs': seed_accs}
        print(f"\n{act.upper()} Summary: Acc = {results[act]['mean_acc']:.2f}% ± {results[act]['std_acc']:.2f}%, Latency = {results[act]['mean_latency']:.2f} ms")
    
    print(f"\n\n{'='*115}\n{'Activation':<15} | {'Params':<12} | {'FLOPs':<12} | {'Latency (ms)':<15} | {'Mean Acc (%)':<15} | {'Std Acc':<8}\n{'-' * 115}")
    for act, res in results.items():
        print(f"{act.upper():<15} | {res['params']:>12,} | {res['flops']:>12,} | {res['mean_latency']:>15.2f} | {res['mean_acc']:>15.2f} | {res['std_acc']:>8.2f}")
    print(f"{'='*115}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'outputs/results/review_experiments_standalone_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run()
