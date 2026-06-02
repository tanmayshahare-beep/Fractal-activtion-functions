"""
Standalone Analysis: EFTA vs REAct-EFTA on ResNet-18 with Gini Coefficient.

This script focuses on:
1. Exponential Fractal Tree Activation (EFTA)
2. Rational Exponential Fractal Tree Activation (REAct-EFTA)
3. Gini Coefficient analysis of leaf usage (branch specialization).

Usage:
    python efta_react_gini_resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
import json
import gc
from datetime import datetime

# Try to import fvcore for FLOPs
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

# =============================================================================
# ACTIVATIONS WITH GINI TRACKING
# =============================================================================

class ExponentialFTA(nn.Module):
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        
        self.leaf_params = nn.Parameter(torch.randn(self.num_leaves, num_units, 3) * 0.1)
        with torch.no_grad():
            self.leaf_params[:, :, 0].fill_(1.0)  # alpha
            self.leaf_params[:, :, 1].fill_(0.1)  # beta
            self.leaf_params[:, :, 2].fill_(1.0)  # gamma

    def forward(self, x, return_usage=False):
        original_shape = x.shape
        x_flat = x.reshape(-1, self.num_units) if len(x.shape) == 4 else x

        current_max = None
        winning_leaves = None if not return_usage else torch.zeros(x_flat.shape[0], self.num_units, device=x.device, dtype=torch.long)
        
        for i in range(self.num_leaves):
            alpha, beta, gamma = self.leaf_params[i, :, 0], self.leaf_params[i, :, 1], self.leaf_params[i, :, 2]
            leaf_val = torch.where(x_flat < 0, alpha * (torch.exp(torch.clamp(beta * x_flat, -10, 10)) - 1.0), gamma * x_flat)
            
            if current_max is None:
                current_max = leaf_val
                if return_usage: winning_leaves.fill_(i)
            else:
                if return_usage:
                    is_better = leaf_val > current_max
                    winning_leaves[is_better] = i
                current_max = torch.max(current_max, leaf_val)

        out = current_max.reshape(original_shape)
        return (out, winning_leaves) if return_usage else out

class ReActEFTA(nn.Module):
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None, clamp_value=5.0):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.clamp_value = clamp_value

        self.leaf_params = nn.Parameter(torch.randn(self.num_leaves, num_units, 4) * 0.05)
        with torch.no_grad():
            self.leaf_params.fill_(1.0)

    def forward(self, x, return_usage=False):
        original_shape = x.shape
        x_flat = x.reshape(-1, self.num_units) if len(x.shape) == 4 else x

        current_max = None
        winning_leaves = None if not return_usage else torch.zeros(x_flat.shape[0], self.num_units, device=x.device, dtype=torch.long)
        
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
                if return_usage: winning_leaves.fill_(i)
            else:
                if return_usage:
                    is_better = leaf_val > current_max
                    winning_leaves[is_better] = i
                current_max = torch.max(current_max, leaf_val)

        out = current_max.reshape(original_shape)
        return (out, winning_leaves) if return_usage else out

# =============================================================================
# RESNET-18 ARCHITECTURE
# =============================================================================

class ActivationWrapper(nn.Module):
    def __init__(self, activation_layer):
        super().__init__()
        self.activation = activation_layer

    def forward(self, x, return_usage=False):
        # ResNet-18 Conv layers are (B, C, H, W), activations expect (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        if return_usage:
            x, usage = self.activation(x, return_usage=True)
            x = x.permute(0, 3, 1, 2).contiguous()
            return x, usage
        else:
            x = self.activation(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='efta'):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ActivationWrapper(ExponentialFTA(planes, input_dim=planes) if act_type=='efta' else ReActEFTA(planes, input_dim=planes))
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ActivationWrapper(ExponentialFTA(planes, input_dim=planes) if act_type=='efta' else ReActEFTA(planes, input_dim=planes))
        self.downsample = downsample

    def forward(self, x, return_usage=False):
        identity = x
        usage_data = []

        out = self.conv1(x)
        out = self.bn1(out)
        if return_usage:
            out, u1 = self.act1(out, return_usage=True)
            usage_data.append(u1)
        else:
            out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if return_usage:
            out, u2 = self.act2(out, return_usage=True)
            usage_data.append(u2)
        else:
            out = self.act2(out)

        return (out, usage_data) if return_usage else out

class ResNet18Gini(nn.Module):
    def __init__(self, act_type='efta', num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = ActivationWrapper(ExponentialFTA(64, input_dim=64) if act_type=='efta' else ReActEFTA(64, input_dim=64))
        
        self.layer1 = self._make_layer(64, 2, act_type=act_type)
        self.layer2 = self._make_layer(128, 2, stride=2, act_type=act_type)
        self.layer3 = self._make_layer(256, 2, stride=2, act_type=act_type)
        self.layer4 = self._make_layer(512, 2, stride=2, act_type=act_type)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1, act_type='efta'):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, act_type=act_type)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, act_type=act_type))
        return nn.Sequential(*layers)

    def forward(self, x, return_usage=False):
        usage_all = []
        x = self.conv1(x)
        x = self.bn1(x)
        if return_usage:
            x, u = self.act1(x, return_usage=True)
            usage_all.append(u)
        else:
            x = self.act1(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if return_usage:
                    x, u_list = block(x, return_usage=True)
                    usage_all.extend(u_list)
                else:
                    x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (x, usage_all) if return_usage else x

# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

def compute_gini(frequencies):
    n = len(frequencies)
    if n <= 1: return 0.0
    sorted_freq = np.sort(frequencies)
    # Gini = (2 * sum(i * fi) / (n * sum(fi))) - (n + 1) / n
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_freq))) / (n * np.sum(sorted_freq)) - (n + 1) / n
    return max(0.0, min(1.0, gini))

def analyze_gini(model, loader, device, num_leaves=4):
    model.eval()
    total_usage = np.zeros(num_leaves)
    print("  Calculating branch usage for Gini...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i > 20: break  # Sample 20 batches for efficiency
            inputs = inputs.to(device)
            _, usage_list = model(inputs, return_usage=True)
            for usage in usage_list:
                # usage is (Batch, H*W, Units)
                flat_usage = usage.cpu().numpy().flatten()
                counts = np.bincount(flat_usage, minlength=num_leaves)
                total_usage += counts
    
    gini = compute_gini(total_usage)
    return gini, total_usage

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='./datasets/CIFAR10', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    results = {}
    for act in ['efta', 'react_efta']:
        print(f"\n{'='*60}\nEvaluating: {act.upper()}\n{'='*60}")
        model = ResNet18Gini(act_type=act).to(device)
        
        # Static Analysis
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {params:,}")
        
        # Training (Quick run: 10 epochs for demonstration, use 50 for full results)
        epochs = 10
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            correct, total = 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
            
            print(f"  Epoch {epoch+1}/{epochs} | Acc: {100.*correct/total:.2f}%")

        # Final Analysis
        gini, usage = analyze_gini(model, test_loader, device)
        
        # Latency
        model.eval()
        dummy = torch.randn(128, 3, 32, 32).to(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(50): _ = model(dummy)
        latency = (time.time() - start) / 50 * 1000
        
        results[act] = {
            'params': params,
            'gini': gini,
            'usage_dist': usage.tolist(),
            'latency': latency
        }
        
        print(f"\n{act.upper()} RESULTS:")
        print(f"  Gini Coefficient: {gini:.4f}")
        print(f"  Branch Usage:    {usage}")
        print(f"  Latency:         {latency:.2f} ms")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n\n{'='*80}\n{'Activation':<15} | {'Params':<12} | {'Gini':<10} | {'Latency':<10}\n{'-'*80}")
    for act, res in results.items():
        print(f"{act.upper():<15} | {res['params']:>12,} | {res['gini']:>10.4f} | {res['latency']:>8.2f} ms")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_experiment()
