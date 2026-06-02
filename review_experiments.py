"""
Reviewer Experiments: ResNet-18 Scaling, Parameter Efficiency, and Inference Latency.

This script addresses the reviewer's comments:
1. ResNet-18 scaling for FTA/EFTA (Comment 4).
2. Parameter efficiency validation (Comment 3).
3. Inference latency and profiling (Comment 5).
4. REAct-EFTA performance investigation (Comment 2).

Usage:
    python review_experiments.py
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
from fvcore.nn import FlopCountAnalysis

# Import local activations
from src.activations import FractalTreeActivation, ExponentialFTA, MaxoutLayer
from src.activations.react_efta import ReActEFTA

# =============================================================================
# GPU Setup
# =============================================================================

def setup_device():
    if torch.cuda.is_available():
        # Memory limit removed for high-performance servers
        return torch.device('cuda:0')
    return torch.device('cpu')

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# =============================================================================
# ResNet-18 with Custom Activation
# =============================================================================

class ActivationWrapper(nn.Module):
    """
    Wraps FTA/EFTA/REAct-EFTA to handle (B, C, H, W) <-> (B, H, W, C) conversion
    and provide a standard interface for ResNet.
    """
    def __init__(self, activation_layer):
        super().__init__()
        self.activation = activation_layer
        self.is_fta_family = isinstance(activation_layer, (FractalTreeActivation, ExponentialFTA, ReActEFTA))

    def forward(self, x):
        if self.is_fta_family and len(x.shape) == 4:
            # (B, C, H, W) -> (B, H, W, C)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.activation(x)
            # (B, H, W, C) -> (B, C, H, W)
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
        # Using FTA with depth=1 as equivalent to Maxout
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
        self.stride = stride

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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, act_type=act_type))
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, device):
    """Measures FLOPs using fvcore analysis."""
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)
    flops = FlopCountAnalysis(model, x)
    return flops.total()

def measure_latency(model, device, input_size=(128, 3, 32, 32), repetitions=100):
    model.eval()
    x = torch.randn(*input_size).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / repetitions * 1000  # ms
    return avg_latency

# =============================================================================
# Training Logic
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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, device, epochs=50, act_name='relu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        grad_norms = []
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for fair comparison and stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Diagnostic: Gradient Norm Logging for REAct-EFTA (First 5 epochs)
            if act_name == 'react_efta' and epoch < 5:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if act_name == 'react_efta' and epoch < 5 and grad_norms:
            print(f"    [Epoch {epoch+1}] Mean Grad Norm: {np.mean(grad_norms):.6f}")
            
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        acc = 100. * test_correct / test_total
        if acc > best_acc:
            best_acc = acc
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(trainloader):.3f} | Train Acc: {100.*correct/total:.2f}% | Test Acc: {acc:.2f}%")
        
    return best_acc

# =============================================================================
# Main Experiment
# =============================================================================

def run_review_experiments():
    device = setup_device()
    print(f"Using device: {device}")
    
    os.makedirs('outputs/results', exist_ok=True)
    trainloader, testloader = get_loaders()
    
    act_types = ['relu', 'maxout', 'fta', 'efta', 'react_efta']
    seeds = [42, 123, 456, 789, 1011]
    epochs = 50
    results = {}
    
    for act in act_types:
        print(f"\n{'='*60}")
        print(f"Experiment: ResNet-18 with {act.upper()}")
        print(f"{'='*60}")
        
        # Calculate static metrics once per activation
        temp_model = resnet18_custom(act_type=act).to(device)
        params = count_parameters(temp_model)
        flops = count_flops(temp_model, device)
        print(f"Parameters: {params:,}")
        print(f"FLOPs: {flops:,}")
        del temp_model
        clear_memory()
        
        seed_accs = []
        latencies = []
        
        for i, seed in enumerate(seeds):
            print(f"\n--- Run {i+1}/5 (Seed: {seed}) ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            clear_memory()
            model = resnet18_custom(act_type=act).to(device)
            
            best_acc = train_model(model, trainloader, testloader, device, epochs=epochs, act_name=act)
            seed_accs.append(best_acc)
            
            # Measure latency on the trained model
            latency = measure_latency(model, device)
            latencies.append(latency)
            
            print(f"Seed {seed} Result: {best_acc:.2f}%")
            
            del model
            clear_memory()
        
        results[act] = {
            'params': params,
            'flops': flops,
            'mean_acc': np.mean(seed_accs),
            'std_acc': np.std(seed_accs),
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'all_accs': seed_accs
        }
        
        print(f"\n{act.upper()} Summary: Acc = {results[act]['mean_acc']:.2f}% ± {results[act]['std_acc']:.2f}%, Latency = {results[act]['mean_latency']:.2f} ms")
        
    # Output final summary table
    print(f"\n\n{'='*115}")
    print(f"{'Activation':<15} | {'Params':<12} | {'FLOPs':<12} | {'Latency (ms)':<15} | {'Mean Acc (%)':<15} | {'Std Acc':<8}")
    print("-" * 115)
    for act, res in results.items():
        print(f"{act.upper():<15} | {res['params']:>12,} | {res['flops']:>12,} | {res['mean_latency']:>15.2f} | {res['mean_acc']:>15.2f} | {res['std_acc']:>8.2f}")
    print(f"{'='*115}")
    
    # Save results to JSON for reviewer reporting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'outputs/results/review_experiments_5seed_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    run_review_experiments()
