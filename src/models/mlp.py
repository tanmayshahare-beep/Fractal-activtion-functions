"""
Multi-Layer Perceptron (MLP) Model Architectures - PyTorch

This module provides MLP models with various activation functions
for the Dynamic Activation Functions research.
"""

import torch
import torch.nn as nn
from ..activations import FractalTreeActivation, ExponentialFTA, MaxoutLayer


class MLPBaseline(nn.Module):
    """
    Standard MLP baseline model.

    Args:
        input_size: Size of input features (default: 784 for MNIST)
        hidden_size: Size of hidden layer (default: 128)
        num_classes: Number of output classes (default: 10)
        activation: Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh')
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MLPFTA(nn.Module):
    """
    MLP with Fractal Tree Activation (FTA).

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        num_classes: Number of output classes
        depth: Depth of the FTA tree
        branch_factor: Branching factor of the FTA tree
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10,
                 depth=2, branch_factor=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fta = FractalTreeActivation(
            num_units=hidden_size, depth=depth, branch_factor=branch_factor,
            input_dim=hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fta(x)
        x = self.fc2(x)
        return x


class MLPEFTA(nn.Module):
    """
    MLP with Exponential Fractal Tree Activation (EFTA).

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        num_classes: Number of output classes
        depth: Depth of the EFTA tree
        branch_factor: Branching factor of the EFTA tree
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10,
                 depth=2, branch_factor=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.efta = ExponentialFTA(
            num_units=hidden_size, depth=depth, branch_factor=branch_factor,
            input_dim=hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.efta(x)
        x = self.fc2(x)
        return x


class MLPMaxout(nn.Module):
    """
    MLP with Maxout activation.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        num_classes: Number of output classes
        k: Number of affine transformations in Maxout
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, k=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.maxout = MaxoutLayer(
            in_features=hidden_size, out_features=hidden_size, k=k)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.maxout(x)
        x = self.fc2(x)
        return x


# Factory functions for consistency with previous API

def create_mlp_baseline(activation='relu', input_shape=(784,), num_classes=10):
    """Create MLP baseline model."""
    input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    return MLPBaseline(input_size=input_size, num_classes=num_classes, activation=activation)


def create_mlp_fta(depth=2, branch_factor=2, input_shape=(784,), num_classes=10):
    """Create MLP with FTA."""
    input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    return MLPFTA(input_size=input_size, num_classes=num_classes,
                  depth=depth, branch_factor=branch_factor)


def create_mlp_efta(depth=2, branch_factor=2, input_shape=(784,), num_classes=10):
    """Create MLP with EFTA."""
    input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    return MLPEFTA(input_size=input_size, num_classes=num_classes,
                   depth=depth, branch_factor=branch_factor)


def create_mlp_maxout(k=4, input_shape=(784,), num_classes=10):
    """Create MLP with Maxout."""
    input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    return MLPMaxout(input_size=input_size, num_classes=num_classes, k=k)
