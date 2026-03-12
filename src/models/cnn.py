"""
Convolutional Neural Network (CNN) Model Architectures - PyTorch

This module provides CNN models with various activation functions
for the Dynamic Activation Functions research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activations import FractalTreeActivation, ExponentialFTA, MaxoutLayer


class CNNBaseline(nn.Module):
    """
    Standard CNN baseline model.

    Architecture:
        Input -> Conv2D(32) -> BN -> Activation -> MaxPool -> Dropout
              -> Conv2D(64) -> BN -> Activation -> MaxPool -> Dropout
              -> Flatten -> Dense(128) -> BN -> Activation -> Dropout
              -> Dense(num_classes)

    Args:
        activation: Activation function ('relu', 'leaky_relu', etc.)
        input_channels: Number of input channels (default: 1 for MNIST)
        num_classes: Number of output classes
    """

    def __init__(self, activation='relu', input_channels=1, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            # Convert from (batch, H, W, C) to (batch, C, H, W)
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNFTA(nn.Module):
    """
    CNN with Fractal Tree Activation (FTA).

    Architecture:
        Input -> Conv2D(32) -> BN -> FTA -> MaxPool -> Dropout
              -> Conv2D(64) -> BN -> FTA -> MaxPool -> Dropout
              -> Flatten -> Dense(128) -> BN -> FTA -> Dropout
              -> Dense(num_classes)

    Args:
        depth: Depth of the FTA tree
        branch_factor: Branching factor of the FTA tree
        input_channels: Number of input channels
        num_classes: Number of output classes
    """

    def __init__(self, depth=2, branch_factor=2, input_channels=1, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.fta1 = FractalTreeActivation(num_units=32, depth=depth,
                                           branch_factor=branch_factor, input_dim=32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fta2 = FractalTreeActivation(num_units=64, depth=depth,
                                           branch_factor=branch_factor, input_dim=64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fta3 = FractalTreeActivation(num_units=128, depth=depth,
                                           branch_factor=branch_factor, input_dim=128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # FTA expects (B,H,W,C)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fta3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNEFTA(nn.Module):
    """
    CNN with Exponential Fractal Tree Activation (EFTA).

    Architecture:
        Input -> Conv2D(32) -> BN -> EFTA -> MaxPool -> Dropout
              -> Conv2D(64) -> BN -> EFTA -> MaxPool -> Dropout
              -> Flatten -> Dense(128) -> BN -> EFTA -> Dropout
              -> Dense(num_classes)

    Args:
        depth: Depth of the EFTA tree
        branch_factor: Branching factor of the EFTA tree
        input_channels: Number of input channels
        num_classes: Number of output classes
    """

    def __init__(self, depth=2, branch_factor=2, input_channels=1, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.efta1 = ExponentialFTA(num_units=32, depth=depth,
                                     branch_factor=branch_factor, input_dim=32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.efta2 = ExponentialFTA(num_units=64, depth=depth,
                                     branch_factor=branch_factor, input_dim=64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.efta3 = ExponentialFTA(num_units=128, depth=depth,
                                     branch_factor=branch_factor, input_dim=128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # EFTA expects (B,H,W,C)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNMaxout(nn.Module):
    """
    CNN with Maxout activation.

    Args:
        k: Number of affine transformations in Maxout
        input_channels: Number of input channels
        num_classes: Number of output classes
    """

    def __init__(self, k=4, input_channels=1, num_classes=10):
        super().__init__()
        self.k = k

        # Block 1 - Conv outputs k*32 channels, then maxout reduces to 32
        self.conv1 = nn.Conv2d(input_channels, 32 * k, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32 * k)
        self.maxout1 = MaxoutLayer(in_features=32, out_features=32, k=k)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64 * k, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64 * k)
        self.maxout2 = MaxoutLayer(in_features=64, out_features=64, k=k)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Ensure input is (batch, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # Apply maxout: need to reshape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height, width, 32, self.k)
        x = torch.max(x, dim=4)[0]  # Max over k
        x = x.reshape(batch_size, height, width, 32)
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height, width, 64, self.k)
        x = torch.max(x, dim=4)[0]
        x = x.reshape(batch_size, height, width, 64)
        x = x.permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Factory functions

def create_cnn_baseline(activation='relu', input_shape=(1, 28, 28), num_classes=10):
    """Create CNN baseline model."""
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 1
    return CNNBaseline(activation=activation, input_channels=input_channels,
                       num_classes=num_classes)


def create_cnn_fta(depth=2, branch_factor=2, input_shape=(1, 28, 28), num_classes=10):
    """Create CNN with FTA."""
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 1
    return CNNFTA(depth=depth, branch_factor=branch_factor,
                  input_channels=input_channels, num_classes=num_classes)


def create_cnn_efta(depth=2, branch_factor=2, input_shape=(1, 28, 28), num_classes=10):
    """Create CNN with EFTA."""
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 1
    return CNNEFTA(depth=depth, branch_factor=branch_factor,
                   input_channels=input_channels, num_classes=num_classes)


def create_cnn_maxout(k=4, input_shape=(1, 28, 28), num_classes=10):
    """Create CNN with Maxout."""
    input_channels = input_shape[0] if isinstance(input_shape, tuple) else 1
    return CNNMaxout(k=k, input_channels=input_channels, num_classes=num_classes)
