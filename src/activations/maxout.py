"""
Maxout Activation Layer - PyTorch Implementation

This module implements the standard Maxout activation function:
z = max(W_1·x + b_1, W_2·x + b_2, ..., W_k·x + b_k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxoutLayer(nn.Module):
    """
    Maxout Activation Layer.

    Maxout computes k linear transformations and takes the element-wise maximum:
        z = max(W_1·x + b_1, W_2·x + b_2, ..., W_k·x + b_k)

    Args:
        in_features: number of input features
        out_features: number of output neurons
        k: number of affine transformations to combine

    Example:
        >>> maxout = MaxoutLayer(in_features=784, out_features=128, k=4)
        >>> x = torch.randn(32, 784)
        >>> output = maxout(x)
        >>> output.shape
        torch.Size([32, 128])
    """

    def __init__(self, in_features, out_features, k=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # Weight: (in_features, out_features, k)
        self.weight = nn.Parameter(torch.randn(in_features, out_features, k) * (2.0 / in_features) ** 0.5)
        # Bias: (out_features, k)
        self.bias = nn.Parameter(torch.zeros(out_features, k))

    def forward(self, x):
        # x: (batch, in_features)
        # weight: (in_features, out_features, k)
        # bias: (out_features, k)

        # Compute: x @ weight + bias
        # Result: (batch, out_features, k)
        output = torch.einsum('bi,iok->bok', x, self.weight) + self.bias

        # Take max over k dimension
        # Result: (batch, out_features)
        return torch.max(output, dim=2)[0]
