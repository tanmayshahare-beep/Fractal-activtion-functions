"""
Exponential Fractal Tree Activation (EFTA) Layer - PyTorch Implementation

This module implements the Exponential Fractal Tree Activation function:
a tree of exponential branches combined via max operation.

Each leaf uses: f(x) = α*(exp(β*x)-1) for x<0, γ*x for x>=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialFTA(nn.Module):
    """
    Exponential Fractal Tree Activation (EFTA).
    Optimized for memory efficiency using iterative max and torch.where.
    """

    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim

        if input_dim is None:
            raise ValueError("input_dim must be specified for ExponentialFTA")

        # Parameters for each leaf: (alpha, beta, gamma) per output unit
        self.leaf_params = nn.Parameter(torch.randn(self.num_leaves, num_units, 3) * 0.1)
        
        with torch.no_grad():
            self.leaf_params[:, :, 0].fill_(1.0)  # alpha
            self.leaf_params[:, :, 1].fill_(0.1)  # beta
            self.leaf_params[:, :, 2].fill_(1.0)  # gamma

    def _leaf_function(self, x, alpha, beta, gamma):
        """Memory-efficient leaf function using torch.where."""
        # Using torch.where is more memory efficient than masking
        # alpha, beta, gamma are (num_units,) and x is (..., num_units)
        return torch.where(
            x < 0,
            alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0),
            gamma * x
        )

    def forward(self, x):
        """
        Apply EFTA to inputs.
        Iterative max reduces memory footprint by avoiding large stacked tensors.
        """
        original_shape = x.shape
        if len(x.shape) == 4:
            # (B, H, W, C) -> (B*H*W, C)
            batch_size, height, width, channels = x.shape
            x_flat = x.reshape(-1, channels)
        else:
            x_flat = x

        # Iterative max to find the tree result
        # Note: This is equivalent to a full tree max since max(a,b,c,d) = max(a,max(b,max(c,d)))
        current_max = None
        
        for leaf_idx in range(self.num_leaves):
            alpha = self.leaf_params[leaf_idx, :, 0]
            beta = self.leaf_params[leaf_idx, :, 1]
            gamma = self.leaf_params[leaf_idx, :, 2]
            
            leaf_val = self._leaf_function(x_flat, alpha, beta, gamma)
            
            if current_max is None:
                current_max = leaf_val
            else:
                current_max = torch.max(current_max, leaf_val)

        # Restore original shape
        return current_max.reshape(original_shape)
