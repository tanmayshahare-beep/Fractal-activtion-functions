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

    Each leaf uses a learnable exponential function:
        f(x) = α*(exp(β*x)-1)  for x < 0
        f(x) = γ*x              for x >= 0

    Leaves are combined via max through a tree structure.

    Args:
        num_units: number of output neurons/channels
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
        input_dim: dimension of input features (required for first layer)

    Attributes:
        num_leaves: total number of leaf nodes (branch_factor ** depth)

    Example:
        >>> efta = ExponentialFTA(num_units=128, depth=2, branch_factor=2, input_dim=784)
        >>> x = torch.randn(32, 784)
        >>> output = efta(x)
        >>> output.shape
        torch.Size([32, 128])
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
        # Shape: (num_leaves, num_units, 3)
        # Initialize to identity-like: alpha=1, beta=0 (small), gamma=1
        self.leaf_params = nn.Parameter(torch.randn(
            self.num_leaves, num_units, 3) * 0.1)
        # Initialize: alpha=1, beta=0.1, gamma=1
        with torch.no_grad():
            self.leaf_params[:, :, 0].fill_(1.0)  # alpha
            self.leaf_params[:, :, 1].fill_(0.1)  # beta
            self.leaf_params[:, :, 2].fill_(1.0)  # gamma

    def _leaf_function(self, x, alpha, beta, gamma):
        """
        Apply exponential branch to a single leaf.

        Args:
            x: input tensor
            alpha, beta, gamma: learnable parameters (scalars or broadcastable)

        Returns:
            Exponential activation output
        """
        # Split into negative and positive parts for stability
        neg_mask = (x < 0).to(x.dtype)
        pos_mask = (x >= 0).to(x.dtype)

        # Negative part: α * (exp(β * x) - 1)
        # Use clipping to avoid overflow
        neg_part = alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0)

        # Positive part: γ * x
        pos_part = gamma * x

        return neg_mask * neg_part + pos_mask * pos_part

    def forward(self, x):
        """
        Apply EFTA to inputs.

        Handles both 2D (batch, features) and 4D (batch, h, w, channels).
        Note: For 4D inputs, expects (batch, height, width, channels) format.
        """
        if len(x.shape) == 4:
            # Convolutional: (batch, height, width, channels)
            batch_size, height, width, channels = x.shape

            # Flatten spatial dimensions
            flat_inputs = x.reshape(batch_size, height * width, channels)
        else:
            # Dense: (batch, features)
            flat_inputs = x
            batch_size = x.shape[0]
            height = width = None

        # Compute leaf values for all leaves
        leaf_values_list = []
        for leaf_idx in range(self.num_leaves):
            alpha = self.leaf_params[leaf_idx, :, 0]  # (num_units,)
            beta = self.leaf_params[leaf_idx, :, 1]   # (num_units,)
            gamma = self.leaf_params[leaf_idx, :, 2]  # (num_units,)

            # Apply leaf function (broadcasting handles the parameters)
            leaf_val = self._leaf_function(flat_inputs, alpha, beta, gamma)
            leaf_values_list.append(leaf_val)

        # Stack leaves: (num_leaves, batch, N, num_units) where N = spatial elements
        leaf_values = torch.stack(leaf_values_list, dim=0)

        # Reshape for convolutional input
        if len(x.shape) == 4:
            leaf_values = leaf_values.reshape(
                self.num_leaves, batch_size, height, width, self.num_units)

        # Recursive max combination through tree
        current = leaf_values
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            if len(x.shape) == 4:
                # 4D: (num_leaves, batch, height, width, num_units)
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, height, width, self.num_units)
            else:
                # 2D: (num_leaves, batch, num_units)
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, self.num_units)
            # Take max over children (dim=1)
            current = torch.max(current, dim=1)[0]

        # After all levels: reshape to final shape
        if len(x.shape) == 4:
            return current.reshape(batch_size, height, width, self.num_units)
        else:
            return current.squeeze(0)
