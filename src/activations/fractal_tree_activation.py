"""
Fractal Tree Activation (FTA) Layer - PyTorch Implementation

This module implements the Fractal Tree Activation function:
a tree of linear transforms combined via max operation.

This creates a hierarchical activation where each neuron has an internal
tree of linear transformations, enabling multi-scale feature learning
within a single activation function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalTreeActivation(nn.Module):
    """
    Fractal Tree Activation: a tree of linear transforms combined via max.

    This creates a hierarchical activation where each neuron has an internal
    tree of linear transformations, enabling multi-scale feature learning
    within a single activation function.

    Args:
        num_units: number of output neurons
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
        share_weights: if True, all nodes at the same depth share weights
        input_dim: dimension of input features (required for first layer)

    Attributes:
        num_leaves: total number of leaf nodes (branch_factor ** depth)

    Example:
        >>> fta = FractalTreeActivation(num_units=128, depth=2, branch_factor=2, input_dim=784)
        >>> x = torch.randn(32, 784)
        >>> output = fta(x)
        >>> output.shape
        torch.Size([32, 128])
    """

    def __init__(self, num_units, depth=2, branch_factor=2, share_weights=False, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.share_weights = share_weights
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim

        if input_dim is None:
            raise ValueError("input_dim must be specified for FractalTreeActivation")

        if share_weights:
            # One set of weights per depth level (parameter efficient)
            self.leaf_weights = nn.ModuleList()
            self.leaf_biases = nn.ParameterList()
            for d in range(depth):
                w = nn.Parameter(torch.randn(input_dim, num_units) * (2.0 / input_dim) ** 0.5)
                b = nn.Parameter(torch.zeros(num_units))
                self.leaf_weights.append(w)
                self.leaf_biases.append(b)
        else:
            # Independent weights for each leaf (more expressive)
            self.leaf_weights = nn.Parameter(torch.randn(
                self.num_leaves, input_dim, num_units) * (2.0 / input_dim) ** 0.5)
            self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        """
        Compute fractal tree activation recursively.

        For each leaf node: compute linear transformation
        For internal nodes: take max of children
        Root output is the activation value

        Handles both 2D (batch, features) and 4D (batch, h, w, channels) inputs.
        """
        if len(x.shape) == 4:
            # Convolutional: (batch, height, width, channels)
            batch_size, height, width, channels = x.shape

            # Flatten spatial dimensions
            flat_inputs = x.reshape(batch_size, height * width, channels)

            if self.share_weights:
                leaf_values = []
                for i in range(self.num_leaves):
                    # Use first weight set for all leaves in shared mode
                    leaf_val = F.linear(flat_inputs, self.leaf_weights[0], self.leaf_biases[0])
                    leaf_values.append(leaf_val)
                leaf_values = torch.stack(leaf_values, dim=0)
            else:
                # Compute all leaf values: (num_leaves, batch*height*width, num_units)
                leaf_values = torch.stack([
                    F.linear(flat_inputs, self.leaf_weights[i], self.leaf_biases[i])
                    for i in range(self.num_leaves)
                ], dim=0)

            # Reshape for convolutional: (num_leaves, batch, height, width, num_units)
            leaf_values = leaf_values.reshape(
                self.num_leaves, batch_size, height, width, self.num_units)

            # Recursive max combination through tree
            current = leaf_values
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                # Reshape: (num_nodes, branch_factor, batch, height, width, num_units)
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, height, width, self.num_units)
                # Take max over children (dim=1)
                current = torch.max(current, dim=1)[0]

            # Reshape back to (batch, height, width, num_units)
            return current.reshape(batch_size, height, width, self.num_units)
        else:
            # Dense: (batch, features)
            batch_size = x.shape[0]

            if self.share_weights:
                leaf_values = []
                for i in range(self.num_leaves):
                    leaf_val = F.linear(x, self.leaf_weights[0], self.leaf_biases[0])
                    leaf_values.append(leaf_val)
                leaf_values = torch.stack(leaf_values, dim=0)
            else:
                leaf_values = torch.stack([
                    F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
                    for i in range(self.num_leaves)
                ], dim=0)

            # Recursive max combination through tree
            current = leaf_values  # (num_leaves, batch, num_units)
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                # Reshape: (num_nodes, branch_factor, batch, num_units)
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, self.num_units)
                # Take max over children (dim=1)
                current = torch.max(current, dim=1)[0]

            # Squeeze to (batch, num_units)
            return current.squeeze(0)

    def get_leaf_values(self, x):
        """
        Get leaf values before max pooling for analysis.
        
        Args:
            x: Input tensor (batch, features) or (batch, H, W, C)
        
        Returns:
            Leaf values tensor (num_leaves, batch, num_units) or 
            (num_leaves, batch, H, W, num_units) for 4D input
        """
        if len(x.shape) == 4:
            # Convolutional: (batch, height, width, channels)
            batch_size, height, width, channels = x.shape
            flat_inputs = x.reshape(batch_size, height * width, channels)

            if self.share_weights:
                leaf_values = []
                for i in range(self.num_leaves):
                    leaf_val = F.linear(flat_inputs, self.leaf_weights[0], self.leaf_biases[0])
                    leaf_values.append(leaf_val)
                leaf_values = torch.stack(leaf_values, dim=0)
            else:
                leaf_values = torch.stack([
                    F.linear(flat_inputs, self.leaf_weights[i], self.leaf_biases[i])
                    for i in range(self.num_leaves)
                ], dim=0)

            # Reshape: (num_leaves, batch, height, width, num_units)
            return leaf_values.reshape(
                self.num_leaves, batch_size, height, width, self.num_units)
        else:
            # Dense: (batch, features)
            batch_size = x.shape[0]

            if self.share_weights:
                leaf_values = []
                for i in range(self.num_leaves):
                    leaf_val = F.linear(x, self.leaf_weights[0], self.leaf_biases[0])
                    leaf_values.append(leaf_val)
                leaf_values = torch.stack(leaf_values, dim=0)
            else:
                leaf_values = torch.stack([
                    F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
                    for i in range(self.num_leaves)
                ], dim=0)

            return leaf_values  # (num_leaves, batch, num_units)
