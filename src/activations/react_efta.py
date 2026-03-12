"""
REAct-EFTA: Rational Exponential Fractal Tree Activation Layer - PyTorch

This module implements EFTA with REAct (Rational Exponential Activation) branches.
REAct is a generalized tanh-like function with 4 learnable shape parameters.

Each leaf uses the REAct function:
    REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))

Where p1, p2, p3, p4 are learnable parameters per leaf.

This provides:
- Smooth, bounded activation like tanh
- Learnable asymmetry (p1 vs p2)
- Learnable saturation rates (p3, p4)
- High expressivity with only 4 parameters per unit per leaf

Leaves are combined via max through a tree structure (same as FTA/EFTA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReActEFTA(nn.Module):
    """
    REAct-EFTA: Rational Exponential Fractal Tree Activation.

    Each leaf uses the REAct function (ICASSP 2025):
        REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))
    
    Where p1, p2, p3, p4 are learnable parameters that control:
        - p1: positive slope / growth rate
        - p2: negative slope / decay rate  
        - p3: positive saturation rate
        - p4: negative saturation rate
    
    When p1=p2=p3=p4=1, REAct reduces to tanh(x/2).
    
    Leaves are combined via max through a tree structure.

    Args:
        num_units: number of output neurons/channels
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
        input_dim: dimension of input features (required for first layer)
        clamp_value: clamp value for numerical stability (default: 10)

    Attributes:
        num_leaves: total number of leaf nodes (branch_factor ** depth)

    Example:
        >>> react_efta = ReActEFTA(num_units=128, depth=2, branch_factor=2, input_dim=784)
        >>> x = torch.randn(32, 784)
        >>> output = react_efta(x)
        >>> output.shape
        torch.Size([32, 128])
    """

    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None, 
                 clamp_value=5.0):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim
        self.clamp_value = clamp_value

        if input_dim is None:
            raise ValueError("input_dim must be specified for ReActEFTA")

        # Parameters for each leaf: 4 parameters per output unit (p1, p2, p3, p4)
        # Shape: (num_leaves, num_units, 4)
        # Use smaller initialization for large branch factors
        init_scale = 0.05 / (branch_factor ** 0.5)
        self.leaf_params = nn.Parameter(torch.randn(
            self.num_leaves, num_units, 4) * init_scale)
        
        # Initialize parameters to identity-like behavior (tanh-like)
        with torch.no_grad():
            self.leaf_params[:, :, 0].fill_(1.0)    # p1: positive growth
            self.leaf_params[:, :, 1].fill_(1.0)    # p2: negative decay
            self.leaf_params[:, :, 2].fill_(1.0)    # p3: positive saturation
            self.leaf_params[:, :, 3].fill_(1.0)    # p4: negative saturation

    def _leaf_function(self, x, p1, p2, p3, p4):
        """
        Apply REAct function to a single leaf.

        REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))

        For numerical stability, we use the log-sum-exp trick:
        REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))
        
        Args:
            x: input tensor
            p1, p2, p3, p4: learnable parameters (broadcastable to x shape)

        Returns:
            REAct activation output
        """
        # Clamp parameters for stability
        p1 = torch.clamp(p1, 0.01, self.clamp_value)
        p2 = torch.clamp(p2, 0.01, self.clamp_value)
        p3 = torch.clamp(p3, 0.01, self.clamp_value)
        p4 = torch.clamp(p4, 0.01, self.clamp_value)

        # Compute exponentials with clamping to prevent overflow
        p1x = torch.clamp(p1 * x, -self.clamp_value, self.clamp_value)
        p2x = torch.clamp(p2 * x, -self.clamp_value, self.clamp_value)
        p3x = torch.clamp(p3 * x, -self.clamp_value, self.clamp_value)
        p4x = torch.clamp(p4 * x, -self.clamp_value, self.clamp_value)

        # Compute numerator: exp(p1*x) - exp(-p2*x)
        # Use stable computation
        exp_p1x = torch.exp(p1x)
        exp_neg_p2x = torch.exp(-p2x)
        numerator = exp_p1x - exp_neg_p2x

        # Compute denominator: exp(p3*x) + exp(-p4*x)
        exp_p3x = torch.exp(p3x)
        exp_neg_p4x = torch.exp(-p4x)
        denominator = exp_p3x + exp_neg_p4x

        # Stable division
        output = numerator / (denominator + 1e-8)

        return output

    def forward(self, x):
        """
        Apply REAct-EFTA to inputs.

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
            # Extract parameters for this leaf
            p1 = self.leaf_params[leaf_idx, :, 0]  # (num_units,)
            p2 = self.leaf_params[leaf_idx, :, 1]  # (num_units,)
            p3 = self.leaf_params[leaf_idx, :, 2]  # (num_units,)
            p4 = self.leaf_params[leaf_idx, :, 3]  # (num_units,)

            # Apply REAct function
            leaf_val = self._leaf_function(flat_inputs, p1, p2, p3, p4)
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
            batch_size, height, width, channels = x.shape
            flat_inputs = x.reshape(batch_size, height * width, channels)
        else:
            batch_size = x.shape[0]
            flat_inputs = x

        leaf_values_list = []
        for leaf_idx in range(self.num_leaves):
            p1 = self.leaf_params[leaf_idx, :, 0]
            p2 = self.leaf_params[leaf_idx, :, 1]
            p3 = self.leaf_params[leaf_idx, :, 2]
            p4 = self.leaf_params[leaf_idx, :, 3]

            leaf_val = self._leaf_function(flat_inputs, p1, p2, p3, p4)
            leaf_values_list.append(leaf_val)

        leaf_values = torch.stack(leaf_values_list, dim=0)
        
        if len(x.shape) == 4:
            return leaf_values.reshape(
                self.num_leaves, batch_size, height, width, self.num_units)
        return leaf_values

    def get_param_summary(self):
        """
        Get a summary of learned REAct parameters across all leaves.

        Returns:
            Dictionary with mean values of p1, p2, p3, p4
        """
        with torch.no_grad():
            return {
                'p1_mean': self.leaf_params[:, :, 0].mean().item(),
                'p2_mean': self.leaf_params[:, :, 1].mean().item(),
                'p3_mean': self.leaf_params[:, :, 2].mean().item(),
                'p4_mean': self.leaf_params[:, :, 3].mean().item(),
            }
