"""
Enhanced REAct-EFTA: Rational Exponential Fractal Tree Activation Layer - PyTorch

This module implements an enhanced version of REAct-EFTA with:
- Improved parameter initialization strategies
- Weight decay regularization
- Support for various tree configurations
- Parameter summary and analysis tools

Each leaf uses the REAct function (ICASSP 2025):
    REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedReActEFTA(nn.Module):
    """
    Enhanced REAct-EFTA with improved initialization and regularization.

    Each leaf uses the REAct function:
        REAct(x) = (exp(p1*x) - exp(-p2*x)) / (exp(p3*x) + exp(-p4*x))
    
    Enhanced features:
        - Multiple initialization strategies
        - Weight decay regularization
        - Parameter constraints for stability
    
    Args:
        num_units: number of output neurons/channels
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
        input_dim: dimension of input features (required for first layer)
        init_strategy: 'tanh' (all=1), 'asymmetric' (p1/p2=1.5, p3/p4=0.8), 
                       'sharp' (p1/p2=2, p3/p4=1.5), 'smooth' (all=0.5)
        weight_decay: L2 regularization on leaf parameters
        clamp_value: clamp value for numerical stability
        constrain_params: if True, clamp parameters during forward pass

    Example:
        >>> react_efta = EnhancedReActEFTA(num_units=128, depth=2, branch_factor=7, 
        ...                                input_dim=784, init_strategy='asymmetric')
        >>> x = torch.randn(32, 784)
        >>> output = react_efta(x)
    """

    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None,
                 init_strategy='tanh', weight_decay=0.0, clamp_value=5.0,
                 constrain_params=True):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim
        self.init_strategy = init_strategy
        self.weight_decay = weight_decay
        self.clamp_value = clamp_value
        self.constrain_params = constrain_params

        if input_dim is None:
            raise ValueError("input_dim must be specified for EnhancedReActEFTA")

        # Parameters for each leaf: 4 parameters per output unit (p1, p2, p3, p4)
        # Shape: (num_leaves, num_units, 4)
        init_scale = 0.05 / (branch_factor ** 0.5)
        self.leaf_params = nn.Parameter(torch.randn(
            self.num_leaves, num_units, 4) * init_scale)
        
        # Initialize based on strategy
        self._initialize_parameters(init_strategy)

    def _initialize_parameters(self, strategy):
        """Initialize parameters based on strategy."""
        with torch.no_grad():
            if strategy == 'tanh':
                # All parameters = 1.0 (reduces to tanh-like)
                self.leaf_params[:, :, 0].fill_(1.0)
                self.leaf_params[:, :, 1].fill_(1.0)
                self.leaf_params[:, :, 2].fill_(1.0)
                self.leaf_params[:, :, 3].fill_(1.0)
            
            elif strategy == 'asymmetric':
                # Encourage asymmetry between positive/negative
                self.leaf_params[:, :, 0].fill_(1.5)  # p1: steeper positive
                self.leaf_params[:, :, 1].fill_(0.8)  # p2: gentler negative
                self.leaf_params[:, :, 2].fill_(1.2)  # p3: moderate saturation
                self.leaf_params[:, :, 3].fill_(0.8)  # p4: slower negative saturation
            
            elif strategy == 'sharp':
                # Sharper transition
                self.leaf_params[:, :, 0].fill_(2.0)  # p1: steep positive
                self.leaf_params[:, :, 1].fill_(2.0)  # p2: steep negative
                self.leaf_params[:, :, 2].fill_(1.5)  # p3: faster saturation
                self.leaf_params[:, :, 3].fill_(1.5)  # p4: faster saturation
            
            elif strategy == 'smooth':
                # Smoother, more linear initially
                self.leaf_params[:, :, 0].fill_(0.5)
                self.leaf_params[:, :, 1].fill_(0.5)
                self.leaf_params[:, :, 2].fill_(0.5)
                self.leaf_params[:, :, 3].fill_(0.5)
            
            elif strategy == 'diverse':
                # Initialize different leaves with different strategies
                for leaf_idx in range(self.num_leaves):
                    strategy_idx = leaf_idx % 4
                    if strategy_idx == 0:  # tanh-like
                        self.leaf_params[leaf_idx, :, 0].fill_(1.0)
                        self.leaf_params[leaf_idx, :, 1].fill_(1.0)
                        self.leaf_params[leaf_idx, :, 2].fill_(1.0)
                        self.leaf_params[leaf_idx, :, 3].fill_(1.0)
                    elif strategy_idx == 1:  # asymmetric
                        self.leaf_params[leaf_idx, :, 0].fill_(1.5)
                        self.leaf_params[leaf_idx, :, 1].fill_(0.8)
                        self.leaf_params[leaf_idx, :, 2].fill_(1.2)
                        self.leaf_params[leaf_idx, :, 3].fill_(0.8)
                    elif strategy_idx == 2:  # sharp
                        self.leaf_params[leaf_idx, :, 0].fill_(2.0)
                        self.leaf_params[leaf_idx, :, 1].fill_(2.0)
                        self.leaf_params[leaf_idx, :, 2].fill_(1.5)
                        self.leaf_params[leaf_idx, :, 3].fill_(1.5)
                    else:  # smooth
                        self.leaf_params[leaf_idx, :, 0].fill_(0.5)
                        self.leaf_params[leaf_idx, :, 1].fill_(0.5)
                        self.leaf_params[leaf_idx, :, 2].fill_(0.5)
                        self.leaf_params[leaf_idx, :, 3].fill_(0.5)
            
            elif strategy == 'randomized':
                # Random initialization around tanh-like values
                self.leaf_params[:, :, 0].fill_(1.0)
                self.leaf_params[:, :, 1].fill_(1.0)
                self.leaf_params[:, :, 2].fill_(1.0)
                self.leaf_params[:, :, 3].fill_(1.0)
                # Add noise
                self.leaf_params += torch.randn_like(self.leaf_params) * 0.2

    def _leaf_function(self, x, p1, p2, p3, p4):
        """
        Apply REAct function with optional parameter constraints.
        """
        # Constrain parameters during forward pass if enabled
        if self.constrain_params:
            p1 = torch.clamp(p1, 0.01, self.clamp_value)
            p2 = torch.clamp(p2, 0.01, self.clamp_value)
            p3 = torch.clamp(p3, 0.01, self.clamp_value)
            p4 = torch.clamp(p4, 0.01, self.clamp_value)

        # Compute exponentials with clamping
        p1x = torch.clamp(p1 * x, -self.clamp_value, self.clamp_value)
        p2x = torch.clamp(p2 * x, -self.clamp_value, self.clamp_value)
        p3x = torch.clamp(p3 * x, -self.clamp_value, self.clamp_value)
        p4x = torch.clamp(p4 * x, -self.clamp_value, self.clamp_value)

        exp_p1x = torch.exp(p1x)
        exp_neg_p2x = torch.exp(-p2x)
        exp_p3x = torch.exp(p3x)
        exp_neg_p4x = torch.exp(-p4x)

        numerator = exp_p1x - exp_neg_p2x
        denominator = exp_p3x + exp_neg_p4x

        output = numerator / (denominator + 1e-8)
        return output

    def forward(self, x):
        """Apply REAct-EFTA to inputs."""
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            flat_inputs = x.reshape(batch_size, height * width, channels)
        else:
            flat_inputs = x
            batch_size = x.shape[0]
            height = width = None

        # Compute leaf values
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
            leaf_values = leaf_values.reshape(
                self.num_leaves, batch_size, height, width, self.num_units)

        # Recursive max combination through tree
        current = leaf_values
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            if len(x.shape) == 4:
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, height, width, self.num_units)
            else:
                current = current.reshape(
                    num_nodes, self.branch_factor, batch_size, self.num_units)
            current = torch.max(current, dim=1)[0]

        if len(x.shape) == 4:
            return current.reshape(batch_size, height, width, self.num_units)
        else:
            return current.squeeze(0)

    def get_param_summary(self):
        """Get summary statistics of learned parameters."""
        with torch.no_grad():
            return {
                'p1_mean': self.leaf_params[:, :, 0].mean().item(),
                'p1_std': self.leaf_params[:, :, 0].std().item(),
                'p2_mean': self.leaf_params[:, :, 1].mean().item(),
                'p2_std': self.leaf_params[:, :, 1].std().item(),
                'p3_mean': self.leaf_params[:, :, 2].mean().item(),
                'p3_std': self.leaf_params[:, :, 2].std().item(),
                'p4_mean': self.leaf_params[:, :, 3].mean().item(),
                'p4_std': self.leaf_params[:, :, 3].std().item(),
                'p1_min': self.leaf_params[:, :, 0].min().item(),
                'p1_max': self.leaf_params[:, :, 0].max().item(),
            }

    def get_leaf_activation_stats(self, x):
        """
        Get statistics about which leaves are most active.
        
        Returns:
            Dictionary with leaf activation frequencies and mean values
        """
        if len(x.shape) == 4:
            batch_size, height, width, channels = x.shape
            flat_inputs = x.reshape(batch_size, height * width, channels)
        else:
            flat_inputs = x
            batch_size = x.shape[0]

        leaf_values_list = []
        for leaf_idx in range(self.num_leaves):
            p1 = self.leaf_params[leaf_idx, :, 0]
            p2 = self.leaf_params[leaf_idx, :, 1]
            p3 = self.leaf_params[leaf_idx, :, 2]
            p4 = self.leaf_params[leaf_idx, :, 3]
            leaf_val = self._leaf_function(flat_inputs, p1, p2, p3, p4)
            leaf_values_list.append(leaf_val)

        leaf_values = torch.stack(leaf_values_list, dim=0)  # (num_leaves, batch, units)
        
        # Find which leaf wins the max operation most often
        current = leaf_values
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = current.reshape(
                num_nodes, self.branch_factor, batch_size, self.num_units)
            current = torch.max(current, dim=1)[0]

        # Count how often each leaf is selected at the root
        with torch.no_grad():
            # Get argmax at each level to trace which leaf wins
            current = leaf_values
            indices_list = []
            for level in range(self.depth):
                current = current.reshape(
                    -1, self.branch_factor, batch_size * self.num_units)
                _, indices = torch.max(current, dim=1)
                indices_list.append(indices)
                current = current.gather(1, indices.unsqueeze(1)).squeeze(1)

        return {
            'leaf_values': leaf_values,
            'final_output': current,
        }

    def regularize_loss(self, loss):
        """Add L2 regularization on leaf parameters."""
        if self.weight_decay > 0:
            reg = 0.5 * self.weight_decay * torch.sum(self.leaf_params ** 2)
            return loss + reg
        return loss
