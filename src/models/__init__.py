"""
Models package for Dynamic Activation Functions research (PyTorch).

This package contains model architectures using custom activations:
- MLP: Multi-layer perceptron models
- CNN: Convolutional neural network models
"""

from .mlp import (
    create_mlp_baseline,
    create_mlp_fta,
    create_mlp_efta,
    create_mlp_maxout,
    MLPBaseline,
    MLPFTA,
    MLPEFTA,
    MLPMaxout
)

from .cnn import (
    create_cnn_baseline,
    create_cnn_fta,
    create_cnn_efta,
    create_cnn_maxout,
    CNNBaseline,
    CNNFTA,
    CNNEFTA,
    CNNMaxout
)

__all__ = [
    'create_mlp_baseline',
    'create_mlp_fta',
    'create_mlp_efta',
    'create_mlp_maxout',
    'MLPBaseline',
    'MLPFTA',
    'MLPEFTA',
    'MLPMaxout',
    'create_cnn_baseline',
    'create_cnn_fta',
    'create_cnn_efta',
    'create_cnn_maxout',
    'CNNBaseline',
    'CNNFTA',
    'CNNEFTA',
    'CNNMaxout'
]
