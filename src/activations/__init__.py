"""
Activation Functions Package (PyTorch)

This package contains custom activation functions for neural networks:
- FractalTreeActivation (FTA): Tree of linear branches combined via max
- ExponentialFTA (EFTA): Tree of exponential branches combined via max
- ReActEFTA: Tree of REAct (Rational Exponential) branches combined via max
- EnhancedReActEFTA: Enhanced REAct with better initialization and regularization
- MaxoutLayer: Standard maxout activation
"""

from .fractal_tree_activation import FractalTreeActivation
from .exponential_fta import ExponentialFTA
from .react_efta import ReActEFTA
from .enhanced_react_efta import EnhancedReActEFTA
from .maxout import MaxoutLayer

__all__ = [
    'FractalTreeActivation',
    'ExponentialFTA',
    'ReActEFTA',
    'EnhancedReActEFTA',
    'MaxoutLayer'
]
