import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_mnist_local, NumpyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

# Load data
(x_train, y_train), _ = load_mnist_local('datasets/MNIST')
print('Original data:')
print('  Shape:', x_train.shape)
print('  Dtype:', x_train.dtype)
print('  Range:', x_train.min(), '-', x_train.max())
print('  Sample pixel (first image, [0,0]):', x_train[0, 0, 0])

# Keep as float32 0-255
x_train = x_train.astype('float32')
print('\nAfter astype float32:')
print('  Range:', x_train.min(), '-', x_train.max())

# Create dataset WITHOUT transform first
dataset_raw = NumpyDataset(x_train[:10], y_train[:10], transform=None)
img_raw, label_raw = dataset_raw[0]
print('\nFrom NumpyDataset (no transform):')
print('  Type:', type(img_raw))
print('  Shape:', img_raw.shape if hasattr(img_raw, 'shape') else 'N/A')
print('  Range:', img_raw.min() if hasattr(img_raw, 'min') else 'N/A', '-', img_raw.max() if hasattr(img_raw, 'max') else 'N/A')

# Now with ToTensor transform
transform = transforms.Compose([transforms.ToTensor()])
dataset = NumpyDataset(x_train[:10], y_train[:10], transform=transform)
img, label = dataset[0]
print('\nFrom NumpyDataset (with ToTensor):')
print('  Type:', type(img))
print('  Shape:', img.shape)
print('  Range:', img.min().item(), '-', img.max().item())
print('  Mean:', img.mean().item())
print('  Label:', label)

# Check if values make sense for MNIST
# MNIST has white digits on black background
# So mean should be low (mostly black) with some high values (white digits)
print('\nExpected for MNIST:')
print('  Mean should be ~0.1-0.3 (mostly black background)')
print('  Range should be 0.0 - 1.0')
if img.mean().item() < 0.05:
    print('WARNING: Mean is too low - image might be all black!')
elif img.max().item() < 0.1:
    print('WARNING: Max is very low - data might not be scaled correctly!')
else:
    print('OK: Data looks reasonable')
