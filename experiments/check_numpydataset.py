import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_mnist_local, NumpyDataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Load data
(x_train, y_train), _ = load_mnist_local('datasets/MNIST')

print('Original data shape:', x_train.shape)  # (60000, 28, 28, 1)
print('Original data range:', x_train.min(), '-', x_train.max())

# Check what NumpyDataset does with the data
# It calls Image.fromarray(image.squeeze(), mode='L') for (N,H,W,C) shape

img_0 = x_train[0]  # Shape: (28, 28, 1)
print('\nFirst image:')
print('  Shape:', img_0.shape)
print('  Mean before squeeze:', img_0.mean())

img_squeezed = img_0.squeeze()  # Shape: (28, 28)
print('  Shape after squeeze:', img_squeezed.shape)
print('  Mean after squeeze:', img_squeezed.mean())

# Convert to PIL
pil_img = Image.fromarray(img_squeezed.astype(np.uint8), mode='L')
print('\nPIL Image:')
print('  Mode:', pil_img.mode)
print('  Size:', pil_img.size)  # Should be (28, 28)

# Convert back to numpy to check
pil_as_np = np.array(pil_img)
print('  PIL as numpy - shape:', pil_as_np.shape)
print('  PIL as numpy - mean:', pil_as_np.mean())
print('  PIL as numpy - range:', pil_as_np.min(), '-', pil_as_np.max())

# Apply ToTensor
transform = transforms.ToTensor()
tensor = transform(pil_img)
print('\nAfter ToTensor():')
print('  Shape:', tensor.shape)
print('  Range:', tensor.min().item(), '-', tensor.max().item())
print('  Mean:', tensor.mean().item())

# The issue: ToTensor() on PIL image with uint8 data divides by 255
# But if the PIL image was created from float data, it might not work correctly

# Let's check if the issue is with float vs uint8
print('\n--- Testing with float32 data ---')
x_train_float = x_train.astype('float32')
img_0_float = x_train_float[0]
img_squeezed_float = img_0_float.squeeze()

# When creating PIL from float, it expects values in 0-255 range
pil_img_float = Image.fromarray(img_squeezed_float, mode='L')
pil_as_np_float = np.array(pil_img_float)
print('PIL from float - mean:', pil_as_np_float.mean())

# The problem: Image.fromarray with float32 expects values 0-255,
# but saves them as-is. When ToTensor() converts, it divides by 255.
# So if original was 240, PIL stores 240.0, ToTensor makes it 240/255 = 0.94

# But wait - let me check what actually happens
tensor_float = transform(pil_img_float)
print('ToTensor from float PIL - mean:', tensor_float.mean().item())
print('ToTensor from float PIL - range:', tensor_float.min().item(), '-', tensor_float.max().item())
