import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_mnist_local
import numpy as np

# Load data
(x_train, y_train), _ = load_mnist_local('datasets/MNIST')

print('Checking MNIST data...')
print('Shape:', x_train.shape)

# Check first few images
for i in range(3):
    img = x_train[i]
    print(f'\nImage {i} (label {y_train[i]}):')
    print(f'  Corner pixels: TL={img[0,0]}, TR={img[0,-1]}, BL={img[-1,0]}, BR={img[-1,-1]}')
    print(f'  Center pixels: {img[14,14]}')
    print(f'  Mean: {img.mean():.1f}')
    print(f'  Range: {img.min()} - {img.max()}')

# MNIST should have:
# - Black background (corners should be 0 or near 0)
# - White digits (center should have high values)
# If corners are 255 and center is 0, the image is inverted

print('\n--- DIAGNOSIS ---')
first_img = x_train[0]
corner_mean = (first_img[0,0] + first_img[0,-1] + first_img[-1,0] + first_img[-1,-1]) / 4
center = first_img[14, 14]

print(f'Corner mean: {corner_mean:.1f}')
print(f'Center: {center:.1f}')

if corner_mean > 200 and center < 50:
    print('INVERTED: White background, black digits')
    print('FIX: Use 255 - x to invert back to standard MNIST')
elif corner_mean < 50 and center > 100:
    print('CORRECT: Black background, white digits')
else:
    print('UNCLEAR: Check images manually')
