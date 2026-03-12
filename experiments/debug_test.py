"""
Debug script to test data pipeline and model forward pass.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from src.utils import load_mnist_local, NumpyDataset
from src.models import create_cnn_baseline, create_cnn_efta
from src.activations import ExponentialFTA, FractalTreeActivation
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============== TEST 1: Data Pipeline ==============
print('\n' + '='*60)
print('TEST 1: Data Pipeline')
print('='*60)

(x_train, y_train), (x_test, y_test) = load_mnist_local('datasets/MNIST')
print(f'Loaded MNIST: {len(x_train)} train, {len(x_test)} test')
print(f'Original data range: {x_train.min()} - {x_train.max()}')

x_train = x_train.astype('float32')

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = NumpyDataset(x_train[:500], y_train[:500], transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

images, labels = next(iter(loader))
print(f'After transform: {images.min().item():.3f} - {images.max().item():.3f}')
print(f'Batch shape: {images.shape}')
print(f'Labels sample: {labels[:10]}')

if images.min().item() < 0.0 or images.max().item() > 1.0:
    print('WARNING: Data not in [0, 1] range!')
else:
    print('OK: Data range')

# ============== TEST 2: ReLU Baseline ==============
print('\n' + '='*60)
print('TEST 2: ReLU Baseline Forward Pass')
print('='*60)

model_relu = create_cnn_baseline(activation='relu')
model_relu = model_relu.to(device)
model_relu.train()

params = sum(p.numel() for p in model_relu.parameters())
print(f'Model parameters: {params:,}')

test_input = images[:4].to(device)
test_labels = labels[:4].to(device)

with torch.no_grad():
    output = model_relu(test_input)

print(f'Input shape: {test_input.shape}')
print(f'Output shape: {output.shape}')
print(f'Output range: {output.min().item():.2f} - {output.max().item():.2f}')
print(f'Output sample (first row): {output[0][:5]}')

# Check if outputs are logits (not probabilities)
if output.min().item() < -1 or output.max().item() > 10:
    print('OK: Outputs are logits (good for CrossEntropyLoss)')
else:
    print('WARNING: Outputs might be probabilities (should be logits)')

# Test loss
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, test_labels)
print(f'Initial loss: {loss.item():.4f} (should be ~2.3)')

# Test backward pass - need to re-run without no_grad
model_relu.zero_grad()
output2 = model_relu(test_input)
loss2 = criterion(output2, test_labels)
loss2.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in model_relu.parameters() if p.grad is not None)**0.5
print(f'Gradient norm: {grad_norm:.4f} (should be > 0.1)')

if grad_norm < 0.001:
    print('WARNING: Gradients are vanishing!')
else:
    print('OK: Gradients')

# ============== TEST 3: EFTA Model ==============
print('\n' + '='*60)
print('TEST 3: EFTA Model Forward Pass')
print('='*60)

model_efta = create_cnn_efta(depth=2, branch_factor=2)
model_efta = model_efta.to(device)
model_efta.train()

params = sum(p.numel() for p in model_efta.parameters())
print(f'Model parameters: {params:,}')

# Test activation layer directly
print('\nTesting EFTA activation layer directly:')
test_conv_output = torch.randn(4, 32, 28, 28).to(device)  # Simulate conv output
print(f'Conv output shape: {test_conv_output.shape}')

# EFTA expects (B, H, W, C) format
test_conv_output_perm = test_conv_output.permute(0, 2, 3, 1)
print(f'Permuted to: {test_conv_output_perm.shape}')

efta_layer = ExponentialFTA(num_units=32, depth=2, branch_factor=2, input_dim=32).to(device)
with torch.no_grad():
    efta_out = efta_layer(test_conv_output_perm)
print(f'EFTA output shape: {efta_out.shape}')
print('OK: EFTA layer works')

# Test full model
with torch.no_grad():
    output_efta = model_efta(test_input)

print(f'\nFull model output shape: {output_efta.shape}')
print(f'Output range: {output_efta.min().item():.2f} - {output_efta.max().item():.2f}')

loss_efta = criterion(output_efta, test_labels)
print(f'Initial loss: {loss_efta.item():.4f}')

# ============== TEST 4: Mini Training ==============
print('\n' + '='*60)
print('TEST 4: Mini Training (2 epochs on 500 samples)')
print('='*60)

model = create_cnn_baseline(activation='relu').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_images, batch_labels in loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()
    
    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%')

print('\n' + '='*60)
print('DIAGNOSIS COMPLETE')
print('='*60)
print('If all tests passed, the model and data pipeline are working correctly.')
print('Run train_efta.py for full EFTA training.')
