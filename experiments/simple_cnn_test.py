import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_mnist_local, NumpyDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Simple known-working CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device('cuda')
model = SimpleCNN().to(device)
print('SimpleCNN params:', sum(p.numel() for p in model.parameters()))

# Load data - keep as uint8!
(x_train, y_train), _ = load_mnist_local('datasets/MNIST')
# DO NOT convert to float32 - PIL doesn't handle it correctly
transform = transforms.Compose([transforms.ToTensor()])
dataset = NumpyDataset(x_train, y_train, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print('Training SimpleCNN on MNIST...')
for epoch in range(5):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    print(f'Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%')

print('DONE - SimpleCNN should reach >95% after 5 epochs')
