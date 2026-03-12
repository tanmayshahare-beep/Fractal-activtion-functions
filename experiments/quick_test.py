import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import create_cnn_baseline
from src.utils import load_mnist_local, NumpyDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda')
print('Device:', device)

# Load full data
(x_train, y_train), _ = load_mnist_local('datasets/MNIST')
x_train = x_train.astype('float32')

transform = transforms.Compose([transforms.ToTensor()])
dataset = NumpyDataset(x_train, y_train, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = create_cnn_baseline(activation='relu').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print('Training ReLU baseline on full MNIST...')
for epoch in range(5):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
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
    print(f'Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%')

print('DONE - if Acc > 95%, model works correctly')
