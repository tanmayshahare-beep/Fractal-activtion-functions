import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import idx2numpy

# Set device to CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


class MNISTLocalDataset(torch.utils.data.Dataset):
    """Custom dataset to load MNIST from local idx files."""
    
    def __init__(self, images_path, labels_path, transform=None):
        # Read idx files
        images = idx2numpy.convert_from_file(images_path)
        labels = idx2numpy.convert_from_file(labels_path)
        
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# 1. Set up data loading with local MNIST dataset
dataset_dir = r'.\MNIST dataset'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = MNISTLocalDataset(
    images_path=os.path.join(dataset_dir, 'train-images.idx3-ubyte'),
    labels_path=os.path.join(dataset_dir, 'train-labels.idx1-ubyte'),
    transform=transform
)

test_dataset = MNISTLocalDataset(
    images_path=os.path.join(dataset_dir, 't10k-images.idx3-ubyte'),
    labels_path=os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. Define the baseline network
class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)          # flatten
        x = self.sigmoid(self.fc1(x))   # hidden layer with sigmoid
        x = self.fc2(x)                  # output layer (logits)
        return x


model = BaselineNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 3. Training loop
def train(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')


# 4. Evaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    print("=== PyTorch Baseline Model ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    train(epochs=5)
    evaluate()
