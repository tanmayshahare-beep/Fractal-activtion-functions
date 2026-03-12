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


# ----------------------------
# 1. Maxout activation as a module
# ----------------------------
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        """
        in_features: number of input features
        out_features: number of output features (neurons)
        k: number of linear pieces per neuron
        """
        super().__init__()
        self.k = k
        # Use He initialization for better gradient flow
        self.weight = nn.Parameter(torch.randn(out_features, k, in_features) * np.sqrt(2.0 / (in_features * k)))
        self.bias   = nn.Parameter(torch.zeros(out_features, k))

    def forward(self, x):
        # x shape: (batch, in_features)
        # Compute all linear combinations: (batch, out_features, k)
        out = torch.einsum('bi,ojk->boj', x, self.weight) + self.bias.unsqueeze(0)
        # Max over the k pieces: (batch, out_features)
        out, _ = torch.max(out, dim=2)
        return out


# ----------------------------
# 2. Network with Maxout hidden layer
# ----------------------------
class MaxoutNet(nn.Module):
    def __init__(self, k=4):  # try k=4 branches per neuron
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128, bias=False)  # bias is handled inside Maxout
        self.bn1 = nn.BatchNorm1d(128)                 # batch norm for stability
        self.maxout = Maxout(128, 128, k=k)           # 128 neurons, each with k branches
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)        # batch normalization
        x = self.maxout(x)       # branching activation
        x = self.fc2(x)
        return x


# ----------------------------
# 3. Data loading (using local MNIST dataset)
# ----------------------------
dataset_dir = r'.\MNIST dataset'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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


model = MaxoutNet(k=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower LR with Adam for stability


# ----------------------------
# 4. Training & evaluation (same as baseline)
# ----------------------------
def train(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}!")
                break
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')


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
    print("=== PyTorch Maxout Model (k=4) ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    train(epochs=5)
    evaluate()
