"""
Phase 1: Train a simple CNN with EFTA on MNIST.
Saves the learned EFTA leaf parameters (α, β, γ) for transfer learning.

Usage:
    python phase1_train_mnist_efta.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.activations.exponential_fta import ExponentialFTA


# ──────────────────────────────────────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────────────────────────────────────

class SimpleCNNEFTA(nn.Module):
    """Simple CNN with EFTA for MNIST (consistent with project architecture)."""

    def __init__(self, depth=2, branch_factor=2, input_channels=1, num_classes=10):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.efta1 = ExponentialFTA(num_units=32, depth=depth,
                                     branch_factor=branch_factor, input_dim=32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.efta2 = ExponentialFTA(num_units=64, depth=depth,
                                     branch_factor=branch_factor, input_dim=64)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.efta3 = ExponentialFTA(num_units=128, depth=depth,
                                     branch_factor=branch_factor, input_dim=128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, 2)
        x = self.dropout2d(x)

        # Dense
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.efta3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_efta_layers(self):
        """Return list of all EFTA layers."""
        return [self.efta1, self.efta2, self.efta3]


# ──────────────────────────────────────────────────────────────────────────────
# Training Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_efta_parameters(model, save_path):
    """
    Extract and save EFTA leaf parameters (α, β, γ) to JSON.
    
    Structure:
    {
        "efta_config": {"depth": 2, "branch_factor": 2},
        "layers": {
            "efta1": {"num_units": 32, "num_leaves": 4, "params": [...]},
            "efta2": {"num_units": 64, "num_leaves": 4, "params": [...]},
            "efta3": {"num_units": 128, "num_leaves": 4, "params": [...]}
        }
    }
    """
    efta_layers = model.get_efta_layers()
    export = {
        "efta_config": {
            "depth": model.depth,
            "branch_factor": model.branch_factor,
            "num_leaves": model.branch_factor ** model.depth
        },
        "layers": {}
    }

    for i, efta in enumerate(efta_layers):
        layer_name = f"efta{i+1}"
        num_units = efta.num_units
        num_leaves = efta.num_leaves

        # Extract parameters: shape (num_leaves, num_units, 3)
        params = efta.leaf_params.detach().cpu().numpy()

        # Compute summary statistics
        alpha = params[:, :, 0]
        beta = params[:, :, 1]
        gamma = params[:, :, 2]

        layer_export = {
            "num_units": num_units,
            "num_leaves": num_leaves,
            "param_shape": list(params.shape),  # [num_leaves, num_units, 3]
            "statistics": {
                "alpha": {
                    "mean": float(np.mean(alpha)),
                    "std": float(np.std(alpha)),
                    "min": float(np.min(alpha)),
                    "max": float(np.max(alpha))
                },
                "beta": {
                    "mean": float(np.mean(beta)),
                    "std": float(np.std(beta)),
                    "min": float(np.min(beta)),
                    "max": float(np.max(beta))
                },
                "gamma": {
                    "mean": float(np.mean(gamma)),
                    "std": float(np.std(gamma)),
                    "min": float(np.min(gamma)),
                    "max": float(np.max(gamma))
                }
            },
            # Save full parameters as nested lists for reloading
            "params": params.tolist()
        }
        export["layers"][layer_name] = layer_export

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"\nEFTA parameters saved to: {save_path}")
    print(f"  Config: depth={export['efta_config']['depth']}, k={export['efta_config']['branch_factor']}")
    for layer_name, layer_data in export["layers"].items():
        stats = layer_data["statistics"]
        print(f"  {layer_name} ({layer_data['num_units']} units, {layer_data['num_leaves']} leaves):")
        print(f"    α: mean={stats['alpha']['mean']:.4f}, std={stats['alpha']['std']:.4f}")
        print(f"    β: mean={stats['beta']['mean']:.4f}, std={stats['beta']['std']:.4f}")
        print(f"    γ: mean={stats['gamma']['mean']:.4f}, std={stats['gamma']['std']:.4f}")

    return export


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration ──
    config = {
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "depth": 2,
        "branch_factor": 2,
        "seed": 42,
        "patience": 7,  # early stopping
        "save_dir": "./outputs/mnist_efta_phase1"
    }

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 70)
    print("PHASE 1: Train CNN+EFTA on MNIST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"EFTA config: depth={config['depth']}, k={config['branch_factor']}")
    print(f"Leaves: {config['branch_factor'] ** config['depth']}")
    print(f"Training: {config['epochs']} epochs, lr={config['lr']}, batch={config['batch_size']}")
    print()

    # ── Data Loading ──
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./datasets', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./datasets', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=0
    )

    # ── Model Setup ──
    print("\nBuilding model...")
    model = SimpleCNNEFTA(
        depth=config["depth"],
        branch_factor=config["branch_factor"],
        input_channels=1,
        num_classes=10
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_test_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}. Best test acc: {best_test_acc:.4f}")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best test accuracy: {best_test_acc:.4f}")

    # ── Restore Best Model ──
    model.load_state_dict(best_state)
    _, final_acc = evaluate(model, device, test_loader, criterion)
    print(f"Final restored model test accuracy: {final_acc:.4f}")

    # ── Save EFTA Parameters ──
    print("\n" + "=" * 70)
    print("Saving Learned EFTA Parameters")
    print("=" * 70)

    params_path = os.path.join(config["save_dir"], "mnist_efta_learned_params.json")
    save_efta_parameters(model, params_path)

    # Also save full model checkpoint
    checkpoint_path = os.path.join(config["save_dir"], "mnist_efta_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_accuracy': best_test_acc,
        'history': history
    }, checkpoint_path)
    print(f"Full model checkpoint saved to: {checkpoint_path}")

    # ── Save Training History ──
    history_path = os.path.join(config["save_dir"], "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Best MNIST Test Accuracy: {best_test_acc:.4f}")
    print(f"EFTA params saved to: {params_path}")
    print(f"Model checkpoint saved to: {checkpoint_path}")
    print("\nNext step: Run phase2_train_tinyimagenet_efta.py to transfer these parameters!")


if __name__ == "__main__":
    main()
