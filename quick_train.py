"""
Quick training script to generate trained FTA/EFTA models for visualization.

Trains models on Wine and Adult datasets with 1 seed each,
saves checkpoints, then runs branch visualizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.datasets import load_wine, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import time

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Using device: {device}")
print(f"{'='*70}\n")


# =============================================================================
# Model Definitions
# =============================================================================

class FractalTreeActivation(nn.Module):
    """Fractal Tree Activation (FTA) for MLP."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_units, self.input_dim) * (2.0 / self.input_dim) ** 0.5)
        self.leaf_biases = nn.Parameter(torch.zeros(self.num_leaves, num_units))

    def forward(self, x):
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)
        
        stacked = torch.stack(leaf_outputs, dim=0)
        current = stacked
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = current.view(num_nodes, self.branch_factor, batch_size, self.num_units)
            current = current.max(dim=1)[0]
        
        return current.squeeze(0)
    
    def get_leaf_values(self, x):
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            leaf_val = F.linear(x, self.leaf_weights[i], self.leaf_biases[i])
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class ExponentialFTA(nn.Module):
    """Exponential Fractal Tree Activation (EFTA) for MLP."""
    def __init__(self, num_units, depth=2, branch_factor=2, input_dim=None):
        super().__init__()
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth
        self.input_dim = input_dim if input_dim else num_units
        
        self.leaf_params = nn.Parameter(torch.ones(self.num_leaves, num_units, 3))
        nn.init.constant_(self.leaf_params[:, :, 0], 1.0)
        nn.init.constant_(self.leaf_params[:, :, 1], 0.5)
        nn.init.constant_(self.leaf_params[:, :, 2], 1.0)

    def _leaf_function(self, x, alpha, beta, gamma):
        neg_mask = (x < 0).float()
        pos_mask = (x >= 0).float()
        neg_part = alpha * (torch.exp(torch.clamp(beta * x, -10, 10)) - 1.0)
        pos_part = gamma * x
        return neg_mask * neg_part + pos_mask * pos_part

    def forward(self, x):
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            alpha = self.leaf_params[i, :, 0]
            beta = self.leaf_params[i, :, 1]
            gamma = self.leaf_params[i, :, 2]
            leaf_val = self._leaf_function(x, alpha, beta, gamma)
            leaf_outputs.append(leaf_val)
        
        stacked = torch.stack(leaf_outputs, dim=0)
        current = stacked
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = current.view(num_nodes, self.branch_factor, batch_size, self.num_units)
            current = current.max(dim=1)[0]
        
        return current.squeeze(0)
    
    def get_leaf_values(self, x):
        batch_size = x.shape[0]
        leaf_outputs = []
        for i in range(self.num_leaves):
            alpha = self.leaf_params[i, :, 0]
            beta = self.leaf_params[i, :, 1]
            gamma = self.leaf_params[i, :, 2]
            leaf_val = self._leaf_function(x, alpha, beta, gamma)
            leaf_outputs.append(leaf_val)
        return torch.stack(leaf_outputs, dim=0)


class MLPWithFTA(nn.Module):
    """MLP with FTA activation."""
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(FractalTreeActivation(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class MLPWithEFTA(nn.Module):
    """MLP with EFTA activation."""
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3,
                 depth=2, branch_factor=2):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(ExponentialFTA(h_dim, depth, branch_factor, input_dim=h_dim))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Training Function
# =============================================================================

def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=100, lr=0.001, patience=15):
    """Train model and return best state."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item()
                pred = out.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        scheduler.step(100.0 - val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            pred = out.argmax(dim=1)
            test_correct += (pred == yb).sum().item()
            test_total += yb.size(0)
    
    test_acc = 100.0 * test_correct / test_total
    return test_acc, best_model_state


# =============================================================================
# Data Loading
# =============================================================================

def load_wine_data():
    """Load Wine dataset."""
    data = load_wine()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_adult_data():
    """Load Adult dataset."""
    print("Loading Adult dataset...")
    adult = fetch_openml(data_id=1590, as_frame=True)
    X, y = adult.data, adult.target
    y = (y == '>50K').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, y_train.values, X_val_proc, y_val.values, X_test_proc, y_test.values


def get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create DataLoaders."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                     torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                     torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                     torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Main Training
# =============================================================================

def train_wine_models():
    """Train FTA and EFTA on Wine dataset."""
    print("\n" + "="*70)
    print("TRAINING WINE MODELS")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_wine_data()
    train_loader, val_loader, test_loader = get_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    models = {}
    
    # Train FTA
    print("\nTraining FTA (d=2,k=2)...")
    fta_model = MLPWithFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                          depth=2, branch_factor=2).to(device)
    fta_acc, fta_state = train_model(fta_model, train_loader, val_loader, test_loader, device,
                                     epochs=150, patience=20)
    print(f"  Test Accuracy: {fta_acc:.2f}%")
    models['fta_d2k2'] = fta_state
    
    # Train EFTA
    print("\nTraining EFTA (d=2,k=2)...")
    efta_model = MLPWithEFTA(input_dim=13, hidden_dims=[64, 32], output_dim=3,
                            depth=2, branch_factor=2).to(device)
    efta_acc, efta_state = train_model(efta_model, train_loader, val_loader, test_loader, device,
                                       epochs=150, patience=20)
    print(f"  Test Accuracy: {efta_acc:.2f}%")
    models['efta_d2k2'] = efta_state
    
    # Save checkpoints
    save_dir = './outputs/wine_results'
    os.makedirs(save_dir, exist_ok=True)
    
    for name, state in models.items():
        path = f'{save_dir}/model_{name}.pt'
        torch.save({'model_state_dict': state, 'config': name}, path)
        print(f"Saved: {path}")
    
    return models, X_test, y_test


def train_adult_models():
    """Train FTA and EFTA on Adult dataset."""
    print("\n" + "="*70)
    print("TRAINING ADULT MODELS")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_adult_data()
    train_loader, val_loader, test_loader = get_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )
    
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    models = {}
    
    # Train FTA
    print("\nTraining FTA (d=2,k=2)...")
    fta_model = MLPWithFTA(input_dim=108, hidden_dims=[128, 64, 32], output_dim=2,
                          depth=2, branch_factor=2).to(device)
    fta_acc, fta_state = train_model(fta_model, train_loader, val_loader, test_loader, device,
                                     epochs=100, patience=15)
    print(f"  Test Accuracy: {fta_acc:.2f}%")
    models['fta_d2k2'] = fta_state
    
    # Train EFTA
    print("\nTraining EFTA (d=2,k=2)...")
    efta_model = MLPWithEFTA(input_dim=108, hidden_dims=[128, 64, 32], output_dim=2,
                            depth=2, branch_factor=2).to(device)
    efta_acc, efta_state = train_model(efta_model, train_loader, val_loader, test_loader, device,
                                       epochs=100, patience=15)
    print(f"  Test Accuracy: {efta_acc:.2f}%")
    models['efta_d2k2'] = efta_state
    
    # Save checkpoints
    save_dir = './outputs/adult_results'
    os.makedirs(save_dir, exist_ok=True)
    
    for name, state in models.items():
        path = f'{save_dir}/model_{name}.pt'
        torch.save({'model_state_dict': state, 'config': name}, path)
        print(f"Saved: {path}")
    
    return models, X_test, y_test


def main():
    """Train all models."""
    print("\n" + "="*70)
    print("QUICK TRAINING FOR VISUALIZATION")
    print("="*70)
    
    # Train Wine models
    wine_models, wine_X_test, wine_y_test = train_wine_models()
    
    # Train Adult models
    adult_models, adult_X_test, adult_y_test = train_adult_models()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nCheckpoints saved to:")
    print("  - ./outputs/wine_results/model_*.pt")
    print("  - ./outputs/adult_results/model_*.pt")
    print("\nNow run: python run_branch_viz.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
