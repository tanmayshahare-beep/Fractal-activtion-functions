"""
Phase 2: Freeze MNIST-learned EFTA parameters, train only conv/dense weights on TinyImageNet.

HYPOTHESIS: EFTA activation functions learned on simple MNIST data capture general
nonlinearities that transfer to complex datasets. By freezing α, β, γ parameters,
we dramatically reduce trainable parameters while maintaining near-normal accuracy.

Usage:
    python phase2_train_tinyimagenet_efta.py                          # compare all modes
    python phase2_train_tinyimagenet_efta.py --mode scratch            # train everything from scratch
    python phase2_train_tinyimagenet_efta.py --mode frozen             # freeze MNIST EFTA params
    python phase2_train_tinyimagenet_efta.py --mode finetune           # unfreeze, fine-tune everything
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.activations.exponential_fta import ExponentialFTA


# ──────────────────────────────────────────────────────────────────────────────
# Stronger CNN for TinyImageNet
# ──────────────────────────────────────────────────────────────────────────────

class TinyImageNetCNNEFTA(nn.Module):
    """
    Memory-efficient CNN for TinyImageNet (64x64, 200 classes).
    
    Architecture:
        Input -> Conv2D(48, 3x3) -> BN -> EFTA -> MaxPool -> Dropout
              -> Conv2D(96, 3x3) -> BN -> EFTA -> MaxPool -> Dropout
              -> Conv2D(128, 3x3) -> BN -> EFTA -> AdaptiveAvgPool -> Flatten
              -> Dense(256) -> BN -> EFTA -> Dropout
              -> Dense(200)
    """

    def __init__(self, depth=2, branch_factor=2, input_channels=3, num_classes=200):
        super().__init__()
        self.depth = depth
        self.branch_factor = branch_factor

        # Block 1: 48 channels
        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.efta1 = ExponentialFTA(num_units=48, depth=depth,
                                     branch_factor=branch_factor, input_dim=48)

        # Block 2: 96 channels
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.efta2 = ExponentialFTA(num_units=96, depth=depth,
                                     branch_factor=branch_factor, input_dim=96)

        # Block 3: 128 channels
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.efta3 = ExponentialFTA(num_units=128, depth=depth,
                                     branch_factor=branch_factor, input_dim=128)

        # Dense
        self.fc1 = nn.Linear(128, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.efta4 = ExponentialFTA(num_units=256, depth=depth,
                                     branch_factor=branch_factor, input_dim=256)
        self.fc2 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4 and x.shape[-1] <= 4 and x.shape[-1] != 64:
            x = x.permute(0, 3, 1, 2)

        # Block 1: 64x64 -> 32x32
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.efta1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        x = self.dropout2d(x)

        # Block 2: 32x32 -> 16x16
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.efta2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        x = self.dropout2d(x)

        # Block 3: 16x16 -> 1x1
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.efta3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Dense
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.efta4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_efta_layers(self):
        """Return list of all EFTA layers."""
        return [self.efta1, self.efta2, self.efta3, self.efta4]


# ──────────────────────────────────────────────────────────────────────────────
# EFTA Parameter Transfer & Freezing
# ──────────────────────────────────────────────────────────────────────────────

def freeze_efta_parameters(model, verbose=True):
    """
    Freeze all EFTA leaf parameters (α, β, γ) so they won't be trained.
    
    Returns:
        dict with freeze statistics
    """
    efta_layers = model.get_efta_layers()
    frozen_params = 0
    total_efta_params = 0

    for i, efta in enumerate(efta_layers):
        efta.leaf_params.requires_grad = False
        params_count = efta.leaf_params.numel()
        frozen_params += params_count
        total_efta_params += params_count
        
        if verbose:
            print(f"  Frozen: efta{i+1} ({efta.num_units} units, {efta.num_leaves} leaves) — {params_count:,} params")

    return {
        "frozen_params": frozen_params,
        "total_efta_params": total_efta_params
    }


def load_and_freeze_mnist_efta_params(model, params_path, verbose=True):
    """
    Load MNIST-learned EFTA parameters into the TinyImageNet model, then freeze them.
    
    Strategy:
    - For layers where num_units match (64, 128), directly copy the learned params
    - For new layers (256, 512), use statistical initialization from learned params
      (match mean and std of α, β, γ distributions)
    - The key insight: EFTA parameters capture activation shape characteristics,
      not dataset-specific features, so transfer should be meaningful
    - After loading, freeze all leaf_params so they won't be updated during training
    
    Returns:
        dict with transfer statistics
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"EFTA params not found: {params_path}")

    with open(params_path, 'r') as f:
        mnist_data = json.load(f)

    efta_layers = model.get_efta_layers()
    transfer_stats = {}

    # Map MNIST layers to TinyImageNet layers
    # MNIST: efta1(32), efta2(64), efta3(128)
    # TinyImageNet: efta1(48), efta2(96), efta3(128), efta4(256)
    mnist_layer_names = list(mnist_data["layers"].keys())

    for i, (target_efta, target_name) in enumerate(zip(efta_layers, model.get_efta_names())):
        target_units = target_efta.num_units
        target_leaves = target_efta.num_leaves
        target_params = target_efta.leaf_params.detach().cpu().numpy()

        # Find best MNIST source layer
        # Direct match by units count
        source_name = None
        source_data = None
        for src_name in mnist_layer_names:
            src_units = mnist_data["layers"][src_name]["num_units"]
            if src_units == target_units:
                source_name = src_name
                source_data = mnist_data["layers"][src_name]
                break

        if source_data is not None:
            # Direct parameter copy (same num_units)
            src_params = np.array(source_data["params"])
            # src_params shape: (num_leaves_src, num_units, 3)
            # target_params shape: (num_leaves_tgt, num_units, 3)

            if src_params.shape[0] == target_params.shape[0]:
                # Same number of leaves - direct copy
                target_efta.leaf_params.data = torch.tensor(
                    src_params, dtype=torch.float32, device=target_efta.leaf_params.device
                )
                transfer_stats[target_name] = {
                    "method": "direct_copy",
                    "source": source_name,
                    "num_units": target_units,
                    "num_leaves": target_leaves
                }
                if verbose:
                    print(f"  {target_name} ({target_units} units): Direct copy from {source_name}")
            else:
                # Different leaves - replicate or truncate
                if src_params.shape[0] < target_params.shape[0]:
                    # Replicate source leaves
                    repeats = target_leaves // src_params.shape[0] + 1
                    src_expanded = np.tile(src_params, (repeats, 1, 1))[:target_leaves]
                else:
                    # Truncate
                    src_expanded = src_params[:target_leaves]
                
                target_efta.leaf_params.data = torch.tensor(
                    src_expanded, dtype=torch.float32, device=target_efta.leaf_params.device
                )
                transfer_stats[target_name] = {
                    "method": "replicate" if src_params.shape[0] < target_params.shape[0] else "truncate",
                    "source": source_name,
                    "src_leaves": src_params.shape[0],
                    "tgt_leaves": target_leaves,
                    "num_units": target_units
                }
                if verbose:
                    print(f"  {target_name} ({target_units} units): {transfer_stats[target_name]['method']} from {source_name}")
        else:
            # No direct match - use statistical initialization from all MNIST params
            all_alphas, all_betas, all_gammas = [], [], []
            for src_name in mnist_layer_names:
                src_params = np.array(mnist_data["layers"][src_name]["params"])
                all_alphas.extend(src_params[:, :, 0].flatten())
                all_betas.extend(src_params[:, :, 1].flatten())
                all_gammas.extend(src_params[:, :, 2].flatten())

            # Match distribution
            alpha_mean, alpha_std = np.mean(all_alphas), np.std(all_alphas)
            beta_mean, beta_std = np.mean(all_betas), np.std(all_betas)
            gamma_mean, gamma_std = np.mean(all_gammas), np.std(all_gammas)

            new_params = np.zeros_like(target_params)
            new_params[:, :, 0] = np.random.normal(alpha_mean, alpha_std, 
                                                    (target_leaves, target_units))
            new_params[:, :, 1] = np.random.normal(beta_mean, beta_std,
                                                    (target_leaves, target_units))
            new_params[:, :, 2] = np.random.normal(gamma_mean, gamma_std,
                                                    (target_leaves, target_units))

            target_efta.leaf_params.data = torch.tensor(
                new_params, dtype=torch.float32, device=target_efta.leaf_params.device
            )
            transfer_stats[target_name] = {
                "method": "statistical_init",
                "source": "all_mnist_layers",
                "num_units": target_units,
                "alpha_mean": float(alpha_mean),
                "beta_mean": float(beta_mean),
                "gamma_mean": float(gamma_mean)
            }
            if verbose:
                print(f"  {target_name} ({target_units} units): Statistical init from all MNIST layers")

    # ── Freeze all EFTA parameters ──
    frozen_params = 0
    total_efta_params = 0
    for efta in model.get_efta_layers():
        efta.leaf_params.requires_grad = False
        frozen_params += efta.leaf_params.numel()
        total_efta_params += efta.leaf_params.numel()

    transfer_stats["frozen"] = True
    transfer_stats["frozen_params"] = frozen_params
    transfer_stats["total_efta_params"] = total_efta_params

    if verbose:
        print(f"\n  ✓ Froze {frozen_params:,} EFTA parameters ({total_efta_params:,} total in EFTA layers)")
        print(f"  These will NOT be updated during training.")

    return transfer_stats


# Patch get_efta_names onto the model class
def get_efta_names(self):
    return ["efta1", "efta2", "efta3", "efta4"]

TinyImageNetCNNEFTA.get_efta_names = get_efta_names


# ──────────────────────────────────────────────────────────────────────────────
# Training Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, device, train_loader, optimizer, criterion, grad_clip=1.0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def run_experiment(mode, params_path=None, save_dir=None, args=None):
    """
    Run a complete training experiment.

    Args:
        mode: 'scratch' (train all), 'frozen' (load+freeze MNIST EFTA), or 'finetune' (train all with MNIST init)
        params_path: path to MNIST EFTA params (for frozen/finetune modes)
        save_dir: directory to save results
        args: command-line arguments
    """
    if save_dir is None:
        save_dir = f"./outputs/tinyimagenet_efta_phase2_{mode}"
    os.makedirs(save_dir, exist_ok=True)

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 70)
    print(f"PHASE 2: TinyImageNet EFTA - Mode: {mode.upper()}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"EFTA config: depth={args.depth}, k={args.branch_factor}")
    print(f"Mode: {mode}")
    if mode in ("frozen", "finetune"):
        print(f"MNIST EFTA params from: {params_path}")
    if mode == "frozen":
        print("  → EFTA α,β,γ will be FROZEN (not trained)")
    print()

    # ── Data ──
    print("Loading TinyImageNet dataset...")
    data_dir = './datasets/tiny-imagenet-200'
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # ── Model ──
    print("\nBuilding model...")
    model = TinyImageNetCNNEFTA(
        depth=args.depth,
        branch_factor=args.branch_factor,
        input_channels=3,
        num_classes=200
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Load & Freeze EFTA Parameters (if applicable) ──
    transfer_stats = None
    if mode in ("frozen", "finetune") and params_path:
        print("\nLoading MNIST-learned EFTA parameters...")
        transfer_stats = load_and_freeze_mnist_efta_params(model, params_path)

        # For finetune mode, unfreeze after loading (they're trained from MNIST init)
        if mode == "finetune":
            print("\nFine-tune mode: unfreezing EFTA parameters for training...")
            for efta in model.get_efta_layers():
                efta.leaf_params.requires_grad = True

        # Recompute trainable params after freeze/unfreeze
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_count = num_params - trainable_after
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_after:,} (frozen: {frozen_count:,})")
        transfer_stats["trainable_params"] = trainable_after
        transfer_stats["frozen_params"] = frozen_count

        # Save transfer stats
        with open(os.path.join(save_dir, "transfer_stats.json"), 'w') as f:
            json.dump(transfer_stats, f, indent=2)

    # ── Optimizer & Scheduler ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, args.grad_clip
        )
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f}")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # ── Restore Best Model ──
    model.load_state_dict(best_state)
    _, final_acc = evaluate(model, device, val_loader, criterion)
    print(f"Final restored model val accuracy: {final_acc:.4f}")

    # ── Save EFTA Parameters ──
    print(f"\nSaving EFTA parameters (after fine-tuning)...")
    from phase1_train_mnist_efta import save_efta_parameters
    
    params_save_path = os.path.join(save_dir, "tinyimagenet_efta_params.json")
    save_efta_parameters(model, params_save_path)

    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, "tinyimagenet_efta_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'mode': mode,
        'best_accuracy': best_val_acc,
        'history': history
    }, checkpoint_path)

    # Save history
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to: {save_dir}")
    return best_val_acc, history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: TinyImageNet EFTA - Frozen vs Full Training")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["scratch", "frozen", "finetune", "compare"],
                        help="Training mode: scratch=fine-tune all, frozen=freeze EFTA, finetune=init from MNIST+train all, compare=all 3")
    parser.add_argument("--params_path", type=str, default=None,
                        help="Path to MNIST EFTA parameters (for frozen/finetune modes)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--branch_factor", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Default params path
    if args.params_path is None:
        args.params_path = "./outputs/mnist_efta_phase1/mnist_efta_learned_params.json"

    mode = args.mode

    if mode == "compare":
        print("\n" + "#" * 70)
        print("# COMPARISON: Scratch vs Frozen EFTA vs Fine-tune")
        print("#" * 70)

        results = {}

        # Mode 1: From scratch (train everything)
        print("\n\n")
        acc_scratch, hist_scratch = run_experiment(
            "scratch", save_dir="./outputs/tinyimagenet_efta_phase2_scratch", args=args
        )
        results["scratch"] = {"accuracy": acc_scratch}

        # Clear memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()

        # Mode 2: Frozen EFTA (load MNIST params, freeze, train only conv/dense)
        print("\n\n")
        if not os.path.exists(args.params_path):
            print(f"⚠ MNIST EFTA params not found at {args.params_path}")
            print("  Skipping frozen/finetune modes. Run phase1 first!")
        else:
            acc_frozen, hist_frozen = run_experiment(
                "frozen",
                params_path=args.params_path,
                save_dir="./outputs/tinyimagenet_efta_phase2_frozen",
                args=args
            )
            results["frozen"] = {"accuracy": acc_frozen}

            # Clear memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            # Mode 3: Fine-tune (init from MNIST, but train everything)
            print("\n\n")
            acc_finetune, hist_finetune = run_experiment(
                "finetune",
                params_path=args.params_path,
                save_dir="./outputs/tinyimagenet_efta_phase2_finetune",
                args=args
            )
            results["finetune"] = {"accuracy": acc_finetune}

            # Summary
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            print(f"{'Mode':<20} {'Val Accuracy':>12} {'vs Scratch':>12}")
            print("-" * 70)
            print(f"{'From scratch':<20} {acc_scratch:>11.4f} {'—':>12}")
            print(f"{'Frozen EFTA':<20} {acc_frozen:>11.4f} {acc_frozen-acc_scratch:>+11.4f}")
            print(f"{'Fine-tune (MNIST init)':<20} {acc_finetune:>11.4f} {acc_finetune-acc_scratch:>+11.4f}")
            print()

            # Parameter efficiency
            # Load transfer stats to get frozen param count
            with open("./outputs/tinyimagenet_efta_phase2_frozen/transfer_stats.json") as f:
                frozen_stats = json.load(f)
            frozen_count = frozen_stats.get("frozen_params", 0)
            trainable_frozen = frozen_stats.get("trainable_params", 0)

            print(f"Parameter Efficiency:")
            print(f"  From scratch:  all params trainable")
            print(f"  Frozen EFTA:   {trainable_frozen:,} trainable / {frozen_count:,} frozen = {trainable_frozen + frozen_count:,} total")
            reduction = (frozen_count / (trainable_frozen + frozen_count)) * 100
            print(f"  → {reduction:.1f}% of EFTA params NOT trained (reused from MNIST)")

            if acc_frozen >= acc_scratch * 0.98:
                print(f"\n✓ Frozen EFTA achieves near-normal accuracy ({acc_frozen/acc_scratch*100:.1f}%) while reducing trainable params!")
            else:
                gap = (acc_scratch - acc_frozen) / acc_scratch * 100
                print(f"\n✗ Frozen EFTA accuracy gap: {gap:.1f}% below scratch")
                print(f"  (EFTA learned on MNIST may not generalize well to TinyImageNet)")

    elif mode == "scratch":
        run_experiment("scratch", args=args)

    elif mode == "frozen":
        if not os.path.exists(args.params_path):
            print(f"Error: MNIST EFTA params not found at {args.params_path}")
            print("Run phase1_train_mnist_efta.py first!")
            return
        run_experiment("frozen", params_path=args.params_path, args=args)

    elif mode == "finetune":
        if not os.path.exists(args.params_path):
            print(f"Error: MNIST EFTA params not found at {args.params_path}")
            print("Run phase1_train_mnist_efta.py first!")
            return
        run_experiment("finetune", params_path=args.params_path, args=args)


if __name__ == "__main__":
    main()
