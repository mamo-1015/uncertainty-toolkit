"""
demo/run_demo.py
----------------
End-to-end demonstration on Fashion-MNIST — Parts 1, 2, and 3.

Steps
~~~~~
1. Download Fashion-MNIST (cached after first run).
2. Train a small ResNet classifier (or load a cached checkpoint).
3. Part 1 — Run EpistemicEstimator (MC Dropout) on the full test set.
4. Part 2 — Run AleatoricEstimator (augmentation disagreement) on the
            full test set.  Uses the same frozen model, no fine-tuning.
5. Part 3 — Produce all three diagnostic plots and save to
            ``visualizations/``.

Run
~~~
    python demo/run_demo.py [--epochs N] [--passes T] [--augmentations K]
                            [--batch B] [--dropout P] [--no-cache]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from uncertainty_toolkit import EpistemicEstimator, AleatoricEstimator
from uncertainty_toolkit.model import FashionMNISTResNet
from uncertainty_toolkit.visualizations import generate_all


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DATA_DIR  = Path(__file__).parent.parent / "data"
CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "fashion_mnist.pt"
VIZ_DIR   = Path(__file__).parent.parent / "visualizations"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 256):
    """Download (if needed) and return train/test DataLoaders."""
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    train_ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[data] train: {len(train_ds):,} samples   test: {len(test_ds):,} samples")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: nn.Module, loader: DataLoader, epochs: int, device: torch.device) -> None:
    """Train model in-place."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=3e-2,
        steps_per_epoch=len(loader), epochs=epochs,
    )
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        t0 = time.time()

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

        acc = correct / total
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:>2}/{epochs}  "
            f"loss={running_loss/total:.4f}  "
            f"train_acc={acc:.3f}  "
            f"lr={lr_now:.2e}  "
            f"({elapsed:.1f}s)"
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Deterministic test-set accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="R3AL.AI Uncertainty Toolkit Demo — Fashion-MNIST")
    p.add_argument("--epochs",        type=int,   default=15,  help="Training epochs (default: 15)")
    p.add_argument("--passes",        type=int,   default=50,  help="MC Dropout forward passes T (default: 50)")
    p.add_argument("--augmentations", type=int,   default=20,  help="Augmented views per sample K (default: 20)")
    p.add_argument("--batch",         type=int,   default=256, help="Batch size (default: 256)")
    p.add_argument("--dropout",       type=float, default=0.3, help="Dropout probability (default: 0.3)")
    p.add_argument("--no-cache",      action="store_true",     help="Ignore saved checkpoint and retrain")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  R3AL.AI Uncertainty Toolkit — Fashion-MNIST Demo")
    print(f"{'='*60}")
    print(f"  device        : {device}")
    print(f"  epochs        : {args.epochs}")
    print(f"  MC passes (T) : {args.passes}")
    print(f"  aug views (K) : {args.augmentations}")
    print(f"  dropout       : {args.dropout}")
    print(f"{'='*60}\n")

    # ── Step 1: Data ──────────────────────────────────────────────────
    print("[step 1/5] Loading Fashion-MNIST …")
    train_loader, test_loader = get_dataloaders(batch_size=args.batch)

    # ── Step 2: Model ─────────────────────────────────────────────────
    model = FashionMNISTResNet(num_classes=10, dropout_p=args.dropout)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CKPT_PATH.exists() and not args.no_cache:
        print(f"[step 2/5] Loading checkpoint from {CKPT_PATH}")
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        model.to(device)
    else:
        print(f"[step 2/5] Training for {args.epochs} epochs …")
        model.to(device)
        train(model, train_loader, args.epochs, device)
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  → checkpoint saved to {CKPT_PATH}")

    acc = evaluate(model, test_loader, device)
    print(f"  → deterministic test accuracy: {acc:.3f}")

    # ── Step 3: Part 1 — Epistemic uncertainty (MC Dropout) ──────────
    print(f"\n[step 3/5] Part 1 — EpistemicEstimator  (T={args.passes} passes) …")
    t0 = time.time()
    ep_estimator = EpistemicEstimator(
        model,
        n_passes=args.passes,
        device=device,
        keep_raw=False,
    )
    ep_result = ep_estimator.estimate(test_loader)
    print(f"  → elapsed               : {time.time()-t0:.1f}s")
    print(f"  → MC Dropout accuracy   : {ep_result.correct.float().mean():.3f}")
    print(f"  → mean predictive H     : {ep_result.predictive_entropy.mean():.4f}")
    print(f"  → mean mutual info (Ue) : {ep_result.mutual_information.mean():.4f}")

    ue = ep_result.mutual_information   # (N,) — epistemic uncertainty

    # ── Step 4: Part 2 — Aleatoric uncertainty (augmentation disagreement)
    print(f"\n[step 4/5] Part 2 — AleatoricEstimator  (K={args.augmentations} views) …")
    t0 = time.time()
    al_estimator = AleatoricEstimator(
        model,
        n_augmentations=args.augmentations,
        device=device,
        # Default augmentation pipeline: flip + affine + blur + colour jitter
     
    )
    al_result = al_estimator.estimate(test_loader)
    print(f"  → elapsed               : {time.time()-t0:.1f}s")
    print(f"  → Aug-disagree accuracy : {al_result.correct.float().mean():.3f}")
    print(f"  → mean aug variance (Ua): {al_result.augmentation_variance.mean():.6f}")
    print(f"  → mean aug entropy      : {al_result.mean_entropy.mean():.4f}")
    print(f"  → conditional entropy (Ua) : {al_result.conditional_entropy.mean():.4f}")
    # ua = al_result.augmentation_variance   # (N,) — aleatoric uncertainty

    ua = al_result.conditional_entropy   # (N,) — aleatoric uncertainty

  
    correct = ep_result.correct
    labels  = ep_result.labels

    # ── Step 5: Part 3 — Visualisations ──────────────────────────────
    print(f"\n[step 5/5] Part 3 — Generating diagnostic plots → {VIZ_DIR}/")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    paths = generate_all(
        ue=ue,
        ua=ua,
        correct=correct,
        labels=labels,
        output_dir=VIZ_DIR,
        class_names=FASHION_MNIST_CLASSES,
    )

    print(f"\n{'='*60}")
    print(f"  Demo complete!  {len(paths)} plots saved:")
    for p in paths:
        print(f"    • {p}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
