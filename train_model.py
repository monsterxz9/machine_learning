"""训练入口:统一模型 + Fourier 频率编码 (PyTorch)。"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    GEO_DIM,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    NORM_STATS_PATH,
    NUM_FOURIER,
    SEED,
    VAL_RATIO,
    WEIGHT_DECAY,
)
from model import MicrostripMLP
from preprocessing import build_datasets


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopper:
    """监控 val_loss,patience 个 epoch 没改善就停,并保留 best weights。"""

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for geo, freq, target in loader:
        geo = geo.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(geo, freq)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        bs = geo.size(0)
        total += loss.item() * bs
        n += bs
    return total / n


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for geo, freq, target in loader:
        geo = geo.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(geo, freq)
        bs = geo.size(0)
        total += loss_fn(pred, target).item() * bs
        n += bs
    return total / n


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    print("Loading data...")
    train_ds, val_ds, stats = build_datasets(seed=SEED, val_ratio=VAL_RATIO)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MicrostripMLP(
        geo_dim=GEO_DIM,
        num_fourier=NUM_FOURIER,
        hidden_layers=HIDDEN_LAYERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)

    history = {"train_loss": [], "val_loss": []}
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if stopper.step(val_loss, model):
            print(f"Early stop @ epoch {epoch} (best val={stopper.best:.6f})")
            break

    stopper.restore(model)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": model.export_config()},
        MODEL_PATH,
    )
    np.savez(NORM_STATS_PATH, **stats)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved stats: {NORM_STATS_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Training Curve")
    plt.tight_layout()
    curve_path = os.path.join(MODEL_DIR, "training_curve.png")
    plt.savefig(curve_path)
    print(f"Saved curve: {curve_path}")


if __name__ == "__main__":
    main()
