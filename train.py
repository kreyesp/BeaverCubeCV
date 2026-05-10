import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)
import torch.nn.functional as F
from model import SmallCloudNet
from dataset import CloudSEN12Dataset
from config import (
    BATCH_SIZE, NUM_WORKERS, PATCH_SIZE,
    NUM_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, CHECKPOINT,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
    NUM_CLASSES, CLASS_NAMES,
)


def get_splits(dataset):
    total      = len(dataset)
    train_size = int(TRAIN_RATIO * total)
    val_size   = int(VAL_RATIO   * total)
    test_size  = total - train_size - val_size
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )


def get_metrics(device):
    return MetricCollection({
        "mean_iou":     MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="macro"),
        "per_class_iou": MulticlassJaccardIndex(num_classes=NUM_CLASSES, average="none"),
        "mean_f1":      MulticlassF1Score(num_classes=NUM_CLASSES, average="macro"),
        "accuracy":     MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro"),
    }).to(device)


def run_epoch(model, loader, criterion, device, metrics, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    metrics.reset()

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)   # (B, NUM_CLASSES, H, W) -- same spatial size as input
            logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)

            if torch.isnan(loss):
                print("  WARNING: NaN loss detected, skipping batch")
                if optimizer is not None:
                    optimizer.zero_grad()
                continue

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            preds       = logits.argmax(dim=1)
            masks_clean = masks.clone()
            masks_clean[masks_clean == -1] = 0
            metrics.update(preds, masks_clean)

    avg_loss    = total_loss / len(loader)
    metric_vals = metrics.compute()
    return avg_loss, metric_vals


def print_metrics(split, loss, metric_vals):
    print(f"\n  {split} Loss:     {loss:.4f}")
    print(f"  Mean IoU:       {metric_vals['mean_iou'].item():.4f}")
    print(f"  Mean F1:        {metric_vals['mean_f1'].item():.4f}")
    print(f"  Accuracy:       {metric_vals['accuracy'].item():.4f}")
    print("  Per-class IoU:")
    for name, iou in zip(CLASS_NAMES, metric_vals["per_class_iou"]):
        print(f"    {name:<15} {iou.item():.4f}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on {PATCH_SIZE}×{PATCH_SIZE} patches")

    full_dataset                             = CloudSEN12Dataset()
    train_dataset, val_dataset, test_dataset = get_splits(full_dataset)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    # val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model     = SmallCloudNet(in_ch=3, num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    class_weights = torch.tensor([0.5, 2.0, 2.5, 3.0], device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    metrics           = get_metrics(device)
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, metrics, optimizer)
        val_loss,   val_metrics   = run_epoch(model, val_loader,   criterion, device, metrics)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} — Patience: {epochs_no_improve}/{PATIENCE}")
        print_metrics("Train", train_loss, train_metrics)
        print_metrics("Val",   val_loss,   val_metrics)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             best_val_loss,
                "val_mean_iou":         val_metrics["mean_iou"].item(),
                "val_mean_f1":          val_metrics["mean_f1"].item(),
            }, CHECKPOINT)
            print(f"\n  ✓ Saved new best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement.")
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device, metrics)
    print_metrics("Test", test_loss, test_metrics)
    print(f"\nTraining complete. Best model saved to {CHECKPOINT}")


if __name__ == "__main__":
    train()