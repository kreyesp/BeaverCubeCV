import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import random_split
import torch.nn.functional as F
from model import SmallCloudNet
from dataset import CloudSEN12Dataset
from config import (
    SAVE_DIR, CHECKPOINT, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    NUM_CLASSES, CLASS_NAMES,
)

LABEL_COLORS = {
    0: (0.15, 0.15, 0.15),
    1: (0.8,  0.8,  1.0),
    2: (1.0,  1.0,  1.0),
    3: (0.4,  0.7,  1.0),
}

def label_to_rgb(lbl_hw):
    out = np.zeros((*lbl_hw.shape, 3), dtype=np.float32)
    for cls, col in LABEL_COLORS.items():
        out[lbl_hw == cls] = col
    return out

def to_display(arr_hwc, percentile=98):
    hi = np.percentile(arr_hwc, percentile)
    return np.clip(arr_hwc / (hi + 1e-8), 0, 1)


# ── Load dataset and grab 2 random test samples ───────────────────────────────
full_dataset = CloudSEN12Dataset(add_noise=False)
total      = len(full_dataset)
train_size = int(TRAIN_RATIO * total)
val_size   = int(VAL_RATIO   * total)
test_size  = total - train_size - val_size

_, _, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED),
)

sample_indices = random.sample(range(len(test_dataset)), 2)
print(f"Visualising test samples at positions: {sample_indices}")

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SmallCloudNet(in_ch=3, num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(val_loss={checkpoint['val_loss']:.4f})")

# ── Run inference ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
col_titles = ["Input (simulated BlueFOX)", "Ground Truth", "Prediction"]

for row, sample_idx in enumerate(sample_indices):
    img_tensor, mask_tensor = test_dataset[sample_idx]

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        logits = F.interpolate(logits, size=img_tensor.shape[1:], mode="bilinear", align_corners=False)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    img  = img_tensor.numpy().transpose(1, 2, 0)
    mask = mask_tensor.numpy()
    mask[mask == -1] = 0

    axes[row, 0].imshow(to_display(img))
    axes[row, 1].imshow(label_to_rgb(mask))
    axes[row, 2].imshow(label_to_rgb(pred))

    for col in range(3):
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=10)
    axes[row, 0].set_ylabel(f"sample {sample_idx}", fontsize=9)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=col, edgecolor="gray", label=CLASS_NAMES[cls])
    for cls, col in LABEL_COLORS.items()
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=9, framealpha=0.8)

plt.suptitle(f"Inference on random test samples — val_mean_iou: {checkpoint.get('val_mean_iou', 'N/A'):.4f}", fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("inference_viz.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: inference_viz.png")