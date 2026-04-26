import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
import os
import random
from model import get_model
from dataset import CloudSEN12Dataset
from config import SAVE_DIR, CLASS_NAMES, SEED, CAMERA_H, CAMERA_W, CHECKPOINT, NUM_CLASSES

COLORS = {
    -1: [0.2, 0.2, 0.2],   # no-data — dark grey
     0: [0.2, 0.6, 0.2],   # clear — green
     1: [0.9, 0.9, 0.9],   # thick cloud — white
     2: [0.6, 0.8, 1.0],   # thin cloud — light blue
     3: [0.4, 0.4, 0.4],   # shadow — dark grey
}


def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for val, color in COLORS.items():
        mask = label == val
        rgb[mask] = color
    return rgb


def run_inference(n_samples=6, seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load best checkpoint
    model = get_model().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    # Pick random samples from disk
    img_dir = os.path.join(SAVE_DIR, "images")
    lbl_dir = os.path.join(SAVE_DIR, "labels")
    random.seed(seed)
    files   = os.listdir(img_dir)
    chosen  = random.sample(files, n_samples)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    fig.suptitle("CloudSEN12 — RGB | Ground Truth | Prediction", fontsize=14)

    for i, fname in enumerate(chosen):
        # Load
        img   = np.load(os.path.join(img_dir, fname))            # uint8  (H, W, 3)
        label = np.load(os.path.join(lbl_dir, fname)).astype(np.int64)
        label[label == 99] = -1

        # Prepare tensor for model
        img_tensor = torch.tensor(img.astype(np.float32) / 255.0)  # (H, W, 3)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            outputs = model(pixel_values=img_tensor)
            logits  = F.interpolate(
                outputs.logits,
                size=(CAMERA_H, CAMERA_W),
                mode="bilinear",
                align_corners=False
            )
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{fname} — RGB")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_to_rgb(label))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(label_to_rgb(pred))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS[-1], label="No-data"),
        mpatches.Patch(color=COLORS[0],  label="Clear"),
        mpatches.Patch(color=COLORS[1],  label="Thick cloud"),
        mpatches.Patch(color=COLORS[2],  label="Thin cloud"),
        mpatches.Patch(color=COLORS[3],  label="Shadow"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=11)

    plt.tight_layout()
    plt.savefig("inference_results.png", bbox_inches="tight", dpi=150)
    print("Saved to inference_results.png")
    plt.show()


if __name__ == "__main__":
    run_inference()